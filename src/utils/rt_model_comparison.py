"""Compare hierarchical RT model variants against Lasso baselines (4-way)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import shutil
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from create_synthetic_data import create_metabolomics_data
from src.compassign.datasets import SyntheticDataset
from src.compassign.rt_hierarchical_experimental import HierarchicalRTModel
from src.utils.rt_baseline import SpeciesCompoundLassoBaseline, ClusterCompoundLassoBaseline
from src.compassign.chem_embeddings import load_chemberta_pca20

# Hardcoded path to the descriptor artifact; descriptors are always enabled here
EMBEDDINGS_PATH = REPO_ROOT / "resources" / "metabolites" / "embeddings_chemberta_pca20.parquet"


@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float
    median_ae: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare hierarchical RT model variants against Lasso baselines "
            "using a species-holdout split by default (multicell heterogeneity)."
        )
    )
    parser.add_argument("--n-compounds", type=int, default=50, help="Number of compounds")
    parser.add_argument("--n-species", type=int, default=6, help="Number of species")
    parser.add_argument(
        "--n-internal-standards",
        type=int,
        default=8,
        help="Number of run-level covariates (internal standards)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-samples", type=int, default=1000, help="Posterior draws per chain")
    parser.add_argument("--n-tune", type=int, default=1000, help="NUTS tuning steps")
    parser.add_argument("--n-chains", type=int, default=4, help="Number of MCMC chains")
    # We now compute both baselines (cluster and per-compound) by default.
    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=800,
        help="Posterior draws to use for prediction (None = all)",
    )
    # Hardening knobs (optional)
    parser.add_argument(
        "--rare-train-max",
        type=int,
        default=None,
        help="Cap train rows per rare compound (cross-run)",
    )
    parser.add_argument(
        "--rare-species-holdout",
        action="store_true",
        help="Hold out species for rare test rows when possible",
    )
    parser.add_argument(
        "--rare-test-extreme-quantile",
        type=float,
        default=None,
        help="Select rare test rows from runs far from train (quantile of min L2 distance in covariate space)",
    )
    parser.add_argument(
        "--anchor-train-frac",
        type=float,
        default=None,
        help="Fraction of anchor rows to place in train (overrides test_size for anchors)",
    )
    # Generator passthroughs
    parser.add_argument(
        "--gamma-scale",
        type=float,
        default=None,
        help="Amplify per-run gamma contribution in generator",
    )
    parser.add_argument(
        "--sigma-y", type=float, default=None, help="Override observation noise in generator"
    )
    parser.add_argument("--anchor-budget-min", type=int, default=None)
    parser.add_argument("--anchor-budget-max", type=int, default=None)
    parser.add_argument("--rare-budget-min", type=int, default=None)
    parser.add_argument("--rare-budget-max", type=int, default=None)
    parser.add_argument("--pair-radius-quantile", type=float, default=None)
    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.99,
        help="Target acceptance rate for NUTS",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rt_comparison"),
        help="Directory for comparison artefacts",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quicker MCMC settings (fewer samples/tuning steps)",
    )
    parser.add_argument(
        "--cross-species",
        action="store_true",
        help="Enable species-holdout split and species-specific gamma heterogeneity",
    )
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        median_ae=float(np.median(np.abs(y_true - y_pred))),
    )


def prepare_predictions(y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return prediction array, mask of finite entries, and missing count.

    We intentionally do not impute missing predictions so downstream metrics
    reflect true coverage. Callers can use the returned mask/count to apply
    penalties or annotations explicitly.
    """
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(pred)
    n_missing = int(np.sum(~mask))
    return pred, mask, n_missing


def hierarchical_diagnostics(
    y_true: np.ndarray, y_pred: np.ndarray, pred_std: np.ndarray
) -> Dict[str, float]:
    sigma = np.maximum(pred_std, 1e-6)
    residuals = y_true - y_pred
    z_scores = residuals / sigma
    coverage_95 = float(
        np.mean((y_true >= y_pred - 1.96 * sigma) & (y_true <= y_pred + 1.96 * sigma))
    )
    nll = float(
        np.mean(0.5 * np.log(2.0 * np.pi * sigma**2) + 0.5 * (residuals**2) / (sigma**2))
    )
    return {
        "coverage_95": coverage_95,
        "z_mean": float(np.mean(z_scores)),
        "z_std": float(np.std(z_scores)),
        "avg_interval_95_width": float(np.mean(2.0 * 1.96 * sigma)),
        "gaussian_nll": nll,
    }


def main() -> None:
    args = parse_args()
    # Timestamp and output paths early so we can tee logs immediately
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use a stable log file name so it's easy to tail
    log_path = out_dir / "rt_model_comparison.log"
    try:
        # Truncate the log at the start of each run
        with log_path.open("w", encoding="utf-8") as lf:
            lf.write("")
    except Exception:
        pass

    def _log(msg: str) -> None:
        print(msg, flush=True)
        try:
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
        except Exception:
            pass

    if args.quick:
        # Quick profile: moderate MCMC settings but still robust
        # 500 draws, 500 tune, 4 chains
        args.n_samples = 500
        args.n_tune = 500
        args.n_chains = 4
        args.target_accept = 0.99
        args.posterior_samples = min(200, args.posterior_samples)
    # Single profile only (modelling-focused, full coverage)
    # Defaults sized for laptops: ~50*6*K rows before split
    args.n_compounds = max(args.n_compounds, 50)
    args.n_species = max(args.n_species, 6)
    args.test_size = 0.2

    _log(f"Starting RT comparison @ {timestamp}")
    _log(
        f"Config: quick={bool(args.quick)}, n_compounds={args.n_compounds}, n_species={args.n_species}, IS={args.n_internal_standards}"
    )
    _log(
        f"MCMC: chains={args.n_chains}, samples={args.n_samples}, tune={args.n_tune}, target_accept={args.target_accept}"
    )
    # Ensure a README is present at the rt_comparison root each run
    try:
        root_dir = out_dir.parent
        root_dir.mkdir(parents=True, exist_ok=True)
        tpl = REPO_ROOT / "templates" / "rt_comparison_README.md"
        if tpl.exists():
            shutil.copyfile(tpl, root_dir / "README.md")
    except Exception:
        pass
    _log(
        f"MCMC: chains={args.n_chains}, samples={args.n_samples}, tune={args.n_tune}, target_accept={args.target_accept}"
    )

    _log("Generating synthetic dataset...")
    # Modelling-focused: force a dense per-(species×compound) replication internally
    fixed_k = 10
    peak_df, compound_df, _ta, _rtu, hierarchical_params = create_metabolomics_data(
        n_compounds=args.n_compounds,
        n_species=args.n_species,
        n_internal_standards=args.n_internal_standards,
        anchor_budget_min=args.anchor_budget_min,
        anchor_budget_max=args.anchor_budget_max,
        rare_budget_min=args.rare_budget_min,
        rare_budget_max=args.rare_budget_max,
        pair_radius_quantile=args.pair_radius_quantile,
        gamma_scale=args.gamma_scale,
        sigma_y_override=args.sigma_y,
        fixed_runs_per_species_compound=fixed_k,
        species_gamma_sd=(0.15 if args.cross_species else 0.0),
    )
    _log(f"Generated peaks={len(peak_df)}, compounds={len(compound_df)}")
    dataset = SyntheticDataset(
        peak_df=peak_df,
        compound_df=compound_df,
        true_assignments=_ta,
        rt_uncertainties=_rtu,
        hierarchical_params=hierarchical_params,
    )

    rt_df = dataset.peak_df[dataset.peak_df["true_compound"].notna()].copy()
    rt_df = rt_df.rename(columns={"true_compound": "compound"})
    rt_df["species"] = rt_df["species"].astype(int)
    rt_df["compound"] = rt_df["compound"].astype(int)
    if "run" not in rt_df.columns:
        raise ValueError("synthetic data must include a 'run' column")
    rt_df["run"] = rt_df["run"].astype(int)
    rt_df = rt_df[["species", "compound", "run", "rt"]]

    # Extract run metadata and simple stratified split (rich only)
    run_meta = dataset.run_meta()
    run_df = run_meta.df
    run_covariate_cols = run_meta.covariate_columns

    if args.cross_species:
        # Species holdout (multicell generalization)
        species_ids = np.sort(rt_df["species"].unique())
        rng = np.random.RandomState(args.seed)
        n_holdout = max(1, int(round(float(args.test_size) * len(species_ids))))
        test_species = np.sort(rng.choice(species_ids, size=n_holdout, replace=False))
        train_df = rt_df[~rt_df["species"].isin(test_species)].reset_index(drop=True)
        test_df = rt_df[rt_df["species"].isin(test_species)].reset_index(drop=True)
        _log(
            f"Species-holdout split -> test_species={test_species.tolist()} | train={len(train_df)}, test={len(test_df)}"
        )
    else:
        # Run holdout within each species (production-like per-species training)
        run_df = dataset.run_meta().df
        rng = np.random.RandomState(args.seed)
        holdout_runs = []
        for s in np.sort(run_df["species"].unique()):
            runs_s = np.sort(run_df.loc[run_df["species"] == s, "run"].unique())
            n_hold_s = max(1, int(round(float(args.test_size) * len(runs_s))))
            chosen = rng.choice(runs_s, size=n_hold_s, replace=False)
            holdout_runs.extend(list(chosen))
        holdout_runs = np.array(sorted(set(holdout_runs)), dtype=int)
        train_df = rt_df[~rt_df["run"].isin(holdout_runs)].reset_index(drop=True)
        test_df = rt_df[rt_df["run"].isin(holdout_runs)].reset_index(drop=True)
        _log(
            f"Run-holdout split -> heldout_runs={holdout_runs.tolist()} | train={len(train_df)}, test={len(test_df)}"
        )

    # Train hierarchical RT model

    # Load descriptor features and align to the synthetic compound count
    # Prefer generator‑aligned features if available to avoid misalignment
    Z = None
    gen_Z = (
        hierarchical_params.get("compound_features")
        if isinstance(hierarchical_params, dict)
        else None
    )
    if gen_Z is not None:
        Z = np.asarray(gen_Z, dtype=float)
        if Z.ndim != 2 or Z.shape[0] != args.n_compounds:
            _log(
                f"Generator-provided features have shape {getattr(Z, 'shape', None)}; expected ({args.n_compounds}, d). Falling back to library."
            )
            Z = None
        else:
            _log("Using generator-aligned compound features (ChemBERTa PCA).")

    if Z is None:
        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                f"Missing ChemBERTa embedding artifact at {EMBEDDINGS_PATH}.\n"
                f"Build it with: scripts/build_chem_library.sh"
            )
        emb = load_chemberta_pca20(EMBEDDINGS_PATH)
        if emb.features.shape[0] < args.n_compounds:
            raise ValueError(
                f"Embedding library too small ({emb.features.shape[0]}) for n_compounds={args.n_compounds}"
            )
        Z = emb.features[: args.n_compounds]
        _log("Using library-sliced compound features (first N rows).")

    species_idx = test_df["species"].to_numpy(dtype=int)
    compound_idx = test_df["compound"].to_numpy(dtype=int)
    run_idx = test_df["run"].to_numpy(dtype=int)

    # Hierarchical + chemistry: β descriptors ON, γ class-pooled
    _log("Fitting hierarchical (chemistry) model...")
    m_chem = HierarchicalRTModel(
        n_clusters=hierarchical_params["n_clusters"],
        n_species=args.n_species,
        n_classes=hierarchical_params["n_classes"],
        n_compounds=args.n_compounds,
        species_cluster=hierarchical_params["species_cluster"],
        compound_class=hierarchical_params["compound_class"],
        run_metadata=run_df,
        run_covariate_columns=run_covariate_cols,
        compound_features=Z,
        global_gamma=False,
    )
    m_chem.build_model(train_df)
    tr_chem = m_chem.sample(
        n_samples=args.n_samples,
        n_tune=args.n_tune,
        n_chains=args.n_chains,
        target_accept=args.target_accept,
        random_seed=args.seed,
    )
    _log("Done: hierarchical (chemistry) sampling")
    draws_chem = tr_chem.posterior["mu0"].values.flatten().shape[0]
    n_pred_samples_chem = (
        min(args.posterior_samples, draws_chem) if args.posterior_samples else None
    )
    hchem_mean, hchem_std = m_chem.predict_new(
        species_idx=species_idx,
        compound_idx=compound_idx,
        run_idx=run_idx,
        n_samples=n_pred_samples_chem,
    )

    # Hierarchical (no chemistry): β OFF, γ global
    _log("Fitting hierarchical (plain; no chemistry) model...")
    m_plain = HierarchicalRTModel(
        n_clusters=hierarchical_params["n_clusters"],
        n_species=args.n_species,
        n_classes=hierarchical_params["n_classes"],
        n_compounds=args.n_compounds,
        species_cluster=hierarchical_params["species_cluster"],
        compound_class=hierarchical_params["compound_class"],
        run_metadata=run_df,
        run_covariate_columns=run_covariate_cols,
        compound_features=None,
        include_class_hierarchy=False,
        global_gamma=True,
    )
    m_plain.build_model(train_df)
    tr_plain = m_plain.sample(
        n_samples=args.n_samples,
        n_tune=args.n_tune,
        n_chains=args.n_chains,
        target_accept=args.target_accept,
        random_seed=args.seed + 1,
    )
    _log("Done: hierarchical (plain) sampling")
    draws_plain = tr_plain.posterior["mu0"].values.flatten().shape[0]
    n_pred_samples_plain = (
        min(args.posterior_samples, draws_plain) if args.posterior_samples else None
    )
    hplain_mean, hplain_std = m_plain.predict_new(
        species_idx=species_idx,
        compound_idx=compound_idx,
        run_idx=run_idx,
        n_samples=n_pred_samples_plain,
    )

    # Train baselines:
    # - SpeciesCompoundLassoBaseline: one model per (species × compound)
    # - ClusterCompoundLassoBaseline: one model per (cluster × compound)
    _log("Fitting baselines (species×compound and cluster×compound)...")
    baseline_sc = SpeciesCompoundLassoBaseline(
        species_cluster=None,
    )
    baseline_sc.fit(train_df, run_df=run_df, covariate_columns=run_covariate_cols)
    bsc_mean = baseline_sc.predict(
        species_idx=test_df["species"].to_numpy(),
        compound_idx=test_df["compound"].to_numpy(),
        run_idx=test_df["run"].to_numpy(),
    )

    baseline_pc = ClusterCompoundLassoBaseline(
        species_cluster=hierarchical_params["species_cluster"],
    )
    baseline_pc.fit(train_df, run_df=run_df, covariate_columns=run_covariate_cols)
    bpc_mean = baseline_pc.predict(
        species_idx=test_df["species"].to_numpy(),
        compound_idx=test_df["compound"].to_numpy(),
        run_idx=test_df["run"].to_numpy(),
    )
    _log("Done: baselines fit + predict")

    y_true = test_df["rt"].to_numpy()

    # Identify finite predictions per arm (no imputation)
    hchem_pred, hchem_mask, miss_hchem = prepare_predictions(hchem_mean)
    hplain_pred, hplain_mask, miss_hplain = prepare_predictions(hplain_mean)
    bsc_pred, bsc_mask, miss_bsc = prepare_predictions(bsc_mean)
    bpc_pred, bpc_mask, miss_bpc = prepare_predictions(bpc_mean)
    total_rows = int(len(y_true))

    def metrics_oncov(pred: np.ndarray, mask: np.ndarray) -> tuple[Metrics, int]:
        if np.any(mask):
            return compute_metrics(y_true[mask], pred[mask]), int(np.sum(mask))
        return Metrics(float("nan"), float("nan"), float("nan"), float("nan")), 0

    met_hchem_cov, n_hchem_cov = metrics_oncov(hchem_pred, hchem_mask)
    met_hplain_cov, n_hplain_cov = metrics_oncov(hplain_pred, hplain_mask)
    met_bsc_cov, n_bsc_cov = metrics_oncov(bsc_pred, bsc_mask)
    met_bpc_cov, n_bpc_cov = metrics_oncov(bpc_pred, bpc_mask)

    # Intersection-only metrics (valid rows common to all arms)
    valid_all = hchem_mask & hplain_mask & bsc_mask & bpc_mask
    n_inter = int(np.sum(valid_all))

    def metrics_on_mask(pred: np.ndarray, mask: np.ndarray) -> Metrics:
        return (
            compute_metrics(y_true[mask], pred[mask])
            if np.any(mask)
            else Metrics(float("nan"), float("nan"), float("nan"), float("nan"))
        )

    met_hchem_inter = metrics_on_mask(hchem_pred, valid_all)
    met_hplain_inter = metrics_on_mask(hplain_pred, valid_all)
    met_bsc_inter = metrics_on_mask(bsc_pred, valid_all)
    met_bpc_inter = metrics_on_mask(bpc_pred, valid_all)

    cov = {
        "hierarchical_chem": {"valid": n_hchem_cov, "missing": miss_hchem},
        "hierarchical_plain": {"valid": n_hplain_cov, "missing": miss_hplain},
        "baseline_species_compound": {"valid": n_bsc_cov, "missing": miss_bsc},
        "baseline_cluster_compound": {"valid": n_bpc_cov, "missing": miss_bpc},
        "total_y": total_rows,
    }
    coverage_summary = ", ".join(
        f"{arm}: {vals['valid']}/{total_rows} valid"
        if vals["missing"] == 0
        else f"{arm}: {vals['valid']}/{total_rows} valid ({vals['missing']} missing)"
        for arm, vals in cov.items()
        if arm != "total_y"
    )
    _log(f"Prediction coverage per arm -> {coverage_summary}")
    diag_hchem = hierarchical_diagnostics(y_true, hchem_mean, hchem_std)
    diag_hplain = hierarchical_diagnostics(y_true, hplain_mean, hplain_std)

    # Rare-compound stratified metrics (threshold fixed at 2)
    train_counts = train_df.groupby("compound").size()
    support = test_df["compound"].map(train_counts).fillna(0).astype(int)
    rare_mask = support <= 2
    common_mask = ~rare_mask

    rare_mask_np = rare_mask.to_numpy()
    common_mask_np = common_mask.to_numpy()

    def _metrics_slice(
        mask_np: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray
    ) -> Dict[str, float]:
        rows = mask_np & valid_mask
        if np.any(rows):
            res = compute_metrics(y_true[rows], pred[rows])
            return {**asdict(res), "n": int(np.sum(rows))}
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "median_ae": float("nan"),
            "n": 0,
        }

    # Stratified by group using per-arm coverage (no imputation)
    stratified = {
        "rare_threshold": 2,
        "hierarchical_plain": {
            "rare": _metrics_slice(rare_mask_np, hplain_pred, hplain_mask),
            "common": _metrics_slice(common_mask_np, hplain_pred, hplain_mask),
        },
        "hierarchical_chem": {
            "rare": _metrics_slice(rare_mask_np, hchem_pred, hchem_mask),
            "common": _metrics_slice(common_mask_np, hchem_pred, hchem_mask),
        },
        "baseline_species_compound": {
            "rare": _metrics_slice(rare_mask_np, bsc_pred, bsc_mask),
            "common": _metrics_slice(common_mask_np, bsc_pred, bsc_mask),
        },
        "baseline_cluster_compound": {
            "rare": _metrics_slice(rare_mask_np, bpc_pred, bpc_mask),
            "common": _metrics_slice(common_mask_np, bpc_pred, bpc_mask),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.output_dir / f"rt_model_comparison_{timestamp}.json"
    group_plot_path = args.output_dir / f"rt_model_comparison_groups_{timestamp}.png"
    parity_plot_path = args.output_dir / f"rt_model_comparison_parity_{timestamp}.png"
    per_species_plot_path = args.output_dir / f"rt_model_comparison_per_species_{timestamp}.png"

    # Per-species metrics on coverage (no imputation)
    species_vals = np.sort(test_df["species"].unique())
    per_species: Dict[str, Any] = {}
    for s in species_vals:
        mask_s = test_df["species"].to_numpy() == s
        per_species[str(int(s))] = {
            "hierarchical_plain": asdict(
                compute_metrics(y_true[mask_s & hplain_mask], hplain_pred[mask_s & hplain_mask])
            )
            if np.any(mask_s & hplain_mask)
            else {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "median_ae": float("nan"),
            },
            "hierarchical_chem": asdict(
                compute_metrics(y_true[mask_s & hchem_mask], hchem_pred[mask_s & hchem_mask])
            )
            if np.any(mask_s & hchem_mask)
            else {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "median_ae": float("nan"),
            },
            "baseline_species_compound": asdict(
                compute_metrics(y_true[mask_s & bsc_mask], bsc_pred[mask_s & bsc_mask])
            )
            if np.any(mask_s & bsc_mask)
            else {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "median_ae": float("nan"),
            },
            "baseline_cluster_compound": asdict(
                compute_metrics(y_true[mask_s & bpc_mask], bpc_pred[mask_s & bpc_mask])
            )
            if np.any(mask_s & bpc_mask)
            else {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "median_ae": float("nan"),
            },
        }

    # Simple per-species MAE bar plot
    try:
        species_labels = [str(int(s)) for s in species_vals]
        mae_hp = [per_species[s]["hierarchical_plain"]["mae"] for s in species_labels]
        mae_hc = [per_species[s]["hierarchical_chem"]["mae"] for s in species_labels]
        mae_bsc = [per_species[s]["baseline_species_compound"]["mae"] for s in species_labels]
        mae_bpc = [per_species[s]["baseline_cluster_compound"]["mae"] for s in species_labels]
        import matplotlib.pyplot as plt
        import numpy as _np

        x = _np.arange(len(species_labels))
        w = 0.2
        fig, ax = plt.subplots(figsize=(max(6, len(species_labels) * 0.8), 4))
        ax.bar(x - 1.5 * w, mae_hp, width=w, label="hierarchical_plain")
        ax.bar(x - 0.5 * w, mae_hc, width=w, label="hierarchical_chem")
        ax.bar(x + 0.5 * w, mae_bsc, width=w, label="baseline_sp×cp")
        ax.bar(x + 1.5 * w, mae_bpc, width=w, label="baseline_cl×cp")
        ax.set_xticks(x)
        ax.set_xticklabels(species_labels)
        ax.set_ylabel("MAE (min)")
        ax.set_xlabel("Species")
        ax.set_title("Per-species MAE (on-coverage)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(per_species_plot_path)
        plt.close(fig)
    except Exception:
        pass

    report: Dict[str, Any] = {
        "config": {
            "n_compounds": args.n_compounds,
            "n_species": args.n_species,
            "n_internal_standards": args.n_internal_standards,
            "test_size": args.test_size,
            "seed": args.seed,
            "n_samples": args.n_samples,
            "n_tune": args.n_tune,
            "n_chains": args.n_chains,
            "posterior_samples": {
                "hierarchical_plain": n_pred_samples_plain,
                "hierarchical_chem": n_pred_samples_chem,
            },
            "quick": bool(args.quick),
            "split": "species_holdout" if args.cross_species else "run_holdout",
            **({"test_species": test_species.tolist()} if args.cross_species else {}),
        },
        "metrics_on_coverage": {
            "hierarchical_plain": {
                **asdict(met_hplain_cov),
                "n": n_hplain_cov,
                "missing": miss_hplain,
            },
            "hierarchical_chem": {
                **asdict(met_hchem_cov),
                "n": n_hchem_cov,
                "missing": miss_hchem,
            },
            "baseline_species_compound": {
                **asdict(met_bsc_cov),
                "n": n_bsc_cov,
                "missing": miss_bsc,
            },
            "baseline_cluster_compound": {
                **asdict(met_bpc_cov),
                "n": n_bpc_cov,
                "missing": miss_bpc,
            },
        },
        "metrics_intersection": {
            "n": n_inter,
            "hierarchical_plain": asdict(met_hplain_inter),
            "hierarchical_chem": asdict(met_hchem_inter),
            "baseline_species_compound": asdict(met_bsc_inter),
            "baseline_cluster_compound": asdict(met_bpc_inter),
        },
        "hierarchical_plain": {"diagnostics": diag_hplain},
        "hierarchical_chem": {"diagnostics": diag_hchem},
        "stratified": stratified,
        "coverage": cov,
        "per_species": per_species,
    }

    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("RT model comparison complete.")
    _log("RT model comparison complete. Key on-coverage metrics (MAE):")
    _log(
        f"  Hier (plain): {met_hplain_cov.mae:.3f} (n={n_hplain_cov}, missing={miss_hplain}); "
        f"Hier (chem): {met_hchem_cov.mae:.3f} (n={n_hchem_cov}, missing={miss_hchem})"
    )
    _log(
        f"  Baseline species×comp: {met_bsc_cov.mae:.3f} (n={n_bsc_cov}, missing={miss_bsc}); "
        f"Baseline cluster×comp: {met_bpc_cov.mae:.3f} (n={n_bpc_cov}, missing={miss_bpc})"
    )
    print(
        f"Hier (plain) MAE={met_hplain_cov.mae:.3f}, RMSE={met_hplain_cov.rmse:.3f}, R2={met_hplain_cov.r2:.3f} "
        f"[n={n_hplain_cov}, missing={miss_hplain}]"
    )
    print(
        f"Hier (chem)  MAE={met_hchem_cov.mae:.3f}, RMSE={met_hchem_cov.rmse:.3f}, R2={met_hchem_cov.r2:.3f} "
        f"[n={n_hchem_cov}, missing={miss_hchem}]"
    )
    print(
        f"Baseline species×comp MAE={met_bsc_cov.mae:.3f}, RMSE={met_bsc_cov.rmse:.3f}, R2={met_bsc_cov.r2:.3f} "
        f"[n={n_bsc_cov}, missing={miss_bsc}]"
    )
    print(
        f"Baseline cluster×comp MAE={met_bpc_cov.mae:.3f}, RMSE={met_bpc_cov.rmse:.3f}, R2={met_bpc_cov.r2:.3f} "
        f"[n={n_bpc_cov}, missing={miss_bpc}]"
    )
    print(f"Results saved to {report_path}")

    # Define arm order/colors for the remaining plots
    arms_info = [
        ("Hier (chem)", "hierarchical_chem", hchem_pred, hchem_mask),
        ("Hier (plain)", "hierarchical_plain", hplain_pred, hplain_mask),
        ("Base (species×comp)", "baseline_species_compound", bsc_pred, bsc_mask),
        ("Base (cluster×comp)", "baseline_cluster_compound", bpc_pred, bpc_mask),
    ]
    arms = [info[0] for info in arms_info]
    colors = ["#2ca02c", "#4c78a8", "#54a24b", "#e45756"]

    # Grouped MAE plot (Common only)
    group_name, mask_np = "Common", common_mask_np
    group_key = "common"
    vals: list[float] = []
    counts: list[int] = []
    for _, strat_key, *_ in arms_info:
        metrics = stratified[strat_key][group_key]
        vals.append(metrics["mae"])
        counts.append(int(metrics["n"]))
    idx = np.arange(len(arms))
    fig2, axg = plt.subplots(1, 1, figsize=(6, 4))
    bars = axg.bar(idx, vals, color=colors, alpha=0.9)
    axg.set_xticks(idx)
    axg.set_xticklabels(arms, rotation=20, ha="right")
    finite_vals = [v for v in vals if np.isfinite(v)]
    ymax = max(finite_vals) * 1.2 if finite_vals else 1.0
    if ymax <= 0:
        ymax = 1.0
    axg.set_ylim(0, ymax)
    axg.set_ylabel("MAE (min)")
    axg.set_title(f"{group_name} (n={int(mask_np.sum())})")
    axg.grid(axis="y", alpha=0.3)
    for bar_idx, b in enumerate(bars):
        val = vals[bar_idx]
        count = counts[bar_idx]
        if np.isfinite(val) and count > 0:
            axg.text(
                b.get_x() + b.get_width() / 2,
                val + ymax * 0.02,
                f"{val:.3f}\n(n={count})",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig2.tight_layout()
    fig2.savefig(group_plot_path, dpi=150)
    plt.close(fig2)
    _log(f"Grouped MAE (Common only) plot: {group_plot_path}")

    # Delta plot: ΔMAE vs Hier (chem) to make benefits obvious
    delta_plot_path = args.output_dir / f"rt_model_comparison_deltas_{timestamp}.png"
    labels_delta = [label for label, *_ in arms_info[1:]]
    colors_delta = [colors[i] for i in range(1, len(arms_info))]

    def delta_vs_hchem(
        mask_np: np.ndarray, pred: np.ndarray, mask_other: np.ndarray
    ) -> tuple[float, int]:
        shared = mask_np & mask_other & hchem_mask
        if np.any(shared):
            mae_other = compute_metrics(y_true[shared], pred[shared]).mae
            mae_hchem_shared = compute_metrics(y_true[shared], hchem_pred[shared]).mae
            return mae_other - mae_hchem_shared, int(np.sum(shared))
        return float("nan"), 0

    # Common-only deltas
    deltas: list[float] = []
    counts: list[int] = []
    for _, _, pred_other, mask_other in arms_info[1:]:
        dval, shared_n = delta_vs_hchem(common_mask_np, pred_other, mask_other)
        deltas.append(dval)
        counts.append(shared_n)
    figd, axd = plt.subplots(1, 1, figsize=(6, 4), sharey=True)
    x = np.arange(len(labels_delta))
    bars = axd.bar(x, deltas, color=colors_delta, alpha=0.9)
    axd.axhline(0.0, color="k", linewidth=1)
    axd.set_xticks(x)
    axd.set_xticklabels(labels_delta, rotation=20, ha="right")
    axd.set_ylabel("ΔMAE vs Hier (chem) [min]")
    axd.set_title("Common")
    axd.grid(axis="y", alpha=0.3)
    finite = [abs(v) for v in deltas if np.isfinite(v)]
    span = max(finite) if finite else 0.1
    axd.set_ylim(-span - 0.1, span + 0.1)
    for idx, b in enumerate(bars):
        val = deltas[idx]
        shared_n = counts[idx]
        if np.isfinite(val) and shared_n > 0:
            offset = 0.02 if val >= 0 else -0.02
            axd.text(
                b.get_x() + b.get_width() / 2,
                val + offset,
                f"{val:.3f}\n(shared n={shared_n})",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8,
            )
    figd.suptitle("ΔMAE relative to Hier (chem): positive = Hier (chem) better")
    figd.tight_layout()
    figd.savefig(delta_plot_path, dpi=150)
    plt.close(figd)
    _log(f"Delta (Common only) plot: {delta_plot_path}")

    # Additional overall metric summaries: on-coverage and intersection-only
    plot_oncov_path = args.output_dir / f"rt_model_comparison_metrics_oncov_{timestamp}.png"
    plot_inter_path = args.output_dir / f"rt_model_comparison_metrics_intersection_{timestamp}.png"

    # On-coverage
    vals_plain_cov = [met_hplain_cov.mae, met_hplain_cov.rmse, met_hplain_cov.r2]
    vals_chem_cov = [met_hchem_cov.mae, met_hchem_cov.rmse, met_hchem_cov.r2]
    vals_bsc_cov = [met_bsc_cov.mae, met_bsc_cov.rmse, met_bsc_cov.r2]
    vals_bpc_cov = [met_bpc_cov.mae, met_bpc_cov.rmse, met_bpc_cov.r2]
    series_cov = [vals_plain_cov, vals_chem_cov, vals_bsc_cov, vals_bpc_cov]
    figc, axc_list = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for i, label in enumerate(["MAE", "RMSE", "R²"]):
        axc = axc_list[i]
        vals = [series_cov[j][i] for j in range(len(series_cov))]
        x = np.arange(len(arms))
        bars = axc.bar(x, vals, color=colors, alpha=0.9)
        axc.set_xticks(x)
        axc.set_xticklabels(arms, rotation=20, ha="right")
        axc.set_title(f"{label} (on-coverage)")
        if label != "R²":
            axc.set_ylabel("Minutes")
        axc.grid(axis="y", alpha=0.3)
        finite_vals = [v for v in vals if np.isfinite(v)]
        if label != "R²":
            ymax = max(finite_vals) * 1.2 if finite_vals else 1.0
            if ymax <= 0:
                ymax = 1.0
            axc.set_ylim(0, ymax)
        for idx_bar, b in enumerate(bars):
            h = b.get_height()
            if np.isfinite(h):
                axc.text(
                    b.get_x() + b.get_width() / 2,
                    h,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            else:
                ylim = axc.get_ylim()
                axc.text(
                    b.get_x() + b.get_width() / 2,
                    0.1 * (ylim[1] - ylim[0]) + ylim[0],
                    "no data",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="0.4",
                )
                b.set_facecolor("#c7c7c7")
                b.set_edgecolor("#999999")
                b.set_alpha(0.7)
                b.set_hatch("//")
        if not any(np.isfinite(vals)):
            axc.text(
                0.5,
                0.5,
                "no valid predictions",
                transform=axc.transAxes,
                ha="center",
                va="center",
                color="0.4",
            )
    figc.tight_layout()
    figc.savefig(plot_oncov_path, dpi=150)
    plt.close(figc)
    _log(f"On-coverage metrics plot: {plot_oncov_path}")

    # Intersection-only
    vals_plain_inter = [met_hplain_inter.mae, met_hplain_inter.rmse, met_hplain_inter.r2]
    vals_chem_inter = [met_hchem_inter.mae, met_hchem_inter.rmse, met_hchem_inter.r2]
    vals_bsc_inter = [met_bsc_inter.mae, met_bsc_inter.rmse, met_bsc_inter.r2]
    vals_bpc_inter = [met_bpc_inter.mae, met_bpc_inter.rmse, met_bpc_inter.r2]
    series_inter = [vals_plain_inter, vals_chem_inter, vals_bsc_inter, vals_bpc_inter]
    figi, axi_list = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for i, label in enumerate(["MAE", "RMSE", "R²"]):
        axi = axi_list[i]
        vals = [series_inter[j][i] for j in range(len(series_inter))]
        x = np.arange(len(arms))
        bars = axi.bar(x, vals, color=colors, alpha=0.9)
        axi.set_xticks(x)
        axi.set_xticklabels(arms, rotation=20, ha="right")
        axi.set_title(f"{label} (intersection-only, n={n_inter})")
        if label != "R²":
            axi.set_ylabel("Minutes")
        axi.grid(axis="y", alpha=0.3)
        finite_vals = [v for v in vals if np.isfinite(v)]
        if label != "R²":
            ymax = max(finite_vals) * 1.2 if finite_vals else 1.0
            if ymax <= 0:
                ymax = 1.0
            axi.set_ylim(0, ymax)
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                axi.text(
                    b.get_x() + b.get_width() / 2,
                    h,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            else:
                ylim = axi.get_ylim()
                axi.text(
                    b.get_x() + b.get_width() / 2,
                    0.1 * (ylim[1] - ylim[0]) + ylim[0],
                    "no data",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="0.4",
                )
                b.set_facecolor("#c7c7c7")
                b.set_edgecolor("#999999")
                b.set_alpha(0.7)
                b.set_hatch("//")
        if not any(np.isfinite(vals)):
            note = "undefined (no shared coverage)" if n_inter == 0 else "undefined (n<2)"
            axi.text(0.5, 0.5, note, transform=axi.transAxes, ha="center", va="center", color="0.4")
    figi.tight_layout()
    figi.savefig(plot_inter_path, dpi=150)
    plt.close(figi)
    _log(f"Intersection metrics plot: {plot_inter_path}")

    # Parity plots (y_true vs prediction) for the intersection only
    # Parity uses only valid (non-missing) predictions per arm and shows n=valid count
    names = ["Hier (plain)", "Hier (chem)", "Base (species×comp)", "Base (cluster×comp)"]
    preds_raw = [hplain_mean, hchem_mean, bsc_mean, bpc_mean]
    val_masks = [np.isfinite(p) for p in preds_raw]
    fig3, axes = plt.subplots(2, 2, figsize=(8, 7))
    axes = axes.ravel()
    y_min, y_max = float(np.min(y_true)), float(np.max(y_true))
    pad = 0.05 * (y_max - y_min + 1e-6)
    lo, hi = y_min - pad, y_max + pad
    for axp, pred, mask, nm in zip(axes, preds_raw, val_masks, names):
        yv = y_true[mask]
        pv = pred[mask]
        npts = int(np.sum(mask))
        if npts > 0:
            axp.scatter(yv, pv, s=14 if npts < 200 else 8, alpha=0.6)
        axp.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        axp.set_xlim(lo, hi)
        axp.set_ylim(lo, hi)
        axp.set_title(f"{nm} (n={npts})")
        axp.set_xlabel("Observed RT")
        axp.set_ylabel("Predicted RT")
        axp.grid(alpha=0.3)
    plt.tight_layout()
    fig3.savefig(parity_plot_path, dpi=150)
    plt.close(fig3)
    _log(f"Parity plot: {parity_plot_path}")

    # Coverage plot removed for the 'rich' focus on modelling; metrics above remain for completeness.

    # (Secondary plots removed per request to reduce clutter)


if __name__ == "__main__":
    main()
