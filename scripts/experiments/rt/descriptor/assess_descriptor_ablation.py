#!/usr/bin/env python
"""
Descriptor ablation experiment: with vs without descriptors on synthetic RT.

Outputs JSON + plots with overall and stratified metrics and MAE vs nearest-anchor distance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Matplotlib safe for headless
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Add repo scripts path for generator
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_prep.create_synthetic_data import create_metabolomics_data  # noqa: E402
from src.compassign.rt_hierarchical_experimental import HierarchicalRTModel  # noqa: E402
from src.compassign.datasets import SyntheticDataset  # noqa: E402
# No cluster inference needed; use generator clusters


@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float
    median_ae: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Descriptor ablation: with vs without descriptors on synthetic RT. "
            "Aligned with other RT runners (replication, quick profile, plot-only)."
        )
    )
    # Replication and sampling
    p.add_argument("--seed", type=int, default=42, help="Base seed; reps use seed+i")
    p.add_argument("--reps", type=int, default=5, help="Number of replicates")
    p.add_argument("--draws", type=int, default=None)
    p.add_argument("--tune", type=int, default=None)
    p.add_argument("--chains", type=int, default=None)
    p.add_argument("--predict-draws", type=int, default=None, help="Cap on prediction draws")
    p.add_argument("--quick", action="store_true", help="Use draws=500, tune=500, chains=4")

    # Dataset size knobs (kept for parity with original ablation script)
    p.add_argument("--n-compounds", type=int, default=60)
    p.add_argument("--n-species", type=int, default=15)
    p.add_argument("--n-internal-standards", type=int, default=10)

    # Generator overrides (defaults aligned with other descriptor scripts)
    p.add_argument(
        "--tau-beta", type=float, default=0.30, help="Descriptor strength τβ (default 0.30)"
    )
    p.add_argument(
        "--sigma-compound",
        type=float,
        default=0.30,
        help="Residual β sd when descriptors active (default 0.30)",
    )
    p.add_argument("--sigma-y", type=float, default=None, help="Override observation noise σ_y")
    p.add_argument("--anchor-min", type=int, default=None, help="Min anchor labels per compound")
    p.add_argument("--anchor-max", type=int, default=None, help="Max anchor labels per compound")
    p.add_argument("--rare-budget-min", type=int, default=None, help="Min rare labels per compound")
    p.add_argument("--rare-budget-max", type=int, default=None, help="Max rare labels per compound")
    p.add_argument(
        "--pair-radius-quantile",
        type=float,
        default=None,
        help="Within-cluster distance quantile for neighbor pairing radius",
    )

    # Outputs and plotting
    p.add_argument("--output-dir", type=Path, default=Path("output/rt_descriptor_ablation"))
    p.add_argument(
        "--no-covariates",
        action="store_true",
        help="Ablate run covariates: disable covariate effects in the model",
    )
    p.add_argument(
        "--arms",
        type=str,
        default="all",
        choices=["all", "full", "nocov_desc", "cov_nod"],
        help=(
            "Which arms to run: 'full' (cov+desc), 'nocov_desc' (no covariates + desc), "
            "'cov_nod' (covariates + no descriptors), or 'all' (default)."
        ),
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip sampling and regenerate plots from an aggregate JSON",
    )
    p.add_argument(
        "--plot-input",
        type=Path,
        default=None,
        help="Path to descriptor_ablation_tri_arms_*.json (used with --plot-only)",
    )
    p.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Override destination for the combined plot",
    )
    return p.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    # R2 against mean baseline
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    median_ae = float(np.median(np.abs(y_true - y_pred)))
    return Metrics(mae=mae, rmse=rmse, r2=r2, median_ae=median_ae)


def hierarchical_diagnostics(
    y_true: np.ndarray, y_pred: np.ndarray, pred_std: np.ndarray
) -> Dict[str, float]:
    sigma = np.maximum(pred_std, 1e-6)
    residuals = y_true - y_pred
    z = residuals / sigma
    cov95 = float(np.mean((y_true >= y_pred - 1.96 * sigma) & (y_true <= y_pred + 1.96 * sigma)))
    return {
        "coverage_95": cov95,
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "avg_interval_95_width": float(np.mean(2.0 * 1.96 * sigma)),
    }


def split_group_aware(
    rt_df: pd.DataFrame, compound_df: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group-aware split with cross-run generalization for rare compounds.

    - Anchors: random 80/20 split for stability
    - Rare: cap train to ≤2 rows per compound; test rows must come from runs
      that are not used in the train rows for that compound (cross-run)
    - Unseen: all rows to test
    """
    grp_map = dict(zip(compound_df["compound_id"], compound_df["compound_group"]))
    comp_groups = rt_df["compound"].map(grp_map)
    unseen_mask = comp_groups == "unseen"
    rare_mask = comp_groups == "rare"
    anchor_mask = comp_groups == "anchor"

    unseen_rows = rt_df[unseen_mask]
    rare_rows = rt_df[rare_mask]
    train_rare_parts: list[pd.DataFrame] = []
    test_rare_parts: list[pd.DataFrame] = []
    for _cid, sub in rare_rows.groupby("compound"):
        sub = sub.sample(frac=1.0, random_state=seed)
        runs = sub["run"].astype(int).unique().tolist()
        # Target: ≥10 training rows overall and ~80/20 run split when possible
        if len(sub) >= 12 and len(runs) >= 3:
            # Choose ~80% of distinct runs for train, leave at least 1 run for test
            n_train_runs = max(2, int(round(0.8 * len(runs))))
            n_train_runs = min(n_train_runs, len(runs) - 1)
            rng = np.random.default_rng(seed + int(_cid))
            train_run_sel = set(rng.choice(runs, size=n_train_runs, replace=False).tolist())
            train_take = sub[sub["run"].astype(int).isin(train_run_sel)]
            # If still <10 rows in train (e.g., skewed runs), top up from remaining
            if len(train_take) < 10:
                remain_rows = sub.loc[~sub.index.isin(train_take.index)]
                need = (
                    min(10 - len(train_take), len(remain_rows) - 1) if len(remain_rows) > 1 else 0
                )
                if need > 0:
                    extra = remain_rows.sample(n=need, random_state=seed)
                    train_take = pd.concat([train_take, extra], ignore_index=True)
        else:
            # No fallback top-up; leave rare-compound train rows empty when insufficient data
            train_take = sub.iloc[0:0]

        train_runs = set(train_take["run"].astype(int).tolist())
        # Test rows must be from runs not seen in train_take
        remain = sub.loc[~sub.index.isin(train_take.index)]
        test_take = remain[~remain["run"].astype(int).isin(train_runs)]
        train_rare_parts.append(train_take)
        if len(test_take) > 0:
            test_rare_parts.append(test_take)
    train_rare = (
        pd.concat(train_rare_parts, ignore_index=True) if train_rare_parts else rare_rows.iloc[0:0]
    )
    test_rare = (
        pd.concat(test_rare_parts, ignore_index=True) if test_rare_parts else rare_rows.iloc[0:0]
    )

    anchors = rt_df[anchor_mask]
    # Stratify anchors (80/20) for stability
    rng = np.random.default_rng(seed)
    mask = anchors.index.isin(
        rng.choice(anchors.index.to_numpy(), size=int(0.8 * len(anchors)), replace=False)
    )
    train_anchor = anchors[mask]
    test_anchor = anchors[~mask]

    train_df = pd.concat([train_anchor, train_rare], ignore_index=True)
    test_df = pd.concat([test_anchor, test_rare, unseen_rows], ignore_index=True)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def nearest_anchor_distances(
    Z: np.ndarray, comp_groups: List[str], comp_idx: np.ndarray
) -> np.ndarray:
    anchors = np.where(np.array(comp_groups) == "anchor")[0]
    if anchors.size == 0:
        return np.zeros_like(comp_idx, dtype=float)
    # Compute min L2 distance to any anchor for each test compound
    dists = np.empty(len(comp_idx), dtype=float)
    for i, c in enumerate(comp_idx):
        vec = Z[int(c)]
        da = np.linalg.norm(Z[anchors] - vec[None, :], axis=1)
        dists[i] = float(da.min())
    return dists


def _bootstrap_ci_mean(
    values: List[float], *, B: int = 2000, alpha: float = 0.05, seed: int = 777
) -> Tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    clean = vals[np.isfinite(vals)]
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    if clean.size == 1:
        m = float(clean[0])
        return m, m, m
    rng = np.random.RandomState(int(seed))
    idx = rng.randint(0, clean.size, size=(int(B), clean.size))
    means = clean[idx].mean(axis=1)
    m = float(np.mean(clean))
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return m, lo, hi


def _pick_latest(json_dir: Path, pattern: str) -> Path | None:
    cand = sorted(json_dir.glob(pattern))
    return cand[-1] if cand else None


# (removed old _combined_plot that used support-based rare/nonrare)


def main() -> None:
    args = parse_args()

    # Align sampler defaults
    if args.quick:
        args.draws = 500
        args.tune = 500
        args.chains = 4
    else:
        args.draws = args.draws or 1000
        args.tune = args.tune or 1000
        args.chains = args.chains or 4

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot-only: load existing JSON and regenerate combined tri-arm plot
    if args.plot_only:
        json_path = args.plot_input or _pick_latest(out_dir, "descriptor_ablation_tri_arms_*.json")
        if not json_path:
            raise SystemExit("No tri-arm JSON found. Provide --plot-input or run sampling first.")
        with Path(json_path).open("r") as fp:
            payload = json.load(fp)
        reps = payload.get("replicates", {})

        def collect_triple(key: str):
            """Collect replicate MAEs for the three arms from a section key.

            Key can be:
              - 'overall'
              - 'by_group.rare' | 'by_group.unseen' | 'by_group.anchor'
            """
            a, b, c = [], [], []
            for rep in reps.values():
                if key == "overall":
                    sect = rep.get("overall", {})
                elif key.startswith("by_group."):
                    grp = key.split(".", 1)[1]
                    sect = rep.get("by_group", {}).get(grp, {})
                else:
                    sect = {}
                if sect:
                    a.append(float(sect.get("full", {}).get("mae", float("nan"))))
                    b.append(float(sect.get("nocov_desc", {}).get("mae", float("nan"))))
                    c.append(float(sect.get("cov_nod", {}).get("mae", float("nan"))))
            return a, b, c

        o_full, o_nocov, o_covnod = collect_triple("overall")
        r_full, r_nocov, r_covnod = collect_triple("by_group.rare")

        def plot_tri(groups, labels, output_path: Path):
            x = np.arange(len(groups))
            width = 0.22
            fig, ax = plt.subplots(figsize=(8, 4.5))
            # Fix colors per arm across all groups
            colors = {
                "full": "#1f77b4",         # blue (C0)
                "nocov_desc": "#ff7f0e",   # orange (C1)
                "cov_nod": "#2ca02c",      # green (C2)
            }
            for i, (full_vals, nocov_vals, covnod_vals) in enumerate(groups):

                def pack(vals):
                    m, lo, hi = _bootstrap_ci_mean(vals)
                    return (
                        m,
                        (m - lo if np.isfinite(lo) else np.nan),
                        (hi - m if np.isfinite(hi) else np.nan),
                    )

                m1, l1, h1 = pack(full_vals)
                m2, l2, h2 = pack(nocov_vals)
                m3, l3, h3 = pack(covnod_vals)
                ax.bar(
                    x[i] - width,
                    m1,
                    width,
                    yerr=[[l1], [h1]],
                    capsize=3,
                    label="full" if i == 0 else None,
                    color=colors["full"],
                )
                ax.bar(
                    x[i],
                    m2,
                    width,
                    yerr=[[l2], [h2]],
                    capsize=3,
                    label="no cov + desc" if i == 0 else None,
                    color=colors["nocov_desc"],
                )
                ax.bar(
                    x[i] + width,
                    m3,
                    width,
                    yerr=[[l3], [h3]],
                    capsize=3,
                    label="cov + no desc" if i == 0 else None,
                    color=colors["cov_nod"],
                )
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel("MAE")
            ax.set_title("Descriptor ablation: full vs ablations")
            ax.legend()
            fig.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=160)
            plt.close(fig)

        plot_path = args.plot_output or (out_dir / "combined_tri_mae.png")
        plot_tri(
            groups=[
                (o_full, o_nocov, o_covnod),
                (r_full, r_nocov, r_covnod),
            ],
            labels=["All Test Compounds", "Rare Compounds"],
            output_path=plot_path,
        )
        print(f"Saved combined tri-arm plot to {plot_path}")
        return

    # Select arms
    selected = ["full", "nocov_desc", "cov_nod"] if args.arms == "all" else [args.arms]

    replicates: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for i in range(int(args.reps)):
        seed_i = int(args.seed) + i
        # Generate dataset
        peak_df, compound_df, _ta, _rtu, hp = create_metabolomics_data(
            n_compounds=args.n_compounds,
            n_species=args.n_species,
            n_internal_standards=args.n_internal_standards,
            desc_tau_beta=args.tau_beta,
            desc_sigma_compound=args.sigma_compound,
            sigma_y_override=args.sigma_y,
            anchor_budget_min=args.anchor_min,
            anchor_budget_max=args.anchor_max,
            rare_budget_min=args.rare_budget_min,
            rare_budget_max=args.rare_budget_max,
            pair_radius_quantile=args.pair_radius_quantile,
            seed=seed_i,
        )
        dataset = SyntheticDataset(
            peak_df=peak_df,
            compound_df=compound_df,
            true_assignments=_ta,
            rt_uncertainties=_rtu,
            hierarchical_params=hp,
        )

        run_meta = dataset.run_meta()
        run_df = run_meta.df
        run_covariate_cols = run_meta.covariate_columns

        # Build RT observation frames
        rt_df = dataset.peak_df[dataset.peak_df["true_compound"].notna()].copy()
        rt_df = rt_df.rename(columns={"true_compound": "compound"})
        rt_df["species"] = rt_df["species"].astype(int)
        rt_df["compound"] = rt_df["compound"].astype(int)
        rt_df["run"] = rt_df["run"].astype(int)
        rt_df = rt_df[["species", "compound", "run", "rt"]]

        train_df, test_df = split_group_aware(rt_df, compound_df, seed=seed_i)

        # Descriptor features and species clusters (use generator clusters)
        Z = hp.get("compound_features")
        n_species = int(dataset.peak_df["species"].nunique())
        species_cluster = np.asarray(hp["species_cluster"], dtype=int)

        # Build model arguments; ablate covariates by supplying an empty feature matrix
        model_kwargs = dict(
            n_clusters=int(hp["n_clusters"]),
            n_species=n_species,
            n_classes=int(hp["n_classes"]),
            n_compounds=int(args.n_compounds),
            species_cluster=np.asarray(species_cluster, dtype=int),
            compound_class=np.asarray(hp["compound_class"], dtype=int),
        )
        if bool(args.no_covariates):
            n_runs = int(run_df["run"].max()) + 1
            run_species_vec = np.zeros(n_runs, dtype=int)
            idx = run_df["run"].to_numpy(dtype=int)
            run_species_vec[idx] = run_df["species"].to_numpy(dtype=int)
            model_kwargs.update(
                run_features=np.zeros((n_runs, 0), dtype=float),
                run_species=run_species_vec,
            )
        else:
            model_kwargs.update(
                run_metadata=run_df,
                run_covariate_columns=run_covariate_cols,
            )

        # Evaluate triple arms as requested
        preds: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
        test_species = test_df["species"].to_numpy(dtype=int)
        test_comp = test_df["compound"].to_numpy(dtype=int)
        test_run = test_df["run"].to_numpy(dtype=int)

        # Helper to prepare kwargs per arm
        def prepare_kwargs(use_covariates: bool) -> dict:
            if use_covariates:
                return dict(
                    model_kwargs, run_metadata=run_df, run_covariate_columns=run_covariate_cols
                )
            else:
                n_runs = int(run_df["run"].max()) + 1
                run_species_vec = np.zeros(n_runs, dtype=int)
                idx = run_df["run"].to_numpy(dtype=int)
                run_species_vec[idx] = run_df["species"].to_numpy(dtype=int)
                return dict(
                    model_kwargs,
                    run_features=np.zeros((n_runs, 0), dtype=float),
                    run_species=run_species_vec,
                )

        if "full" in selected:
            mk = prepare_kwargs(True)
            m = HierarchicalRTModel(
                **mk, compound_features=Z, include_class_hierarchy=True, global_gamma=False
            )
            m.build_model(train_df)
            _ = m.sample(
                n_samples=int(args.draws),
                n_tune=int(args.tune),
                n_chains=int(args.chains),
                target_accept=0.99,
                random_seed=seed_i,
            )
            preds["full"] = m.predict_new(
                test_species,
                test_comp,
                run_idx=test_run,
                n_samples=int(args.predict_draws) if args.predict_draws else None,
            )

        if "nocov_desc" in selected:
            mk = prepare_kwargs(False)
            # For no-covariates arm, species_cluster already set via model_kwargs when args.no_covariates, but we want arm-specific
            m = HierarchicalRTModel(
                **mk, compound_features=Z, include_class_hierarchy=True, global_gamma=False
            )
            m.build_model(train_df)
            _ = m.sample(
                n_samples=int(args.draws),
                n_tune=int(args.tune),
                n_chains=int(args.chains),
                target_accept=0.99,
                random_seed=seed_i + 1,
            )
            preds["nocov_desc"] = m.predict_new(
                test_species,
                test_comp,
                run_idx=test_run,
                n_samples=int(args.predict_draws) if args.predict_draws else None,
            )

        if "cov_nod" in selected:
            mk = prepare_kwargs(True)
            global_comp_class = np.zeros(int(args.n_compounds), dtype=int)
            m = HierarchicalRTModel(
                **{**mk, "n_classes": 1, "compound_class": global_comp_class},
                compound_features=None,
                include_class_hierarchy=False,
                global_gamma=True,
            )
            m.build_model(train_df)
            _ = m.sample(
                n_samples=int(args.draws),
                n_tune=int(args.tune),
                n_chains=int(args.chains),
                target_accept=0.99,
                random_seed=seed_i + 2,
            )
            preds["cov_nod"] = m.predict_new(
                test_species,
                test_comp,
                run_idx=test_run,
                n_samples=int(args.predict_draws) if args.predict_draws else None,
            )

        # Metrics overall and by generator groups
        y_true = test_df["rt"].to_numpy(dtype=float)
        m_overall: Dict[str, Dict[str, float]] = {
            k: asdict(compute_metrics(y_true, pm[0])) for k, pm in preds.items()
        }
        # Map compounds to generator groups
        grp_map = dict(
            zip(
                compound_df["compound_id"].to_numpy(dtype=int),
                compound_df.get(
                    "compound_group", pd.Series(["anchor"]).repeat(len(compound_df)).values
                ),
            )
        )
        test_groups = test_df["compound"].map(grp_map).astype(str)
        m_by_group: Dict[str, Dict[str, float]] = {}
        by_group: Dict[str, Dict[str, Dict[str, float]]] = {}
        for grp in ("anchor", "rare", "unseen"):
            mask = test_groups == grp
            if mask.any():
                grp_metrics: Dict[str, Dict[str, float]] = {}
                for k, (y_mean, _) in preds.items():
                    grp_metrics[k] = asdict(compute_metrics(y_true[mask], y_mean[mask]))
                by_group[grp] = grp_metrics
        # Include support summary per group (number of train rows per compound)
        support_summary: Dict[str, Dict[str, float]] = {}
        train_counts = train_df.groupby("compound").size()
        for grp in ("anchor", "rare", "unseen"):
            grp_comps = compound_df.loc[
                compound_df.get("compound_group", "anchor").astype(str) == grp, "compound_id"
            ].to_numpy(dtype=int)
            sup = train_counts.reindex(grp_comps).fillna(0).astype(int).to_numpy()
            if sup.size:
                q1 = float(np.percentile(sup, 25))
                q3 = float(np.percentile(sup, 75))
                support_summary[grp] = {
                    "n_compounds": int(len(grp_comps)),
                    "support_median": float(np.median(sup)),
                    "support_q1": q1,
                    "support_q3": q3,
                    "support_min": int(np.min(sup)),
                    "support_max": int(np.max(sup)),
                }

        replicates[str(i)] = {
            "overall": m_overall,
            "by_group": by_group,
            "support_by_group": support_summary,
        }

    # Persist a single summary JSON for this run
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"descriptor_ablation_tri_arms_{timestamp}.json"
    results = {
        "config": {
            "seed": int(args.seed),
            "reps": int(args.reps),
            "n_compounds": int(args.n_compounds),
            "n_species": int(args.n_species),
            "n_internal_standards": int(args.n_internal_standards),
            "draws": int(args.draws),
            "tune": int(args.tune),
            "chains": int(args.chains),
            "predict_draws": int(args.predict_draws) if args.predict_draws else None,
            "quick": bool(args.quick),
        },
        "replicates": replicates,
    }
    with json_path.open("w") as fp:
        json.dump(results, fp, indent=2)

    # Build combined tri-arm plot from replicate metrics
    def collect(rep_key: str, arm: str) -> list[float]:
        """Collect MAE values for a given arm, supporting dot-path keys.

        rep_key examples:
        - "overall" → rep["overall"][arm]
        - "by_group.rare" → rep["by_group"]["rare"][arm]
        """
        vals: list[float] = []
        path = rep_key.split(".") if rep_key else []
        for rep in replicates.values():
            node: dict = rep
            ok = True
            for key in path:
                node = node.get(key, {}) if isinstance(node, dict) else {}
                if not node:
                    ok = False
                    break
            if not ok or not isinstance(node, dict):
                continue
            entry = node.get(arm)
            if isinstance(entry, dict):
                m = entry.get("mae")
                if m is not None:
                    vals.append(float(m))
        return vals

    groups = [
        (
            collect("overall", "full"),
            collect("overall", "nocov_desc"),
            collect("overall", "cov_nod"),
        ),
        (
            collect("by_group.rare", "full"),
            collect("by_group.rare", "nocov_desc"),
            collect("by_group.rare", "cov_nod"),
        ),
    ]
    # Reuse the plot-only tri routine for convenience
    def _plot_tri_runtime(groups, labels, output_path: Path):
        x = np.arange(len(groups))
        width = 0.22
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = {
            "full": "#1f77b4",
            "nocov_desc": "#ff7f0e",
            "cov_nod": "#2ca02c",
        }
        for i, (full_vals, nocov_vals, covnod_vals) in enumerate(groups):

            def pack(vals):
                m, lo, hi = _bootstrap_ci_mean(vals)
                return (
                    m,
                    (m - lo if np.isfinite(lo) else np.nan),
                    (hi - m if np.isfinite(hi) else np.nan),
                )

            m1, l1, h1 = pack(full_vals)
            m2, l2, h2 = pack(nocov_vals)
            m3, l3, h3 = pack(covnod_vals)
            ax.bar(
                x[i] - width,
                m1,
                width,
                yerr=[[l1], [h1]],
                capsize=3,
                label="full" if i == 0 else None,
                color=colors["full"],
            )
            ax.bar(
                x[i],
                m2,
                width,
                yerr=[[l2], [h2]],
                capsize=3,
                label="no cov + desc" if i == 0 else None,
                color=colors["nocov_desc"],
            )
            ax.bar(
                x[i] + width,
                m3,
                width,
                yerr=[[l3], [h3]],
                capsize=3,
                label="cov + no desc" if i == 0 else None,
                color=colors["cov_nod"],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("MAE")
        ax.set_title("Descriptor ablation: full vs ablations")
        ax.legend()
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    plot_path = args.plot_output or (out_dir / "combined_tri_mae.png")
    try:
        _plot_tri_runtime(groups, ["All Test Compounds", "Rare Compounds"], plot_path)
        print(f"Saved combined tri-arm plot to {plot_path}")
    except Exception as exc:
        print(f"Warning: could not generate tri-arm plot ({exc})")

    print(f"Saved replicate summary: {json_path}")


if __name__ == "__main__":
    main()
