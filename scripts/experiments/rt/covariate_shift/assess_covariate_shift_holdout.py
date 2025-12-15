#!/usr/bin/env python
"""Assess robustness to covariate shift with fixed-train, binned test.

Design:
- Compute run far-ness (min L2 in standardized IS space) per dataset replicate.
- Fix training set to in-support runs (<= train_quantile, default 0.60).
- Partition out-of-support runs into equal-size distance bins: 60–70, 70–80, 80–90, 90–95, 95–100.
- Train each method once on the fixed training set; evaluate on each test bin.
- Aggregate across replicates (default 1 seed) and plot mean MAE with 95% CIs per bin.

Methods:
- Hierarchical (global_gamma=True, include_class_hierarchy=False)
- Baseline species×compound Lasso
- Baseline cluster×compound Lasso

Outputs:
- JSON summary with per-replicate bin metrics and aggregate means/CIs
- MAE vs bin plot showing mean ± 95% CI error bars (replicate dots overlaid)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_prep.create_synthetic_data import create_metabolomics_data  # type: ignore  # noqa: E402
from src.compassign.utils import SyntheticDataset  # noqa: E402
from src.compassign.rt.hierarchical import HierarchicalRTModel  # noqa: E402
from src.compassign.rt.baselines import (  # noqa: E402
    SpeciesCompoundLassoBaseline,
    ClusterCompoundLassoBaseline,
)
# No cluster inference needed; use generator clusters  # noqa: E402


@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float
    median_ae: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        median_ae=float(np.median(np.abs(y_true - y_pred))),
    )


def compute_metrics_on_finite(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    mask = np.isfinite(y_pred)
    if not np.any(mask):
        return Metrics(mae=float("nan"), rmse=float("nan"), r2=float("nan"), median_ae=float("nan"))
    return compute_metrics(y_true[mask], y_pred[mask])


def _t_crit_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    t_lookup = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    if df in t_lookup:
        return float(t_lookup[df])
    # Approximate with normal for larger df
    return 1.96


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed-train, covariate-shift binned test for RT models"
    )
    p.add_argument("--n-compounds", type=int, default=50)
    p.add_argument("--n-species", type=int, default=6)
    p.add_argument("--n-internal-standards", type=int, default=8)
    p.add_argument("--seed", type=int, default=42, help="Base seed; replicates use seed+i")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument(
        "--train-quantile", type=float, default=0.60, help="Quantile for in-support training runs"
    )
    p.add_argument("--quick", action="store_true", help="Use draws=500, tune=500, chains=4")
    p.add_argument("--draws", type=int, default=None)
    p.add_argument("--tune", type=int, default=None)
    p.add_argument("--chains", type=int, default=None)
    p.add_argument(
        "--predict-draws",
        type=int,
        default=None,
        help="Optional cap on posterior draws for prediction (default: use all)",
    )
    p.add_argument(
        "--species-gamma-sd",
        type=float,
        default=0.1,
        help="Species-specific slope heterogeneity in the generator (0 for none)",
    )
    p.add_argument(
        "--anchors",
        type=str,
        default="0,3,5",
        help="Comma-separated anchor counts per test run for intercept calibration",
    )
    p.add_argument(
        "--runs-per-species",
        type=int,
        default=8,
        help="Number of runs simulated per species (controls holdout size)",
    )
    p.add_argument("--output-dir", type=Path, default=Path("output/rt_covshift_holdout"))
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip sampling; regenerate plots from an existing JSON",
    )
    p.add_argument(
        "--plot-input",
        type=Path,
        default=None,
        help=(
            "Path to covshift_holdout_*.json to plot (used with --plot-only). "
            "If omitted, picks latest under --output-dir"
        ),
    )
    return p.parse_args()


def _bootstrap_ci_mean(
    values: List[float], *, B: int = 2000, alpha: float = 0.05, seed: int = 1234
) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for the mean. Returns (lo, hi).
    For n==0 returns (nan, nan). For n==1, all resamples are identical so lo==hi==value.
    """
    import numpy as _np

    vals = _np.asarray(values, dtype=float)
    clean = vals[_np.isfinite(vals)]
    n = clean.size
    if n == 0:
        return float("nan"), float("nan")
    rng = _np.random.RandomState(int(seed))
    # Draw bootstrap indices (B × n) and compute means
    idx = rng.randint(0, n, size=(int(B), n))
    means = clean[idx].mean(axis=1)
    lo = float(_np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(_np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def make_dataset(
    n_compounds: int,
    n_species: int,
    n_is: int,
    *,
    seed: int,
    n_runs_per_species: int,
    species_gamma_sd: float,
) -> Tuple[SyntheticDataset, Dict[str, Any]]:
    fixed_k = 10
    peak_df, compound_df, _ta, _rtu, hp = create_metabolomics_data(
        n_compounds=max(n_compounds, 50),
        n_species=max(n_species, 6),
        n_internal_standards=n_is,
        fixed_runs_per_species_compound=fixed_k,
        n_runs_per_species=n_runs_per_species,
        species_gamma_sd=float(species_gamma_sd),
        seed=int(seed),
    )
    dataset = SyntheticDataset(
        peak_df=peak_df,
        compound_df=compound_df,
        true_assignments=_ta,
        rt_uncertainties=_rtu,
        hierarchical_params=hp,
    )
    return dataset, hp


def far_scores_global(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    Z = (X - mu) / sd
    n = Z.shape[0]
    dmin = np.full(n, np.inf)
    for i in range(n):
        di = np.sqrt(((Z[i] - Z) ** 2).sum(axis=1))
        di[i] = np.inf
        dmin[i] = float(np.min(di))
    rank_idx = np.argsort(dmin)  # ascending (in-support first)
    return dmin, rank_idx


def train_hier(
    dataset: SyntheticDataset,
    hp: Dict[str, Any],
    train_df,
    *,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
    n_pred: int,
):
    run_meta = dataset.run_meta()
    # Use oracle species clusters from the generator (shared for all models)
    inferred_clusters = np.asarray(hp["species_cluster"], dtype=int)
    n_species = int(dataset.peak_df["species"].nunique())
    cfg = dict(
        n_clusters=int(hp["n_clusters"]),
        n_species=n_species,
        n_classes=int(hp["n_classes"]),
        n_compounds=int(dataset.compound_df.shape[0]),
        species_cluster=np.asarray(inferred_clusters, dtype=int),
        compound_class=np.asarray(hp["compound_class"], dtype=int),
        run_metadata=run_meta.df,
        run_covariate_columns=run_meta.covariate_columns,
        # Enable descriptors by default to leverage chemistry signal
        compound_features=hp.get("compound_features", None),
        include_class_hierarchy=True,
        global_gamma=False,
    )
    m = HierarchicalRTModel(**cfg)
    m.build_model(train_df)
    tr = m.sample(
        n_samples=draws, n_tune=tune, n_chains=chains, target_accept=0.99, random_seed=seed
    )
    draws_total = tr.posterior["mu0"].values.flatten().shape[0]
    n_use = min(n_pred, draws_total) if n_pred else None
    return m, n_use


def train_baselines(dataset: SyntheticDataset, train_df, *, seed: int):
    run_meta = dataset.run_meta()
    cov_cols = run_meta.covariate_columns
    hp = dataset.hierarchical_params  # type: ignore[attr-defined]
    inferred_clusters = np.asarray(hp["species_cluster"], dtype=int)
    b_sc = SpeciesCompoundLassoBaseline()
    b_sc.fit(train_df, run_df=run_meta.df, covariate_columns=cov_cols)
    b_pc = ClusterCompoundLassoBaseline(species_cluster=np.asarray(inferred_clusters, dtype=int))  # type: ignore[attr-defined]
    b_pc.fit(train_df, run_df=run_meta.df, covariate_columns=cov_cols)
    return b_sc, b_pc


def _apply_intercept_calibration(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    run_idx: np.ndarray,
    anchors_per_run: int,
    *,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate predictions per run using N anchors; return (y_pred_adj, mask_eval).

    - For each run, randomly select up to N points as anchors (without replacement).
    - Compute Δ_run = mean(y_true_anchor − y_pred_anchor) and add to all predictions in that run.
    - Exclude anchors from evaluation mask so metrics are on held-out rows only.
    """
    if anchors_per_run <= 0:
        return y_pred, np.ones_like(y_pred, dtype=bool)
    y_adj = y_pred.copy()
    mask_eval = np.ones_like(y_pred, dtype=bool)
    runs = np.unique(run_idx)
    for r in runs:
        idx = np.where(run_idx == r)[0]
        if idx.size == 0:
            continue
        n_a = int(min(anchors_per_run, idx.size))
        # Deterministic choice given rng state
        sel = rng.choice(idx, size=n_a, replace=False)
        delta = float(np.mean(y_true[sel] - y_pred[sel]))
        y_adj[idx] = y_pred[idx] + delta
        mask_eval[sel] = False
    return y_adj, mask_eval


def eval_on_bin(
    dataset: SyntheticDataset,
    model,
    n_use: int,
    b_sc,
    b_pc,
    bin_runs: np.ndarray,
    *,
    anchors_per_run_list: list[int],
    rng_seed: int,
):
    # Build test df for the selected runs
    rt_df = dataset.peak_df[dataset.peak_df["true_compound"].notna()].copy()
    rt_df = rt_df.rename(columns={"true_compound": "compound"})
    rt_df["species"], rt_df["compound"], rt_df["run"] = (
        rt_df["species"].astype(int),
        rt_df["compound"].astype(int),
        rt_df["run"].astype(int),
    )
    test_df = rt_df[rt_df["run"].isin(bin_runs)].reset_index(drop=True)
    species_idx = test_df["species"].to_numpy(dtype=int)
    compound_idx = test_df["compound"].to_numpy(dtype=int)
    run_idx = test_df["run"].to_numpy(dtype=int)
    y_true = test_df["rt"].to_numpy(dtype=float)

    # Hierarchical predictions + calibration diag
    y_h_mean, y_h_std = model.predict_new(
        species_idx=species_idx, compound_idx=compound_idx, run_idx=run_idx, n_samples=n_use
    )
    # Baselines
    y_sc = b_sc.predict(species_idx=species_idx, compound_idx=compound_idx, run_idx=run_idx)
    y_pc = b_pc.predict(species_idx=species_idx, compound_idx=compound_idx, run_idx=run_idx)

    out: Dict[str, Any] = {"sizes": {"test": int(len(test_df))}, "anchors": {}}
    for a in anchors_per_run_list:
        rng = np.random.RandomState(rng_seed + a + int(np.sum(bin_runs)))
        # Hier calibration
        y_h_adj, mask_eval = _apply_intercept_calibration(y_h_mean, y_true, run_idx, a, rng=rng)
        met_h = compute_metrics(y_true[mask_eval], y_h_adj[mask_eval])
        z = (y_true[mask_eval] - y_h_adj[mask_eval]) / np.maximum(y_h_std[mask_eval], 1e-6)
        diag_h = {
            "coverage_95": float(
                np.mean(
                    (y_true[mask_eval] >= y_h_adj[mask_eval] - 1.96 * y_h_std[mask_eval])
                    & (y_true[mask_eval] <= y_h_adj[mask_eval] + 1.96 * y_h_std[mask_eval])
                )
            ),
            "z_mean": float(np.mean(z)),
            "z_std": float(np.std(z)),
            "avg_interval_95_width": float(np.mean(2.0 * 1.96 * y_h_std[mask_eval])),
        }
        # Baselines calibration (same anchors per model)
        rng_sc = np.random.RandomState(rng_seed + a + int(np.sum(bin_runs)))
        rng_pc = np.random.RandomState(rng_seed + a + int(np.sum(bin_runs)))
        y_sc_adj, mask_eval_sc = _apply_intercept_calibration(y_sc, y_true, run_idx, a, rng=rng_sc)
        y_pc_adj, mask_eval_pc = _apply_intercept_calibration(y_pc, y_true, run_idx, a, rng=rng_pc)
        met_sc = compute_metrics_on_finite(y_true[mask_eval_sc], y_sc_adj[mask_eval_sc])
        met_pc = compute_metrics_on_finite(y_true[mask_eval_pc], y_pc_adj[mask_eval_pc])
        out["anchors"][str(a)] = {
            "hier": asdict(met_h),
            "diag_hier": diag_h,
            "baseline_species_compound": asdict(met_sc),
            "baseline_cluster_compound": asdict(met_pc),
        }
    return out


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Single output location; no per-profile subfolder
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "covshift_holdout.log"

    def _log(msg: str) -> None:
        print(msg, flush=True)
        try:
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
        except Exception:
            pass

    if args.plot_only:
        # Load an existing summary JSON and rebuild plots only
        src_json: Path | None = args.plot_input
        if src_json is None:
            candidates = sorted(
                out_dir.glob("covshift_holdout_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                src_json = candidates[0]
        if src_json is None or not src_json.exists():
            raise FileNotFoundError(
                "No summary JSON found. Provide --plot-input or ensure covshift_holdout_*.json exists under --output-dir"
            )
        with src_json.open("r", encoding="utf-8") as f:
            results = json.load(f)
        # Require aggregate-provided bin labels; do not fallback
        agg_meta = results.get("aggregate", {})
        bin_labels = agg_meta.get("bins")
        if not bin_labels:
            raise ValueError(
                "Aggregate bin labels missing in summary JSON; regenerate results with bins captured."
            )
    else:
        draws = args.draws
        tune = args.tune
        chains = args.chains
        # Profiles: quick = 500/500/4, full (default) = 1000/1000/4
        if args.quick and (draws is None and tune is None and chains is None):
            draws, tune, chains = 500, 500, 4
        if draws is None:
            draws = 1000
        if tune is None:
            tune = 1000
        if chains is None:
            chains = 4
        n_pred = int(args.predict_draws) if args.predict_draws else None

        # Define bins (quantile ranges over runs) and make labels
        q_train = float(args.train_quantile)
        bin_edges = [(0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 1.00)]
        bin_labels = ["60–70%", "70–80%", "80–90%", "90–95%", "95–100%"]
        anchors_list = [int(x.strip()) for x in args.anchors.split(",") if x.strip()]

        _log(
            f"Covariate-shift binned assessment @ {timestamp} | quick={bool(args.quick)} draws={draws} tune={tune} chains={chains} | reps={args.reps} | train_q={q_train}"
        )

        results: Dict[str, Any] = {
            "config": {
                "n_compounds": int(max(args.n_compounds, 50)),
                "n_species": int(max(args.n_species, 6)),
                "n_internal_standards": int(args.n_internal_standards),
                "seed": int(args.seed),
                "reps": int(args.reps),
                "quick": bool(args.quick),
                "draws": int(draws),
                "tune": int(tune),
                "chains": int(chains),
                "predict_draws": int(args.predict_draws) if args.predict_draws else None,
                "train_quantile": q_train,
                "bins": bin_labels,
                "anchors": anchors_list,
                "runs_per_species": int(max(args.runs_per_species, 1)),
            },
            "replicates": {},
            "aggregate": {},
        }

        # Aggregation buffers per anchor budget
        agg: Dict[str, Any] = {
            str(a): {
                "hier": [[] for _ in bin_edges],
                "baseline_species_compound": [[] for _ in bin_edges],
                "baseline_cluster_compound": [[] for _ in bin_edges],
            }
            for a in anchors_list
        }

        for rep in range(int(args.reps)):
            rep_seed = int(args.seed + rep)
            _log("")
            _log(f"Replicate {rep+1}/{args.reps} (seed={rep_seed})")
            dataset, hp = make_dataset(
                args.n_compounds,
                args.n_species,
                args.n_internal_standards,
                seed=rep_seed,
                n_runs_per_species=int(max(args.runs_per_species, 1)),
                species_gamma_sd=float(args.species_gamma_sd),
            )
            run_meta = dataset.run_meta()
            run_df = run_meta.df.copy()
            X = run_df[run_meta.covariate_columns].to_numpy(dtype=float)
            dmin, rank_idx = far_scores_global(X)
            n_runs = int(len(rank_idx))
            # Training runs: lowest far-ness (<= train quantile)
            n_train = int(np.floor(q_train * n_runs))
            train_runs = run_df.loc[rank_idx[:n_train], "run"].astype(int).to_numpy()
            # Build train_df
            rt_df_all = dataset.peak_df[dataset.peak_df["true_compound"].notna()].copy()
            rt_df_all = rt_df_all.rename(columns={"true_compound": "compound"})
            rt_df_all["species"], rt_df_all["compound"], rt_df_all["run"] = (
                rt_df_all["species"].astype(int),
                rt_df_all["compound"].astype(int),
                rt_df_all["run"].astype(int),
            )
            train_df = rt_df_all[rt_df_all["run"].isin(train_runs)].reset_index(drop=True)

            # Train once per method
            hier_model, n_use = train_hier(
                dataset,
                hp,
                train_df,
                draws=draws,
                tune=tune,
                chains=chains,
                seed=rep_seed,
                n_pred=n_pred,
            )
            b_sc, b_pc = train_baselines(dataset, train_df, seed=rep_seed)

            # Evaluate per bin
            rep_entry: Dict[str, Any] = {"seed": rep_seed, "bins": {}}
            for b_idx, (lo, hi) in enumerate(bin_edges):
                start = int(np.floor(lo * n_runs))
                end = int(np.floor(hi * n_runs))
                # Restrict to out-of-support region (exclude any overlap with train)
                bin_rank_slice = rank_idx[start:end]
                mask_oo = np.isin(bin_rank_slice, rank_idx[n_train:])
                bin_idx = bin_rank_slice[mask_oo]
                bin_runs = run_df.loc[bin_idx, "run"].astype(int).to_numpy()
                if bin_runs.size == 0:
                    rep_entry["bins"][bin_labels[b_idx]] = {"sizes": {"test": 0}, "anchors": {}}
                    for a in anchors_list:
                        rep_entry["bins"][bin_labels[b_idx]]["anchors"][str(a)] = {
                            "hier": asdict(
                                Metrics(float("nan"), float("nan"), float("nan"), float("nan"))
                            ),
                            "diag_hier": {
                                "coverage_95": float("nan"),
                                "z_mean": float("nan"),
                                "z_std": float("nan"),
                                "avg_interval_95_width": float("nan"),
                            },
                            "baseline_species_compound": asdict(
                                Metrics(float("nan"), float("nan"), float("nan"), float("nan"))
                            ),
                            "baseline_cluster_compound": asdict(
                                Metrics(float("nan"), float("nan"), float("nan"), float("nan"))
                            ),
                        }
                    continue
                res_bin = eval_on_bin(
                    dataset,
                    hier_model,
                    n_use,
                    b_sc,
                    b_pc,
                    bin_runs,
                    anchors_per_run_list=anchors_list,
                    rng_seed=rep_seed + b_idx,
                )
                rep_entry["bins"][bin_labels[b_idx]] = res_bin
                # Aggregate per anchor
                for a in anchors_list:
                    key = str(a)
                    res_a = res_bin["anchors"][key]
                    agg[key]["hier"][b_idx].append(res_a["hier"]["mae"])
                    agg[key]["baseline_species_compound"][b_idx].append(
                        res_a["baseline_species_compound"]["mae"]
                    )
                    agg[key]["baseline_cluster_compound"][b_idx].append(
                        res_a["baseline_cluster_compound"]["mae"]
                    )

            results["replicates"][str(rep)] = rep_entry

        # Aggregate summary (use bootstrap percentile CI for the mean)
        agg_out_all: Dict[str, Any] = {"bins": bin_labels, "anchors": anchors_list, "series": {}}
        for key, per_anchor in agg.items():
            series: Dict[str, Any] = {}
            for label in ["hier", "baseline_species_compound", "baseline_cluster_compound"]:
                means: List[float] = []
                stds: List[float] = []
                ci_lo: List[float] = []
                ci_hi: List[float] = []
                counts: List[int] = []
                rep_vals: List[List[float]] = per_anchor[label]
                for b_idx, vals in enumerate(rep_vals):
                    clean = [float(v) for v in vals if np.isfinite(v)]
                    n = len(clean)
                    counts.append(n)
                    if n == 0:
                        means.append(float("nan"))
                        stds.append(float("nan"))
                        ci_lo.append(float("nan"))
                        ci_hi.append(float("nan"))
                        continue
                    mean = float(np.mean(clean))
                    std = float(np.std(clean, ddof=1)) if n > 1 else 0.0
                    # Use a single constant seed for simplicity across facets
                    lo, hi = _bootstrap_ci_mean(clean, B=2000, alpha=0.05, seed=777)
                    means.append(mean)
                    stds.append(std)
                    ci_lo.append(lo)
                    ci_hi.append(hi)
                series[label] = {
                    "mae_mean": means,
                    "mae_std": stds,
                    "mae_ci95_lo": ci_lo,
                    "mae_ci95_hi": ci_hi,
                    "count": counts,
                    "mae_values": rep_vals,
                }
            agg_out_all["series"][key] = series
        results["aggregate"] = agg_out_all

    # Plots
    x = np.arange(len(bin_labels))
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in results["aggregate"]["series"].keys():
        series = results["aggregate"]["series"][key]
        # Two-panel plot per anchor budget:
        #  - Left: grouped bars for baselines vs each other (with CI whiskers)
        #  - Right: bar chart of (cluster − hier) per bin with paired-diff CI
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4))

        # Left panel: grouped bars for baselines
        width = 0.35
        offsets = [-width / 2.0, width / 2.0]
        left_methods = [
            ("baseline_species_compound", "#4c78a8", "Base (species×comp)"),
            ("baseline_cluster_compound", "#e45756", "Base (cluster×comp)"),
        ]
        # Blended transform so x is in data coords while y is in axes coords
        from matplotlib.transforms import blended_transform_factory as _btf  # type: ignore

        trans_left = _btf(ax_left.transData, ax_left.transAxes)
        for (method_key, color, label), off in zip(left_methods, offsets):
            m = np.array(series[method_key]["mae_mean"])  # per-bin means
            ci_lo = np.array(series[method_key].get("mae_ci95_lo", [np.nan] * len(m)))
            ci_hi = np.array(series[method_key].get("mae_ci95_hi", [np.nan] * len(m)))
            counts = np.array(series[method_key].get("count", [0] * len(m)))
            # Validate we have enough replicates to form a CI in every bin
            insufficient = np.where(counts < 2)[0]
            if insufficient.size > 0:
                bad_bins = ", ".join(bin_labels[i] for i in insufficient.tolist())
                _log(
                    f"Error: insufficient replicates for CI (anchors/run={key}, method={label}). "
                    f"Bins with <2 replicates: {bad_bins}. Increase --reps or adjust settings."
                )
                raise RuntimeError("Cannot compute bootstrap CI: insufficient replicates")
            if not (np.all(np.isfinite(ci_lo)) and np.all(np.isfinite(ci_hi))):
                _log(
                    f"Error: non-finite CI bounds for (anchors/run={key}, method={label}). "
                    f"Re-run with more replicates or full settings."
                )
                raise RuntimeError("Non-finite CI bounds")
            lower = np.maximum(0.0, m - ci_lo)
            upper = np.maximum(0.0, ci_hi - m)
            yerr = np.vstack([lower, upper])
            bars = ax_left.bar(
                x + off, m, width=width, color=color, alpha=0.9, label=label, zorder=3
            )
            ax_left.errorbar(
                x + off,
                m,
                yerr=yerr,
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=3,
                zorder=4,
            )
            # Value labels placed at a fixed offset above the x-axis inside the bar,
            # using an axes-y transform to avoid overlapping with error bars.
            for bx, by in zip(x + off, m):
                ax_left.text(
                    bx,
                    0.04,
                    f"{by:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="white",
                    transform=trans_left,
                )
        ax_left.set_xticks(x)
        ax_left.set_xticklabels(bin_labels, rotation=0)
        ax_left.set_xlabel("Covariate-shift bins")
        ax_left.set_ylabel("MAE (min)")
        ax_left.set_title(f"Baselines (anchors/run={key})")
        ax_left.grid(True, axis="y", alpha=0.3, zorder=0)
        ax_left.set_ylim(bottom=0.0)
        ax_left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

        # Right panel: grouped bars for Cluster vs Hierarchical
        right_methods = [
            ("baseline_cluster_compound", "#e45756", "Base (cluster×comp)"),
            ("hier", "#2ca02c", "Hier (chemistry γ)"),
        ]
        trans_right = _btf(ax_right.transData, ax_right.transAxes)
        for (method_key, color, label), off in zip(right_methods, offsets):
            m = np.array(series[method_key]["mae_mean"])  # per-bin means
            ci_lo = np.array(series[method_key].get("mae_ci95_lo", [np.nan] * len(m)))
            ci_hi = np.array(series[method_key].get("mae_ci95_hi", [np.nan] * len(m)))
            counts = np.array(series[method_key].get("count", [0] * len(m)))
            insufficient = np.where(counts < 2)[0]
            if insufficient.size > 0:
                bad_bins = ", ".join(bin_labels[i] for i in insufficient.tolist())
                _log(
                    f"Error: insufficient replicates for CI (anchors/run={key}, method={label}). "
                    f"Bins with <2 replicates: {bad_bins}. Increase --reps or adjust settings."
                )
                raise RuntimeError("Cannot compute bootstrap CI: insufficient replicates")
            if not (np.all(np.isfinite(ci_lo)) and np.all(np.isfinite(ci_hi))):
                _log(
                    f"Error: non-finite CI bounds for (anchors/run={key}, method={label}). "
                    f"Re-run with more replicates or full settings."
                )
                raise RuntimeError("Non-finite CI bounds")
            lower = np.maximum(0.0, m - ci_lo)
            upper = np.maximum(0.0, ci_hi - m)
            yerr = np.vstack([lower, upper])
            bars = ax_right.bar(
                x + off, m, width=width, color=color, alpha=0.9, label=label, zorder=3
            )
            ax_right.errorbar(
                x + off,
                m,
                yerr=yerr,
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=3,
                zorder=4,
            )
            # Value labels at a fixed offset above the x-axis inside the bar
            for bx, by in zip(x + off, m):
                ax_right.text(
                    bx,
                    0.04,
                    f"{by:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="white",
                    transform=trans_right,
                )
        ax_right.set_xticks(x)
        ax_right.set_xticklabels(bin_labels, rotation=0)
        ax_right.set_xlabel("Covariate-shift bins")
        ax_right.set_ylabel("MAE (min)")
        ax_right.set_title(f"Cluster vs Hier (anchors/run={key})")
        ax_right.grid(True, axis="y", alpha=0.3, zorder=0)
        ax_right.set_ylim(bottom=0.0)
        ax_right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

        fig.tight_layout()
        # Leave room for legends placed below each subplot
        fig.subplots_adjust(bottom=0.25)
        plot_mae = out_dir / f"covshift_holdout_mae_anchors{key}_{timestamp}.png"
        fig.savefig(plot_mae, dpi=150)
        plt.close(fig)

    # Save JSON only if we ran sampling (not in plot-only mode)
    if not args.plot_only:
        out_json = out_dir / f"covshift_holdout_{timestamp}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        _log(f"Saved results: {out_json}")


if __name__ == "__main__":
    main()
