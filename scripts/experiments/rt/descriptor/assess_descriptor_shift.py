#!/usr/bin/env python
"""Assess descriptor-informed RT under run-level drift with replication and plotting.

Design (aligned with other RT runners):
- Fix training runs to the most in-support runs by run-distance quantile.
- Evaluate three methods on the same held-out runs:
  1) Hierarchical with descriptors (class hierarchy on, global_gamma=False)
  2) Hierarchical without descriptors (no class hierarchy, global_gamma=True)
  3) Species×compound Lasso baseline
- Score under two regimes on identical held-out rows:
  - Observed run covariates
  - Drifted covariates (per-run perturbations along an IS-weighted direction)

Outputs:
- Replicate JSON aggregating metrics across seeds
- Combined plot (observed vs drifted) with mean±95% CI for the selected metric
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_prep.create_synthetic_data import create_metabolomics_data  # type: ignore
from src.compassign.utils import SyntheticDataset
from src.compassign.rt.hierarchical import HierarchicalRTModel
from src.compassign.rt.baselines import SpeciesCompoundLassoBaseline, ClusterCompoundLassoBaseline
# No cluster inference needed; use generator clusters


@dataclass
class MetricSummary:
    """Compact container for scalar performance metrics."""

    mae: float
    rmse: float
    bias: float
    mae_p95: float
    coverage_95: float | None = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (replication + plotting aligned with other RT runners)."""

    parser = argparse.ArgumentParser(
        description=(
            "Descriptor-informed RT under run-level drift with replication and combined plotting"
        )
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed; replicates use seed+i")
    parser.add_argument("--reps", type=int, default=5, help="Number of replicates")
    parser.add_argument("--train-quantile", type=float, default=0.60, help="Fraction of runs kept")
    parser.add_argument("--desc-dim", type=int, default=8, help="Descriptor dimensions")
    parser.add_argument(
        "--drift",
        type=float,
        default=0.5,
        help="Scale applied to mean covariate SD along the average gamma direction",
    )
    parser.add_argument(
        "--drift-jitter-sigma",
        type=float,
        default=0.25,
        help=(
            "Lognormal jitter sigma applied per-run to the drift magnitude; set 0 for fixed "
            "magnitude across runs (reduces between-replicate variance)"
        ),
    )
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--tune", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--predict-draws", type=int, default=None, help="Cap on prediction draws")
    parser.add_argument("--quick", action="store_true", help="Use draws=500, tune=500, chains=4")
    parser.add_argument(
        "--desc-tau-beta",
        type=float,
        default=0.30,
        help="Descriptor strength (tau_beta) used by the generator",
    )
    parser.add_argument(
        "--desc-sigma-compound",
        type=float,
        default=0.30,
        help="Residual compound sd when descriptors are active",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/rt_descriptor_shift"))
    parser.add_argument(
        "--plot-metric",
        type=str,
        default="mae",
        choices=["mae", "rmse", "bias", "mae_p95"],
        help="Metric to visualise in the combined plot.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip sampling and only build the combined plot from an aggregate JSON",
    )
    parser.add_argument(
        "--plot-input",
        type=Path,
        default=None,
        help="Path to descriptor_shift_reps_*.json to plot (used with --plot-only)",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help=(
            "Override destination for the combined plot (defaults to "
            "<output-dir>/combined_<metric>.png)."
        ),
    )
    return parser.parse_args()


def far_scores_global(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return L2 distance-to-nearest-neighbour scores and ascending rank indices."""

    X = np.asarray(X, dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    Z = (X - mu) / sd
    n = Z.shape[0]
    dmin = np.full(n, np.inf)
    for i in range(n):
        distances = np.sqrt(((Z[i] - Z) ** 2).sum(axis=1))
        distances[i] = np.inf
        dmin[i] = float(np.min(distances))
    rank_idx = np.argsort(dmin)
    return dmin, rank_idx


def make_descriptors(
    n_compounds: int,
    n_classes: int,
    compound_class: Iterable[int],
    dim: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic descriptor embeddings aligned with classes."""

    class_centroids = rng.normal(loc=0.0, scale=1.0, size=(n_classes, dim))
    descriptors = np.zeros((n_compounds, dim), dtype=float)
    for cid in range(n_compounds):
        cls = int(compound_class[cid])
        descriptors[cid] = class_centroids[cls] + rng.normal(loc=0.0, scale=0.2, size=dim)
    descriptors -= descriptors.mean(axis=0, keepdims=True)
    descriptors /= descriptors.std(axis=0, keepdims=True) + 1e-8
    return descriptors


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray | None = None
) -> MetricSummary:
    """Compute MAE, RMSE, bias, 95th percentile MAE, and optional coverage."""

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return MetricSummary(
            mae=float("nan"),
            rmse=float("nan"),
            bias=float("nan"),
            mae_p95=float("nan"),
            coverage_95=float("nan") if y_std is not None else None,
        )
    err = y_pred[mask] - y_true[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    bias = float(np.mean(err))
    mae_p95 = float(np.percentile(np.abs(err), 95))
    coverage = None
    if y_std is not None:
        y_std_masked = y_std[mask]
        if np.all(np.isfinite(y_std_masked)):
            lower = y_pred[mask] - 1.96 * y_std_masked
            upper = y_pred[mask] + 1.96 * y_std_masked
            coverage = float(np.mean((y_true[mask] >= lower) & (y_true[mask] <= upper)))
    return MetricSummary(mae=mae, rmse=rmse, bias=bias, mae_p95=mae_p95, coverage_95=coverage)


def train_hierarchical(
    dataset: SyntheticDataset,
    hp: Dict[str, Any],
    train_df,
    *,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
    predict_draws: int | None,
    species_cluster: np.ndarray,
    descriptors: np.ndarray | None,
    include_class_hierarchy: bool,
    global_gamma: bool,
) -> Tuple[HierarchicalRTModel, int]:
    """Fit the hierarchical model with optional descriptors and return prediction draw count."""

    run_meta = dataset.run_meta()
    model = HierarchicalRTModel(
        n_clusters=int(hp["n_clusters"]),
        n_species=int(dataset.peak_df["species"].nunique()),
        n_classes=int(hp["n_classes"]),
        n_compounds=int(dataset.compound_df.shape[0]),
        species_cluster=np.asarray(species_cluster, dtype=int),
        compound_class=np.asarray(hp["compound_class"], dtype=int),
        run_metadata=run_meta.df,
        run_covariate_columns=run_meta.covariate_columns,
        compound_features=descriptors,
        include_class_hierarchy=include_class_hierarchy,
        global_gamma=global_gamma,
    )
    model.build_model(train_df)
    trace = model.sample(
        n_samples=draws,
        n_tune=tune,
        n_chains=chains,
        target_accept=0.99,
        random_seed=seed,
    )
    n_total = trace.posterior["mu0"].values.flatten().shape[0]
    n_use = min(int(predict_draws), n_total) if predict_draws else n_total
    return model, n_use


def evaluate_hierarchical(
    model: HierarchicalRTModel,
    df,
    *,
    n_samples: int,
    run_idx: np.ndarray | None = None,
    run_features: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict means and std for provided observations."""

    species_idx = df["species"].to_numpy(dtype=int)
    compound_idx = df["compound"].to_numpy(dtype=int)
    if run_idx is None:
        run_idx_arg = None
    else:
        run_idx_arg = np.asarray(run_idx, dtype=int)
    pred_mean, pred_std = model.predict_new(
        species_idx,
        compound_idx,
        run_idx=run_idx_arg,
        run_features=run_features,
        n_samples=n_samples,
    )
    return pred_mean, pred_std


def _t_crit_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    lookup = {
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
    return float(lookup.get(df, 1.96))


def _bootstrap_ci_mean(
    values: List[float], *, B: int = 2000, alpha: float = 0.05, seed: int = 777
) -> Tuple[float, float, float]:
    """Bootstrap CI for mean (parity with other RT scripts). Returns (mean, lo, hi)."""
    vals = np.asarray(values, dtype=float)
    clean = vals[np.isfinite(vals)]
    n = clean.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        m = float(clean[0])
        return m, m, m
    rng = np.random.RandomState(int(seed))
    idx = rng.randint(0, n, size=(int(B), n))
    means = clean[idx].mean(axis=1)
    m = float(np.mean(clean))
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return m, lo, hi


def _pick_latest(json_dir: Path, pattern: str) -> Path | None:
    cand = sorted(json_dir.glob(pattern))
    return cand[-1] if cand else None


def _create_combined_plot(
    obs_vals: Dict[str, List[float]],
    drift_vals: Dict[str, List[float]],
    metric: str,
    output_path: Path,
) -> None:
    labels = [
        "hierarchical_descriptors",
        "hierarchical_no_descriptors",
        "baseline_species_compound",
        "baseline_cluster_compound",
    ]
    x = np.arange(len(labels))
    width = 0.18

    def pack(vals: Dict[str, List[float]]):
        means, los, his = [], [], []
        for k in labels:
            m, lo, hi = _bootstrap_ci_mean(vals.get(k, []), B=2000, alpha=0.05, seed=777)
            means.append(m)
            los.append(m - lo if np.isfinite(lo) else np.nan)
            his.append(hi - m if np.isfinite(hi) else np.nan)
        return np.array(means), np.array(los), np.array(his)

    m_obs, lo_obs, hi_obs = pack(obs_vals)
    m_drf, lo_drf, hi_drf = pack(drift_vals)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, m_obs, width, yerr=[lo_obs, hi_obs], capsize=3, label="observed")
    ax.bar(x + width / 2, m_drf, width, yerr=[lo_drf, hi_drf], capsize=3, label="drifted")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "desc",
            "no-desc",
            "base sp×comp",
            "base cl×comp",
        ]
    )
    ax.set_ylabel(metric.upper())
    ax.set_title("Descriptor drift: observed vs drifted")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _single_rep(
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    peak_df, compound_df, true_assignments, rt_uncertainties, hp = create_metabolomics_data(
        n_compounds=60,
        n_species=6,
        n_internal_standards=8,
        fixed_runs_per_species_compound=10,
        species_gamma_sd=0.0,
        desc_tau_beta=float(args.desc_tau_beta),
        desc_sigma_compound=float(args.desc_sigma_compound),
        seed=seed,
    )
    dataset = SyntheticDataset(
        peak_df=peak_df,
        compound_df=compound_df,
        true_assignments=true_assignments,
        rt_uncertainties=rt_uncertainties,
        hierarchical_params=hp,
    )

    run_meta = dataset.run_meta()
    run_features = run_meta.features
    rt_df = dataset.peak_df[dataset.peak_df["true_compound"].notna()].copy()
    rt_df = rt_df.rename(columns={"true_compound": "compound"})
    rt_df["compound"] = rt_df["compound"].astype(int)

    distances, rank_idx = far_scores_global(run_features)
    n_runs = run_features.shape[0]
    n_train = max(1, int(np.floor(args.train_quantile * n_runs)))
    train_runs = np.sort(rank_idx[:n_train])
    test_runs = np.sort(rank_idx[n_train:])

    train_df = rt_df[rt_df["run"].isin(train_runs)].copy()
    test_df = rt_df[rt_df["run"].isin(test_runs)].copy()

    descriptor_source = hp.get("compound_features")
    if descriptor_source is not None:
        descriptors = np.asarray(descriptor_source, dtype=float)
    else:
        descriptors = make_descriptors(
            n_compounds=dataset.compound_df.shape[0],
            n_classes=int(hp["n_classes"]),
            compound_class=hp["compound_class"],
            dim=int(args.desc_dim),
            rng=rng,
        )

    draws = int(args.draws)
    tune = int(args.tune)
    chains = int(args.chains)

    # Use oracle species clusters from the generator for this experiment
    species_cluster_inferred = np.asarray(hp["species_cluster"], dtype=int)

    hier_with_desc, n_use_desc = train_hierarchical(
        dataset,
        hp,
        train_df,
        draws=draws,
        tune=tune,
        chains=chains,
        seed=seed,
        predict_draws=args.predict_draws,
        species_cluster=species_cluster_inferred,
        descriptors=descriptors,
        include_class_hierarchy=True,
        global_gamma=False,
    )

    hier_no_desc, n_use_no_desc = train_hierarchical(
        dataset,
        hp,
        train_df,
        draws=draws,
        tune=tune,
        chains=chains,
        seed=seed + 1,
        predict_draws=args.predict_draws,
        species_cluster=species_cluster_inferred,
        descriptors=None,
        include_class_hierarchy=False,
        global_gamma=True,
    )

    baseline = SpeciesCompoundLassoBaseline()
    baseline.fit(train_df, run_df=run_meta.df, covariate_columns=run_meta.covariate_columns)

    # Cluster×compound baseline
    baseline_cluster = ClusterCompoundLassoBaseline(species_cluster=species_cluster_inferred)
    baseline_cluster.fit(train_df, run_df=run_meta.df, covariate_columns=run_meta.covariate_columns)

    test_run_idx = test_df["run"].to_numpy(dtype=int)
    y_true = test_df["rt"].to_numpy(dtype=float)

    pred_desc_obs, std_desc_obs = evaluate_hierarchical(
        hier_with_desc,
        test_df,
        n_samples=n_use_desc,
        run_idx=test_run_idx,
    )
    pred_nodesc_obs, std_nodesc_obs = evaluate_hierarchical(
        hier_no_desc,
        test_df,
        n_samples=n_use_no_desc,
        run_idx=test_run_idx,
    )
    baseline_pred_obs = baseline.predict(
        species_idx=test_df["species"].to_numpy(dtype=int),
        compound_idx=test_df["compound"].to_numpy(dtype=int),
        run_idx=test_run_idx,
    )
    baseline_cluster_pred_obs = baseline_cluster.predict(
        species_idx=test_df["species"].to_numpy(dtype=int),
        compound_idx=test_df["compound"].to_numpy(dtype=int),
        run_idx=test_run_idx,
    )

    weights_raw = hp.get("compound_internal_std_weights")
    cov_std = run_features.std(axis=0) + 1e-8
    rng_drift = np.random.default_rng(seed + 13_357)
    weights = np.asarray(weights_raw, dtype=float) if weights_raw is not None else None
    weight_mean = weights.mean(axis=0) if weights is not None else None
    run_drift_vectors: Dict[int, np.ndarray] = {}
    drifted_features_rows = np.empty_like(run_features[test_run_idx])

    test_species = test_df["species"].to_numpy(dtype=int)
    test_compounds = test_df["compound"].to_numpy(dtype=int)

    for idx, run in enumerate(test_run_idx):
        run_int = int(run)
        drift_vec = run_drift_vectors.get(run_int)
        if drift_vec is None:
            if weights is not None:
                comps = test_compounds[test_run_idx == run_int]
                if comps.size > 0:
                    direction = weights[comps].mean(axis=0)
                else:
                    direction = weight_mean
            else:
                direction = None

            if direction is None or float(np.linalg.norm(direction)) < 1e-8:
                direction = rng_drift.normal(size=cov_std.shape)
            direction = np.asarray(direction, dtype=float)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm < 1e-8:
                direction = rng_drift.normal(size=cov_std.shape)
                direction_norm = float(np.linalg.norm(direction))
            direction /= direction_norm

            if float(args.drift_jitter_sigma) > 0.0:
                magnitude = float(args.drift) * float(
                    rng_drift.lognormal(mean=0.0, sigma=float(args.drift_jitter_sigma))
                )
            else:
                magnitude = float(args.drift)
            drift_vec = direction * (cov_std * magnitude)
            run_drift_vectors[run_int] = drift_vec
        drifted_features_rows[idx] = run_features[test_run_idx[idx]] + drift_vec

    drifted_features = drifted_features_rows

    pred_desc_drift, std_desc_drift = evaluate_hierarchical(
        hier_with_desc,
        test_df,
        n_samples=n_use_desc,
        run_idx=None,
        run_features=drifted_features,
    )
    pred_nodesc_drift, std_nodesc_drift = evaluate_hierarchical(
        hier_no_desc,
        test_df,
        n_samples=n_use_no_desc,
        run_idx=None,
        run_features=drifted_features,
    )
    baseline_run_features_drift = baseline._run_features.copy()  # type: ignore[union-attr]
    for run_int, vec in run_drift_vectors.items():
        baseline_run_features_drift[run_int] = baseline_run_features_drift[run_int] + vec
    baseline_pred_drift = baseline.predict(
        species_idx=test_species,
        compound_idx=test_compounds,
        run_idx=test_run_idx,
        run_features_override=baseline_run_features_drift,
    )
    # Apply drift to cluster×compound baseline by temporarily overriding stored run features
    baseline_cluster_features_orig = baseline_cluster._run_features.copy()  # type: ignore[attr-defined]
    baseline_cluster_run_features_drift = baseline_cluster_features_orig.copy()
    for run_int, vec in run_drift_vectors.items():
        baseline_cluster_run_features_drift[run_int] = (
            baseline_cluster_run_features_drift[run_int] + vec
        )
    baseline_cluster._run_features = baseline_cluster_run_features_drift  # type: ignore[attr-defined]
    baseline_cluster_pred_drift = baseline_cluster.predict(
        species_idx=test_species,
        compound_idx=test_compounds,
        run_idx=test_run_idx,
    )
    baseline_cluster._run_features = baseline_cluster_features_orig  # restore

    metrics_observed = {
        "hierarchical_descriptors": asdict(compute_metrics(y_true, pred_desc_obs, std_desc_obs)),
        "hierarchical_no_descriptors": asdict(
            compute_metrics(y_true, pred_nodesc_obs, std_nodesc_obs)
        ),
        "baseline_species_compound": asdict(compute_metrics(y_true, baseline_pred_obs, None)),
        "baseline_cluster_compound": asdict(
            compute_metrics(y_true, baseline_cluster_pred_obs, None)
        ),
    }
    metrics_drifted = {
        "hierarchical_descriptors": asdict(
            compute_metrics(y_true, pred_desc_drift, std_desc_drift)
        ),
        "hierarchical_no_descriptors": asdict(
            compute_metrics(y_true, pred_nodesc_drift, std_nodesc_drift)
        ),
        "baseline_species_compound": asdict(compute_metrics(y_true, baseline_pred_drift, None)),
        "baseline_cluster_compound": asdict(
            compute_metrics(y_true, baseline_cluster_pred_drift, None)
        ),
    }

    return {
        "split": {
            "train_runs": train_runs.tolist(),
            "test_runs": test_runs.tolist(),
            "distance_scores": distances[test_runs].tolist(),
        },
        "metrics": {
            "observed": metrics_observed,
            "drifted": metrics_drifted,
        },
    }


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

    # Plot-only path
    if args.plot_only:
        json_path = args.plot_input or _pick_latest(out_dir, "descriptor_shift_reps_*.json")
        if not json_path:
            raise SystemExit("No aggregate JSON found. Provide --plot-input or run sampling first.")
        with Path(json_path).open("r") as fp:
            payload = json.load(fp)
        reps = payload.get("replicates", {})
        metric = str(args.plot_metric)
        obs_vals = {
            k: []
            for k in [
                "hierarchical_descriptors",
                "hierarchical_no_descriptors",
                "baseline_species_compound",
            ]
        }
        drift_vals = {k: [] for k in obs_vals.keys()}
        for rep in reps.values():
            for label, m in rep.get("observed", {}).items():
                obs_vals.setdefault(label, []).append(float(m.get(metric, float("nan"))))
            for label, m in rep.get("drifted", {}).items():
                drift_vals.setdefault(label, []).append(float(m.get(metric, float("nan"))))
        plot_path = args.plot_output or (out_dir / f"combined_{metric}.png")
        _create_combined_plot(obs_vals, drift_vals, metric, plot_path)
        print(f"Saved combined plot to {plot_path}")
        return

    # Replication loop
    replicates: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for i in range(int(args.reps)):
        seed = int(args.seed) + i
        rep = _single_rep(seed=seed, args=args)
        # Store only metrics per scenario for aggregation/plotting
        replicates[str(i)] = {
            "observed": rep["metrics"]["observed"],
            "drifted": rep["metrics"]["drifted"],
        }

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"descriptor_shift_reps_{timestamp}.json"
    results = {
        "config": {
            "seed": int(args.seed),
            "reps": int(args.reps),
            "train_quantile": float(args.train_quantile),
            "descriptor_dim": int(args.desc_dim),
            "drift_scale_sd": float(args.drift),
            "desc_tau_beta": float(args.desc_tau_beta),
            "desc_sigma_compound": float(args.desc_sigma_compound),
            "draws": int(args.draws),
            "tune": int(args.tune),
            "chains": int(args.chains),
            "predict_draws": int(args.predict_draws) if args.predict_draws else None,
            "plot_metric": str(args.plot_metric),
        },
        "replicates": replicates,
    }
    with json_path.open("w") as fp:
        json.dump(results, fp, indent=2)

    # Print brief summary: mean across replicates for both regimes
    metric = str(args.plot_metric)

    def collect(vals_key: str, label: str) -> List[float]:
        out: List[float] = []
        for rep in replicates.values():
            m = rep.get(vals_key, {}).get(label, {})
            v = m.get(metric)
            if v is not None:
                out.append(float(v))
        return out

    labels = [
        "hierarchical_descriptors",
        "hierarchical_no_descriptors",
        "baseline_species_compound",
        "baseline_cluster_compound",
    ]
    print("==== Descriptor Drift (replicate means) ====")
    for scen in ("observed", "drifted"):
        print(f"\n[{scen}]")
        for label in labels:
            vals = collect(scen, label)
            if vals:
                print(f"  {label:>30}: {np.mean(vals):.3f} (n={len(vals)})")

    # Produce combined plot
    obs_vals = {k: collect("observed", k) for k in labels}
    drift_vals = {k: collect("drifted", k) for k in labels}
    plot_path = args.plot_output or (out_dir / f"combined_{metric}.png")
    try:
        _create_combined_plot(obs_vals, drift_vals, metric, plot_path)
        print(f"Saved combined plot to {plot_path}")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: could not generate combined plot ({exc})")

    print(f"Saved replicate summary: {json_path}")


if __name__ == "__main__":
    main()
