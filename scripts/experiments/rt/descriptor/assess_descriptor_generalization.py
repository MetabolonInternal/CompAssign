#!/usr/bin/env python
"""Evaluate descriptor-enabled hierarchies on cold-start and shifted compound mixes,
then produce a single combined figure with two subplots (cold_start and mix_shift)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_prep.create_synthetic_data import create_metabolomics_data  # type: ignore
from src.compassign.utils import SyntheticDataset
from src.compassign.rt.hierarchical import HierarchicalRTModel
# No cluster inference needed; use generator clusters


@dataclass
class MetricSummary:
    mae: float
    rmse: float
    bias: float
    mae_p95: float
    coverage_95: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe descriptor utility under cold-start and shifted compound mix scenarios, "
            "running both and emitting a combined plot by default"
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reps", type=int, default=5, help="Number of replicates; seeds use seed+i"
    )
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--tune", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Use draws=500, tune=500, chains=4")
    parser.add_argument("--desc-tau-beta", type=float, default=0.30)
    parser.add_argument("--desc-sigma-compound", type=float, default=0.30)
    parser.add_argument(
        "--mix-k",
        type=int,
        default=5,
        help=(
            "Primary observations per species–compound in mix_shift (fixed_runs_per_species_compound); "
            "lower reduces runtime."
        ),
    )
    parser.add_argument(
        "--holdout-frac", type=float, default=0.25, help="Fraction of classes held out in mix_shift"
    )
    parser.add_argument(
        "--unseen-eval", type=int, default=4, help="Evaluation draws per unseen compound"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/rt_descriptor_generalization")
    )
    parser.add_argument(
        "--plot-metric",
        type=str,
        default="mae",
        choices=["mae", "rmse", "bias", "mae_p95"],
        help="Metric to visualise in the combined plot.",
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
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip sampling and just build the combined plot from an existing summary JSON",
    )
    parser.add_argument(
        "--plot-input",
        type=Path,
        default=None,
        help="Path to a descriptor_generalization_reps_*.json to plot (used with --plot-only)",
    )
    parser.add_argument(
        "--predict-draws",
        type=int,
        default=None,
        help="Optional cap on posterior draws used for prediction (default: use all)",
    )
    return parser.parse_args()


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray | None = None
) -> MetricSummary:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return MetricSummary(
            mae=np.nan, rmse=np.nan, bias=np.nan, mae_p95=np.nan, coverage_95=np.nan
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
    hp: Dict[str, np.ndarray],
    train_df: pd.DataFrame,
    *,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
    species_cluster: np.ndarray,
    descriptors: np.ndarray | None,
    include_class_hierarchy: bool,
    global_gamma: bool,
) -> Tuple[HierarchicalRTModel, int]:
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
    n_use = n_total  # Prediction draw cap applied by caller
    return model, n_use


def evaluate_model(
    model: HierarchicalRTModel,
    df: pd.DataFrame,
    *,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    species_idx = df["species"].to_numpy(dtype=int)
    compound_idx = df["compound"].to_numpy(dtype=int)
    run_idx = df["run"].to_numpy(dtype=int)
    pred_mean, pred_std = model.predict_new(
        species_idx,
        compound_idx,
        run_idx=run_idx,
        n_samples=n_samples,
    )
    return pred_mean, pred_std


def prepare_cold_start(
    args: argparse.Namespace,
    *,
    seed: int,
) -> Tuple[
    SyntheticDataset,
    Dict[str, np.ndarray],
    pd.DataFrame,
    Dict[str, pd.DataFrame],
    Dict[str, List[int]],
]:
    peak_df, compound_df, true_assignments, rt_uncertainties, hp = create_metabolomics_data(
        n_compounds=80,
        n_species=6,
        n_internal_standards=8,
        fixed_runs_per_species_compound=None,
        anchor_free_frac=0.25,
        unseen_eval_per_compound=args.unseen_eval,
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
    rt_df = dataset.peak_df[dataset.peak_df["true_compound"].notna()].copy()
    rt_df = rt_df.rename(columns={"true_compound": "compound"})
    rt_df["compound"] = rt_df["compound"].astype(int)
    groups = dataset.compound_df.get(
        "compound_group", pd.Series(["anchor"] * len(dataset.compound_df))
    )
    unseen_ids = sorted(dataset.compound_df.index[groups == "unseen"].tolist())
    if not unseen_ids:
        raise RuntimeError(
            "Synthetic generator produced no unseen compounds; adjust anchor_free_frac."
        )
    train_df = rt_df[~rt_df["compound"].isin(unseen_ids)].copy()
    eval_sets: Dict[str, pd.DataFrame] = {
        "unseen_compounds": rt_df[rt_df["compound"].isin(unseen_ids)].copy(),
        "anchor_compounds": rt_df[
            rt_df["compound"].isin(dataset.compound_df.index[groups == "anchor"])
        ].copy(),
    }
    metadata = {"unseen_compound_ids": unseen_ids}
    return dataset, hp, train_df, eval_sets, metadata


def prepare_mix_shift(
    args: argparse.Namespace,
    *,
    seed: int,
) -> Tuple[
    SyntheticDataset,
    Dict[str, np.ndarray],
    pd.DataFrame,
    Dict[str, pd.DataFrame],
    Dict[str, List[int]],
]:
    peak_df, compound_df, true_assignments, rt_uncertainties, hp = create_metabolomics_data(
        n_compounds=80,
        n_species=6,
        n_internal_standards=8,
        fixed_runs_per_species_compound=int(args.mix_k),
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
    rt_df = dataset.peak_df[dataset.peak_df["true_compound"].notna()].copy()
    rt_df = rt_df.rename(columns={"true_compound": "compound"})
    rt_df["compound"] = rt_df["compound"].astype(int)
    compound_classes = np.asarray(hp["compound_class"], dtype=int)
    unique_classes = sorted(set(compound_classes.tolist()))
    holdout_count = max(1, int(round(len(unique_classes) * float(args.holdout_frac))))
    holdout_classes = unique_classes[-holdout_count:]
    rt_df["class"] = compound_classes[rt_df["compound"].to_numpy(dtype=int)]
    mask_holdout = rt_df["class"].isin(holdout_classes)
    train_df = rt_df[~mask_holdout].copy()
    eval_sets: Dict[str, pd.DataFrame] = {
        "heldout_classes": rt_df[mask_holdout].copy(),
        "seen_classes": rt_df[~mask_holdout].copy(),
    }
    metadata = {"heldout_classes": holdout_classes}
    return dataset, hp, train_df, eval_sets, metadata


def _bootstrap_ci_mean(
    values: List[float], *, B: int = 2000, alpha: float = 0.05, seed: int = 1234
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean. Returns (lo, hi)."""
    import numpy as _np

    vals = _np.asarray(values, dtype=float)
    clean = vals[_np.isfinite(vals)]
    n = clean.size
    if n == 0:
        return float("nan"), float("nan")
    rng = _np.random.RandomState(int(seed))
    idx = rng.randint(0, n, size=(int(B), n))
    means = clean[idx].mean(axis=1)
    lo = float(_np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(_np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _load_reps_for_plot(
    summary_json: Path, scenario: str, metric: str
) -> Tuple[List[float], List[float]]:
    """Load replicate values from a single summary JSON produced by this script.

    Returns descriptor and no_descriptor arrays for the scenario's target label.
    """
    with summary_json.open() as fp:
        data = json.load(fp)
    target_key = {"cold_start": "unseen_compounds", "mix_shift": "heldout_classes"}[scenario]
    desc_vals: List[float] = []
    nod_vals: List[float] = []
    reps = data.get("replicates", {})
    for _, rep_payload in sorted(
        reps.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0
    ):
        scen = rep_payload.get(scenario, {})
        lab = scen.get(target_key, {})
        d = lab.get("descriptor", {})
        n = lab.get("no_descriptor", {})
        if metric in d and metric in n:
            desc_vals.append(float(d[metric]))
            nod_vals.append(float(n[metric]))
    if not desc_vals:
        raise RuntimeError(f"No replicate values for {scenario} in {summary_json}")
    return desc_vals, nod_vals


def create_combined_plot(
    *,
    cold_desc_vals: List[float],
    cold_nod_vals: List[float],
    mix_desc_vals: List[float],
    mix_nod_vals: List[float],
    metric: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as _np

    def _summary(vals: List[float]) -> Tuple[float, float, float]:
        mean = float(_np.mean(vals)) if len(vals) > 0 else float("nan")
        # Use a single constant seed for simplicity across panels/methods
        lo, hi = _bootstrap_ci_mean(vals, B=2000, alpha=0.05, seed=777)
        return mean, lo, hi

    def _draw(ax, desc_vals, nod_vals, title: str) -> None:
        mean_d, lo_d, hi_d = _summary(desc_vals)
        mean_n, lo_n, hi_n = _summary(nod_vals)
        x = _np.array([0, 1], dtype=float)
        width = 0.6
        colors = ["#1f77b4", "#ff7f0e"]
        means = _np.array([mean_d, mean_n])
        ci_lo = _np.array([mean_d - lo_d, mean_n - lo_n])
        ci_hi = _np.array([hi_d - mean_d, hi_n - mean_n])
        ax.bar(x, means, width=width, color=colors, zorder=2)
        yerr = _np.vstack([_np.maximum(ci_lo, 0.0), _np.maximum(ci_hi, 0.0)])
        ax.errorbar(
            x, means, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=3
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["Descriptor", "No descriptor"])
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(bottom=0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.5), sharey=True)
    _draw(ax1, cold_desc_vals, cold_nod_vals, "Cold Start")
    _draw(ax2, mix_desc_vals, mix_nod_vals, "Mix Shift")
    fig.suptitle("Descriptor vs No Descriptor", y=0.98)
    fig.supylabel(metric.upper())
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    draws, tune, chains = args.draws, args.tune, args.chains
    # Profiles: quick = 500/500/4, full (default) = 1000/1000/4
    if args.quick and (draws is None and tune is None and chains is None):
        draws, tune, chains = 500, 500, 4
    if draws is None:
        draws = 1000
    if tune is None:
        tune = 1000
    if chains is None:
        chains = 4

    # If requested, skip sampling and just build the aggregated plot from a single summary
    # All artifacts are stored flat under output_dir
    scenario_base = args.output_dir
    if args.plot_only:
        try:
            summary = args.plot_input
            if summary is None:
                # pick latest summary in output_dir
                cands = sorted(scenario_base.glob("descriptor_generalization_reps_*.json"))
                if not cands:
                    raise FileNotFoundError("No summary JSONs found; provide --plot-input")
                summary = cands[-1]
            cold_desc, cold_nod = _load_reps_for_plot(summary, "cold_start", args.plot_metric)
            mix_desc, mix_nod = _load_reps_for_plot(summary, "mix_shift", args.plot_metric)
            plot_path = args.plot_output or (scenario_base / f"combined_{args.plot_metric}.png")
            create_combined_plot(
                cold_desc_vals=cold_desc,
                cold_nod_vals=cold_nod,
                mix_desc_vals=mix_desc,
                mix_nod_vals=mix_nod,
                metric=args.plot_metric,
                output_path=plot_path,
            )
            print(f"Saved combined plot to {plot_path}")
        except Exception as exc:  # pragma: no cover
            print(f"Warning: could not generate combined plot ({exc})")
        return

    # Otherwise, run both scenarios with in-run replicates then build the combined figure
    target_keys = {"cold_start": "unseen_compounds", "mix_shift": "heldout_classes"}
    replicates: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    # For plotting arrays
    plot_vals: Dict[str, Dict[str, List[float]]] = {
        "cold_start": {"descriptor": [], "no_descriptor": []},
        "mix_shift": {"descriptor": [], "no_descriptor": []},
    }

    # Prepare datasets once per scenario per replicate for isolation
    for rep in range(int(args.reps)):
        rep_seed = int(args.seed + rep)
        replicates[str(rep)] = {}
        for scenario_name, prep_fn in ("cold_start", prepare_cold_start), (
            "mix_shift",
            prepare_mix_shift,
        ):
            dataset, hp, train_df, eval_sets, metadata = prep_fn(args, seed=rep_seed)
            descriptor_source = dataset.hierarchical_params.get("compound_features")
            descriptors = (
                np.asarray(descriptor_source, dtype=float)
                if descriptor_source is not None
                else None
            )
            run_meta = dataset.run_meta()
            n_species = int(dataset.peak_df["species"].nunique())
            species_cluster = np.asarray(hp["species_cluster"], dtype=int)

            hier_desc, n_use_desc = train_hierarchical(
                dataset,
                hp,
                train_df=train_df,
                draws=draws,
                tune=tune,
                chains=chains,
                seed=rep_seed,
                species_cluster=species_cluster,
                descriptors=descriptors,
                include_class_hierarchy=True,
                global_gamma=False,
            )
            if args.predict_draws:
                n_use_desc = int(min(n_use_desc, int(args.predict_draws)))

            hier_nodesc, n_use_nodesc = train_hierarchical(
                dataset,
                hp,
                train_df=train_df,
                draws=draws,
                tune=tune,
                chains=chains,
                seed=rep_seed + 1,
                species_cluster=species_cluster,
                descriptors=None,
                include_class_hierarchy=False,
                global_gamma=True,
            )
            if args.predict_draws:
                n_use_nodesc = int(min(n_use_nodesc, int(args.predict_draws)))

            scen_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
            for label, df in eval_sets.items():
                if df.empty:
                    continue
                pred_desc, std_desc = evaluate_model(hier_desc, df, n_samples=n_use_desc)
                pred_nodesc, std_nodesc = evaluate_model(hier_nodesc, df, n_samples=n_use_nodesc)
                y_true = df["rt"].to_numpy(dtype=float)
                scen_metrics[label] = {
                    "descriptor": asdict(compute_metrics(y_true, pred_desc, std_desc)),
                    "no_descriptor": asdict(compute_metrics(y_true, pred_nodesc, std_nodesc)),
                    "n_rows": len(df),
                }

            replicates[str(rep)][scenario_name] = scen_metrics

            # Collect values for plotting (target label only)
            tkey = target_keys[scenario_name]
            if tkey in scen_metrics:
                mdesc = scen_metrics[tkey]["descriptor"].get(args.plot_metric)
                mnod = scen_metrics[tkey]["no_descriptor"].get(args.plot_metric)
                if mdesc is not None and mnod is not None:
                    plot_vals[scenario_name]["descriptor"].append(float(mdesc))
                    plot_vals[scenario_name]["no_descriptor"].append(float(mnod))

    # Persist a single summary JSON for this run
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"descriptor_generalization_reps_{timestamp}.json"
    results = {
        "config": {
            "seed": int(args.seed),
            "reps": int(args.reps),
            "draws": int(draws),
            "tune": int(tune),
            "chains": int(chains),
            "predict_draws": int(args.predict_draws) if args.predict_draws else None,
            "desc_tau_beta": float(args.desc_tau_beta),
            "desc_sigma_compound": float(args.desc_sigma_compound),
            "mix_k": int(args.mix_k),
            "plot_metric": str(args.plot_metric),
        },
        "replicates": replicates,
    }
    with json_path.open("w") as fp:
        json.dump(results, fp, indent=2)

    # Print brief summary for current run
    def fmt(m: Dict[str, float]) -> str:
        return f"MAE {m['mae']:.3f} | RMSE {m['rmse']:.3f} | Bias {m['bias']:.3f} | p95 {m['mae_p95']:.3f}"

    # Print brief summary: mean across replicates for both labels
    print("==== Descriptor Generalization (replicate means) ====")
    for scen in ("cold_start", "mix_shift"):
        print(f"\n[{scen}]")
        # Compute means across reps for all labels present in first replicate
        labels = set()
        for rep_payload in replicates.values():
            labels.update(rep_payload.get(scen, {}).keys())
        for label in sorted(labels):
            desc_vals = []
            nod_vals = []
            for rep_payload in replicates.values():
                m = rep_payload.get(scen, {}).get(label, {})
                d = m.get("descriptor", {})
                n = m.get("no_descriptor", {})
                if args.plot_metric in d and args.plot_metric in n:
                    desc_vals.append(float(d[args.plot_metric]))
                    nod_vals.append(float(n[args.plot_metric]))
            if desc_vals and nod_vals:
                print(
                    f"  {label}  →  desc: {np.mean(desc_vals):.3f} | no-desc: {np.mean(nod_vals):.3f} (n={len(desc_vals)})"
                )

    # Build aggregated plot from replicates in this run
    try:
        cold_desc = plot_vals["cold_start"]["descriptor"]
        cold_nod = plot_vals["cold_start"]["no_descriptor"]
        mix_desc = plot_vals["mix_shift"]["descriptor"]
        mix_nod = plot_vals["mix_shift"]["no_descriptor"]
        plot_path = args.plot_output or (scenario_base / f"combined_{args.plot_metric}.png")
        create_combined_plot(
            cold_desc_vals=cold_desc,
            cold_nod_vals=cold_nod,
            mix_desc_vals=mix_desc,
            mix_nod_vals=mix_nod,
            metric=args.plot_metric,
            output_path=plot_path,
        )
        print(f"Saved combined plot to {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"Warning: could not generate combined plot ({exc})")


if __name__ == "__main__":
    main()
