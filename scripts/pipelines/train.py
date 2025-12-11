#!/usr/bin/env python
"""
CompAssign training pipeline for RT regression and hierarchical Bayesian assignment.

This script trains both the hierarchical RT model and the hierarchical Bayesian assignment model.
Optimized for high precision metabolomics compound assignment with minimal features.
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path
from datetime import datetime

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting optional in headless envs
    matplotlib = None
    plt = None

from src.compassign.rt_hierarchical_experimental import HierarchicalRTModel
from src.compassign.peak_assignment import PeakAssignment
from src.compassign.diagnostic_plots import create_all_diagnostic_plots, create_combined_dashboard

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_prep.create_synthetic_data import (
    create_synthetic_dataset,
)


def print_flush(msg):
    """Print with immediate flush for real-time logging"""
    print(msg, flush=True)
    sys.stdout.flush()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train CompAssign hierarchical Bayesian model for metabolomics " "compound assignment"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Synthetic data includes isomers and near-isobars. See docs for details.",
    )

    # Data generation parameters
    parser.add_argument("--n-clusters", type=int, default=8, help="Number of species clusters")
    parser.add_argument(
        "--n-species", type=int, default=40, help="Number of species (~5 per cluster)"
    )
    parser.add_argument("--n-classes", type=int, default=4, help="Number of compound classes")
    parser.add_argument("--n-compounds", type=int, default=20, help="Number of compounds")
    parser.add_argument(
        "--decoy-fraction",
        type=float,
        default=0.5,
        help="Fraction of compounds that are decoys (never appear in samples)",
    )
    parser.add_argument(
        "--mass-error-ppm",
        type=float,
        default=5.0,
        help="Baseline 1σ mass error used in synthetic generator (ppm)",
    )

    # Sampling parameters
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Number of MCMC samples per chain"
    )
    parser.add_argument(
        "--n-chains", type=int, default=None, help="Number of MCMC chains (default: 4)"
    )
    parser.add_argument("--n-tune", type=int, default=1000, help="Number of tuning (burn-in) steps")
    parser.add_argument(
        "--target-accept", type=float, default=0.99, help="Target acceptance rate for NUTS"
    )

    # Model parameters
    parser.add_argument(
        "--mass-tolerance-ppm",
        type=float,
        default=25.0,
        help="Mass tolerance in ppm (default: 25 ppm)",
    )
    parser.add_argument(
        "--rt-window-k",
        type=float,
        default=2.0,
        help="RT window multiplier k*sigma (default: 2.0 for harder data)",
    )
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=0.7,
        help="Probability threshold for assignment",
    )
    parser.add_argument(
        "--max-predictions-per-peak",
        type=int,
        default=2,
        help="Cap the number of predicted compounds per peak (many-to-many default)",
    )
    parser.add_argument(
        "--skip-threshold-scan",
        action="store_true",
        help="Skip generating the precision-recall threshold sweep",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory for results"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


def test_threshold_impact(assignment_model, output_path):
    """Test different thresholds to show precision-recall tradeoff."""

    print_flush("\n" + "=" * 60)
    print_flush("THRESHOLD IMPACT ANALYSIS")
    print_flush("=" * 60)

    thresholds = [0.0, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    results = []

    for thresh in thresholds:
        result = assignment_model.assign(prob_threshold=thresh)
        assigned_peaks = len([v for v in result.assignments.values() if v])
        results.append(
            {
                "threshold": thresh,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "n_assigned_peaks": assigned_peaks,
            }
        )

        print_flush(f"\nThreshold: {thresh:.2f}")
        print_flush(f"  Precision: {result.precision:.3f}")
        print_flush(f"  Recall:    {result.recall:.3f}")
        print_flush(f"  F1:        {result.f1:.3f}")
        print_flush(f"  Assigned peaks:  {assigned_peaks}")

    df = pd.DataFrame(results)
    results_dir = output_path / "results"
    analysis_path = results_dir / "threshold_analysis.csv"
    df.to_csv(analysis_path, index=False)

    # Compute AUC and save JSON summary
    df_sorted = df.sort_values("recall").reset_index(drop=True)
    auc = float(np.trapz(df_sorted["precision"].to_numpy(), df_sorted["recall"].to_numpy()))
    summary = {
        "threshold_metrics": df_sorted.to_dict(orient="records"),
        "auc_precision_recall": auc,
    }
    (results_dir / "threshold_metrics.json").write_text(json.dumps(summary, indent=2))
    print_flush(f"\nSaved threshold metrics to {analysis_path} (AUC={auc:.3f})")

    # Optional PR plot
    if plt is None:
        print_flush("Warning: could not generate PR plot (matplotlib unavailable)")
    else:
        try:
            plt.figure(figsize=(6, 4))
            plt.plot(df_sorted["recall"], df_sorted["precision"], marker="o", linewidth=1.5)
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.05)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve (AUC={auc:.3f})")
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            plot_path = output_path / "plots" / "precision_recall_curve.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print_flush(f"Saved PR curve to {plot_path}")
        except Exception as exc:  # pragma: no cover
            print_flush(f"Warning: could not generate PR plot ({exc})")

    print_flush("\n" + "=" * 60)
    print_flush("Higher thresholds increase precision at the cost of recall")
    print_flush("=" * 60)


def main():
    """Main training pipeline."""
    args = parse_args()

    # Setup output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "models").mkdir(exist_ok=True)
    (output_path / "plots").mkdir(exist_ok=True)
    (output_path / "results").mkdir(exist_ok=True)

    # Save configuration
    config = vars(args)
    config["timestamp"] = datetime.now().isoformat()
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Start training

    # 1. Generate synthetic data
    print_flush("\n1. Generating synthetic data...")
    dataset = create_synthetic_dataset(
        n_compounds=args.n_compounds,
        n_species=args.n_species,
        n_peaks_per_compound=3,
        n_noise_peaks=max(200, args.n_compounds * 10),  # More noise peaks
        isomer_fraction=0.4,  # 40% of compounds are isomers (increased)
        near_isobar_fraction=0.3,  # 30% are near-isobars (increased)
        mass_error_ppm=args.mass_error_ppm,
        rt_uncertainty_range=(0.2, 0.8),  # Higher RT uncertainty
        decoy_fraction=args.decoy_fraction,  # Use command-line parameter
    )
    peak_df = dataset.peak_df
    compound_df = dataset.compound_df
    hierarchical_params = dataset.hierarchical_params

    # Extract run-level metadata (covariates embedded in peak_df)
    run_meta = dataset.run_meta()
    run_df = run_meta.df
    run_covariate_cols = run_meta.covariate_columns

    # Adapt to expected format
    compound_info = compound_df
    # Create RT observations from peaks (for RT model training)
    rt_df = peak_df[peak_df["true_compound"].notna()].copy()
    rt_df = rt_df.rename(columns={"true_compound": "compound"})
    rt_df = rt_df.reset_index(drop=True)
    if "run" not in rt_df.columns:
        raise ValueError("Synthetic generator must provide a 'run' column")
    rt_df["run"] = rt_df["run"].astype(int)
    rt_df["species"] = rt_df["species"].astype(int)
    rt_df["compound"] = rt_df["compound"].astype(int)
    rt_df = rt_df[["species", "compound", "run", "rt"]]

    # Print data summary
    true_assignments = peak_df["true_compound"].notna().sum()
    noise_peaks = peak_df["true_compound"].isna().sum()
    print_flush(f"  Observations: {len(rt_df)}")
    print_flush(f"  Peaks: {len(peak_df)}")
    print_flush(f"  True assignments: {true_assignments}")
    print_flush(f"  Noise peaks: {noise_peaks}")
    print_flush(f"  Compounds: {args.n_compounds}")

    # Count isomers and near-isobars
    isomer_count = 0
    near_isobar_count = 0
    for i in range(len(compound_info)):
        for j in range(i + 1, len(compound_info)):
            mass_diff = abs(compound_info.iloc[i]["true_mass"] - compound_info.iloc[j]["true_mass"])
            if mass_diff < 0.001:  # Same formula (isomers)
                isomer_count += 1
            elif mass_diff < 0.01:  # Near-isobars
                near_isobar_count += 1

    print_flush(f"  - Isomers: {isomer_count}")
    print_flush(f"  - Near-isobars: {near_isobar_count}")

    # 2. Train RT model
    print_flush("\n2. Training hierarchical RT model...")

    # Optional descriptor features (20D) from synthetic generator
    compound_features = hierarchical_params.get("compound_features")

    rt_model = HierarchicalRTModel(
        n_clusters=hierarchical_params["n_clusters"],
        n_species=args.n_species,
        n_classes=hierarchical_params["n_classes"],
        n_compounds=args.n_compounds,
        species_cluster=hierarchical_params["species_cluster"],
        compound_class=hierarchical_params["compound_class"],
        run_metadata=run_df,
        run_covariate_columns=run_covariate_cols,
        compound_features=compound_features,
    )

    # Build model with RT observations
    rt_model.build_model(rt_df)

    print_flush(f"Sampling with {args.n_chains or 4} chains, {args.n_samples} samples each...")
    used_target_accept = args.target_accept  # Honor CLI parameter as-is
    print_flush(f"Target accept: {used_target_accept}, Max treedepth: 15")

    sample_kwargs = {
        "n_samples": args.n_samples,
        "n_tune": args.n_tune,
        "target_accept": used_target_accept,
        "max_treedepth": 15,
        "random_seed": args.seed,
    }
    # Always pass an explicit chain count to keep logs consistent
    sample_kwargs["n_chains"] = args.n_chains or 4

    trace_rt = rt_model.sample(**sample_kwargs)

    # Save RT trace
    trace_rt.to_netcdf(output_path / "models" / "rt_trace.nc")

    # 2b. MCMC convergence/divergence diagnostics (gating)
    print_flush("\n2b. Checking RT MCMC diagnostics...")
    try:
        diag = {}
        summary = az.summary(trace_rt)
        rhat = summary["r_hat"].dropna()
        ess = summary["ess_bulk"].dropna()
        rhat_threshold = 1.01
        ess_threshold = 200
        rhat_viol = rhat[rhat > rhat_threshold].to_dict()
        ess_viol = ess[ess < ess_threshold].to_dict()
        divergences = (
            int(trace_rt.sample_stats["diverging"].values.sum())
            if "diverging" in trace_rt.sample_stats
            else 0
        )

        diag["rhat_max"] = float(rhat.max()) if len(rhat) else None
        diag["ess_min"] = float(ess.min()) if len(ess) else None
        diag["rhat_threshold"] = rhat_threshold
        diag["ess_threshold"] = ess_threshold
        diag["rhat_violations"] = {k: float(v) for k, v in list(rhat_viol.items())[:20]}
        diag["ess_violations"] = {k: float(v) for k, v in list(ess_viol.items())[:20]}
        diag["divergences"] = divergences

        gating_failed = divergences > 0 or len(rhat_viol) > 0 or len(ess_viol) > 0
        diag["gating_failed"] = gating_failed

        with open(output_path / "results" / "rt_mcmc_diagnostics.json", "w") as f:
            json.dump(diag, f, indent=2)

        if gating_failed:
            print_flush("  WARNING: MCMC diagnostics outside thresholds:")
            print_flush(f"    divergences: {divergences}")
            if rhat_viol:
                message = (
                    f"    r_hat violations (>{rhat_threshold}): {len(rhat_viol)} vars; "
                    f"max={diag['rhat_max']:.3f}"
                )
                print_flush(message)
            if ess_viol:
                message = (
                    f"    ESS violations (<{ess_threshold}): {len(ess_viol)} vars; "
                    f"min={diag['ess_min']:.1f}"
                )
                print_flush(message)
        else:
            print_flush(
                "  MCMC diagnostics look good (no divergences; r_hat/ESS within thresholds)"
            )
    except Exception as e:
        print_flush(f"  Warning: failed to compute MCMC diagnostics: {e}")

    # 3. Posterior predictive checks + diagnostic plots
    print_flush("\n3. RT posterior predictive checks + diagnostics...")
    ppc_results = rt_model.posterior_predictive_check(rt_df, random_seed=args.seed)

    # Compute per-species PPC metrics
    try:
        y_true = ppc_results["y_true"]
        pred_mean = ppc_results["pred_mean"]
        pred_lo = ppc_results["pred_lower_95"]
        pred_hi = ppc_results["pred_upper_95"]
        species_arr = rt_df["species"].values

        by_species = {}
        for s in np.unique(species_arr):
            m = species_arr == s
            if m.sum() == 0:
                continue
            rmse_s = float(np.sqrt(np.mean((pred_mean[m] - y_true[m]) ** 2)))
            mae_s = float(np.mean(np.abs(pred_mean[m] - y_true[m])))
            cov_s = float(np.mean((y_true[m] >= pred_lo[m]) & (y_true[m] <= pred_hi[m])))
            by_species[int(s)] = {"rmse": rmse_s, "mae": mae_s, "coverage_95": cov_s}

        ppc_summary = {
            "overall": {
                "rmse": float(ppc_results["rmse"]),
                "mae": float(ppc_results["mae"]),
                "coverage_95": float(ppc_results["coverage_95"]),
            },
            "by_species": by_species,
            "timestamp": datetime.now().isoformat(),
        }

        (output_path / "results").mkdir(exist_ok=True)
        with open(output_path / "results" / "rt_ppc_summary.json", "w") as f:
            json.dump(ppc_summary, f, indent=2)

        print_flush(
            f"  RT RMSE: {ppc_summary['overall']['rmse']:.3f}, "
            f"MAE: {ppc_summary['overall']['mae']:.3f}, "
            f"95% coverage: {ppc_summary['overall']['coverage_95']*100:.1f}%"
        )

        # Consolidate RT metrics into single file
        rt_metrics = {
            "rmse": float(ppc_results["rmse"]),
            "mae": float(ppc_results["mae"]),
            "coverage_95": float(ppc_results["coverage_95"]),
            "rhat_max": float(diag.get("rhat_max", 0.0)) if "diag" in locals() else None,
            "n_divergences": int(diag.get("divergences", 0)) if "diag" in locals() else 0,
            "by_species": by_species,
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_path / "results" / "rt_metrics.json", "w") as f:
            json.dump(rt_metrics, f, indent=2)

    except Exception as e:
        print_flush(f"  Warning: failed to compute per-species PPC metrics: {e}")
        rt_metrics = {"error": str(e), "timestamp": datetime.now().isoformat()}

    # Create diagnostic plots with PPC (including pooling diagnostics)
    create_all_diagnostic_plots(trace_rt, ppc_results, output_path, obs_df=rt_df)

    # 4. Train softmax assignment model
    print_flush("\n4. Training peak assignment model...")

    # Initialize hierarchical Bayesian model
    assignment_model = PeakAssignment(
        mass_tolerance_ppm=args.mass_tolerance_ppm,
        rt_window_k=args.rt_window_k,
        random_seed=args.seed,
    )

    # Compute RT predictions
    rt_predictions = assignment_model.compute_rt_predictions(
        trace_rt=trace_rt,
        n_species=args.n_species,
        n_compounds=args.n_compounds,
        run_metadata=run_df,
        run_covariate_columns=run_covariate_cols,
        rt_model=rt_model,
    )

    # Generate training data
    train_pack = assignment_model.generate_training_data(
        peak_df=peak_df,
        compound_mass=compound_info["true_mass"].values,
        n_compounds=args.n_compounds,
        compound_info=compound_info,  # Pass compound_info to skip decoys during training
        initial_labeled_fraction=0.8,  # 80% labeled for training
    )

    # Report candidate complexity (training = no decoys, test = with decoys)
    mask_train = train_pack["mask"]
    mask_test = train_pack.get("mask_test", mask_train)
    train_candidate_counts = mask_train[:, 1:].sum(axis=1)
    test_candidate_counts = mask_test[:, 1:].sum(axis=1)

    def print_candidate_stats(label: str, counts: np.ndarray) -> None:
        counts = counts.astype(int)
        if counts.size == 0:
            return
        print_flush(
            f"  {label} candidates -> mean {np.mean(counts):.2f}, median {np.median(counts):.1f}, "
            f"min {counts.min()}, max {counts.max()}"
        )
        print_flush(
            f"    distribution: 0={np.sum(counts == 0)}, 1={np.sum(counts == 1)}, "
            f"2+={np.sum(counts >= 2)}, 5+={np.sum(counts >= 5)}, 10+={np.sum(counts >= 10)}"
        )

    print_flush("  Candidate set summary:")
    print_candidate_stats("train", train_candidate_counts)
    print_candidate_stats("test", test_candidate_counts)

    # Analyze raw feature distributions (pre-standardization) for true vs. decoy candidates
    feature_diag = getattr(assignment_model, "feature_diagnostics", pd.DataFrame()).copy()
    if not feature_diag.empty:
        feature_cols = assignment_model.feature_names
        summary = feature_diag.groupby("is_true")[feature_cols].agg(["mean", "std", "median"])
        # Flatten multi-index columns for readability
        summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
        summary_path = output_path / "results" / "feature_summary.csv"
        summary.to_csv(summary_path)
        print_flush(f"\n  Saved feature summary to {summary_path}")
        key_metrics = [
            m for m in ("has_isotope", "n_adducts", "rt_cluster_size") if m in feature_cols
        ]
        for metric in key_metrics:
            pos_mean = (
                summary.loc[1.0, f"{metric}_mean"] if (1.0 in summary.index) else float("nan")
            )
            neg_mean = (
                summary.loc[0.0, f"{metric}_mean"] if (0.0 in summary.index) else float("nan")
            )
            print_flush(f"    {metric}: mean(true)={pos_mean:.3f}, mean(false)={neg_mean:.3f}")

    # Calculate RT recall ceiling - what fraction of true compounds survive the mass/RT windows?
    print_flush("\n  Calculating RT recall ceiling...")
    rt_recall_ceiling = 0.0
    n_peaks_with_true = 0

    # Check if true compounds are in the candidate sets
    true_compounds = train_pack.get("true_compounds", np.array([]))
    row_to_candidates_test = train_pack.get("row_to_candidates_test", [])

    # Handle both numpy arrays and lists
    has_true_compounds = (isinstance(true_compounds, np.ndarray) and true_compounds.size > 0) or (
        isinstance(true_compounds, list) and len(true_compounds) > 0
    )
    has_candidates = len(row_to_candidates_test) > 0

    if has_true_compounds and has_candidates:
        for i, true_comp in enumerate(true_compounds):
            if true_comp is not None:  # Skip null peaks
                n_peaks_with_true += 1
                # Check if true compound is in test candidates (position 0 is null)
                if i < len(row_to_candidates_test):
                    candidates = row_to_candidates_test[i][1:]  # Skip null at position 0
                    if true_comp in candidates:
                        rt_recall_ceiling += 1

        if n_peaks_with_true > 0:
            rt_recall_ceiling = rt_recall_ceiling / n_peaks_with_true
            survivors = int(rt_recall_ceiling * n_peaks_with_true)
            message = (
                f"    RT recall ceiling: {rt_recall_ceiling:.3f} "
                f"({survivors}/{n_peaks_with_true} true compounds survive filtering)"
            )
            print_flush(message)
        else:
            print_flush("    No peaks with true compounds to evaluate recall ceiling")
    else:
        print_flush("    Unable to calculate RT recall ceiling (missing data structures)")

    # Store RT recall ceiling in metrics
    if "rt_metrics" in locals():
        rt_metrics["rt_recall_ceiling"] = float(rt_recall_ceiling)
        with open(output_path / "results" / "rt_metrics.json", "w") as f:
            json.dump(rt_metrics, f, indent=2)

    # Save presence prior snapshot
    try:
        assignment_model.presence.save(str(output_path / "models" / "presence_prior.npz"))
    except Exception as e:
        print_flush(f"Warning: could not save presence prior: {e}")

    # Build and sample
    assignment_model.build_model()
    trace_assignment = assignment_model.sample(
        draws=args.n_samples,
        tune=args.n_tune,
        chains=args.n_chains or 4,  # Use 4 chains by default like RT model
        target_accept=args.target_accept,
    )

    # Get predictions (pass compound_info to properly handle decoys)
    results = assignment_model.assign(
        prob_threshold=args.probability_threshold,
        compound_info=compound_info,
        max_predictions_per_peak=args.max_predictions_per_peak,
    )

    # Save traces and results
    trace_assignment.to_netcdf(output_path / "models" / "assignment_trace.nc")

    # Save peak assignments (many-to-many): compound list and compound:prob list
    peak_id_array = assignment_model.train_pack["peak_ids"]
    rows = []
    candidates_map = assignment_model.train_pack.get(
        "row_to_candidates_test", assignment_model.train_pack["row_to_candidates"]
    )
    mask_test = assignment_model.train_pack.get("mask_test", assignment_model.train_pack["mask"])
    for pid in peak_id_array:
        pid = int(pid)
        assigned = results.assignments.get(pid, [])
        row = assignment_model.train_pack["peak_to_row"][pid]
        valid_idx = np.where(mask_test[row])[0]
        probs_vec = results.per_peak_probs.get(pid, np.array([]))
        comp_prob = {}
        for j, k in enumerate(valid_idx):
            if k == 0:
                continue
            c = candidates_map[row][k]
            if c is None:
                continue
            comp_prob[int(c)] = float(probs_vec[j])
        assigned_str = ";".join(str(int(c)) for c in assigned)
        assigned_probs_str = ";".join(
            f"{int(c)}:{comp_prob.get(int(c), 0.0):.4f}" for c in assigned
        )
        rows.append(
            {
                "peak_id": pid,
                "assigned_compounds": assigned_str,
                "assigned_with_probs": assigned_probs_str,
                "top_prob": results.top_prob.get(pid, 0.0),
            }
        )
    assignment_df = pd.DataFrame(rows)
    assignment_df.to_csv(output_path / "results" / "peak_assignments.csv", index=False)

    # Save compound information (including decoy status)
    compound_info.to_csv(output_path / "results" / "compound_info.csv", index=False)

    # Analyze confidence distributions for real vs decoy assignments
    if "is_decoy" in compound_info.columns and results.peaks_by_compound:
        decoy_ids = set(compound_info[compound_info["is_decoy"]]["compound_id"].values)
        real_ids = set(compound_info[~compound_info["is_decoy"]]["compound_id"].values)

        decoy_probs = []
        real_probs = []

        for cid, peak_ids in results.peaks_by_compound.items():
            for pid in peak_ids:
                prob = results.top_prob.get(pid, 0.0)
                if cid in decoy_ids:
                    decoy_probs.append(prob)
                elif cid in real_ids:
                    real_probs.append(prob)

        # Save confidence analysis
        confidence_stats = {
            "real_assignments": {
                "count": len(real_probs),
                "mean_confidence": np.mean(real_probs) if real_probs else 0,
                "std_confidence": np.std(real_probs) if real_probs else 0,
                "min_confidence": min(real_probs) if real_probs else 0,
                "max_confidence": max(real_probs) if real_probs else 0,
            },
            "decoy_assignments": {
                "count": len(decoy_probs),
                "mean_confidence": np.mean(decoy_probs) if decoy_probs else 0,
                "std_confidence": np.std(decoy_probs) if decoy_probs else 0,
                "min_confidence": min(decoy_probs) if decoy_probs else 0,
                "max_confidence": max(decoy_probs) if decoy_probs else 0,
            },
        }

        with open(output_path / "results" / "confidence_analysis.json", "w") as f:
            json.dump(confidence_stats, f, indent=2)

    # Save metrics
    metrics = {
        # Pair-based micro metrics
        "precision": float(results.precision),
        "recall": float(results.recall),
        "f1": float(results.f1),
        # Peak-level macro metrics
        "f1_macro": float(getattr(results, "f1_macro", 0.0)),
        "jaccard_macro": float(getattr(results, "jaccard_macro", 0.0)),
        "far_null": float(getattr(results, "far_null", 0.0)),
        "assignment_rate": float(getattr(results, "assignment_rate", 0.0)),
        # Compound-level metrics (PRIMARY)
        "compound_precision": float(results.compound_precision),
        "compound_recall": float(results.compound_recall),
        "compound_f1": float(results.compound_f1),
        "compounds_identified": results.compound_metrics["identified"],
        "compounds_total": results.compound_metrics["total"],
        "compounds_false_positives": results.compound_metrics.get("false_positives", 0),
        "compounds_decoys_assigned": results.compound_metrics.get("decoys_assigned", 0),
        # Coverage
        "mean_coverage": float(results.mean_coverage),
        # Other metrics
        "ece_micro": float(results.ece),
        "ece_ovr": float(getattr(results, "ece_ovr", 0.0)),
        "brier_ovr": float(getattr(results, "brier_ovr", 0.0)),
        "cardinality_mae": float(getattr(results, "cardinality_mae", 0.0)),
        "true_positives_pairs": int(results.confusion_matrix.get("TP", 0)),
        "false_positives_pairs": int(results.confusion_matrix.get("FP", 0)),
        "false_negatives_pairs": int(results.confusion_matrix.get("FN", 0)),
        "timestamp": datetime.now().isoformat(),
        "model": "hierarchical_bayesian",
        "mass_tolerance_ppm": args.mass_tolerance_ppm,
        "rt_window_k": args.rt_window_k,
        "probability_threshold": args.probability_threshold,
        "max_predictions_per_peak": args.max_predictions_per_peak,
    }

    with open(output_path / "results" / "assignment_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Create combined metrics JSON with stage analysis
    print_flush("\n  Creating combined metrics and stage analysis...")

    # Determine stage weakness
    rt_recall_ceiling_value = (
        rt_metrics.get("rt_recall_ceiling", 1.0) if "rt_metrics" in locals() else 1.0
    )
    assignment_recall_value = metrics["recall"]

    # Stage weakness identification logic
    rt_loss = 1.0 - rt_recall_ceiling_value
    assignment_loss = rt_recall_ceiling_value - assignment_recall_value

    if rt_loss > 0.05:  # RT loses more than 5%
        if assignment_loss > 0.05:  # Assignment also loses more than 5%
            stage_analysis = "Both"
            notes = (
                "Both stages need improvement. "
                f"RT loses {rt_loss*100:.1f}% recall, "
                f"assignment loses additional {assignment_loss*100:.1f}%."
            )
        else:
            stage_analysis = "RT"
            notes = (
                "RT model is the primary bottleneck, "
                f"losing {rt_loss*100:.1f}% of true compounds during filtering."
            )
    else:
        if assignment_loss > 0.10:  # Assignment loses more than 10%
            stage_analysis = "Assignment"
            notes = (
                "Assignment model needs improvement, "
                f"losing {assignment_loss*100:.1f}% despite good RT filtering."
            )
        else:
            stage_analysis = "Good"
            notes = (
                f"Both stages performing well. Total recall: {assignment_recall_value*100:.1f}%."
            )

    # Create combined metrics JSON
    combined_metrics = {
        "rt_metrics": rt_metrics
        if "rt_metrics" in locals()
        else {"error": "RT metrics not available"},
        "assignment_metrics": {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "calibration_error": metrics.get("ece_ovr", 0.0),
            "confusion_matrix": {
                "TP": metrics["true_positives_pairs"],
                "FP": metrics["false_positives_pairs"],
                "FN": metrics["false_negatives_pairs"],
            },
            "brier_ovr": metrics.get("brier_ovr", 0.0),
        },
        "stage_ceiling": {"rt_recall_ceiling": rt_recall_ceiling_value, "analysis": stage_analysis},
        "notes": notes,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path / "results" / "combined_metrics.json", "w") as f:
        json.dump(combined_metrics, f, indent=2)

    print_flush(f"    Stage analysis: {stage_analysis}")
    print_flush(f"    {notes}")

    # Create combined dashboard visualization
    try:
        create_combined_dashboard(
            rt_metrics=rt_metrics if "rt_metrics" in locals() else {},
            assignment_metrics=metrics,
            ppc_results=ppc_results if "ppc_results" in locals() else {},
            output_path=output_path,
        )
    except Exception as e:
        print_flush(f"    Warning: Could not create dashboard: {e}")

    # 5. Print results (concise version)
    print_flush("\n5. Results:")
    message = (
        f"  Pair F1 (micro): {metrics['f1']:.3f}  "
        f"(P={metrics['precision']:.3f}, R={metrics['recall']:.3f})"
    )
    print_flush(message)
    message = (
        "  Peak Jaccard (macro): "
        f"{metrics['jaccard_macro']:.3f}; FAR(null): {metrics['far_null']:.3f}; "
        f"AR: {metrics['assignment_rate']:.3f}"
    )
    print_flush(message)
    message = (
        f"  Compounds: {metrics['compounds_identified']}/{metrics['compounds_total']} "
        f"identified (F1={metrics['compound_f1']:.3f})"
    )
    print_flush(message)
    print_flush(f"  Coverage: {metrics['mean_coverage']:.1%} of peaks per compound")

    # Test threshold impact if requested
    if not args.skip_threshold_scan:
        test_threshold_impact(assignment_model, output_path)

    # Create assignment plots (skip for now - needs fixing)
    # print_flush("\n6. Creating assignment plots...")
    # create_assignment_plots(
    #     results=results,
    #     peak_df=peak_df,
    #     compound_info=compound_info,
    #     output_dir=output_path / "plots" / "assignments"
    # )

    # Print final summary
    print_flush("\n6. Training complete")

    # Calculate dataset statistics
    N, K = assignment_model.train_pack["mask"].shape
    n_valid_slots = assignment_model.train_pack["mask"].sum()
    train_idx = np.where(assignment_model.train_pack["labels"] >= 0)[0]
    n_train = len(train_idx)
    n_test = N - n_train

    print_flush("\nDataset summary:")
    print_flush(f"  Total peaks: {N}")
    print_flush(f"  Max candidates: {K-1} (+ null)")
    print_flush(f"  Valid slots: {n_valid_slots}")
    print_flush(f"  Train peaks: {n_train} ({n_train/N*100:.1f}%)")
    print_flush(f"  Test peaks: {n_test} ({n_test/N*100:.1f}%)")

    print_flush("\nCompound-level performance:")
    message = (
        f"  Compound Precision: {metrics['compound_precision']:.3f} "
        f"({metrics['compound_precision']*100:.1f}%)"
    )
    print_flush(message)
    message = (
        f"  Compound Recall:    {metrics['compound_recall']:.3f} "
        f"({metrics['compound_recall']*100:.1f}%)"
    )
    print_flush(message)
    print_flush(f"  Compound F1:        {metrics['compound_f1']:.3f}")
    print_flush(f"  Mean Coverage:      {metrics['mean_coverage']:.3f}")

    # Show decoy statistics if available
    n_decoys = (
        len(compound_info[compound_info["is_decoy"]]) if "is_decoy" in compound_info.columns else 0
    )
    n_real = (
        len(compound_info[~compound_info["is_decoy"]])
        if "is_decoy" in compound_info.columns
        else len(compound_info)
    )
    decoys_assigned = results.compound_metrics.get("decoys_assigned", 0)

    if n_decoys > 0:
        print_flush("\nDecoy detection:")
        print_flush(f"  Library composition: {n_real} real, {n_decoys} decoys")
        error_rate = decoys_assigned / n_decoys * 100
        message = (
            f"  Decoys incorrectly assigned: {decoys_assigned}/{n_decoys} " f"({error_rate:.1f}%)"
        )
        print_flush(message)
        print_flush(f"  False positives from decoys: {decoys_assigned}")
        print_flush(
            f"  Compound false positives: {results.compound_metrics.get('false_positives', 0)}"
        )

        # Show which decoys were assigned if any
        if decoys_assigned > 0 and hasattr(results, "peaks_by_compound"):
            decoy_ids = set(compound_info[compound_info["is_decoy"]]["compound_id"].values)
            assigned_decoys = [cid for cid in results.peaks_by_compound.keys() if cid in decoy_ids]
            if assigned_decoys:
                preview = sorted(assigned_decoys)[:5]
                suffix = "..." if len(assigned_decoys) > 5 else ""
                print_flush(f"  Decoy IDs assigned: {preview}{suffix}")

        # Show confidence analysis if available
        if Path(output_path / "results" / "confidence_analysis.json").exists():
            with open(output_path / "results" / "confidence_analysis.json") as f:
                conf_stats = json.load(f)

            if conf_stats["real_assignments"]["count"] > 0:
                print_flush("\nConfidence analysis:")
                print_flush("  Real compound assignments:")
                print_flush(
                    f"    Mean confidence: {conf_stats['real_assignments']['mean_confidence']:.3f}"
                )
                conf_range = (
                    conf_stats["real_assignments"]["min_confidence"],
                    conf_stats["real_assignments"]["max_confidence"],
                )
                print_flush(f"    Range: [{conf_range[0]:.3f}, {conf_range[1]:.3f}]")

            if conf_stats["decoy_assignments"]["count"] > 0:
                print_flush("  Decoy compound assignments:")
                print_flush(
                    f"    Mean confidence: {conf_stats['decoy_assignments']['mean_confidence']:.3f}"
                )
                conf_range = (
                    conf_stats["decoy_assignments"]["min_confidence"],
                    conf_stats["decoy_assignments"]["max_confidence"],
                )
                print_flush(f"    Range: [{conf_range[0]:.3f}, {conf_range[1]:.3f}]")

    print_flush("\nPair-level performance:")
    print_flush(f"  Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print_flush(f"  Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print_flush(f"  F1:        {metrics['f1']:.3f}")
    print_flush(f"  ECE (micro): {metrics['ece_micro']:.3f}")
    print_flush(f"  ECE (OVR):      {metrics['ece_ovr']:.3f}")
    print_flush(f"  Brier (OVR):    {metrics['brier_ovr']:.3f}")
    print_flush(f"  Card. MAE:      {metrics['cardinality_mae']:.3f}")

    print_flush(f"\nResults saved to: {output_path}/")


if __name__ == "__main__":
    main()
