#!/usr/bin/env python3
"""Assess active learning benefit versus naive full review."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_prep.create_synthetic_data import create_synthetic_dataset  # noqa: E402

from src.compassign.peak_assignment import PeakAssignment, AssignmentResults
from src.compassign.presence_prior import PresencePrior
from src.compassign.oracles import OptimalOracle
from src.compassign.active_learning import ActiveLearner
from src.compassign.eval_loop import simulate_annotation_round
from src.compassign.rt_hierarchical import HierarchicalRTModel
from src.compassign.run_metadata import RunMetadata

RECALL_MATCH_RATIO = 1.0  # Default to matching 100% of the naive full‑review recall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate active learning vs full-review baseline to count annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default="output/al_assessment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-species", type=int, default=40)
    parser.add_argument("--n-compounds", type=int, default=20)
    # Dataset shape/complexity knobs to better mimic production
    parser.add_argument(
        "--n-peaks-per-compound",
        type=int,
        default=3,
        help="Average primary peaks per compound (more increases presence‑prior leverage)",
    )
    parser.add_argument(
        "--isomer-fraction",
        type=float,
        default=0.4,
        help="Fraction of compounds that are isomers (increases RT ambiguity)",
    )
    parser.add_argument(
        "--near-isobar-fraction",
        type=float,
        default=0.3,
        help="Fraction of compounds that are near‑isobars (increases m/z ambiguity)",
    )
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=1.0,
        help="Scale factor for the number of noise peaks (multiplies default)",
    )
    parser.add_argument(
        "--presence-prob",
        type=float,
        default=0.4,
        help="Probability that a real compound appears in a species (0–1)",
    )
    parser.add_argument(
        "--draws", type=int, default=1000, help="Samples per chain for initial fitting"
    )
    parser.add_argument("--tune", type=int, default=1000, help="Tuning steps for initial fitting")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--mass-error-ppm", type=float, default=5.0)
    parser.add_argument("--decoy-fraction", type=float, default=0.5)
    parser.add_argument(
        "--initial-labeled-fraction",
        type=float,
        default=0.8,
        help="Fraction of peaks seeded with labels before AL (matches training pipeline)",
    )
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=25, help="Annotations per AL round")
    parser.add_argument("--rounds", type=int, default=12, help="Maximum active-learning rounds")
    parser.add_argument(
        "--target-recall-ratio",
        type=float,
        default=RECALL_MATCH_RATIO,
        help=(
            "Fraction of the full-review recall used as the efficiency target. "
            "For example, 0.95 means we measure clicks to reach 95% of the full-review recall."
        ),
    )
    parser.add_argument(
        "--acquisition",
        type=str,
        default="hybrid",
        choices=["hybrid", "entropy", "fp", "lc", "margin", "mi"],
    )
    parser.add_argument(
        "--lambda-fp", type=float, default=0.7, help="FP weight for hybrid acquisition"
    )
    parser.add_argument(
        "--diversity-k", type=int, default=75, help="Top-k pool for diverse sampling"
    )
    parser.add_argument("--quick", action="store_true", help="Use a smaller, faster configuration")
    parser.add_argument("--verbose", action="store_true", help="Print round details from simulator")

    # Candidate-generation knobs (to tune task ambiguity)
    parser.add_argument(
        "--cand-mass-ppm",
        type=float,
        default=25.0,
        help="Mass tolerance (ppm) for candidate matching inside the assignment model",
    )
    parser.add_argument(
        "--cand-rt-k",
        type=float,
        default=2.0,
        help="Retention-time window multiplier (in predictive SD units) for candidate matching",
    )
    return parser.parse_args()


def apply_quick_defaults(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.n_species = 15
    args.n_compounds = 15
    # Principled quick defaults: 4 chains × 500 tune × 500 draws
    args.draws = 500
    args.tune = 500
    args.chains = 4
    args.batch_size = min(args.batch_size, 20)
    args.rounds = min(args.rounds, 10)
    args.initial_labeled_fraction = 0.0


def generate_dataset(
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], RunMetadata]:
    dataset = create_synthetic_dataset(
        n_compounds=args.n_compounds,
        n_species=args.n_species,
        n_peaks_per_compound=args.n_peaks_per_compound,
        n_noise_peaks=int(max(200, args.n_compounds * 10) * args.noise_multiplier),
        isomer_fraction=args.isomer_fraction,
        near_isobar_fraction=args.near_isobar_fraction,
        mass_error_ppm=args.mass_error_ppm,
        rt_uncertainty_range=(0.2, 0.8),
        decoy_fraction=args.decoy_fraction,
        presence_prob=args.presence_prob,
        near_isobar_ppm_max=max(
            5.0,
            min(
                float(args.cand_mass_ppm) * 0.65,
                float(args.cand_mass_ppm) - 1.3 * float(args.mass_error_ppm),
            ),
        ),
        near_isobar_rt_sd=max(0.2, 0.16 * float(args.cand_rt_k)),
    )

    peak_df = dataset.peak_df
    compound_df = dataset.compound_df.sort_values("compound_id").reset_index(drop=True)
    hierarchical_params = dataset.hierarchical_params
    run_meta = dataset.run_meta()
    return peak_df, compound_df, hierarchical_params, run_meta


def compute_rt_predictions(
    peaks_df: pd.DataFrame,
    compound_info: pd.DataFrame,
    hierarchical_params: Dict,
    run_meta: RunMetadata,
    args: argparse.Namespace,
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    rt_model = HierarchicalRTModel(
        n_clusters=hierarchical_params["n_clusters"],
        n_species=args.n_species,
        n_classes=hierarchical_params["n_classes"],
        n_compounds=args.n_compounds,
        species_cluster=hierarchical_params["species_cluster"],
        compound_class=hierarchical_params["compound_class"],
        run_metadata=run_meta.df,
        run_covariate_columns=run_meta.covariate_columns,
        compound_features=hierarchical_params.get("compound_features"),
    )

    rt_df = peaks_df[peaks_df["true_compound"].notna()].copy()
    rt_df = rt_df.rename(columns={"true_compound": "compound"})
    if "run" not in rt_df.columns:
        raise ValueError("peak_df must include a 'run' column")

    rt_model.build_model(rt_df)
    trace_rt = rt_model.sample(
        n_samples=args.draws,
        n_tune=args.tune,
        n_chains=args.chains,
        target_accept=args.target_accept,
        random_seed=args.seed,
    )

    temp_assignment = PeakAssignment(
        mass_tolerance_ppm=25.0,
        rt_window_k=2.0,
        random_seed=args.seed,
    )

    return temp_assignment.compute_rt_predictions(
        trace_rt=trace_rt,
        n_species=args.n_species,
        n_compounds=args.n_compounds,
        run_metadata=run_meta.df,
        run_covariate_columns=run_meta.covariate_columns,
        rt_model=rt_model,
    )


def setup_model(
    peaks_df: pd.DataFrame,
    compound_info: pd.DataFrame,
    compound_mass: np.ndarray,
    rt_predictions: Dict[Tuple[int, int], Tuple[float, float]],
    args: argparse.Namespace,
) -> Tuple[PeakAssignment, Dict[str, np.ndarray]]:
    model = PeakAssignment(
        mass_tolerance_ppm=float(args.cand_mass_ppm),
        rt_window_k=float(args.cand_rt_k),
        random_seed=args.seed,
    )
    model.rt_predictions = dict(rt_predictions)

    n_species = int(peaks_df["species"].nunique())
    presence = PresencePrior.init(n_species, len(compound_mass), smoothing=1.0)
    train_pack = model.generate_training_data(
        peak_df=peaks_df.copy(),
        compound_mass=compound_mass,
        n_compounds=len(compound_mass),
        compound_info=compound_info,
        init_presence=presence,
        initial_labeled_fraction=args.initial_labeled_fraction,
        random_seed=args.seed,
    )

    model.build_model()
    model.sample(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        seed=args.seed,
    )
    patch_sampler(model, args)
    return model, train_pack


def patch_sampler(model: PeakAssignment, args: argparse.Namespace) -> None:
    base_sample = model.__class__.sample

    def custom_sample(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.95,
        seed: int = 42,
    ):
        return base_sample(
            self,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            seed=args.seed,
        )

    model.sample = types.MethodType(custom_sample, model)


def compute_full_metrics(
    model: PeakAssignment,
    threshold: float,
    compound_info: pd.DataFrame,
) -> AssignmentResults:
    labels = model.train_pack["labels"].copy()
    try:
        model.train_pack["labels"][:] = -1
        results = model.assign(
            prob_threshold=threshold,
            compound_info=compound_info,
            max_predictions_per_peak=2,
        )
    finally:
        model.train_pack["labels"][:] = labels
    return results


def build_features_dict(model: PeakAssignment) -> Dict[int, np.ndarray]:
    X = model.train_pack["X"]
    mask = model.train_pack["mask"]
    peak_ids = model.train_pack["peak_ids"]

    p_mean = None
    if hasattr(model, "trace") and model.trace is not None:
        post = model.trace.posterior
        if "r" in post:
            p_mean = post["r"].values.mean(axis=(0, 1))
        elif "p" in post:
            p_mean = post["p"].values.mean(axis=(0, 1))
    if p_mean is None:
        probs_dict = model.predict_probs()
        p_mean = np.zeros_like(mask, dtype=float)
        for i, pid in enumerate(peak_ids):
            probs = probs_dict.get(int(pid))
            if probs is None:
                continue
            valid_k = np.where(mask[i])[0]
            length = min(len(probs), len(valid_k))
            p_mean[i, valid_k[:length]] = probs[:length]

    features_dict: Dict[int, np.ndarray] = {}
    for i, pid in enumerate(peak_ids):
        valid_k = np.where(mask[i])[0]
        k_nonnull = valid_k[valid_k > 0]
        if len(k_nonnull) == 0:
            features_dict[int(pid)] = np.zeros(X.shape[-1])
            continue
        weights = p_mean[i, k_nonnull]
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            weights = np.full_like(weights, 1.0 / len(weights))
        else:
            weights = weights / weight_sum
        feats = (weights[:, None] * X[i, k_nonnull, :]).sum(axis=0)
        features_dict[int(pid)] = feats
    return features_dict


def run_naive_review(
    model: PeakAssignment,
    oracle: OptimalOracle,
    peaks_df: pd.DataFrame,
    compound_mass: np.ndarray,
    threshold: float,
    compound_info: pd.DataFrame,
    verbose: bool,
) -> Tuple[AssignmentResults, AssignmentResults, int]:
    """Simulate a full review once (no progression), used only to get the endpoint target."""
    baseline = compute_full_metrics(model, threshold, compound_info)
    all_peak_ids = [int(pid) for pid in model.train_pack["peak_ids"]]
    round_result = simulate_annotation_round(
        assignment_model=model,
        oracle=oracle,
        batch_peak_ids=all_peak_ids,
        peak_df=peaks_df,
        compound_mass=compound_mass,
        threshold=threshold,
        refit_model=True,
        verbose=verbose,
    )
    metrics_after = compute_full_metrics(model, threshold, compound_info)
    return baseline, metrics_after, round_result.n_annotations


def run_random_review(
    model: PeakAssignment,
    oracle: Oracle,
    peaks_df: pd.DataFrame,
    compound_mass: np.ndarray,
    threshold: float,
    compound_info: pd.DataFrame,
    args: argparse.Namespace,
    total_annotations: int,
    verbose: bool,
) -> Tuple[List[Dict[str, float]], AssignmentResults]:
    rng = np.random.default_rng(args.seed)
    labels = model.train_pack["labels"]
    peak_to_row = model.train_pack["peak_to_row"]
    unlabeled = [
        int(pid) for pid in model.train_pack["peak_ids"] if labels[peak_to_row[int(pid)]] < 0
    ]

    history: List[Dict[str, float]] = []
    cumulative = 0

    initial_metrics = compute_full_metrics(model, threshold, compound_info)
    history.append(
        {
            "round": 0,
            "annotations_this_round": 0,
            "cumulative_annotations": 0,
            "precision": float(initial_metrics.precision),
            "recall": float(initial_metrics.recall),
            "f1": float(initial_metrics.f1),
        }
    )

    remaining = set(unlabeled)
    batch_size = args.batch_size

    for round_idx in range(1, args.rounds + 1):
        available = list(remaining)
        if not available:
            break
        size = min(batch_size, len(available), total_annotations - cumulative)
        if size <= 0:
            break
        batch = rng.choice(available, size=size, replace=False).tolist()

        round_result = simulate_annotation_round(
            assignment_model=model,
            oracle=oracle,
            batch_peak_ids=batch,
            peak_df=peaks_df,
            compound_mass=compound_mass,
            threshold=threshold,
            refit_model=True,
            verbose=verbose,
        )

        cumulative += round_result.n_annotations
        remaining.difference_update(batch)

        metrics_after = compute_full_metrics(model, threshold, compound_info)
        history.append(
            {
                "round": round_idx,
                "annotations_this_round": round_result.n_annotations,
                "cumulative_annotations": cumulative,
                "precision": float(metrics_after.precision),
                "recall": float(metrics_after.recall),
                "f1": float(metrics_after.f1),
            }
        )

        if cumulative >= total_annotations:
            break

    final_metrics = compute_full_metrics(model, threshold, compound_info)
    return history, final_metrics


def run_active_learning(
    model: PeakAssignment,
    oracle: OptimalOracle,
    peaks_df: pd.DataFrame,
    compound_mass: np.ndarray,
    threshold: float,
    compound_info: pd.DataFrame,
    args: argparse.Namespace,
    recall_target: float,
    verbose: bool,
) -> Tuple[List[Dict[str, float]], AssignmentResults]:
    peak_to_row = model.train_pack["peak_to_row"]
    learner = ActiveLearner(
        acquisition_fn=args.acquisition,
        threshold=threshold,
        lambda_fp=args.lambda_fp,
        diversity_k=args.diversity_k,
    )
    history: List[Dict[str, float]] = []
    cumulative = 0

    initial_metrics = compute_full_metrics(model, threshold, compound_info)
    history.append(
        {
            "round": 0,
            "annotations_this_round": 0,
            "cumulative_annotations": 0,
            "precision": float(initial_metrics.precision),
            "recall": float(initial_metrics.recall),
            "f1": float(initial_metrics.f1),
        }
    )

    for round_idx in range(1, args.rounds + 1):
        probs_dict = model.predict_probs()
        labels = model.train_pack["labels"]
        available_probs = {}
        for peak_id, probs in probs_dict.items():
            row = peak_to_row[int(peak_id)]
            if labels[row] < 0:
                available_probs[int(peak_id)] = probs
        if not available_probs:
            break

        features_dict = build_features_dict(model)
        filtered_features = {pid: features_dict[pid] for pid in available_probs.keys()}
        batch_size = min(args.batch_size, len(available_probs))
        prob_samples_dict = None
        if args.acquisition == "mi":
            all_samples = model.predict_prob_samples()
            prob_samples_dict = {pid: all_samples[pid] for pid in available_probs.keys()}
        batch = learner.select_next_batch(
            probs_dict=available_probs,
            batch_size=batch_size,
            features_dict=filtered_features,
            prob_samples_dict=prob_samples_dict,
        )
        if not batch:
            break

        round_result = simulate_annotation_round(
            assignment_model=model,
            oracle=oracle,
            batch_peak_ids=batch,
            peak_df=peaks_df,
            compound_mass=compound_mass,
            threshold=threshold,
            refit_model=True,
            verbose=verbose,
        )
        cumulative += round_result.n_annotations
        metrics_after = compute_full_metrics(model, threshold, compound_info)
        history.append(
            {
                "round": round_idx,
                "annotations_this_round": round_result.n_annotations,
                "cumulative_annotations": cumulative,
                "precision": float(metrics_after.precision),
                "recall": float(metrics_after.recall),
                "f1": float(metrics_after.f1),
            }
        )
        # Do not early-stop; continue to build the full curve for readability/comparison

    final_metrics = compute_full_metrics(model, threshold, compound_info)
    return history, final_metrics


def area_under_recall_curve(points: List[Dict[str, float]]) -> float:
    if len(points) < 2:
        return 0.0
    xs = np.array([p["cumulative_annotations"] for p in points], dtype=float)
    ys = np.array([p["recall"] for p in points], dtype=float)
    return float(np.trapz(ys, xs))


def assign_to_dict(result: AssignmentResults) -> Dict[str, float]:
    return {
        "precision": float(result.precision),
        "recall": float(result.recall),
        "f1": float(result.f1),
        "assignment_rate": float(result.assignment_rate),
        "compound_precision": float(result.compound_precision),
        "compound_recall": float(result.compound_recall),
        "compound_f1": float(result.compound_f1),
        "ece": float(result.ece),
    }


def main() -> None:
    args = parse_args()
    apply_quick_defaults(args)

    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    peaks_df, compound_info, hierarchical_params, run_meta = generate_dataset(args)
    compound_mass = compound_info["true_mass"].to_numpy()

    print("\nDataset snapshot:")
    print(f"  Peaks: {len(peaks_df)}")
    print(f"  Species: {peaks_df['species'].nunique()}")
    print(f"  Compounds: {len(compound_info)}")

    rt_predictions = compute_rt_predictions(
        peaks_df=peaks_df,
        compound_info=compound_info,
        hierarchical_params=hierarchical_params,
        run_meta=run_meta,
        args=args,
    )

    oracle = OptimalOracle()

    naive_model, train_pack = setup_model(
        peaks_df=peaks_df,
        compound_info=compound_info,
        compound_mass=compound_mass,
        rt_predictions=rt_predictions,
        args=args,
    )

    mask_train = train_pack["mask"]
    train_candidate_counts = mask_train[:, 1:].sum(axis=1)
    mean_candidates = float(np.mean(train_candidate_counts)) if train_candidate_counts.size else 0.0
    print(f"  Mean candidates (train): {mean_candidates:.2f}")

    baseline_naive, naive_final, naive_clicks = run_naive_review(
        model=naive_model,
        oracle=oracle,
        peaks_df=peaks_df,
        compound_mass=compound_mass,
        threshold=args.threshold,
        compound_info=compound_info,
        verbose=args.verbose,
    )

    naive_metrics = assign_to_dict(naive_final)
    print("\nNaive full-review baseline:")
    print(f"  Total annotations: {naive_clicks}")
    print(f"  Recall after review: {naive_final.recall:.3f}")

    random_model, _ = setup_model(
        peaks_df=peaks_df,
        compound_info=compound_info,
        compound_mass=compound_mass,
        rt_predictions=rt_predictions,
        args=args,
    )
    random_history, random_final = run_random_review(
        model=random_model,
        oracle=oracle,
        peaks_df=peaks_df,
        compound_mass=compound_mass,
        threshold=args.threshold,
        compound_info=compound_info,
        args=args,
        total_annotations=args.batch_size * args.rounds,
        verbose=args.verbose,
    )

    random_metrics = assign_to_dict(random_final)
    print("\nRandom sampling baseline:")
    print(
        f"  Total annotations: {random_history[-1]['cumulative_annotations'] if random_history else 0}"
    )
    print(f"  Final recall: {random_final.recall:.3f}")

    al_model, _ = setup_model(
        peaks_df=peaks_df,
        compound_info=compound_info,
        compound_mass=compound_mass,
        rt_predictions=rt_predictions,
        args=args,
    )
    history, al_final = run_active_learning(
        model=al_model,
        oracle=oracle,
        peaks_df=peaks_df,
        compound_mass=compound_mass,
        threshold=args.threshold,
        compound_info=compound_info,
        args=args,
        recall_target=float(naive_final.recall),
        verbose=args.verbose,
    )

    clicks_to_target = None
    for entry in history:
        if entry["recall"] >= naive_final.recall:
            clicks_to_target = entry["cumulative_annotations"]
            break

    ratio_target = naive_final.recall * float(args.target_recall_ratio)
    clicks_to_ratio = None
    for entry in history:
        if entry["recall"] >= ratio_target:
            clicks_to_ratio = entry["cumulative_annotations"]
            break

    random_clicks_to_ratio = None
    for entry in random_history:
        if entry["recall"] >= ratio_target:
            random_clicks_to_ratio = entry["cumulative_annotations"]
            break

    final_annotations = history[-1]["cumulative_annotations"] if history else 0
    random_final_annotations = random_history[-1]["cumulative_annotations"] if random_history else 0

    def format_clicks(value: int | None) -> str:
        return str(value) if value is not None else "not reached"

    print("\nActive learning simulation:")
    print(f"  Rounds executed: {len(history) - 1}")
    print("  Clicks to match naive recall: " f"{format_clicks(clicks_to_target)}")
    # With default target ratio = 1.0, this equals clicks_to_target; keep printed only if ratio != 1
    if float(args.target_recall_ratio) != 1.0:
        print("  Clicks to reach target recall ratio: " f"{format_clicks(clicks_to_ratio)}")
    print(f"  Final recall: {al_final.recall:.3f}")
    if random_clicks_to_ratio is not None:
        print(f"  Random baseline reached 95% recall after: {random_clicks_to_ratio} annotations")
    else:
        print("  Random baseline did not reach 95% recall.")

    al_metrics = assign_to_dict(al_final)
    auc = area_under_recall_curve(history)

    improvement_factor = float(naive_clicks) / clicks_to_target if clicks_to_target else None
    improvement_ratio = float(naive_clicks) / clicks_to_ratio if clicks_to_ratio else None

    summary = {
        "config": {
            "seed": args.seed,
            "threshold": args.threshold,
            "batch_size": args.batch_size,
            "rounds": args.rounds,
            "acquisition": args.acquisition,
            "lambda_fp": args.lambda_fp,
            "diversity_k": args.diversity_k,
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "mass_error_ppm": args.mass_error_ppm,
            "decoy_fraction": args.decoy_fraction,
            "initial_labeled_fraction": args.initial_labeled_fraction,
        },
        "dataset": {
            "n_peaks": int(len(peaks_df)),
            "n_species": int(peaks_df["species"].nunique()),
            "n_compounds": int(len(compound_info)),
            "mean_candidates": mean_candidates,
        },
        "naive": {
            "annotations": int(naive_clicks),
            "metrics": naive_metrics,
        },
        "active_learning": {
            "rounds": len(history) - 1,
            "history": history,
            "metrics": al_metrics,
            "clicks_to_target_recall": clicks_to_target,
            "clicks_to_recall_ratio": clicks_to_ratio,
            "recall_auc": auc,
            "final_annotations": final_annotations,
            "relative_efficiency_target": improvement_factor,
            "relative_efficiency_ratio": improvement_ratio,
        },
        "random": {
            "rounds": len(random_history) - 1,
            "history": random_history,
            "metrics": random_metrics,
            "clicks_to_recall_ratio": random_clicks_to_ratio,
            "final_annotations": random_final_annotations,
        },
        "recall_ratio_target": float(args.target_recall_ratio),
    }

    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    pd.DataFrame(history).to_csv(results_dir / "al_progression.csv", index=False)

    print(f"\nSaved summary to {results_dir / 'summary.json'}")
    print(f"Saved AL progression to {results_dir / 'al_progression.csv'}")


if __name__ == "__main__":
    main()
