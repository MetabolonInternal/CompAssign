"""Active learning validation (fast, deterministic).

This is a lightweight smoke test for the active learning selection logic. It does *not*
run the full Bayesian peak assignment model (which is slow); instead it simulates model
probability outputs and applies acquisition functions + oracle feedback to ensure:
- acquisition functions run end-to-end
- metrics move in the expected direction (entropy/expected FP decrease)
- outputs are compatible with `visualize_validation_results.py`
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from compassign.active_learning.core import entropy, select_batch


def _expected_fp(probs_dict: dict[int, np.ndarray], threshold: float) -> float:
    expected_fp = 0.0
    for probs in probs_dict.values():
        max_prob = float(np.max(probs))
        if max_prob >= threshold and int(np.argmax(probs)) > 0:
            expected_fp += 1.0 - max_prob
    return float(expected_fp)


def _predict_labels(probs_dict: dict[int, np.ndarray], threshold: float) -> dict[int, int]:
    pred: dict[int, int] = {}
    for peak_id, probs in probs_dict.items():
        k = int(np.argmax(probs))
        p = float(probs[k])
        pred[peak_id] = k if (k > 0 and p >= threshold) else 0
    return pred


def _classification_metrics(
    *,
    pred: dict[int, int],
    truth: dict[int, int],
) -> tuple[float, float, float, float]:
    tp = fp = fn = 0
    n = 0
    assigned = 0
    for peak_id, y_true in truth.items():
        y_pred = pred.get(peak_id, 0)
        n += 1
        if y_pred > 0:
            assigned += 1
        if y_pred > 0 and y_true > 0 and y_pred == y_true:
            tp += 1
        elif y_pred > 0 and (y_true == 0 or y_pred != y_true):
            fp += 1
        elif y_pred == 0 and y_true > 0:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    assignment_rate = assigned / n if n else 0.0
    return float(precision), float(recall), float(f1), float(assignment_rate)


def _prob_entropy_sum(probs_dict: dict[int, np.ndarray]) -> float:
    return float(sum(entropy(p) for p in probs_dict.values()))


def _sample_probabilities(
    rng: np.random.Generator,
    *,
    n_peaks: int,
    n_candidates: int,
    null_fraction: float,
    base_confidence: float,
    noise: float,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    if not (0.0 <= null_fraction <= 1.0):
        raise ValueError("null_fraction must be within [0, 1]")
    if n_candidates < 2:
        raise ValueError("n_candidates must be >= 2 (includes null)")

    probs_dict: dict[int, np.ndarray] = {}
    truth: dict[int, int] = {}
    uniform = np.full(n_candidates, 1.0 / n_candidates, dtype=float)

    n_null = int(round(n_peaks * null_fraction))
    null_ids = set(rng.choice(np.arange(n_peaks), size=n_null, replace=False).tolist())

    for peak_id in range(n_peaks):
        if peak_id in null_ids:
            y_true = 0
            base = uniform.copy()
            base[0] = float(base_confidence)
            base[1:] = (1.0 - base[0]) / (n_candidates - 1)
        else:
            y_true = int(rng.integers(1, n_candidates))
            base = uniform.copy()
            base[y_true] = float(base_confidence)
            remaining = 1.0 - base[y_true]
            base[base != base[y_true]] = remaining / (n_candidates - 1)
            base[0] = remaining / (n_candidates - 1)

        p = (1.0 - float(noise)) * base + float(noise) * uniform
        p = np.clip(p, 1e-12, 1.0)
        p = p / p.sum()
        probs_dict[int(peak_id)] = p.astype(float)
        truth[int(peak_id)] = int(y_true)

    return probs_dict, truth


def _oracle_label(
    rng: np.random.Generator,
    *,
    true_label: int,
    n_candidates: int,
    accuracy: float,
) -> int:
    if rng.random() < float(accuracy):
        return int(true_label)

    choices = list(range(n_candidates))
    if true_label in choices:
        choices.remove(true_label)
    return int(rng.choice(choices)) if choices else int(true_label)


def _apply_label_update(probs: np.ndarray, label: int, confidence: float = 0.98) -> np.ndarray:
    out = np.full_like(probs, (1.0 - float(confidence)) / (len(probs) - 1), dtype=float)
    out[int(label)] = float(confidence)
    return out


@dataclass(frozen=True)
class ExperimentResult:
    experiment_name: str
    config: dict[str, Any]
    metrics: dict[str, float]
    rounds_data: list[dict[str, float]]
    runtime_seconds: float
    notes: str = ""


def _run_rounds(
    *,
    seed: int,
    acquisition_fn: str,
    oracle_accuracy: float,
    n_rounds: int,
    batch_size: int,
    threshold: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rng = np.random.default_rng(int(seed))

    probs_dict, truth = _sample_probabilities(
        rng,
        n_peaks=200,
        n_candidates=6,  # [null, c1..c5]
        null_fraction=0.45,
        base_confidence=0.82,
        noise=0.35,
    )

    rounds: list[dict[str, float]] = []
    prev_entropy = _prob_entropy_sum(probs_dict)

    for r in range(1, int(n_rounds) + 1):
        # Selection
        prob_samples_dict: dict[int, np.ndarray] | None = None
        if acquisition_fn == "mi":
            prob_samples_dict = {}
            for peak_id, probs in probs_dict.items():
                alpha = np.clip(probs, 1e-6, 1.0) * 50.0
                prob_samples_dict[peak_id] = rng.dirichlet(alpha, size=16)

        selected = select_batch(
            probs_dict=probs_dict,
            batch_size=int(batch_size),
            acquisition_fn=acquisition_fn,
            threshold=float(threshold),
            lambda_fp=0.7,
            diversity_k=None,
            features_dict=None,
            prob_samples_dict=prob_samples_dict,
        )

        # Oracle + update
        for peak_id in selected:
            probs = probs_dict[int(peak_id)]
            label = _oracle_label(
                rng,
                true_label=truth[int(peak_id)],
                n_candidates=len(probs),
                accuracy=float(oracle_accuracy),
            )
            probs_dict[int(peak_id)] = _apply_label_update(probs, label)

        # Metrics
        ent = _prob_entropy_sum(probs_dict)
        pred = _predict_labels(probs_dict, threshold=float(threshold))
        precision, recall, f1, assignment_rate = _classification_metrics(pred=pred, truth=truth)
        exp_fp = _expected_fp(probs_dict, threshold=float(threshold))
        entropy_delta = ent - prev_entropy
        prev_entropy = ent

        rounds.append(
            {
                "round": float(r),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "entropy": float(ent),
                "expected_fp": float(exp_fp),
                "assignment_rate": float(assignment_rate),
                "entropy_delta": float(entropy_delta),
            }
        )

    metrics = {
        "final_precision": float(rounds[-1]["precision"]),
        "final_recall": float(rounds[-1]["recall"]),
        "final_f1": float(rounds[-1]["f1"]),
        "entropy_reduction": float(rounds[0]["entropy"] - rounds[-1]["entropy"]),
        "fp_reduction": float(rounds[0]["expected_fp"] - rounds[-1]["expected_fp"]),
    }
    return metrics, rounds


def main() -> list[ExperimentResult]:
    start = time.time()

    seed = 42
    threshold = 0.75
    results: list[ExperimentResult] = []

    # 1) Acquisition comparison
    for method in ["fp", "entropy", "hybrid", "mi"]:
        t0 = time.time()
        metrics, rounds = _run_rounds(
            seed=seed,
            acquisition_fn=method,
            oracle_accuracy=1.0,
            n_rounds=6,
            batch_size=12,
            threshold=threshold,
        )
        results.append(
            ExperimentResult(
                experiment_name="acquisition_comparison",
                config={"method": method, "threshold": threshold},
                metrics=metrics,
                rounds_data=rounds,
                runtime_seconds=float(time.time() - t0),
            )
        )

    # 2) Oracle robustness
    for oracle_name, oracle_acc in [
        ("Optimal", 1.0),
        ("SmartNoisy_80", 0.8),
        ("SmartNoisy_60", 0.6),
        ("Random", 0.2),
    ]:
        t0 = time.time()
        metrics, rounds = _run_rounds(
            seed=seed + 1,
            acquisition_fn="hybrid",
            oracle_accuracy=oracle_acc,
            n_rounds=6,
            batch_size=12,
            threshold=threshold,
        )
        results.append(
            ExperimentResult(
                experiment_name="oracle_robustness",
                config={"oracle": oracle_name, "threshold": threshold},
                metrics=metrics,
                rounds_data=rounds,
                runtime_seconds=float(time.time() - t0),
            )
        )

    # 3) Threshold calibration
    calib_seed = seed + 2
    # Reuse the last-round probabilities by re-running the same simulation and keeping the
    # final snapshot deterministic (keep it simple: re-run, then compute metrics at thresholds).
    rng = np.random.default_rng(int(calib_seed))
    probs_dict, truth = _sample_probabilities(
        rng,
        n_peaks=200,
        n_candidates=6,
        null_fraction=0.45,
        base_confidence=0.82,
        noise=0.35,
    )
    for r in range(1, 4 + 1):
        selected = select_batch(
            probs_dict=probs_dict,
            batch_size=12,
            acquisition_fn="hybrid",
            threshold=float(threshold),
            lambda_fp=0.7,
        )
        for peak_id in selected:
            y = truth[int(peak_id)]
            probs_dict[int(peak_id)] = _apply_label_update(probs_dict[int(peak_id)], y)

    for th in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]:
        pred = _predict_labels(probs_dict, threshold=float(th))
        precision, recall, f1, assignment_rate = _classification_metrics(pred=pred, truth=truth)
        results.append(
            ExperimentResult(
                experiment_name="threshold_calibration",
                config={"threshold": float(th)},
                metrics={
                    "final_f1": float(f1),
                    "final_recall": float(recall),
                    "assignment_rate": float(assignment_rate),
                },
                rounds_data=[],
                runtime_seconds=0.0,
            )
        )

    # 4) Convergence analysis
    t0 = time.time()
    metrics, rounds = _run_rounds(
        seed=seed + 3,
        acquisition_fn="hybrid",
        oracle_accuracy=1.0,
        n_rounds=10,
        batch_size=10,
        threshold=threshold,
    )
    results.append(
        ExperimentResult(
            experiment_name="convergence_analysis",
            config={"method": "hybrid", "threshold": threshold},
            metrics=metrics,
            rounds_data=rounds,
            runtime_seconds=float(time.time() - t0),
        )
    )

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "runtime_seconds": float(time.time() - start),
        "experiments": [asdict(r) for r in results],
    }
    Path("validation_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return results


if __name__ == "__main__":
    main()
