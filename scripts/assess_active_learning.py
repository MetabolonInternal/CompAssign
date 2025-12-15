"""Compatibility shim for the active-learning assessment CLI.

The canonical implementation lives under `scripts/experiments/active_learning/`.

This shim intentionally defines `main()` so test suites can monkeypatch helpers
like `generate_dataset()` on *this* module while still reusing the canonical CLI.
"""

from __future__ import annotations

from scripts.experiments.active_learning import assess_active_learning as _impl

RECALL_MATCH_RATIO = _impl.RECALL_MATCH_RATIO

# Re-export patch points (tests monkeypatch these on `scripts.assess_active_learning`).
generate_dataset = _impl.generate_dataset
compute_rt_predictions = _impl.compute_rt_predictions
setup_model = _impl.setup_model
run_naive_review = _impl.run_naive_review
run_random_review = _impl.run_random_review
run_active_learning = _impl.run_active_learning
assign_to_dict = _impl.assign_to_dict
area_under_recall_curve = _impl.area_under_recall_curve


def parse_args():  # noqa: D103
    return _impl.parse_args()


def main() -> None:
    # Propagate any monkeypatched functions into the canonical module before delegating.
    _impl.generate_dataset = generate_dataset
    _impl.compute_rt_predictions = compute_rt_predictions
    _impl.setup_model = setup_model
    _impl.run_naive_review = run_naive_review
    _impl.run_random_review = run_random_review
    _impl.run_active_learning = run_active_learning
    _impl.assign_to_dict = assign_to_dict
    _impl.area_under_recall_curve = area_under_recall_curve
    _impl.main()


if __name__ == "__main__":
    main()
