#!/usr/bin/env python3
"""
Active-learning experiment runner (disabled).

This experiment previously depended on the removed legacy non-ridge hierarchical RT model.
It must be ported to the current Stage1CoeffSummaries-based RT ridge pipelines before it can run.

For RT model training/evaluation, use:
  - `./scripts/run_rt_prod.sh`
  - `./scripts/run_rt_prod_eval.sh`
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/experiments/active_learning/assess_active_learning.py is disabled.\n"
        "TODO: port active learning experiments to use Stage1CoeffSummaries-based RT predictions.\n"
        "The legacy hierarchical RT model has been removed."
    )


if __name__ == "__main__":
    main()
