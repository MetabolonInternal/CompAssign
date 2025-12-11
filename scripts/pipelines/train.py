#!/usr/bin/env python3
"""
Legacy synthetic end-to-end training pipeline (disabled).

This script previously trained the removed non-ridge hierarchical RT model together with the
peak-assignment model on synthetic data.

It must be ported to the current ridge RT pipelines before it can be used again.
For now it intentionally halts loudly to avoid accidental use.

Use instead:
  - RT ridge training: `./scripts/run_rt_prod.sh`
  - RT ridge evaluation: `./scripts/run_rt_prod_eval.sh`
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/pipelines/train.py is disabled because it depended on the removed legacy "
        "hierarchical RT model.\n"
        "TODO: port this synthetic pipeline to Stage1CoeffSummaries-based RT predictions.\n"
        "Use `./scripts/run_rt_prod.sh` and `./scripts/run_rt_prod_eval.sh` for the RT component."
    )


if __name__ == "__main__":
    main()
