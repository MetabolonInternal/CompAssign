#!/usr/bin/env python3
"""
Descriptor shift experiment (disabled).

This experiment previously depended on the removed legacy non-ridge hierarchical RT model.
It must be ported to the current ridge RT pipelines before it can run again.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/experiments/rt/descriptor/assess_descriptor_shift.py is disabled.\n"
        "TODO: port descriptor experiments to Stage1CoeffSummaries-based ridge RT models.\n"
        "The legacy hierarchical RT model has been removed."
    )


if __name__ == "__main__":
    main()
