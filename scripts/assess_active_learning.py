"""Active-learning assessment CLI (disabled).

This CLI depended on the removed legacy non-ridge hierarchical RT model.
It must be ported to the current Stage1CoeffSummaries-based ridge RT pipeline before use.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts/assess_active_learning.py is disabled.\n"
        "TODO: port the active-learning pipeline to Stage1CoeffSummaries-based ridge RT predictions.\n"
        "The legacy hierarchical RT model has been removed."
    )


if __name__ == "__main__":
    main()
