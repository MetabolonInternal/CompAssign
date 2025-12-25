"""Compatibility shim for aggregating active-learning run summaries.

The canonical implementation lives under `scripts/experiments/active_learning/`.
This module is kept so older imports like `scripts.aggregate_al_results` continue
to work.
"""

from __future__ import annotations

from scripts.experiments.active_learning.aggregate_al_results import *  # type: ignore  # noqa: F401,F403
