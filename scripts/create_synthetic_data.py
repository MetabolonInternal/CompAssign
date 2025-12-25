"""Compatibility shim for synthetic data generation helpers.

The canonical implementation lives in `scripts/data_prep/create_synthetic_data.py`.
This module is kept so older imports like `scripts.create_synthetic_data` continue
to work (tests and some utilities rely on it).
"""

from __future__ import annotations

from scripts.data_prep.create_synthetic_data import (
    create_metabolomics_data,
    create_synthetic_dataset,
)

__all__ = ["create_metabolomics_data", "create_synthetic_dataset"]
