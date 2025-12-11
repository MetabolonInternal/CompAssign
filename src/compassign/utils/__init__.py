"""Shared utilities across CompAssign subpackages.

This package intentionally holds cross-cutting helpers used by both the RT and
peak-assignment code paths (data containers, feature utilities, plotting).
"""

from .data_features import (  # noqa: F401
    ADDUCT_DEFS,
    ADDUCT_FEATURE_MAP,
    CHEMICAL_RELATION_MASSES,
    FRAGMENT_DEFS,
    ION_TRANSFORMS,
    ISOTOPE_INTENSITY_FACTOR,
    ISOTOPE_SHIFT,
    MATCHING_TRANSFORMS,
    RUN_COVARIATE_PREFIX,
    ChemEmbedding,
    IonTransform,
    RunMetadata,
    RunMetadataError,
    SyntheticDataset,
    ensure_run_metadata,
    extract_run_metadata,
    load_chemberta_pca20,
    compute_all_chemical_features,
)
from .plots import (  # noqa: F401
    create_all_diagnostic_plots,
    create_assignment_plots,
    create_combined_dashboard,
)

__all__ = [
    "RUN_COVARIATE_PREFIX",
    "RunMetadata",
    "RunMetadataError",
    "SyntheticDataset",
    "ensure_run_metadata",
    "extract_run_metadata",
    "IonTransform",
    "ChemEmbedding",
    "load_chemberta_pca20",
    "compute_all_chemical_features",
    "ION_TRANSFORMS",
    "MATCHING_TRANSFORMS",
    "ADDUCT_FEATURE_MAP",
    "CHEMICAL_RELATION_MASSES",
    "ADDUCT_DEFS",
    "FRAGMENT_DEFS",
    "ISOTOPE_SHIFT",
    "ISOTOPE_INTENSITY_FACTOR",
    "create_all_diagnostic_plots",
    "create_assignment_plots",
    "create_combined_dashboard",
]
