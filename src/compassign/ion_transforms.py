"""Shared definitions for ion/adduct mass transforms.

Centralizes the mass shifts and metadata so synthetic-data generation,
feature extraction, and candidate matching stay in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class IonTransform:
    name: str
    mass_shift: float
    feature_name: str
    category: str  # 'adduct', 'isotope', 'fragment'
    intensity_factor: float | None = None


# Keep declaration order stable for feature construction
ION_TRANSFORMS: Dict[str, IonTransform] = {
    "M+H": IonTransform("M+H", 0.0, "is_MH", "adduct", 0.9),
    "M+13C": IonTransform("M+13C", 1.00335, "is_M13C", "isotope", 0.35),
    "M+Na": IonTransform("M+Na", 21.9819, "is_MNa", "adduct", 0.5),
    "M+K": IonTransform("M+K", 37.9559, "is_MK", "adduct", 0.35),
    "M+NH4": IonTransform("M+NH4", 18.0338, "is_MNH4", "adduct", 0.45),
    "M-H2O+H": IonTransform("M-H2O+H", -18.0106, "is_MH2O_loss", "fragment", 0.25),
    "M-CO2+H": IonTransform("M-CO2+H", -43.9898, "is_MCO2_loss", "fragment", 0.2),
}

# Mass shifts used for candidate matching (includes base form at 0.0)
MATCHING_TRANSFORMS: Dict[str, float] = {
    name: transform.mass_shift for name, transform in ION_TRANSFORMS.items()
}

# Order-preserving map used when building feature vectors
ADDUCT_FEATURE_MAP: List[Tuple[str, str]] = [
    (transform.name, transform.feature_name) for transform in ION_TRANSFORMS.values()
]

# Mass deltas for chemical-feature lookups (exclude the zero-shift base peak)
CHEMICAL_RELATION_MASSES: Dict[str, float] = {
    name: transform.mass_shift
    for name, transform in ION_TRANSFORMS.items()
    if transform.mass_shift != 0.0
}

# Convenience accessors for the synthetic data generator
ADDUCT_DEFS: List[Dict[str, float]] = [
    {
        "name": transform.name,
        "delta": transform.mass_shift,
        "intensity_factor": transform.intensity_factor,
    }
    for transform in ION_TRANSFORMS.values()
    if transform.category == "adduct" and transform.mass_shift != 0.0
]

FRAGMENT_DEFS: List[Dict[str, float]] = [
    {
        "name": transform.name,
        "delta": transform.mass_shift,
        "intensity_factor": transform.intensity_factor,
    }
    for transform in ION_TRANSFORMS.values()
    if transform.category == "fragment"
]

ISOTOPE_SHIFT = ION_TRANSFORMS["M+13C"].mass_shift
"""Mass difference used for primary isotope detection."""

ISOTOPE_INTENSITY_FACTOR = ION_TRANSFORMS["M+13C"].intensity_factor or 0.35
"""Default intensity multiplier used when generating isotope peaks."""

__all__ = [
    "IonTransform",
    "ION_TRANSFORMS",
    "MATCHING_TRANSFORMS",
    "ADDUCT_FEATURE_MAP",
    "CHEMICAL_RELATION_MASSES",
    "ADDUCT_DEFS",
    "FRAGMENT_DEFS",
    "ISOTOPE_SHIFT",
    "ISOTOPE_INTENSITY_FACTOR",
]
