from pathlib import Path
import sys

import pandas as pd

# Ensure the project root (containing the `src` package) is importable when tests run standalone
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compassign.utils import compute_all_chemical_features  # noqa: E402


def _build_peak_df():
    """Create a minimal peak table with multiple adduct relationships."""
    species = 0
    return pd.DataFrame(
        [
            {  # Base [M+H]+
                "peak_id": 0,
                "species": species,
                "mass": 500.0,
                "rt": 5.0,
                "intensity": 1.0e6,
                "true_compound": 0,
            },
            {  # Sodium adduct [M+Na]+
                "peak_id": 1,
                "species": species,
                "mass": 521.9819,
                "rt": 5.004,
                "intensity": 5.0e5,
                "true_compound": 0,
            },
            {  # Water-loss fragment [M-H2O+H]+
                "peak_id": 2,
                "species": species,
                "mass": 481.9894,
                "rt": 5.002,
                "intensity": 3.0e5,
                "true_compound": 0,
            },
            {  # Isotope peak at +1.00335 Da with realistic intensity ratio
                "peak_id": 3,
                "species": species,
                "mass": 501.00335,
                "rt": 5.001,
                "intensity": 2.0e5,
                "true_compound": 0,
            },
        ]
    )


def test_adduct_count_detects_bidirectional_matches():
    peak_df = _build_peak_df()
    species = 0

    base = compute_all_chemical_features(500.0, 5.0, 1.0e6, peak_df, species)
    sodium = compute_all_chemical_features(521.9819, 5.004, 5.0e5, peak_df, species)
    water_loss = compute_all_chemical_features(481.9894, 5.002, 3.0e5, peak_df, species)

    assert base["n_adducts"] == 3.0  # Sodium, water loss, and 13C isotope partners
    assert sodium["n_adducts"] == 1.0  # Finds the lighter [M+H]+ partner
    assert water_loss["n_adducts"] == 1.0  # Finds the heavier [M+H]+ partner


def test_isotope_detection_scores_reasonable_ratio():
    peak_df = _build_peak_df()
    species = 0

    base = compute_all_chemical_features(500.0, 5.0, 1.0e6, peak_df, species)

    assert base["has_isotope"] == 1.0
    assert base["isotope_score"] == 1.0  # Intensity ratio sits in the “typical” band
