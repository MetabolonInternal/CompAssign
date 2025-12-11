from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure repository root is importable when running tests directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compassign.assignment import PeakAssignment, PresencePrior  # noqa: E402


def test_presence_prior_balances_null_and_positive_updates():
    """Seeded labels should soften null dominance via log-scaled updates."""

    n_species = 1
    n_compounds = 1

    peak_df = pd.DataFrame(
        [
            {
                "peak_id": 0,
                "species": 0,
                "mass": 100.0,
                "rt": 5.0,
                "intensity": 1e5,
                "true_compound": 0,
            },
            {
                "peak_id": 1,
                "species": 0,
                "mass": 130.0,
                "rt": 6.0,
                "intensity": 2e4,
                "true_compound": np.nan,
            },
            {
                "peak_id": 2,
                "species": 0,
                "mass": 140.0,
                "rt": 6.5,
                "intensity": 1.5e4,
                "true_compound": np.nan,
            },
            {
                "peak_id": 3,
                "species": 0,
                "mass": 150.0,
                "rt": 7.0,
                "intensity": 1.2e4,
                "true_compound": np.nan,
            },
            {
                "peak_id": 4,
                "species": 0,
                "mass": 160.0,
                "rt": 7.5,
                "intensity": 1.0e4,
                "true_compound": np.nan,
            },
        ]
    )

    compound_mass = np.array([100.0])

    model = PeakAssignment(mass_tolerance_ppm=10000.0, rt_window_k=5.0, random_seed=0)
    model.rt_predictions = {(0, 0): (5.0, 0.2)}

    presence = PresencePrior.init(n_species=n_species, n_compounds=n_compounds, smoothing=1.0)

    model.generate_training_data(
        peak_df=peak_df,
        compound_mass=compound_mass,
        n_compounds=n_compounds,
        init_presence=presence,
        initial_labeled_fraction=1.0,
        random_seed=0,
    )

    # Four null peaks for species 0 -> log1p(4) contribution to alpha_null
    expected_alpha_null = 1.0 + np.log1p(4)
    # One non-null label -> log1p(1) contribution to beta_null
    expected_beta_null = 1.0 + np.log1p(1)

    assert model.presence.alpha_null == pytest.approx(expected_alpha_null)
    assert model.presence.beta_null == pytest.approx(expected_beta_null)

    # Positive prior for (species, compound) should reflect one observation
    assert model.presence.alpha[0, 0] == pytest.approx(2.0)
