import numpy as np
import pandas as pd
import pytest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compassign.assignment import PeakAssignment, PresencePrior  # noqa: E402
import arviz as az  # noqa: E402


def _build_simple_assignment():
    peak_df = pd.DataFrame(
        [
            {
                "peak_id": 0,
                "species": 0,
                "mass": 100.000,
                "rt": 5.0,
                "intensity": 1_000.0,
                "true_compound": 0,
            },
            {
                "peak_id": 1,
                "species": 0,
                "mass": 100.050,
                "rt": 5.1,
                "intensity": 800.0,
                "true_compound": 1,
            },
        ]
    )

    compound_mass = np.array([100.000, 100.050], dtype=float)

    pa = PeakAssignment(mass_tolerance_ppm=20000.0, rt_window_k=10.0, random_seed=0)
    pa.rt_predictions = {
        (0, 0): (5.0, 0.5),
        (0, 1): (5.0, 0.5),
    }

    presence = PresencePrior.init(n_species=1, n_compounds=2)

    pa.generate_training_data(
        peak_df=peak_df,
        compound_mass=compound_mass,
        n_compounds=2,
        init_presence=presence,
        initial_labeled_fraction=0.0,
        random_seed=0,
    )

    theta_features = np.zeros((1, 1, len(pa.feature_names)), dtype=float)
    theta0 = np.zeros((1, 1), dtype=float)
    theta_null = np.array([[[-0.5]]], dtype=float)
    sigma_logit = np.zeros((1, 1), dtype=float)

    pa.trace = az.from_dict(
        posterior={
            "theta_features": theta_features,
            "theta0": theta0,
            "theta_null": theta_null,
            "sigma_logit": sigma_logit,
        }
    )

    return pa


def test_predict_prob_samples_matches_mean():
    pa = _build_simple_assignment()

    sample_dict = pa.predict_prob_samples()
    prob_dict = pa.predict_probs()

    assert set(sample_dict.keys()) == set(prob_dict.keys())

    for peak_id, samples in sample_dict.items():
        assert samples.ndim == 2
        assert samples.shape[0] >= 1
        mean = samples.mean(axis=0)
        assert np.allclose(mean, prob_dict[peak_id])
        assert np.allclose(samples.sum(axis=1), np.ones(samples.shape[0]))


def test_predict_prob_samples_respects_max_samples():
    pa = _build_simple_assignment()
    samples = pa.predict_prob_samples(max_samples=1)
    for arr in samples.values():
        assert arr.shape[0] == 1


def test_generate_training_data_validates_inputs():
    pa = PeakAssignment()
    pa.rt_predictions = {(0, 0): (5.0, 0.5)}

    peak_df = pd.DataFrame(
        [
            {
                "peak_id": 0,
                "species": 0,
                "mass": 100.0,
                "rt": 5.0,
                "intensity": 10.0,
                "true_compound": 0,
            }
        ]
    )
    compound_mass = np.array([100.0])

    with pytest.raises(ValueError):
        pa.generate_training_data(peak_df.drop(columns=["mass"]), compound_mass, n_compounds=1)

    with pytest.raises(ValueError):
        pa.generate_training_data(peak_df, np.array([], dtype=float), n_compounds=1)
