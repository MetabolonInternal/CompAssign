import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compassign.rt.hierarchical import HierarchicalRTModel  # noqa: E402


def _small_rt_setup() -> Tuple[HierarchicalRTModel, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n_species = 2
    n_compounds = 2
    species_cluster = np.zeros(n_species, dtype=int)
    compound_class = np.zeros(n_compounds, dtype=int)
    run_species = np.arange(n_species, dtype=int)
    run_features = np.stack(
        [rng.normal(loc=i, scale=0.05, size=2) for i in range(n_species)],
        axis=0,
    )

    rows = []
    for species in range(n_species):
        for compound in range(n_compounds):
            base = 5.0 + species * 0.3 + compound * 0.5
            for _ in range(2):
                rows.append(
                    {
                        "species": species,
                        "compound": compound,
                        "rt": float(rng.normal(loc=base, scale=0.05)),
                    }
                )
    df = pd.DataFrame(rows)
    df["run"] = df["species"].astype(int)

    # Simple descriptor matrix (production model requires compound_features)
    compound_features = rng.normal(size=(n_compounds, 4))

    model = HierarchicalRTModel(
        n_clusters=1,
        n_species=n_species,
        n_classes=1,
        n_compounds=n_compounds,
        species_cluster=species_cluster,
        compound_class=compound_class,
        run_features=run_features,
        run_species=run_species,
        compound_features=compound_features,
    )
    return model, df


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_predictive_std_includes_observation_noise():
    model, df = _small_rt_setup()
    model.build_model(df)
    trace = model.sample(
        n_samples=20,
        n_tune=20,
        n_chains=1,
        target_accept=0.9,
        max_treedepth=6,
        random_seed=123,
    )
    species_idx = np.array([0, 1])
    compound_idx = np.array([0, 1])

    run_idx = np.array([0, 1])
    pred_mean, pred_std = model.predict_new(
        species_idx, compound_idx, run_idx=run_idx, run_features=model.run_features
    )

    assert pred_mean.shape == (2,)
    assert pred_std.shape == (2,)

    # Ensure the posterior-subsampling path runs (historically this raised
    # UnboundLocalError when traces include `sigma_y_group`).
    n_draws = int(trace.posterior["mu0"].values.size)
    pred_mean_sub, pred_std_sub = model.predict_new(
        species_idx,
        compound_idx,
        run_idx=run_idx,
        run_features=model.run_features,
        n_samples=n_draws,
    )
    assert pred_mean_sub.shape == (2,)
    assert pred_std_sub.shape == (2,)

    if "sigma_y_group" in trace.posterior:
        sigma_y_group = trace.posterior["sigma_y_group"].values.reshape(-1, model.n_clusters)
        # Use the average noise variance across clusters as a floor.
        noise_floor = float(np.sqrt(np.mean(np.square(sigma_y_group))))
    else:
        sigma_y = trace.posterior["sigma_y"].values.flatten()
        noise_floor = float(np.sqrt(np.mean(np.square(sigma_y))))

    # Numerical cushion for Monte Carlo noise
    assert np.all(pred_std + 1e-6 >= noise_floor)


def test_build_model_validates_inputs():
    model, df = _small_rt_setup()

    # Missing column
    with pytest.raises(ValueError):
        model.build_model(df.drop(columns=["rt"]))

    # Invalid species index
    bad_df = df.copy()
    bad_df.loc[0, "species"] = 999
    with pytest.raises(ValueError):
        model.build_model(bad_df)

    # Invalid cluster mapping length
    with pytest.raises(ValueError):
        HierarchicalRTModel(
            n_clusters=1,
            n_species=2,
            n_classes=1,
            n_compounds=2,
            species_cluster=np.zeros(1, dtype=int),
            compound_class=np.zeros(2, dtype=int),
            run_features=model.run_features,
            run_species=model.run_species,
        )


def test_logging_does_not_raise(caplog):
    caplog.set_level(logging.INFO)
    model, df = _small_rt_setup()
    model.build_model(df)
    model.sample(n_samples=5, n_tune=5, n_chains=1, random_seed=321)
    model.posterior_predictive_check(df)
    assert any("hierarchical RT model" in record.getMessage() for record in caplog.records)
