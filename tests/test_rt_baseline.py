"""Tests for retention-time baseline utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.compassign.rt.baselines import ClusterLassoBaseline


def test_cluster_lasso_baseline_respects_clusters() -> None:
    """Cluster-specific models should learn distinct regressions."""

    internal_std = np.array(
        [
            [1.0, 0.2],
            [1.5, 0.4],
            [2.0, 0.6],
            [0.1, 1.0],
            [0.2, 1.8],
            [0.4, 2.5],
        ]
    )
    species_cluster = np.array([0, 0, 0, 1, 1, 1])

    def cluster0_rt(species_idx: int) -> float:
        is_vec = internal_std[species_idx]
        return 10.0 + 2.0 * is_vec[0] + 0.5 * is_vec[1]

    def cluster1_rt(species_idx: int) -> float:
        is_vec = internal_std[species_idx]
        return 3.0 - 0.5 * is_vec[0] + 1.5 * is_vec[1]

    rows = []
    for species_idx in range(len(internal_std)):
        for compound_idx in (0, 1):
            if species_cluster[species_idx] == 0:
                rt = cluster0_rt(species_idx)
            else:
                rt = cluster1_rt(species_idx)
            rows.append({"species": species_idx, "compound": compound_idx, "rt": rt})

    train_df = pd.DataFrame(rows)
    # Create a run_df with covariates matching the internal_std rows (one run per species)
    run_df = pd.DataFrame(
        {
            "run": np.arange(len(internal_std), dtype=int),
            "species": np.arange(len(internal_std), dtype=int),
            "run_covariate_0": internal_std[:, 0],
            "run_covariate_1": internal_std[:, 1],
        }
    )

    baseline = ClusterLassoBaseline(
        species_cluster=species_cluster,
        random_state=0,
    )
    baseline.fit(train_df, run_df=run_df, covariate_columns=["run_covariate_0", "run_covariate_1"])

    species = np.arange(len(internal_std))
    compounds = np.zeros_like(species)
    preds = baseline.predict(species_idx=species, compound_idx=compounds)

    expected = np.array(
        [
            cluster0_rt(0),
            cluster0_rt(1),
            cluster0_rt(2),
            cluster1_rt(3),
            cluster1_rt(4),
            cluster1_rt(5),
        ]
    )

    np.testing.assert_allclose(preds, expected, atol=2e-2)
