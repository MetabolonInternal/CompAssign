from __future__ import annotations

from pathlib import Path

import numpy as np

from src.compassign.rt.pymc_collapsed_ridge import Stage1CoeffSummaries


def test_stage1_coeff_summaries_load_npz_defaults_when_missing_group_fields(tmp_path: Path) -> None:
    path = tmp_path / "stage1_old.npz"
    n_groups = 2
    n_features = 2
    n_coefs = n_features + 1
    np.savez(
        path,
        feature_names=np.asarray(["IS1", "RS1"], dtype=object),
        group_keys=np.asarray([(0 << 32) + 10, (1 << 32) + 10], dtype=np.int64),
        species_cluster=np.asarray([0, 1], dtype=np.int64),
        comp_id=np.asarray([10, 10], dtype=np.int64),
        chem_id=np.asarray([100, 100], dtype=np.int64),
        compound_class=np.asarray([1, 1], dtype=np.int64),
        n_obs=np.asarray([10, 10], dtype=np.int64),
        beta_hat=np.zeros((n_groups, n_coefs), dtype=np.float64),
        beta_var_diag=np.zeros((n_groups, n_coefs), dtype=np.float64),
        sigma2_mean=np.ones((n_groups,), dtype=np.float64),
    )

    art = Stage1CoeffSummaries.load_npz(path)
    assert art.group_col == "species_cluster"
    assert art.supercat_id is None


def test_stage1_coeff_summaries_roundtrip_includes_group_fields(tmp_path: Path) -> None:
    art = Stage1CoeffSummaries(
        feature_names=("IS1", "RS1"),
        group_keys=np.asarray([(101 << 32) + 10, (202 << 32) + 10], dtype=np.int64),
        group_col="species",
        species_cluster=np.asarray([101, 202], dtype=np.int64),
        supercat_id=np.asarray([0, 1], dtype=np.int64),
        comp_id=np.asarray([10, 10], dtype=np.int64),
        chem_id=np.asarray([100, 100], dtype=np.int64),
        compound_class=np.asarray([1, 1], dtype=np.int64),
        n_obs=np.asarray([10, 10], dtype=np.int64),
        beta_hat=np.zeros((2, 3), dtype=np.float64),
        beta_var_diag=np.zeros((2, 3), dtype=np.float64),
        beta_cov=None,
        sigma2_mean=np.ones((2,), dtype=np.float64),
    )

    path = tmp_path / "stage1_new.npz"
    art.save_npz(path)
    loaded = Stage1CoeffSummaries.load_npz(path)

    assert loaded.group_col == "species"
    assert loaded.supercat_id is not None
    np.testing.assert_array_equal(
        np.asarray(loaded.supercat_id, dtype=np.int64), np.asarray([0, 1])
    )
