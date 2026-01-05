from __future__ import annotations

from pathlib import Path

import numpy as np

from src.compassign.rt.ridge_stage1 import PartialPoolBackoffSummaries


def test_partial_pool_backoff_summaries_roundtrip(tmp_path: Path) -> None:
    art = PartialPoolBackoffSummaries(
        feature_names=("IS1", "RS1"),
        cluster_ids=np.asarray([10, 20], dtype=np.int64),
        cluster_supercat_id=np.asarray([1, 1], dtype=np.int64),
        comp_ids=np.asarray([100, 101, 102], dtype=np.int64),
        comp_chem_id=np.asarray([1000, 1001, 1002], dtype=np.int64),
        comp_class=np.asarray([3, 3, 4], dtype=np.int64),
        t0=1.5,
        mu_cluster=np.asarray([0.2, -0.2], dtype=np.float64),
        alpha_comp=np.asarray([0.1, -0.1, 0.0], dtype=np.float64),
        w_cluster=np.asarray([[0.5, -0.1], [0.4, -0.2]], dtype=np.float64),
        tau_b=0.3,
        sigma2=0.01,
        lambda_slopes=1e-3,
        alpha_z_center=np.asarray([0.1, -0.2], dtype=np.float64),
        alpha_theta=np.asarray([0.3, 0.4], dtype=np.float64),
        tau_comp=0.25,
    )

    path = tmp_path / "partial_pool_backoff.npz"
    art.save_npz(path)
    loaded = PartialPoolBackoffSummaries.load_npz(path)

    assert loaded.feature_names == art.feature_names
    np.testing.assert_array_equal(loaded.cluster_ids, art.cluster_ids)
    np.testing.assert_array_equal(loaded.cluster_supercat_id, art.cluster_supercat_id)
    np.testing.assert_array_equal(loaded.comp_ids, art.comp_ids)
    np.testing.assert_array_equal(loaded.comp_chem_id, art.comp_chem_id)
    np.testing.assert_array_equal(loaded.comp_class, art.comp_class)
    assert loaded.t0 == art.t0
    np.testing.assert_allclose(loaded.mu_cluster, art.mu_cluster)
    np.testing.assert_allclose(loaded.alpha_comp, art.alpha_comp)
    np.testing.assert_allclose(loaded.w_cluster, art.w_cluster)
    np.testing.assert_allclose(loaded.tau_b, art.tau_b)
    np.testing.assert_allclose(loaded.sigma2, art.sigma2)
    np.testing.assert_allclose(loaded.lambda_slopes, art.lambda_slopes)
    np.testing.assert_allclose(loaded.alpha_z_center, art.alpha_z_center)
    np.testing.assert_allclose(loaded.alpha_theta, art.alpha_theta)
    np.testing.assert_allclose(loaded.tau_comp, art.tau_comp)
