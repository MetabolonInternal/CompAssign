from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.pipelines.train_rt_stage1_coeff_summaries as stage1
from src.compassign.rt.pymc_collapsed_ridge import Stage1CoeffSummaries


def _write_synthetic_rt_csv(path: Path) -> None:
    rng = np.random.default_rng(42)
    n_per_group = 12
    rows = []
    for species_cluster, comp_id, chem_id, compound_class in [
        (0, 10, 100, 1),
        (1, 10, 100, 1),
        (0, 20, 200, 2),
        (1, 20, 200, 2),
    ]:
        for _ in range(n_per_group):
            is1 = float(rng.normal(loc=5.0, scale=0.5))
            rs1 = float(rng.normal(loc=10.0, scale=0.8))
            es_1 = float(rng.normal(loc=1.0, scale=0.2))
            rt = 0.3 + 0.1 * is1 - 0.05 * rs1 + 0.02 * es_1 + float(rng.normal(scale=0.01))
            rows.append(
                {
                    "rt": rt,
                    "species_cluster": species_cluster,
                    "comp_id": comp_id,
                    "compound": chem_id,
                    "compound_class": compound_class,
                    "IS1": is1,
                    "RS1": rs1,
                    "ES_1": es_1,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_synthetic_rt_csv_many_chems(path: Path, *, n_chems: int = 20) -> None:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_chems):
        chem_id = 10_000 + i
        comp_id = 1_000 + i
        compound_class = i % 3
        for species_cluster in [0, 1]:
            for _ in range(3):
                is1 = float(rng.normal(loc=5.0, scale=0.5))
                rs1 = float(rng.normal(loc=10.0, scale=0.8))
                es_1 = float(rng.normal(loc=1.0, scale=0.2))
                rt = 0.3 + 0.1 * is1 - 0.05 * rs1 + 0.02 * es_1 + float(rng.normal(scale=0.01))
                rows.append(
                    {
                        "rt": rt,
                        "species_cluster": species_cluster,
                        "comp_id": comp_id,
                        "compound": chem_id,
                        "compound_class": compound_class,
                        "IS1": is1,
                        "RS1": rs1,
                        "ES_1": es_1,
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_stage1_coeff_summaries_schema_and_determinism(tmp_path: Path, monkeypatch) -> None:
    data_csv = tmp_path / "train_rt.csv"
    _write_synthetic_rt_csv(data_csv)

    out1 = tmp_path / "stage1_run1"
    argv1 = [
        "train_rt_stage1_coeff_summaries",
        "--data-csv",
        str(data_csv),
        "--include-es-all",
        "--output-dir",
        str(out1),
        "--seed",
        "42",
    ]
    monkeypatch.setattr(sys, "argv", argv1)
    stage1.main()

    out2 = tmp_path / "stage1_run2"
    argv2 = [
        "train_rt_stage1_coeff_summaries",
        "--data-csv",
        str(data_csv),
        "--include-es-all",
        "--output-dir",
        str(out2),
        "--seed",
        "42",
    ]
    monkeypatch.setattr(sys, "argv", argv2)
    stage1.main()

    npz1 = out1 / "stage1_coeff_summaries.npz"
    npz2 = out2 / "stage1_coeff_summaries.npz"
    assert npz1.exists()
    assert npz2.exists()

    cfg1 = json.loads((out1 / "config.json").read_text())
    cfg2 = json.loads((out2 / "config.json").read_text())
    assert cfg1["feature_names"] == cfg2["feature_names"]
    assert cfg1["backend"] == "sklearn_ridge"

    art1 = Stage1CoeffSummaries.load_npz(npz1)
    art2 = Stage1CoeffSummaries.load_npz(npz2)

    assert art1.feature_names == ("IS1", "RS1", "ES_1")
    assert art1.feature_center is not None
    assert art1.feature_center.shape == (3,)
    assert art1.group_keys.ndim == 1
    assert art1.beta_hat.ndim == 2
    assert art1.beta_var_diag.ndim == 2
    assert art1.beta_cov is not None
    assert art1.beta_cov.ndim == 3
    assert art1.sigma2_mean.ndim == 1

    n_groups = int(art1.group_keys.size)
    n_features = int(len(art1.feature_names))
    n_coefs = n_features + 1
    assert art1.species_cluster.shape == (n_groups,)
    assert art1.comp_id.shape == (n_groups,)
    assert art1.chem_id.shape == (n_groups,)
    assert art1.compound_class.shape == (n_groups,)
    assert art1.n_obs.shape == (n_groups,)
    assert art1.beta_hat.shape == (n_groups, n_coefs)
    assert art1.beta_var_diag.shape == (n_groups, n_coefs)
    assert art1.beta_cov.shape == (n_groups, n_coefs, n_coefs)
    assert art1.sigma2_mean.shape == (n_groups,)

    # Deterministic ordering (sorted keys) and deterministic values (same seed, same input).
    assert np.all(art1.group_keys[:-1] <= art1.group_keys[1:])
    np.testing.assert_array_equal(art1.group_keys, art2.group_keys)
    np.testing.assert_array_equal(art1.species_cluster, art2.species_cluster)
    np.testing.assert_array_equal(art1.comp_id, art2.comp_id)
    np.testing.assert_array_equal(art1.chem_id, art2.chem_id)
    np.testing.assert_array_equal(art1.compound_class, art2.compound_class)
    np.testing.assert_array_equal(art1.n_obs, art2.n_obs)
    np.testing.assert_array_equal(art1.beta_hat, art2.beta_hat)
    np.testing.assert_array_equal(art1.beta_var_diag, art2.beta_var_diag)
    np.testing.assert_array_equal(art1.beta_cov, art2.beta_cov)
    np.testing.assert_array_equal(art1.sigma2_mean, art2.sigma2_mean)
    np.testing.assert_array_equal(art1.feature_center, art2.feature_center)

    # Full covariance should be symmetric and match the diagonal summaries.
    cov1 = np.asarray(art1.beta_cov, dtype=np.float64)
    cov_sym = np.transpose(cov1, (0, 2, 1))
    np.testing.assert_allclose(cov1, cov_sym, rtol=1e-6, atol=1e-10)
    diag = np.diagonal(cov1, axis1=1, axis2=2)
    np.testing.assert_allclose(
        diag, np.asarray(art1.beta_var_diag, dtype=np.float64), rtol=1e-6, atol=1e-8
    )


def test_stage1_exclusion_artifacts(tmp_path: Path, monkeypatch) -> None:
    data_csv = tmp_path / "train_rt_many_chems.csv"
    _write_synthetic_rt_csv_many_chems(data_csv, n_chems=20)

    out = tmp_path / "stage1_exclusions"
    argv = [
        "train_rt_stage1_coeff_summaries",
        "--data-csv",
        str(data_csv),
        "--include-es-all",
        "--output-dir",
        str(out),
        "--seed",
        "42",
        "--exclude-clusters",
        "0",
        "--exclude-chems-frac",
        "0.5",
        "--exclude-chems-seed",
        "42",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    stage1.main()

    cfg = json.loads((out / "config.json").read_text())
    assert cfg["excluded_clusters_n"] == 1
    assert cfg["exclude_clusters"] == [0]
    assert int(cfg["excluded_cluster_rows"]) > 0
    assert cfg["excluded_clusters_csv"] is not None
    assert cfg["excluded_chems_csv"] is not None
    assert int(cfg["excluded_chems_n"]) > 0
    assert int(cfg["excluded_chems_n"]) < 20

    excluded_clusters = pd.read_csv(out / "results" / "excluded_clusters.csv")
    assert excluded_clusters["species_cluster"].astype(int).tolist() == [0]

    excluded_chems = pd.read_csv(out / "results" / "excluded_chems.csv")
    assert len(excluded_chems) == int(cfg["excluded_chems_n"])

    art = Stage1CoeffSummaries.load_npz(out / "stage1_coeff_summaries.npz")
    assert 0 not in set(int(v) for v in np.unique(art.species_cluster).tolist())
