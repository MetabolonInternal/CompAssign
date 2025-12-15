from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.pipelines.eval_rt_coeff_summaries_by_support as eval_support
from src.compassign.rt.pymc_collapsed_ridge import Stage1CoeffSummaries


def _write_synthetic_csv(path: Path) -> None:
    rows = []
    # Two seen groups plus one missing-group row at the end.
    groups = [
        (0, 10, 0.3, 0.1, -0.05),
        (1, 10, 0.2, 0.05, 0.02),
    ]
    for species_cluster, comp_id, b, w1, w2 in groups:
        for i in range(10):
            is1 = 5.0 + 0.1 * i
            rs1 = 10.0 - 0.2 * i
            rt = b + w1 * is1 + w2 * rs1
            rows.append(
                {
                    "rt": rt,
                    "species_cluster": species_cluster,
                    "comp_id": comp_id,
                    "IS1": is1,
                    "RS1": rs1,
                }
            )
    rows.append(
        {
            "rt": 0.0,
            "species_cluster": 9,
            "comp_id": 999,
            "IS1": 1.0,
            "RS1": 1.0,
        }
    )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_synthetic_csv_species(path: Path) -> None:
    rows = []
    groups = [
        (101, 10, 0.3, 0.1, -0.05),
        (202, 10, 0.2, 0.05, 0.02),
    ]
    for species, comp_id, b, w1, w2 in groups:
        for i in range(10):
            is1 = 5.0 + 0.1 * i
            rs1 = 10.0 - 0.2 * i
            rt = b + w1 * is1 + w2 * rs1
            rows.append(
                {
                    "rt": rt,
                    "species": species,
                    "comp_id": comp_id,
                    "IS1": is1,
                    "RS1": rs1,
                }
            )
    rows.append({"rt": 0.0, "species": 999, "comp_id": 999, "IS1": 1.0, "RS1": 1.0})
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_synthetic_csv_cluster_species(path: Path) -> None:
    rows = []
    groups = [
        (0, 9, 10, 0.3, 0.1, -0.05),
        (1, 2, 10, 0.2, 0.05, 0.02),
    ]
    for species_cluster, species, comp_id, b, w1, w2 in groups:
        for i in range(10):
            is1 = 5.0 + 0.1 * i
            rs1 = 10.0 - 0.2 * i
            rt = b + w1 * is1 + w2 * rs1
            rows.append(
                {
                    "rt": rt,
                    "species_cluster": species_cluster,
                    "species": species,
                    "comp_id": comp_id,
                    "IS1": is1,
                    "RS1": rs1,
                }
            )
    rows.append(
        {
            "rt": 0.0,
            "species_cluster": 9,
            "species": 999,
            "comp_id": 999,
            "IS1": 1.0,
            "RS1": 1.0,
        }
    )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_row_variance_from_beta_cov_matches_einsum() -> None:
    rng = np.random.default_rng(123)
    n = 9
    d = 4
    n_groups = 3

    x1 = rng.normal(size=(n, d))
    group_idx = rng.integers(0, n_groups, size=n, dtype=np.int64)

    a = rng.normal(size=(n_groups, d, d))
    beta_cov = a @ np.swapaxes(a, 1, 2)

    cov_per_row = beta_cov[group_idx]
    expected = np.einsum("ni,nij,nj->n", x1, cov_per_row, x1, optimize=True)
    got = eval_support._row_variance_from_beta_cov(x1=x1, group_idx=group_idx, beta_cov=beta_cov)

    assert np.allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_eval_support_summaries_seen_group_metrics(tmp_path: Path, monkeypatch) -> None:
    test_csv = tmp_path / "test_rt.csv"
    _write_synthetic_csv(test_csv)

    # Build a perfect coefficient artifact for the two seen groups.
    feature_names = ("IS1", "RS1")
    keys = np.asarray([(0 << 32) + 10, (1 << 32) + 10], dtype=np.int64)
    beta_hat = np.asarray(
        [
            [0.3, 0.1, -0.05],
            [0.2, 0.05, 0.02],
        ],
        dtype=np.float64,
    )
    cov = np.zeros((2, 3, 3), dtype=np.float64)
    beta_var_diag = np.diagonal(cov, axis1=1, axis2=2)
    sigma2 = np.full((2,), 1e-12, dtype=np.float64)
    art = Stage1CoeffSummaries(
        feature_names=feature_names,
        group_keys=keys,
        species_cluster=np.asarray([0, 1], dtype=np.int64),
        comp_id=np.asarray([10, 10], dtype=np.int64),
        chem_id=np.asarray([100, 100], dtype=np.int64),
        compound_class=np.asarray([1, 1], dtype=np.int64),
        n_obs=np.asarray([10, 10], dtype=np.int64),
        beta_hat=beta_hat,
        beta_var_diag=beta_var_diag,
        beta_cov=cov,
        sigma2_mean=sigma2,
    )
    coeff_npz = tmp_path / "stage1_coeff_summaries.npz"
    art.save_npz(coeff_npz)

    out_dir = tmp_path / "results"
    argv = [
        "eval_rt_coeff_summaries_by_support",
        "--coeff-npz",
        str(coeff_npz),
        "--test-csv",
        str(test_csv),
        "--output-dir",
        str(out_dir),
        "--chunk-size",
        "7",
        "--label",
        "unit",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    eval_support.main()

    metrics = json.loads((out_dir / "rt_eval_coeff_summaries_by_support_unit.json").read_text())
    assert metrics["n_test"] == 21
    assert metrics["skipped_missing_group"] == 1
    assert metrics["n_used"] == 20
    assert metrics["metrics"]["n_obs"] == 20
    assert metrics["metrics"]["rmse"] < 1e-6

    by_group = pd.read_csv(out_dir / "rt_eval_coeff_summaries_by_group_unit.csv")
    assert set(by_group["species_cluster"].astype(int).tolist()) == {0, 1}
    assert float(by_group["rmse"].max()) < 1e-6
    assert set(by_group["support_bin"].astype(str).tolist()) == {"6-10"}

    by_support = pd.read_csv(out_dir / "rt_eval_coeff_summaries_by_support_unit.csv")
    row = by_support.loc[by_support["support_bin"].astype(str) == "6-10"].iloc[0]
    assert int(row["n_groups_with_test"]) == 2
    assert int(row["n_obs_test"]) == 20
    assert float(row["rmse"]) < 1e-6


def test_eval_support_respects_group_col_species(tmp_path: Path, monkeypatch) -> None:
    test_csv = tmp_path / "test_rt_species.csv"
    _write_synthetic_csv_species(test_csv)

    feature_names = ("IS1", "RS1")
    keys = np.asarray([(101 << 32) + 10, (202 << 32) + 10], dtype=np.int64)
    beta_hat = np.asarray(
        [
            [0.3, 0.1, -0.05],
            [0.2, 0.05, 0.02],
        ],
        dtype=np.float64,
    )
    cov = np.zeros((2, 3, 3), dtype=np.float64)
    beta_var_diag = np.diagonal(cov, axis1=1, axis2=2)
    sigma2 = np.full((2,), 1e-12, dtype=np.float64)
    art = Stage1CoeffSummaries(
        feature_names=feature_names,
        group_keys=keys,
        group_col="species",
        species_cluster=np.asarray([101, 202], dtype=np.int64),
        supercat_id=np.asarray([0, 1], dtype=np.int64),
        comp_id=np.asarray([10, 10], dtype=np.int64),
        chem_id=np.asarray([100, 100], dtype=np.int64),
        compound_class=np.asarray([1, 1], dtype=np.int64),
        n_obs=np.asarray([10, 10], dtype=np.int64),
        beta_hat=beta_hat,
        beta_var_diag=beta_var_diag,
        beta_cov=cov,
        sigma2_mean=sigma2,
    )
    coeff_npz = tmp_path / "stage1_coeff_summaries_species.npz"
    art.save_npz(coeff_npz)

    out_dir = tmp_path / "results_species"
    argv = [
        "eval_rt_coeff_summaries_by_support",
        "--coeff-npz",
        str(coeff_npz),
        "--test-csv",
        str(test_csv),
        "--output-dir",
        str(out_dir),
        "--chunk-size",
        "7",
        "--label",
        "unit_species",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    eval_support.main()

    metrics = json.loads(
        (out_dir / "rt_eval_coeff_summaries_by_support_unit_species.json").read_text()
    )
    assert metrics["group_col"] == "species"
    assert metrics["n_test"] == 21
    assert metrics["skipped_missing_group"] == 1
    assert metrics["n_used"] == 20
    assert metrics["metrics"]["rmse"] < 1e-6

    by_group = pd.read_csv(out_dir / "rt_eval_coeff_summaries_by_group_unit_species.csv")
    assert set(by_group["species"].astype(int).tolist()) == {101, 202}
    assert set(by_group["supercat_id"].astype(int).tolist()) == {0, 1}
    assert float(by_group["rmse"].max()) < 1e-6


def test_eval_support_rejects_unknown_group_col(tmp_path: Path, monkeypatch) -> None:
    test_csv = tmp_path / "test_rt_cluster.csv"
    _write_synthetic_csv(test_csv)

    feature_names = ("IS1", "RS1")
    keys = np.asarray([(0 << 32) + 10, (1 << 32) + 10], dtype=np.int64)
    beta_hat = np.asarray([[0.3, 0.1, -0.05], [0.2, 0.05, 0.02]], dtype=np.float64)
    cov = np.zeros((2, 3, 3), dtype=np.float64)
    beta_var_diag = np.diagonal(cov, axis1=1, axis2=2)
    sigma2 = np.full((2,), 1e-12, dtype=np.float64)
    art = Stage1CoeffSummaries(
        feature_names=feature_names,
        group_keys=keys,
        group_col="bogus_group_col",
        species_cluster=np.asarray([0, 1], dtype=np.int64),
        supercat_id=None,
        comp_id=np.asarray([10, 10], dtype=np.int64),
        chem_id=np.asarray([100, 100], dtype=np.int64),
        compound_class=np.asarray([1, 1], dtype=np.int64),
        n_obs=np.asarray([10, 10], dtype=np.int64),
        beta_hat=beta_hat,
        beta_var_diag=beta_var_diag,
        beta_cov=cov,
        sigma2_mean=sigma2,
    )
    coeff_npz = tmp_path / "stage1_coeff_summaries_bad_group_col.npz"
    art.save_npz(coeff_npz)

    out_dir = tmp_path / "results_bad_group_col"
    argv = [
        "eval_rt_coeff_summaries_by_support",
        "--coeff-npz",
        str(coeff_npz),
        "--test-csv",
        str(test_csv),
        "--output-dir",
        str(out_dir),
        "--chunk-size",
        "7",
        "--label",
        "unit_bad_group_col",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="Unsupported group_col"):
        eval_support.main()
