from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.pipelines.eval_rt_coeff_summaries_by_species_cluster as eval_coeff
from src.compassign.rt.pymc_collapsed_ridge import Stage1CoeffSummaries


def _write_synthetic_csv(path: Path) -> None:
    rows = []
    # Two seen groups plus one missing-group row at the end.
    groups = [
        (0, 10, 100, 1, 0.3, 0.1, -0.05),
        (1, 10, 100, 1, 0.2, 0.05, 0.02),
    ]
    for species_cluster, comp_id, chem_id, compound_class, b, w1, w2 in groups:
        for i in range(10):
            is1 = 5.0 + 0.1 * i
            rs1 = 10.0 - 0.2 * i
            rt = b + w1 * is1 + w2 * rs1
            rows.append(
                {
                    "rt": rt,
                    "species_cluster": species_cluster,
                    "comp_id": comp_id,
                    "compound": chem_id,
                    "compound_class": compound_class,
                    "IS1": is1,
                    "RS1": rs1,
                }
            )
    rows.append(
        {
            "rt": 0.0,
            "species_cluster": 9,
            "comp_id": 999,
            "compound": 999,
            "compound_class": 9,
            "IS1": 1.0,
            "RS1": 1.0,
        }
    )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_eval_coeff_summaries_seen_group_metrics(tmp_path: Path, monkeypatch) -> None:
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
        "eval_rt_coeff_summaries_by_species_cluster",
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
    eval_coeff.main()

    metrics = json.loads((out_dir / "rt_eval_coeff_summaries_unit.json").read_text())
    assert metrics["n_test"] == 21
    assert metrics["skipped_missing_group"] == 1
    assert metrics["n_used"] == 20
    assert metrics["metrics"]["n_obs"] == 20
    assert metrics["metrics"]["rmse"] < 1e-6

    by_cluster = pd.read_csv(out_dir / "rt_eval_coeff_summaries_by_species_cluster_unit.csv")
    assert set(by_cluster["species_cluster"].astype(int).tolist()) == {0, 1}
    assert float(by_cluster["rmse"].max()) < 1e-6
