from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.pipelines.train_rt_pymc_collapsed_ridge as trainer
from src.compassign.rt.pymc_collapsed_ridge import Stage1CoeffSummaries


def _write_synthetic_csv(path: Path) -> None:
    rng = np.random.default_rng(42)
    rows = []
    groups = [
        (0, 10, 100, 1, 0.3, 0.1, -0.05),
        (1, 10, 100, 1, 0.2, 0.05, 0.02),
    ]
    for species_cluster, comp_id, chem_id, compound_class, b, w1, w2 in groups:
        for _ in range(20):
            is1 = float(rng.normal(loc=5.0, scale=0.4))
            rs1 = float(rng.normal(loc=10.0, scale=0.6))
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
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("COMPASSIGN_RUN_SLOW", "") != "1",
    reason="Set COMPASSIGN_RUN_SLOW=1 to run PyMC smoke tests",
)
def test_train_pymc_collapsed_ridge_smoke(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("pymc")
    pytest.importorskip("pytensor")

    data_csv = tmp_path / "train_rt.csv"
    _write_synthetic_csv(data_csv)

    out_dir = tmp_path / "pymc_single"
    argv = [
        "train_rt_pymc_collapsed_ridge",
        "--data-csv",
        str(data_csv),
        "--output-dir",
        str(out_dir),
        "--feature-center",
        "none",
        "--lambda-mode",
        "fixed",
        "--lambda-slopes",
        "1e-3",
        "--method",
        "map",
        "--map-maxeval",
        "5000",
        "--seed",
        "42",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    trainer.main()

    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["artifact_type"] == "rt_pymc_collapsed_ridge"
    assert (out_dir / "models" / "stage1_coeff_summaries_posterior.npz").exists()

    art = Stage1CoeffSummaries.load_npz(out_dir / "models" / "stage1_coeff_summaries_posterior.npz")
    assert art.feature_names == ("IS1", "RS1")
    assert art.beta_hat.shape[1] == 3
    assert art.beta_cov is not None
    assert art.beta_cov.shape[1:] == (3, 3)
