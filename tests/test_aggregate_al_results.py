"""Tests for aggregate active-learning results helper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import scripts.aggregate_al_results as aggregate


def test_aggregate_al_results_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / "experiment" / "seed_1"
    base_dir.mkdir(parents=True)
    summary_path = base_dir / "summary.json"
    payload = {
        "config": {
            "seed": 7,
            "threshold": 0.4,
            "batch_size": 10,
            "rounds": 3,
            "acquisition": "hybrid",
            "lambda_fp": 0.5,
            "mass_error_ppm": 10.0,
            "decoy_fraction": 0.2,
            "initial_labeled_fraction": 0.1,
        },
        "dataset": {"n_peaks": 20, "mean_candidates": 3.0},
        "naive": {
            "metrics": {"precision": 0.7, "recall": 0.6},
        },
        "active_learning": {
            "metrics": {"precision": 0.8, "recall": 0.75, "f1": 0.77},
            "clicks_to_recall_ratio": 25,
            "clicks_to_target_recall": 45,
            "final_annotations": 60,
        },
        "random": {
            "metrics": {"precision": 0.6, "recall": 0.55, "f1": 0.57},
            "clicks_to_recall_ratio": 80,
            "final_annotations": 90,
        },
        "recall_ratio_target": 1.0,
    }
    summary_path.write_text(json.dumps(payload))

    outdir = tmp_path / "aggregate"
    argv = [
        "aggregate_al_results",
        "--base-dirs",
        str(tmp_path),
        "--outdir",
        str(outdir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    aggregate.main()

    csv_path = outdir / "aggregate.csv"
    json_path = outdir / "aggregate.json"

    assert csv_path.exists()
    assert json_path.exists()

    df = pd.read_csv(csv_path)
    assert {"seed", "al_precision", "rand_precision"}.issubset(df.columns)
    assert df.loc[0, "al_precision"] == 0.8
    assert df.loc[0, "rand_clicks_to_ratio"] == 80
