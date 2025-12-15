"""Tests for the retention-time model comparison CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import src.compassign.rt.model_comparison as rtc


class _DummyPosterior:
    def __getitem__(self, item: str):
        return _DummyPosteriorEntry()


class _DummyPosteriorEntry:
    def __init__(self) -> None:
        self.values = np.ones((1, 1, 1))


class _DummyTrace:
    def __init__(self) -> None:
        self.posterior = _DummyPosterior()


def test_rt_model_comparison_writes_report(tmp_path: Path, monkeypatch) -> None:
    def fake_create_metabolomics_data(**_: object):
        peak_rows = []
        for species in range(3):
            for compound in range(2):
                peak_rows.append(
                    {
                        "species": species,
                        "true_compound": compound,
                        "rt": float(species + compound + 1.0),
                        "run": species,
                    }
                )
        peak_df = pd.DataFrame(peak_rows)
        compound_df = pd.DataFrame(
            {
                "compound_id": [0, 1],
                "predicted_rt": [1.0, 2.0],
                "rt_prediction_std": [0.2, 0.2],
            }
        )
        hierarchical_params = {
            "n_clusters": 1,
            "n_classes": 1,
            "species_cluster": np.zeros(3, dtype=int),
            "compound_class": np.zeros(2, dtype=int),
            "run_covariate_columns": ["run_covariate_0"],
        }
        run_df = pd.DataFrame(
            {
                "run": [0, 1, 2],
                "species": [0, 1, 2],
                "run_covariate_0": [0.0, 0.1, 0.2],
            }
        )
        peak_df["run_covariate_0"] = peak_df["run"].map(
            dict(zip(run_df["run"], run_df["run_covariate_0"]))
        )
        return peak_df, compound_df, None, None, hierarchical_params

    class DummyRTModel:
        def __init__(self, *args, **kwargs):
            self._built = False
            self.run_metadata = kwargs.get("run_metadata")
            self.run_features = kwargs.get("run_features")

        def build_model(self, train_df: pd.DataFrame) -> None:
            assert "run" in train_df.columns
            self._built = True
            self._train_df = train_df

        def sample(self, **_: object):
            return _DummyTrace()

        def predict_new(
            self,
            species_idx: np.ndarray,
            compound_idx: np.ndarray,
            run_idx: np.ndarray,
            run_features: np.ndarray | None = None,
            n_samples=None,
        ):
            assert run_idx.shape == species_idx.shape
            preds = species_idx.astype(float) + compound_idx.astype(float) + 1.0
            return preds, np.full_like(preds, 0.1, dtype=float)

    class DummyBaseline:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def fit(self, train_df: pd.DataFrame, **_: object):
            self._mean_rt = float(train_df["rt"].mean())
            return self

        def predict(self, species_idx: np.ndarray, compound_idx: np.ndarray, run_idx: np.ndarray | None = None) -> np.ndarray:
            _ = species_idx, compound_idx, run_idx
            return np.full(species_idx.shape, self._mean_rt, dtype=float)

    monkeypatch.setattr(rtc, "create_metabolomics_data", fake_create_metabolomics_data)
    monkeypatch.setattr(rtc, "HierarchicalRTModel", DummyRTModel)
    monkeypatch.setattr(rtc, "SpeciesCompoundLassoBaseline", DummyBaseline)

    output_dir = tmp_path / "comparison"
    argv = [
        "rt_model_comparison",
        "--n-species",
        "3",
        "--n-compounds",
        "2",
        "--test-size",
        "0.5",
        "--output-dir",
        str(output_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    rtc.main()

    json_files = list(output_dir.glob("rt_model_comparison_*.json"))
    assert json_files, "expected a comparison report to be written"

    report = json.loads(json_files[0].read_text())
    hier_mae = report["metrics_on_coverage"]["hierarchical_chem"]["mae"]
    baseline_mae = report["metrics_on_coverage"]["baseline_species_compound"]["mae"]

    assert hier_mae <= 1e-12
    assert baseline_mae > hier_mae
