"""Smoke tests for the active-learning assessment CLI with stubbed dependencies."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import scripts.assess_active_learning as assess
from src.compassign.utils import RunMetadata


def test_assess_active_learning_summary_includes_random_section(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_generate_dataset(args):
        peaks = pd.DataFrame(
            {
                "peak_id": [0, 1, 2, 3],
                "species": [0, 0, 1, 1],
                "true_compound": [0, 1, 0, 1],
                "rt": [1.0, 1.2, 2.0, 2.1],
            }
        )
        compound_info = pd.DataFrame(
            {
                "compound_id": [0, 1],
                "true_mass": [100.0, 110.0],
                "predicted_rt": [1.1, 2.05],
                "rt_prediction_std": [0.2, 0.3],
            }
        )
        hierarchical_params = {
            "n_clusters": 1,
            "n_classes": 1,
            "species_cluster": np.zeros(args.n_species, dtype=int),
            "compound_class": np.zeros(args.n_compounds, dtype=int),
        }
        run_df = pd.DataFrame(
            {
                "run": [0, 1, 2, 3],
                "species": [0, 0, 1, 1],
                "run_covariate_0": [0.0, 0.1, 0.2, 0.3],
            }
        )
        run_meta = RunMetadata(
            df=run_df,
            features=run_df[["run_covariate_0"]].to_numpy(dtype=float),
            species=run_df["species"].to_numpy(dtype=int),
            covariate_columns=["run_covariate_0"],
        )
        return peaks, compound_info, hierarchical_params, run_meta

    def fake_compute_rt_predictions(*args, **kwargs):  # noqa: ANN001, ANN002, D401
        return {(0, 0): (1.0, 0.2)}

    class DummyModel:
        def __init__(self, train_pack):
            self.train_pack = train_pack

    def fake_setup_model(*_args, **_kwargs):
        mask = np.array([[True, True, False]])
        train_pack = {"mask": mask}
        model = DummyModel(train_pack=train_pack)
        return model, train_pack

    def fake_run_naive_review(*_args, **_kwargs):
        baseline = SimpleNamespace(precision=0.5, recall=0.4, f1=0.45)
        final = SimpleNamespace(
            precision=0.7,
            recall=0.75,
            f1=0.72,
            assignment_rate=0.6,
            compound_precision=0.65,
            compound_recall=0.7,
            compound_f1=0.68,
            ece=0.1,
        )
        return baseline, final, 100

    def fake_run_random_review(*_args, **_kwargs):
        history = [
            {
                "round": 0,
                "annotations_this_round": 0,
                "cumulative_annotations": 0,
                "precision": 0.5,
                "recall": 0.4,
                "f1": 0.45,
            },
            {
                "round": 1,
                "annotations_this_round": 10,
                "cumulative_annotations": 10,
                "precision": 0.55,
                "recall": 0.6,
                "f1": 0.57,
            },
        ]
        final = SimpleNamespace(
            precision=0.55,
            recall=0.6,
            f1=0.57,
            assignment_rate=0.5,
            compound_precision=0.5,
            compound_recall=0.55,
            compound_f1=0.52,
            ece=0.2,
        )
        return history, final

    def fake_run_active_learning(*_args, **_kwargs):
        history = [
            {
                "round": 0,
                "annotations_this_round": 0,
                "cumulative_annotations": 0,
                "precision": 0.6,
                "recall": 0.5,
                "f1": 0.55,
            },
            {
                "round": 1,
                "annotations_this_round": 5,
                "cumulative_annotations": 5,
                "precision": 0.75,
                "recall": 0.8,
                "f1": 0.77,
            },
        ]
        final = SimpleNamespace(
            precision=0.75,
            recall=0.8,
            f1=0.77,
            assignment_rate=0.7,
            compound_precision=0.78,
            compound_recall=0.8,
            compound_f1=0.79,
            ece=0.08,
        )
        return history, final

    def fake_assign_to_dict(result):
        return {
            "precision": float(result.precision),
            "recall": float(result.recall),
            "f1": float(result.f1),
            "assignment_rate": float(getattr(result, "assignment_rate", 0.0)),
            "compound_precision": float(getattr(result, "compound_precision", 0.0)),
            "compound_recall": float(getattr(result, "compound_recall", 0.0)),
            "compound_f1": float(getattr(result, "compound_f1", 0.0)),
            "ece": float(getattr(result, "ece", 0.0)),
        }

    def fake_area_under_recall_curve(history):
        return sum(entry["recall"] for entry in history)

    monkeypatch.setattr(assess, "generate_dataset", fake_generate_dataset)
    monkeypatch.setattr(assess, "compute_rt_predictions", fake_compute_rt_predictions)
    monkeypatch.setattr(assess, "setup_model", fake_setup_model)
    monkeypatch.setattr(assess, "run_naive_review", fake_run_naive_review)
    monkeypatch.setattr(assess, "run_random_review", fake_run_random_review)
    monkeypatch.setattr(assess, "run_active_learning", fake_run_active_learning)
    monkeypatch.setattr(assess, "assign_to_dict", fake_assign_to_dict)
    monkeypatch.setattr(assess, "area_under_recall_curve", fake_area_under_recall_curve)

    output_dir = tmp_path / "al"
    argv = [
        "assess_active_learning",
        "--output-dir",
        str(output_dir),
        "--n-species",
        "2",
        "--n-compounds",
        "2",
        "--rounds",
        "2",
        "--batch-size",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assess.main()

    summary_files = list((output_dir / "results").glob("summary.json"))
    assert summary_files, "expected assessment summary output"

    summary = json.loads(summary_files[0].read_text())
    assert "random" in summary
    assert summary["random"]["metrics"]["recall"] == 0.6
    assert summary["active_learning"]["clicks_to_target_recall"] == 5
