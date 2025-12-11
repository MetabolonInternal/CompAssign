#!/usr/bin/env python3
"""
Evaluate a trained production RT model on a cap-10 dataset for a quick generalisation check.

This script reuses the production loader and hierarchical RT model to:
  - rebuild the model structure for the training (cap-5) CSV,
  - attach a previously saved trace from `train_rt_prod.py`,
  - apply the model to a cap-10 RT CSV for the same library, and
  - report RMSE / MAE / 95%% coverage on that cap-10 set.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import pandas as pd

# Ensure repository root on path so we can import compassign and pipeline helpers.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipelines import train_rt_prod as train_mod  # type: ignore
from src.compassign.rt_hierarchical import HierarchicalRTModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained hierarchical RT model on a cap-10 RT CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-output-dir",
        type=Path,
        required=True,
        help="Output directory from train_rt_prod.py (contains config.json and models/rt_trace.nc)",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        required=True,
        help="Cap-10 RT production CSV for the same library (e.g. *_lib208_cap10_chemclass_rt_prod.csv)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write evaluation metrics JSON (defaults to <train-output-dir>/results/rt_eval_cap10.json)",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=200,
        help="Maximum number of test rows to evaluate (randomly sampled; set to 0 for all rows)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of posterior samples to use when computing predictions (subsampled from the full trace)",
    )
    return parser.parse_args()


def _load_trained_model(
    train_output_dir: Path,
) -> Tuple[HierarchicalRTModel, train_mod.LoadedData, pd.DataFrame]:
    """Rebuild the RT model for the training CSV and attach a saved trace."""

    config_path = train_output_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"config.json not found under {train_output_dir}")
    with config_path.open("r") as f:
        config = json.load(f)

    data_csv = Path(config["data_csv"])
    if not data_csv.is_absolute():
        data_csv = (REPO_ROOT / data_csv).resolve()
    if not data_csv.exists():
        raise SystemExit(f"Training data CSV from config not found: {data_csv}")

    include_es = bool(config.get("feature_group_map"))
    loaded = train_mod.load_production_csv(data_csv, include_es_group1=include_es)
    train_df = loaded.rt_df.copy()

    compound_features = train_mod.build_compound_features(data_csv, loaded)

    rt_model = HierarchicalRTModel(
        n_clusters=loaded.n_clusters,
        n_species=loaded.n_species,
        n_classes=loaded.n_classes,
        n_compounds=loaded.n_compounds,
        species_cluster=loaded.species_cluster,
        compound_class=loaded.compound_class,
        run_features=loaded.run_features,
        run_species=loaded.run_species,
        run_covariate_columns=loaded.feature_names,
        compound_features=compound_features,
    )
    rt_model.build_model(train_df)

    trace_path = train_output_dir / "models" / "rt_trace.nc"
    if not trace_path.exists():
        raise SystemExit(f"Trace file not found: {trace_path}")
    # Lazily import arviz here to avoid import-time numba/coverage issues when
    # only CLI help is requested.
    import arviz as az  # type: ignore

    rt_model.trace = az.from_netcdf(trace_path)

    return rt_model, loaded, train_df


def main() -> None:
    args = parse_args()

    out_dir = args.train_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"[eval] Loading trained model from {out_dir}")
    rt_model, loaded, _train_df = _load_trained_model(out_dir)

    # Recover species metadata (raw names and groups) from the species mapping CSV
    # used during RT CSV construction. We infer the mapping path from the training
    # data CSV recorded in config.json. Species grouping is attached via
    # sample_set_id, not the integer species index, because the same species can
    # appear in multiple groups (e.g., different matrices).
    config_path = out_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"config.json not found under {out_dir}")
    with config_path.open("r") as f:
        config = json.load(f)
    train_data_csv = Path(config["data_csv"])
    if not train_data_csv.is_absolute():
        train_data_csv = (REPO_ROOT / train_data_csv).resolve()
    stem = train_data_csv.stem  # e.g. merged_training_<hash>_lib208_cap5_chemclass_rt_prod
    root = stem.split("_cap")[0]  # e.g. merged_training_<hash>_lib208
    species_mapping_path = train_data_csv.parent / f"{root}_species_mapping.csv"
    species_mapping_df = None
    species_meta = None
    if species_mapping_path.exists():
        sm_df = pd.read_csv(species_mapping_path)
        # Mapping from sample_set_id -> species_group_raw for group-level metrics
        required_group_cols = {"sample_set_id", "species_group_raw"}
        if required_group_cols.issubset(sm_df.columns):
            species_mapping_df = sm_df[list(required_group_cols)].copy()
        # Per-species metadata for nicer labels
        if {"species", "species_raw"}.issubset(sm_df.columns):
            species_meta = (
                sm_df[["species", "species_raw"]]
                .drop_duplicates(subset="species")
                .copy()
            )

    test_csv = args.test_csv
    if not test_csv.is_absolute():
        test_csv = (REPO_ROOT / test_csv).resolve()
    if not test_csv.exists():
        raise SystemExit(f"Test CSV not found: {test_csv}")

    print(f"[eval] Loading test data from {test_csv}")
    test_raw = pd.read_csv(test_csv)

    # Align species/compound identifiers to the training model's index space.
    species_map = loaded.species_map  # (raw_species, raw_species_cluster) -> species_idx
    compound_map = loaded.compound_map

    if "species_cluster" not in test_raw.columns:
        raise SystemExit("Test CSV must contain 'species_cluster' column")

    # Filter to species/cluster combinations and compounds seen in training.
    known_pairs = set(species_map.keys())
    species_series = test_raw["species"].astype(int)
    cluster_series = test_raw["species_cluster"].astype(int)
    pair_mask = [
        (int(s), int(c)) in known_pairs for s, c in zip(species_series, cluster_series)
    ]
    pair_mask = pd.Series(pair_mask, index=test_raw.index)
    compound_mask = test_raw["compound"].astype(int).isin(compound_map.keys())
    before = len(test_raw)
    mask = pair_mask & compound_mask
    test_df = test_raw[mask].copy()
    dropped = before - len(test_df)
    if dropped > 0:
        print(f"[eval] Dropping {dropped} rows with unseen species/compound ids")
    if len(test_df) == 0:
        raise SystemExit("No overlapping species/compound combinations between train and test data")

    # Optional subsampling for lightweight spot checks
    if args.max_test_rows and args.max_test_rows > 0 and len(test_df) > args.max_test_rows:
        test_df = test_df.sample(n=args.max_test_rows, random_state=42).reset_index(drop=True)
        print(f"[eval] Subsampled test data to {len(test_df)} rows for quick evaluation")

    # Map (species, species_cluster) -> species_idx
    def _map_species(row: pd.Series) -> int:
        key = (int(row["species"]), int(row["species_cluster"]))
        try:
            return species_map[key]
        except KeyError as exc:
            raise SystemExit(f"Unexpected (species, species_cluster) pair {key} in test data") from exc

    test_df["species_idx"] = test_df.apply(_map_species, axis=1)
    test_df["compound_idx"] = test_df["compound"].map(lambda c: compound_map[int(c)])

    # Use the same IS covariate columns as training.
    is_cols = loaded.feature_names
    missing_is = [c for c in is_cols if c not in test_df.columns]
    if missing_is:
        raise SystemExit(f"Test CSV missing required IS covariate columns: {missing_is}")

    X_test = test_df[is_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(X_test)):
        raise SystemExit("Test IS covariate matrix contains non-finite values")

    y_true = test_df["rt"].to_numpy(dtype=float)
    if not np.all(np.isfinite(y_true)):
        raise SystemExit("Test RT values must be finite")

    species_idx = test_df["species_idx"].to_numpy(dtype=int)
    compound_idx = test_df["compound_idx"].to_numpy(dtype=int)

    print("[eval] Computing predictions on cap-10 set")
    pred_mean, pred_std = rt_model.predict_new(
        species_idx=species_idx,
        compound_idx=compound_idx,
        run_features=X_test,
        n_samples=args.n_samples,
    )

    metrics = train_mod.compute_pred_metrics(pred_mean, pred_std, y_true)
    out_path = args.output_json or (results_dir / "rt_eval_cap10.json")

    payload = {"metrics": asdict(metrics), "n_test": int(len(test_df))}
    out_path.write_text(json.dumps(payload, indent=2))

    print(
        f"[eval] cap-10 n={len(test_df)}, RMSE={metrics.rmse:.3f}, "
        f"MAE={metrics.mae:.3f}, coverage95={metrics.coverage_95:.3f}"
    )
    print(f"[eval] Wrote metrics to {out_path}")

    # Per-species and per-species-group diagnostics and RMSE plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None  # type: ignore

    # Attach prediction details to the test frame
    test_df["pred_mean"] = pred_mean
    test_df["pred_std"] = pred_std
    test_df["squared_error"] = (pred_mean - y_true) ** 2
    test_df["abs_error"] = np.abs(pred_mean - y_true)
    lower = pred_mean - 1.96 * pred_std
    upper = pred_mean + 1.96 * pred_std
    test_df["covered_95"] = (y_true >= lower) & (y_true <= upper)

    # Per-species metrics keyed by integer species index
    species_group = (
        test_df.groupby("species")
        .agg(
            n_obs=("rt", "size"),
            rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
            mae=("abs_error", lambda x: float(np.mean(x))),
            coverage_95=("covered_95", lambda x: float(np.mean(x))),
        )
        .reset_index()
    )
    # Attach species_raw (organism) labels where available
    if species_meta is not None:
        species_group = species_group.merge(species_meta, on="species", how="left")
    species_csv = results_dir / "rt_eval_cap10_by_species.csv"
    species_group.to_csv(species_csv, index=False)
    print(f"[eval] Wrote per-species metrics to {species_csv}")

    # Aggregate by species_group_raw (e.g. '1 human blood') using sample_set_id
    # to recover the original matrix group labels.
    if species_mapping_df is not None:
        with_group = test_df.merge(
            species_mapping_df, left_on="sampleset_id", right_on="sample_set_id", how="left"
        )
        group_metrics = (
            with_group.groupby("species_group_raw")
            .agg(
                n_obs=("rt", "size"),
                rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
                mae=("abs_error", lambda x: float(np.mean(x))),
                coverage_95=("covered_95", lambda x: float(np.mean(x))),
            )
            .reset_index()
        )
        # For per-species labels, attach the most common group id (numeric prefix)
        if species_meta is not None:
            species_mode_group = (
                with_group.dropna(subset=["species_group_raw"])
                .groupby("species")["species_group_raw"]
                .agg(lambda x: x.value_counts().idxmax())
                .reset_index()
            )
            species_mode_group["species_group_id"] = (
                species_mode_group["species_group_raw"].astype(str).str.extract(r"^(\d+)")
            )
            species_group = species_group.merge(
                species_mode_group[["species", "species_group_id"]], on="species", how="left"
            )
        group_csv = results_dir / "rt_eval_cap10_by_species_group.csv"
        group_metrics.to_csv(group_csv, index=False)
        print(f"[eval] Wrote per-species-group metrics to {group_csv}")
    else:
        group_metrics = None

    # Per-species RMSE plot (x-axis labelled by species_raw where available)
    if plt is not None and len(species_group) > 0:
        sg_sorted = species_group.sort_values("rmse")
        fig_width = max(6.0, 0.25 * len(sg_sorted))
        plt.figure(figsize=(fig_width, 4.0))
        plt.bar(
            np.arange(len(sg_sorted)),
            sg_sorted["rmse"],
            color="tab:blue",
            alpha=0.8,
        )
        # Compose labels as SPECIES_RAW [group_id] when available, e.g. "HOMO SAPIENS [1]"
        base_labels = sg_sorted.get("species_raw", sg_sorted["species"].astype(str))
        base_labels = base_labels.fillna(sg_sorted["species"].astype(str))
        if "species_group_id" in sg_sorted.columns:
            group_suffix = sg_sorted["species_group_id"].fillna("")
            labels = (
                base_labels.astype(str)
                + group_suffix.map(lambda g: f" [{g}]" if isinstance(g, str) and g else "")
            )
        else:
            labels = base_labels.astype(str)
        plt.xticks(
            np.arange(len(sg_sorted)),
            labels.astype(str),
            rotation=90,
            fontsize=6,
        )
        plt.ylabel("RMSE (min)")
        plt.xlabel("Species")
        plt.title("RT prediction RMSE by species (cap-10)")
        plt.tight_layout()
        plot_path_species = results_dir / "rt_eval_cap10_rmse_by_species.png"
        plt.savefig(plot_path_species, dpi=200)
        plt.close()
        print(f"[eval] Wrote RMSE-by-species plot to {plot_path_species}")

    # Per-species-group RMSE plot (matrix / tissue-level groups)
    if plt is not None and group_metrics is not None and len(group_metrics) > 0:
        species_sorted = group_metrics.sort_values("rmse")
        fig_width = max(6.0, 0.3 * len(species_sorted))
        plt.figure(figsize=(fig_width, 4.0))
        plt.bar(
            np.arange(len(species_sorted)),
            species_sorted["rmse"],
            color="tab:green",
            alpha=0.8,
        )
        plt.xticks(
            np.arange(len(species_sorted)),
            species_sorted["species_group_raw"].astype(str),
            rotation=90,
            fontsize=6,
        )
        plt.ylabel("RMSE (min)")
        plt.xlabel("Species group")
        plt.title("RT prediction RMSE by species group (cap-10)")
        plt.tight_layout()
        plot_path_group = results_dir / "rt_eval_cap10_rmse_by_species_group.png"
        plt.savefig(plot_path_group, dpi=200)
        plt.close()
        print(f"[eval] Wrote RMSE-by-species-group plot to {plot_path_group}")


if __name__ == "__main__":
    main()
