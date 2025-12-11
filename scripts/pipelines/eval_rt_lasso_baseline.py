#!/usr/bin/env python3
"""
Evaluate legacy eslasso RT models as a baseline on cap or real-test RT CSVs.

This script:
  - Loads per-supercategory eslasso regression weights from
    external_repos/sally/new_models/eslasso.
  - Streams an RT production CSV in chunks.
  - For each row, looks up the appropriate (supercategory, lib_id, comp_id)
    lasso model, computes a point prediction and a window-based interval.
  - Aggregates global, per-species, and per-species-group metrics.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class LassoModel:
    features: List[str]
    coefs: np.ndarray
    b: float
    window: float


@dataclass
class AggStats:
    n: int = 0
    sum_sq_err: float = 0.0
    sum_abs_err: float = 0.0
    sum_covered: float = 0.0

    def update(self, sq_err: float, abs_err: float, covered: bool) -> None:
        self.n += 1
        self.sum_sq_err += float(sq_err)
        self.sum_abs_err += float(abs_err)
        self.sum_covered += float(bool(covered))

    def to_metrics(self) -> Dict[str, float]:
        if self.n == 0:
            return {
                "n_obs": 0,
                "rmse": float("nan"),
                "mae": float("nan"),
                "coverage_95": float("nan"),
            }
        rmse = float(np.sqrt(self.sum_sq_err / self.n))
        mae = float(self.sum_abs_err / self.n)
        cov = float(self.sum_covered / self.n)
        return {"n_obs": int(self.n), "rmse": rmse, "mae": mae, "coverage_95": cov}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate eslasso RT baseline on RT production CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-output-dir",
        type=Path,
        required=True,
        help="Output dir for the lib (e.g., output/rt_prod/lib208_cap5) where results will be written.",
    )
    parser.add_argument(
        "--lib-id",
        type=int,
        required=True,
        help="Library id (e.g., 208, 209).",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        required=True,
        help="RT production CSV to evaluate.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Rows per chunk when streaming the RT CSV.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=0,
        help="Maximum number of test rows to evaluate (0 = all).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON for global metrics (default: <train-output-dir>/results/rt_eval_lasso.json).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label for this evaluation (e.g., cap10, realtest) used in output filenames.",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=REPO_ROOT / "external_repos" / "sally" / "new_models" / "eslasso",
        help="Root directory containing per-supercategory eslasso regression CSVs.",
    )
    parser.add_argument(
        "--drop-es",
        action="store_true",
        help="Ignore ES_* covariates by zeroing them before prediction (what-if analysis).",
    )
    return parser.parse_args()


def discover_supercategory_dirs(models_root: Path) -> Dict[int, Path]:
    """Map supercategory number → directory path under models_root."""
    mapping: Dict[int, Path] = {}
    if not models_root.is_dir():
        raise SystemExit(f"Models root not found: {models_root}")
    for d in models_root.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"supercategory_(\d+)_", d.name)
        if not m:
            continue
        num = int(m.group(1))
        mapping[num] = d
    if not mapping:
        raise SystemExit(f"No supercategory_* directories found under {models_root}")
    return mapping


def load_lasso_models(models_root: Path, lib_id: int) -> Dict[Tuple[int, int], LassoModel]:
    """
    Load eslasso regression weights for a given lib into a dictionary
    keyed by (supercategory_number, comp_id).
    """
    super_dirs = discover_supercategory_dirs(models_root)
    models: Dict[Tuple[int, int], LassoModel] = {}

    reserved = {
        "comp_id",
        "b",
        "buffer_around_center",
        "cloud_center_range",
        "test_ws",
        "window",
        "lib_id",
    }

    for super_num, d in super_dirs.items():
        path = d / f"regression_{lib_id}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "comp_id" not in df.columns or "b" not in df.columns or "window" not in df.columns:
            continue

        # Features: all numeric IS*/ES_*/RS* columns; IS selection is shared across models,
        # ES selection can be sparse and differs per lasso, as encoded in the coefficients.
        feature_cols = [
            c for c in df.columns if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
        ]
        feature_cols_sorted = sorted(feature_cols)  # deterministic order

        for _, row in df.iterrows():
            comp_id = int(row["comp_id"])
            coefs = row[feature_cols_sorted].to_numpy(dtype=float)
            model = LassoModel(
                features=feature_cols_sorted,
                coefs=coefs,
                b=float(row["b"]),
                window=float(row["window"]),
            )
            models[(super_num, comp_id)] = model

    if not models:
        raise SystemExit(f"No eslasso models found under {models_root} for lib {lib_id}")
    return models


def find_species_mapping(lib_id: int) -> Path:
    """Best-effort search for the species mapping CSV for a given lib."""
    search_dirs = [
        REPO_ROOT / f"repo_export/lib{lib_id}/species_mapping",
        REPO_ROOT / "repo_export",
    ]
    for base in search_dirs:
        if base.is_dir():
            candidates = sorted(base.glob(f"**/*_lib{lib_id}_species_mapping.csv"))
            if candidates:
                return candidates[0]
    raise SystemExit(f"No species mapping found for lib {lib_id} under repo_export")


def main() -> None:
    args = parse_args()

    out_dir = args.train_output_dir
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    test_csv = args.test_csv
    if not test_csv.is_absolute():
        test_csv = (REPO_ROOT / test_csv).resolve()
    if not test_csv.exists():
        raise SystemExit(f"Test CSV not found: {test_csv}")

    test_name = test_csv.name
    if "_cap10_" in test_name:
        dataset_tag = "cap10"
    elif "realtest" in test_name:
        dataset_tag = "realtest"
    else:
        dataset_tag = "unlabelled"
    label = args.label or dataset_tag
    label_suffix = f"_{label}" if label else ""

    print(f"[lasso] Loading eslasso models from {args.models_root} for lib {args.lib_id}")
    models = load_lasso_models(args.models_root, lib_id=args.lib_id)
    print(f"[lasso] Loaded {len(models):,} compound models across supercategories")

    species_mapping_path = find_species_mapping(args.lib_id)
    print(f"[lasso] Using species mapping: {species_mapping_path}")
    sm_df = pd.read_csv(species_mapping_path)
    # Minimal mapping needed for evaluation
    if not {"sample_set_id", "species_group_raw"}.issubset(sm_df.columns):
        raise SystemExit(
            f"Species mapping {species_mapping_path} missing required columns 'sample_set_id'/'species_group_raw'"
        )
    # Keep species_raw for nicer per-species labels when available
    keep_cols = ["sample_set_id", "species_group_raw"]
    if "species_raw" in sm_df.columns:
        keep_cols.append("species_raw")
    sm_df = sm_df[keep_cols].copy()

    # Build simple lookup dict for species_raw per sample_set_id (optional)
    ssid_to_species_raw: Dict[int, Optional[str]] = {}
    if "species_raw" in sm_df.columns:
        for _, row in sm_df.iterrows():
            ssid_to_species_raw[int(row["sample_set_id"])] = str(row["species_raw"])

    global_stats = AggStats()
    species_stats: Dict[str, AggStats] = {}
    group_stats: Dict[str, AggStats] = {}
    # Per-group chemical coverage (unique compounds with a lasso model)
    group_compounds_total: Dict[str, set[int]] = {}
    group_compounds_modeled: Dict[str, set[int]] = {}

    remaining = args.max_test_rows if args.max_test_rows and args.max_test_rows > 0 else None
    total_rows = 0
    used_rows = 0
    skipped_no_group = 0
    skipped_no_model = 0
    chunk_idx = 0

    print(
        f"[lasso] Streaming {test_csv} with chunk_size={args.chunk_size}, "
        f"max_test_rows={'ALL' if remaining is None else remaining}"
    )

    for chunk in pd.read_csv(test_csv, chunksize=args.chunk_size):
        chunk_idx += 1
        if remaining is not None and remaining <= 0:
            break

        total_rows += len(chunk)

        # Limit to remaining rows if max_test_rows is set
        if remaining is not None and len(chunk) > remaining:
            chunk = chunk.iloc[:remaining].copy()

        # Join species mapping to get species_group_raw (and optional species_raw)
        chunk = chunk.merge(
            sm_df,
            left_on="sampleset_id",
            right_on="sample_set_id",
            how="left",
        )

        for _, row in chunk.iterrows():
            group_raw = row.get("species_group_raw")
            if pd.isna(group_raw):
                skipped_no_group += 1
                continue
            m = re.match(r"\s*(\d+)", str(group_raw))
            if not m:
                skipped_no_group += 1
                continue
            super_num = int(m.group(1))

            try:
                comp_id = int(row["comp_id"])
            except Exception:
                continue

            group_label = str(group_raw)
            total_set = group_compounds_total.setdefault(group_label, set())
            total_set.add(comp_id)

            model = models.get((super_num, comp_id))
            if model is None:
                skipped_no_model += 1
                continue

            # Extract covariate values for this model's features.
            try:
                x_vals = row[model.features].to_numpy(dtype=float)
                if args.drop_es:
                    # Zero out ES_* features to measure their contribution.
                    es_mask = [f.startswith("ES_") for f in model.features]
                    if any(es_mask):
                        x_vals = x_vals.copy()
                        for i, is_es in enumerate(es_mask):
                            if is_es:
                                x_vals[i] = 0.0
            except KeyError as exc:
                missing = [f for f in model.features if f not in row.index]
                raise SystemExit(
                    f"RT CSV missing covariate columns required by lasso model: {missing}"
                ) from exc

            pred_rt = float(model.b + np.dot(model.coefs, x_vals))
            y_true = float(row["rt"])
            err = pred_rt - y_true
            sq_err = err * err
            abs_err = abs(err)
            covered = abs_err <= model.window

            global_stats.update(sq_err, abs_err, covered)
            used_rows += 1

            modeled_set = group_compounds_modeled.setdefault(group_label, set())
            modeled_set.add(comp_id)

            # Species-level key: prefer species_raw (organism) when available
            ssid = int(row["sampleset_id"])
            species_label = ssid_to_species_raw.get(ssid)
            if species_label is None:
                species_label = f"species_{row.get('species', 'NA')}"

            stats_s = species_stats.setdefault(str(species_label), AggStats())
            stats_s.update(sq_err, abs_err, covered)

            # Group-level key: full species_group_raw string
            group_label = str(group_raw)
            stats_g = group_stats.setdefault(group_label, AggStats())
            stats_g.update(sq_err, abs_err, covered)

        if remaining is not None:
            remaining -= len(chunk)

        # Lightweight textual progress for long runs (e.g., lib209 real-test)
        if chunk_idx % 20 == 0:
            print(
                f"[lasso] Progress: chunks={chunk_idx}, total_seen={total_rows:,}, "
                f"evaluated={used_rows:,}, skipped_no_model={skipped_no_model:,}"
            )

    print(
        f"[lasso] Total rows seen={total_rows:,}, evaluated={used_rows:,}, "
        f"skipped_no_group={skipped_no_group:,}, skipped_no_model={skipped_no_model:,}"
    )

    global_metrics = global_stats.to_metrics()
    out_json = args.output_json or (results_dir / f"rt_eval_lasso{label_suffix}.json")
    payload = {
        "metrics": global_metrics,
        "n_test": int(global_stats.n),
        "chunk_size": args.chunk_size,
        "lib_id": args.lib_id,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(
        f"[lasso] Global: n={global_stats.n:,}, RMSE={global_metrics['rmse']:.3f}, "
        f"MAE={global_metrics['mae']:.3f}, coverage95={global_metrics['coverage_95']:.3f}"
    )
    print(f"[lasso] Wrote global metrics to {out_json}")

    # Per-species DataFrame
    species_rows = []
    for label, stats in species_stats.items():
        m = stats.to_metrics()
        row = {"species_label": label, **m}
        species_rows.append(row)
    species_df = pd.DataFrame(species_rows)
    species_csv = results_dir / f"rt_eval_lasso_by_species{label_suffix}.csv"
    species_df.to_csv(species_csv, index=False)
    print(f"[lasso] Wrote per-species metrics to {species_csv}")

    # Per-species-group DataFrame
    group_rows = []
    for label, stats in group_stats.items():
        m = stats.to_metrics()
        total_set = group_compounds_total.get(label, set())
        modeled_set = group_compounds_modeled.get(label, set())
        n_total = len(total_set)
        n_modeled = len(modeled_set)
        coverage = float(n_modeled / n_total) if n_total > 0 else float("nan")
        row = {
            "species_group_raw": label,
            **m,
            "n_compounds_total": int(n_total),
            "n_compounds_modeled": int(n_modeled),
            "compound_coverage": coverage,
        }
        group_rows.append(row)
    group_df = pd.DataFrame(group_rows)
    group_csv = results_dir / f"rt_eval_lasso_by_species_group{label_suffix}.csv"
    group_df.to_csv(group_csv, index=False)
    print(f"[lasso] Wrote per-species-group metrics to {group_csv}")

    # Plots (optional): require matplotlib
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None  # type: ignore

    if plt is not None and not species_df.empty:
        sg_sorted = species_df.sort_values("rmse")
        fig_width = max(6.0, 0.25 * len(sg_sorted))
        plt.figure(figsize=(fig_width, 4.0))
        plt.bar(
            np.arange(len(sg_sorted)),
            sg_sorted["rmse"],
            color="tab:blue",
            alpha=0.8,
        )
        plt.xticks(
            np.arange(len(sg_sorted)),
            sg_sorted["species_label"].astype(str),
            rotation=90,
            fontsize=6,
        )
        plt.ylabel("RMSE (min)")
        plt.xlabel("Species")
        plt.title("Lasso RT prediction RMSE by species")
        plt.tight_layout()
        plot_path_species = results_dir / f"rt_eval_lasso_rmse_by_species{label_suffix}.png"
        plt.savefig(plot_path_species, dpi=200)
        plt.close()
        print(f"[lasso] Wrote RMSE-by-species plot to {plot_path_species}")

    if plt is not None and not group_df.empty:
        g_sorted = group_df.sort_values("rmse")
        fig_width = max(6.0, 0.3 * len(g_sorted))
        plt.figure(figsize=(fig_width, 4.0))
        plt.bar(
            np.arange(len(g_sorted)),
            g_sorted["rmse"],
            color="tab:green",
            alpha=0.8,
        )
        plt.xticks(
            np.arange(len(g_sorted)),
            g_sorted["species_group_raw"].astype(str),
            rotation=90,
            fontsize=6,
        )
        plt.ylabel("RMSE (min)")
        plt.xlabel("Species group")
        plt.title("Lasso RT prediction RMSE by species group")
        plt.tight_layout()
        plot_path_group = results_dir / f"rt_eval_lasso_rmse_by_species_group{label_suffix}.png"
        plt.savefig(plot_path_group, dpi=200)
        plt.close()
        print(f"[lasso] Wrote RMSE-by-species-group plot to {plot_path_group}")


if __name__ == "__main__":
    main()
