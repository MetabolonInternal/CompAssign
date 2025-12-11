#!/usr/bin/env python3
"""
Streaming evaluation for the fast ridge RT production model.

Writes:
  - global metrics JSON
  - per-species-group CSV
  - per-species CSV
  - RMSE plots
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compassign.rt.fast_ridge_prod import (  # noqa: E402
    RUN_KEY_COLS,
    RidgeGroupCompoundRTModel,
    detect_lib_id_from_paths,
    load_species_mapping,
)


REQUIRED_COLS = [
    *RUN_KEY_COLS,
    "rt",
    "comp_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ridge RT model on large CSVs using chunked predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-output-dir",
        type=Path,
        required=True,
        help="Output directory from train_rt_ridge_prod.py (contains config.json and models/rt_ridge_model.npz).",
    )
    parser.add_argument(
        "--test-csv", type=Path, required=True, help="RT production CSV to evaluate."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write evaluation metrics JSON (default: <train-output-dir>/results/rt_eval_ridge.json).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Number of rows per prediction chunk.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=0,
        help="Maximum number of test rows to evaluate (0 = all).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label for this evaluation used in output filenames.",
    )
    parser.add_argument(
        "--use-posterior",
        action="store_true",
        help="Use Bayesian ridge posterior predictive (Student-t) if available in the model file.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.95,
        help="Central interval mass for coverage calculation when using posterior (e.g. 0.95).",
    )
    return parser.parse_args()


@dataclass
class AggStats:
    n: int = 0
    sum_sq_err: float = 0.0
    sum_abs_err: float = 0.0
    sum_covered: float = 0.0

    def update(self, sq_err: np.ndarray, abs_err: np.ndarray, covered: np.ndarray) -> None:
        self.n += int(len(sq_err))
        self.sum_sq_err += float(sq_err.sum())
        self.sum_abs_err += float(abs_err.sum())
        self.sum_covered += float(covered.sum())

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


def _detect_lib_id_from_paths(train_out_dir: Path, test_csv: Path) -> int | None:
    return detect_lib_id_from_paths([train_out_dir, test_csv])


def load_model(train_output_dir: Path) -> tuple[RidgeGroupCompoundRTModel, dict]:
    config_path = train_output_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"config.json not found under {train_output_dir}")
    config = json.loads(config_path.read_text())

    model_path = train_output_dir / "models" / "rt_ridge_model.npz"
    if not model_path.exists():
        raise SystemExit(f"Ridge model file not found: {model_path}")
    model = RidgeGroupCompoundRTModel.load_npz(model_path)
    return model, config


def main() -> None:
    args = parse_args()
    if not (0.0 < float(args.interval) < 1.0):
        raise SystemExit("--interval must be in (0, 1)")

    train_out_dir = args.train_output_dir
    if not train_out_dir.exists():
        raise SystemExit(f"Train output dir not found: {train_out_dir}")
    model, config = load_model(train_out_dir)
    include_es_group1 = bool(config.get("include_es_group1", False))
    use_posterior = bool(args.use_posterior)
    if use_posterior:
        try:
            from scipy import stats as scipy_stats  # type: ignore
        except Exception:
            raise SystemExit("scipy is required for --use-posterior but could not be imported")
        alpha = 0.5 + 0.5 * float(args.interval)
        q_norm = float(scipy_stats.norm.ppf(alpha))

    test_csv = args.test_csv
    if not test_csv.is_absolute():
        test_csv = (REPO_ROOT / test_csv).resolve()
    if not test_csv.exists():
        raise SystemExit(f"Test CSV not found: {test_csv}")

    header = pd.read_csv(test_csv, nrows=0)
    missing_req = [c for c in REQUIRED_COLS if c not in header.columns]
    if missing_req:
        raise SystemExit(f"CSV missing required columns: {missing_req}")

    lib_id = config.get("lib_id")
    if lib_id is None:
        lib_id = _detect_lib_id_from_paths(train_out_dir, test_csv)
    if lib_id is None:
        raise SystemExit("Unable to determine lib_id; cannot load species mapping.")
    lib_id = int(lib_id)

    species_mapping = load_species_mapping(lib_id)
    ssid_to_group_raw = species_mapping.sample_set_id_to_group_raw
    ssid_to_super = species_mapping.sample_set_id_to_supercategory
    group1_ssids = species_mapping.group1_sample_set_ids
    ssid_to_species_raw: Dict[int, Optional[str]] = {
        int(k): str(v) for k, v in species_mapping.sample_set_id_to_species_raw.items()
    }

    results_dir = train_out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    label_suffix = f"_{args.label}" if args.label else ""
    out_json = (
        args.output_json
        if args.output_json is not None
        else results_dir / f"rt_eval_ridge_streaming{label_suffix}.json"
    )
    out_csv_species = results_dir / f"rt_eval_ridge_streaming_by_species{label_suffix}.csv"
    out_csv_group = results_dir / f"rt_eval_ridge_streaming_by_species_group{label_suffix}.csv"

    feature_cols = list(model.feature_names)
    es_cols = [c for c in feature_cols if c.startswith("ES_")]

    remaining = args.max_test_rows if args.max_test_rows and args.max_test_rows > 0 else None

    global_stats = AggStats()
    species_stats: Dict[str, AggStats] = {}
    group_stats: Dict[str, AggStats] = {}
    group_compounds_total: Dict[str, set[int]] = {}
    group_compounds_modeled: Dict[str, set[int]] = {}

    total_rows = 0
    kept_rows = 0
    skipped_no_group = 0
    skipped_no_model = 0
    used_fallback_rows = 0

    print(f"[ridge] Loading model from {train_out_dir}")
    print(f"[ridge] Streaming test data from {test_csv} with chunk_size={args.chunk_size}")

    for chunk in pd.read_csv(test_csv, chunksize=args.chunk_size):
        if remaining is not None and remaining <= 0:
            break

        total_rows += len(chunk)
        if remaining is not None and len(chunk) > remaining:
            chunk = chunk.iloc[:remaining].copy()

        # Map groups.
        group_raw = chunk["sampleset_id"].astype(int).map(ssid_to_group_raw)
        ok_group = group_raw.notna()
        if not ok_group.all():
            skipped_no_group += int((~ok_group).sum())
            chunk = chunk.loc[ok_group].copy()
            group_raw = group_raw.loc[ok_group]

        super_num = chunk["sampleset_id"].astype(int).map(ssid_to_super)
        ok_super = super_num.notna()
        if not ok_super.all():
            skipped_no_group += int((~ok_super).sum())
            chunk = chunk.loc[ok_super].copy()
            group_raw = group_raw.loc[ok_super]
            super_num = super_num.loc[ok_super]

        super_arr = super_num.astype(int).to_numpy(dtype=np.int64, copy=False)

        if es_cols:
            chunk[es_cols] = chunk[es_cols].fillna(0.0)
            if include_es_group1:
                # Mask ES covariates to Group 1 only.
                mask1 = chunk["sampleset_id"].astype(int).isin(group1_ssids)
                chunk.loc[~mask1, es_cols] = 0.0

        x = chunk[feature_cols].to_numpy(dtype=np.float64, copy=False)
        y = chunk["rt"].to_numpy(dtype=np.float64, copy=False)
        comp_id = chunk["comp_id"].to_numpy(dtype=np.int64, copy=False)

        # Track total compound coverage per group before dropping unmodeled rows.
        group_raw_arr_all = group_raw.to_numpy(dtype=object, copy=False)
        unique_groups_all = np.unique(group_raw_arr_all)
        for g in unique_groups_all.tolist():
            mask_g = group_raw_arr_all == g
            total_set = group_compounds_total.setdefault(str(g), set())
            total_set.update(np.unique(comp_id[mask_g]).tolist())

        pred_df = None
        if use_posterior:
            pred_mean, pred_std, pred_df, used_fallback = model.predict_posterior(
                super_num=super_arr,
                comp_id=comp_id,
                x=x,
            )
        else:
            pred_mean, pred_std, used_fallback = model.predict(
                super_num=super_arr,
                comp_id=comp_id,
                x=x,
            )
        modeled = np.isfinite(pred_mean)
        if not modeled.all():
            skipped_no_model += int((~modeled).sum())

        used_fallback_rows += int(used_fallback[modeled].sum())

        pred_mean = pred_mean[modeled]
        pred_std = pred_std[modeled]
        pred_df = pred_df[modeled] if pred_df is not None else None
        y = y[modeled]
        group_raw = group_raw.to_numpy(dtype=object, copy=False)[modeled]
        comp_id = comp_id[modeled]
        ssid = chunk["sampleset_id"].to_numpy(dtype=np.int64, copy=False)[modeled]

        kept_rows += int(len(y))

        err = pred_mean - y
        sq_err = np.square(err)
        abs_err = np.abs(err)
        if pred_df is None:
            covered = abs_err <= (1.96 * pred_std)
        else:
            df = pred_df.astype(np.float64, copy=False)
            # Convert predictive std to Student-t scale parameter.
            scale = pred_std * np.sqrt(np.maximum(df - 2.0, 1.0) / df)
            q = np.full(df.shape, q_norm, dtype=np.float64)
            small = df < 30.0
            if small.any():
                q[small] = scipy_stats.t.ppf(alpha, df[small])  # type: ignore[name-defined]
            covered = abs_err <= (q * scale)

        global_stats.update(sq_err, abs_err, covered)

        # Group metrics and compound coverage sets.
        unique_groups = np.unique(group_raw)
        for g in unique_groups.tolist():
            mask_g = group_raw == g
            stats_g = group_stats.setdefault(str(g), AggStats())
            stats_g.update(sq_err[mask_g], abs_err[mask_g], covered[mask_g])

            modeled_set = group_compounds_modeled.setdefault(str(g), set())
            modeled_set.update(np.unique(comp_id[mask_g]).tolist())

        # Species metrics (optional, uses species_raw if present).
        species_labels = []
        for s in ssid.tolist():
            label = ssid_to_species_raw.get(int(s))
            if label is None:
                label = f"sampleset_{int(s)}"
            species_labels.append(label)
        species_labels_arr = np.asarray(species_labels, dtype=object)
        for s in np.unique(species_labels_arr).tolist():
            mask_s = species_labels_arr == s
            stats_s = species_stats.setdefault(str(s), AggStats())
            stats_s.update(sq_err[mask_s], abs_err[mask_s], covered[mask_s])

        if remaining is not None:
            remaining -= len(chunk)

    print(
        f"[ridge] Total rows seen={total_rows:,}, kept={kept_rows:,}, "
        f"skipped_no_group={skipped_no_group:,}, skipped_no_model={skipped_no_model:,}, "
        f"used_fallback={used_fallback_rows:,}"
    )

    global_metrics = global_stats.to_metrics()
    print(
        f"[ridge] Global: n={global_metrics['n_obs']:,}, RMSE={global_metrics['rmse']:.3f}, "
        f"MAE={global_metrics['mae']:.3f}, coverage95={global_metrics['coverage_95']:.3f}"
    )

    out_json.write_text(
        json.dumps(
            {
                "metrics": global_metrics,
                "n_test": int(total_rows),
                "n_used": int(kept_rows),
                "chunk_size": int(args.chunk_size),
                "lib_id": int(lib_id),
                "use_posterior": bool(use_posterior),
                "interval": float(args.interval),
                "skipped_no_group": int(skipped_no_group),
                "skipped_no_model": int(skipped_no_model),
                "used_fallback_rows": int(used_fallback_rows),
            },
            indent=2,
        )
    )
    print(f"[ridge] Wrote global metrics to {out_json}")

    # Per-species-group metrics CSV (sorted for determinism).
    group_rows = []
    for group_label, stats in group_stats.items():
        m = stats.to_metrics()
        total = group_compounds_total.get(group_label, set())
        modeled = group_compounds_modeled.get(group_label, set())
        group_rows.append(
            {
                "species_group_raw": group_label,
                "n_obs": int(m["n_obs"]),
                "rmse": float(m["rmse"]),
                "mae": float(m["mae"]),
                "coverage_95": float(m["coverage_95"]),
                "n_compounds_total": int(len(total)),
                "n_compounds_modeled": int(len(modeled)),
                "compound_coverage": float(len(modeled) / len(total)) if total else float("nan"),
            }
        )
    group_df = pd.DataFrame(group_rows).sort_values("species_group_raw")
    group_df.to_csv(out_csv_group, index=False)
    print(f"[ridge] Wrote per-species-group metrics to {out_csv_group}")

    # Per-species metrics CSV.
    species_rows = []
    for species_label, stats in species_stats.items():
        m = stats.to_metrics()
        species_rows.append(
            {
                "species": species_label,
                "n_obs": int(m["n_obs"]),
                "rmse": float(m["rmse"]),
                "mae": float(m["mae"]),
                "coverage_95": float(m["coverage_95"]),
            }
        )
    species_df = pd.DataFrame(species_rows).sort_values("rmse")
    species_df.to_csv(out_csv_species, index=False)
    print(f"[ridge] Wrote per-species metrics to {out_csv_species}")

    # Simple plots (best-effort).
    try:
        import matplotlib.pyplot as plt  # type: ignore

        plot_path_group = results_dir / f"rt_eval_ridge_rmse_by_species_group{label_suffix}.png"
        dfp = group_df.sort_values("rmse")
        plt.figure(figsize=(10, 6))
        plt.barh(dfp["species_group_raw"], dfp["rmse"])
        plt.xlabel("RMSE")
        plt.ylabel("Species group")
        plt.title(f"Ridge RT prediction RMSE by species group{label_suffix}")
        plt.tight_layout()
        plt.savefig(plot_path_group, dpi=150)
        plt.close()
        print(f"[ridge] Wrote RMSE-by-species-group plot to {plot_path_group}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
