#!/usr/bin/env python3
"""
Evaluate local per-species-matrix eslasso RT models (Sally local_models) on production RT CSVs.

This is analogous to `eval_rt_lasso_baseline_by_species_cluster.py`, but uses the "local_models"
bundle in `external_repos/sally/local_models/eslasso`, where models are keyed by a small set of
species-matrix types (e.g. human_plasma, human_urine).

Important: these local models do NOT cover all rows in the production CSV. We therefore:
  - map each test row's sampleset_id -> species_raw using repo_export species mapping,
  - infer a species-matrix key from species_raw (heuristics; matches Sally's model bundle names),
  - score only rows whose inferred matrix has a corresponding model bundle and comp_id entry.

Outputs (under --output-dir/results):
  - rt_eval_lasso_local_<label>.json
  - rt_eval_lasso_local_by_species_cluster_<label>.csv
  - rt_eval_lasso_local_by_matrix_key_<label>.csv
  - rt_eval_lasso_local_by_support_<label>.csv (when --support-map-csv is provided)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SUPPORT_BIN_ORDER = ["<= 1", "2-2", "3-5", "6-10", "11-20", "21-50", "51-100", "> 100"]

LOGGER = logging.getLogger(__name__)

RESERVED = {
    "comp_id",
    "b",
    "buffer_around_center",
    "cloud_center_range",
    "test_ws",
    "window",
    "lib_id",
}


@dataclass(frozen=True)
class LassoModels:
    features: List[str]
    comp_ids_sorted: np.ndarray  # (M,) int64
    coefs: np.ndarray  # (M, F) float32
    b: np.ndarray  # (M,) float32
    window: np.ndarray  # (M,) float32 (half-width)


@dataclass
class AggStats:
    n: int = 0
    sum_sq_err: float = 0.0
    sum_abs_err: float = 0.0
    sum_covered: float = 0.0
    sum_interval_width: float = 0.0

    def update(
        self,
        *,
        sq_err: np.ndarray,
        abs_err: np.ndarray,
        covered: np.ndarray,
        interval_width: np.ndarray,
    ) -> None:
        if sq_err.size == 0:
            return
        self.n += int(sq_err.size)
        self.sum_sq_err += float(np.sum(sq_err))
        self.sum_abs_err += float(np.sum(abs_err))
        self.sum_covered += float(np.sum(covered))
        self.sum_interval_width += float(np.sum(interval_width))

    def to_metrics(self) -> Dict[str, float]:
        if self.n == 0:
            return {
                "n_obs": 0,
                "rmse": float("nan"),
                "mae": float("nan"),
                "coverage_95": float("nan"),
                "interval_width_mean": float("nan"),
            }
        rmse = float(np.sqrt(self.sum_sq_err / self.n))
        mae = float(self.sum_abs_err / self.n)
        cov = float(self.sum_covered / self.n)
        width_mean = float(self.sum_interval_width / self.n)
        return {
            "n_obs": int(self.n),
            "rmse": rmse,
            "mae": mae,
            "coverage_95": cov,
            "interval_width_mean": width_mean,
        }


@dataclass
class MatrixRowCounts:
    n_rows_mapped: int = 0
    n_rows_evaluated: int = 0
    n_rows_skipped_no_bundle: int = 0
    n_rows_skipped_unsupported: int = 0
    n_rows_skipped_comp_missing: int = 0

    def to_row(
        self, *, matrix_key: str, model_available: bool, model_supported: bool
    ) -> Dict[str, object]:
        return {
            "matrix_key": str(matrix_key),
            "model_available": bool(model_available),
            "model_supported": bool(model_supported),
            "n_rows_mapped": int(self.n_rows_mapped),
            "n_rows_evaluated": int(self.n_rows_evaluated),
            "n_rows_skipped_no_bundle": int(self.n_rows_skipped_no_bundle),
            "n_rows_skipped_unsupported": int(self.n_rows_skipped_unsupported),
            "n_rows_skipped_comp_missing": int(self.n_rows_skipped_comp_missing),
        }


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local_models eslasso RT models by species_cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory (writes results/*).",
    )
    parser.add_argument("--lib-id", type=int, required=True, help="Library id (e.g. 208, 209).")
    parser.add_argument(
        "--test-csv", type=Path, required=True, help="RT production CSV to evaluate."
    )
    parser.add_argument("--chunk-size", type=int, default=50_000, help="Rows per chunk.")
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
        help="Optional label for output filenames (e.g. realtest).",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=REPO_ROOT / "external_repos" / "sally" / "local_models" / "eslasso",
        help="Root directory containing per-species-matrix eslasso regression CSVs.",
    )
    parser.add_argument(
        "--species-mapping-csv",
        type=Path,
        default=None,
        help="Optional species mapping CSV with columns (sample_set_id, species_raw).",
    )
    parser.add_argument(
        "--support-map-csv",
        type=Path,
        default=None,
        help=(
            "Optional support-bin mapping CSV (e.g. PyMC by-group CSV for species_cluster) used to "
            "write rt_eval_lasso_local_by_support_<label>.csv."
        ),
    )
    return parser.parse_args()


def _load_support_bin_map(
    support_map_csv: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a mapping from (species_cluster, comp_id) -> support_bin from a CSV.

    Expected to accept either:
      - a "by-group" CSV containing {group_key, support_bin}, where group_key encodes
        (species_cluster << 32) + comp_id, or
      - a CSV containing {species_cluster, comp_id, support_bin}.
    """
    if not support_map_csv.exists():
        raise SystemExit(f"Support map CSV not found: {support_map_csv}")
    df = pd.read_csv(support_map_csv)
    if "support_bin" not in df.columns:
        raise SystemExit(f"Support map CSV missing support_bin column: {support_map_csv}")

    if "group_key" in df.columns:
        keys = df["group_key"].to_numpy(dtype=np.int64, copy=False)
    elif {"species_cluster", "comp_id"}.issubset(df.columns):
        species_cluster = df["species_cluster"].to_numpy(dtype=np.int64, copy=False)
        comp_id = df["comp_id"].to_numpy(dtype=np.int64, copy=False)
        keys = (species_cluster.astype(np.int64) << 32) + comp_id.astype(np.int64)
    else:
        raise SystemExit(
            f"Support map CSV must contain either group_key or (species_cluster, comp_id): {support_map_csv}"
        )

    bin_name = df["support_bin"].astype(str).to_numpy(copy=False)
    name_to_idx = {name: i for i, name in enumerate(DEFAULT_SUPPORT_BIN_ORDER)}
    bin_idx = np.asarray([name_to_idx.get(str(x), -1) for x in bin_name], dtype=np.int16)

    keep = bin_idx >= 0
    keys = keys[keep]
    bin_idx = bin_idx[keep]

    order = np.argsort(keys)
    keys_sorted = keys[order]
    bin_idx_sorted = bin_idx[order]

    n_groups_by_bin = np.zeros((len(DEFAULT_SUPPORT_BIN_ORDER),), dtype=np.int64)
    for i in range(len(DEFAULT_SUPPORT_BIN_ORDER)):
        n_groups_by_bin[i] = int(np.sum(bin_idx_sorted == i))

    return keys_sorted, bin_idx_sorted, n_groups_by_bin


def _default_species_mapping_csv(lib_id: int) -> Path:
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


def _discover_local_model_dirs(models_root: Path) -> Dict[str, Path]:
    if not models_root.is_dir():
        raise SystemExit(f"Models root not found: {models_root}")
    out: Dict[str, Path] = {}
    for d in sorted(models_root.iterdir()):
        if d.is_dir():
            out[d.name] = d
    if not out:
        raise SystemExit(f"No model directories found under {models_root}")
    return out


def _load_local_models(
    *, models_root: Path, lib_id: int
) -> tuple[Dict[str, LassoModels], List[str]]:
    model_dirs = _discover_local_model_dirs(models_root)
    out: Dict[str, LassoModels] = {}
    union_features: set[str] = set()

    for name, d in model_dirs.items():
        path = d / f"regression_{lib_id}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not {"comp_id", "b", "window"}.issubset(df.columns):
            continue

        feature_cols = [
            c for c in df.columns if c not in RESERVED and pd.api.types.is_numeric_dtype(df[c])
        ]
        feature_cols_sorted = sorted(str(c) for c in feature_cols)
        if not feature_cols_sorted:
            continue

        keep_cols = ["comp_id", "b", "window", *feature_cols_sorted]
        df = df[keep_cols].copy()
        df = df.dropna(subset=["comp_id", "b", "window"])
        df["comp_id"] = df["comp_id"].astype(int)
        df = df.sort_values("comp_id")

        comp_ids = df["comp_id"].to_numpy(dtype=np.int64, copy=False)
        b = df["b"].to_numpy(dtype=np.float32, copy=False)
        window = df["window"].to_numpy(dtype=np.float32, copy=False)
        coefs = df[feature_cols_sorted].to_numpy(dtype=np.float32, copy=False)

        out[name] = LassoModels(
            features=feature_cols_sorted,
            comp_ids_sorted=comp_ids,
            coefs=coefs,
            b=b,
            window=window,
        )
        union_features.update(feature_cols_sorted)

    if not out:
        raise SystemExit(
            f"No local_models eslasso regression_{lib_id}.csv files found under {models_root}"
        )
    return out, sorted(union_features)


def _load_sampleset_to_species_raw(mapping_csv: Path) -> Dict[int, str]:
    if not mapping_csv.exists():
        raise SystemExit(f"Species mapping not found: {mapping_csv}")
    df = pd.read_csv(mapping_csv)
    if not {"sample_set_id", "species_raw"}.issubset(df.columns):
        raise SystemExit(f"Species mapping {mapping_csv} missing sample_set_id/species_raw columns")
    out: Dict[int, str] = {}
    for _, row in df.iterrows():
        ssid = int(row["sample_set_id"])
        raw = row["species_raw"]
        if pd.isna(raw):
            continue
        out[ssid] = str(raw)
    if not out:
        raise SystemExit(f"Species mapping {mapping_csv} did not yield any species_raw values")
    return out


_PLASMA_TYPES = {"PLASMA", "SERUM", "EDTA PLASMA", "EDTA-PLASMA"}


def _infer_species_matrix_from_species_raw(species_raw: str) -> Optional[str]:
    """
    Heuristic mapping from `species_raw` (e.g. 'PLASMA+...+HUMAN+HOMO SAPIENS') to
    Sally local-model bundle names (human_plasma, rat_plasma, human_urine, ...).
    """
    s = str(species_raw).strip()
    if not s:
        return None
    # Some exports already use the local-model keys directly (e.g. 'human_plasma').
    normalized = s.lower()
    if normalized in {
        "human_plasma",
        "human_urine",
        "human_fecal",
        "human_cells",
        "rat_plasma",
        "rat_liver",
    }:
        return normalized
    parts = [p.strip() for p in s.split("+")]
    p0 = parts[0].upper() if parts else ""
    p1 = parts[1].upper() if len(parts) > 1 else ""
    upper = s.upper()

    is_human = "HUMAN" in upper or "HOMO SAPIENS" in upper
    is_rat = "RAT" in upper
    is_mouse = "MOUSE" in upper
    is_rodent = is_rat or is_mouse

    if p0 in _PLASMA_TYPES:
        if is_human:
            return "human_plasma"
        if is_rodent:
            return "rat_plasma"
        return None

    if "URINE" in p0:
        if is_human:
            return "human_urine"
        return None

    if p0 in {"FECES", "FECAL"}:
        if is_human:
            return "human_fecal"
        return None

    if "CELL" in p0:
        if is_human:
            return "human_cells"
        return None

    if "TISSUE" in p0 and "LIVER" in p1:
        if is_rodent:
            return "rat_liver"
        return None

    # Fallback: sometimes LIVER appears elsewhere.
    if "LIVER" in upper and is_rodent:
        return "rat_liver"

    return None


def _build_sampleset_to_species_matrix(ssid_to_species_raw: Dict[int, str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for ssid, raw in ssid_to_species_raw.items():
        key = _infer_species_matrix_from_species_raw(raw)
        if key is not None:
            out[int(ssid)] = str(key)
    return out


def main() -> None:
    _setup_logging()
    args = parse_args()

    test_csv = args.test_csv
    if not test_csv.is_absolute():
        test_csv = (REPO_ROOT / test_csv).resolve()
    if not test_csv.exists():
        raise SystemExit(f"Test CSV not found: {test_csv}")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    label = args.label
    if label is None:
        name = test_csv.name
        if "realtest" in name:
            label = "realtest"
        elif "_cap10_" in name:
            label = "cap10"
        else:
            label = "dataset"
    label_suffix = f"_{label}" if label else ""

    LOGGER.info("Loading local_models eslasso from %s (lib=%s)", args.models_root, args.lib_id)
    models_by_matrix, union_features = _load_local_models(
        models_root=Path(args.models_root), lib_id=int(args.lib_id)
    )
    LOGGER.info(
        "Loaded %s species-matrix models; union_features=%s",
        len(models_by_matrix),
        len(union_features),
    )

    mapping_csv = args.species_mapping_csv or _default_species_mapping_csv(int(args.lib_id))
    ssid_to_species_raw = _load_sampleset_to_species_raw(mapping_csv)
    ssid_to_matrix = _build_sampleset_to_species_matrix(ssid_to_species_raw)
    LOGGER.info(
        "Using species mapping: %s (n=%s, matrix_mapped=%s)",
        mapping_csv,
        len(ssid_to_species_raw),
        len(ssid_to_matrix),
    )

    header = pd.read_csv(test_csv, nrows=0)
    required_cols = {"sampleset_id", "comp_id", "rt", "species_cluster"}
    missing_req = [c for c in sorted(required_cols) if c not in header.columns]
    if missing_req:
        raise SystemExit(f"Test CSV missing required columns: {missing_req}")
    missing_features = [c for c in union_features if c not in header.columns]
    if missing_features:
        LOGGER.warning(
            "RT CSV missing %s covariate columns referenced by local lasso models; affected matrices will be skipped (first 20): %s",
            len(missing_features),
            missing_features[:20],
        )

    available_features = [c for c in union_features if c in header.columns]
    usecols = ["sampleset_id", "comp_id", "rt", "species_cluster", *available_features]
    remaining = int(args.max_test_rows) if args.max_test_rows and args.max_test_rows > 0 else None

    global_stats = AggStats()
    cluster_stats: Dict[int, AggStats] = {}

    support_keys: np.ndarray | None = None
    support_bin_idx: np.ndarray | None = None
    n_groups_by_bin = np.zeros((len(DEFAULT_SUPPORT_BIN_ORDER),), dtype=np.int64)
    support_stats: list[AggStats] = []
    skipped_missing_support_bin = 0
    support_map_csv_str: str | None = None

    if args.support_map_csv is not None:
        support_map_csv = args.support_map_csv
        if not support_map_csv.is_absolute():
            support_map_csv = (REPO_ROOT / support_map_csv).resolve()
        support_keys, support_bin_idx, n_groups_by_bin = _load_support_bin_map(support_map_csv)
        support_stats = [AggStats() for _ in DEFAULT_SUPPORT_BIN_ORDER]
        support_map_csv_str = str(support_map_csv)
        LOGGER.info(
            "Loaded support-bin map: %s (n_keys=%s)", support_map_csv_str, f"{len(support_keys):,}"
        )

    total_rows = 0
    evaluated_rows = 0
    skipped_no_matrix = 0
    skipped_no_model = 0

    matrix_counts: Dict[str, MatrixRowCounts] = {}

    model_present_cols: Dict[str, List[str]] = {}
    model_present_idx: Dict[str, np.ndarray] = {}
    model_supported: Dict[str, bool] = {}
    header_cols = set(map(str, header.columns))
    for matrix_name, model in models_by_matrix.items():
        missing_model_features = [f for f in model.features if f not in header_cols]
        model_supported[str(matrix_name)] = not missing_model_features
        if missing_model_features:
            LOGGER.warning(
                "Skipping matrix=%s for this CSV because %s required covariates are missing (first 20): %s",
                matrix_name,
                len(missing_model_features),
                missing_model_features[:20],
            )

        present_cols = [f for f in model.features if f in header_cols]
        present_idx = np.asarray(
            [i for i, f in enumerate(model.features) if f in header_cols], dtype=np.int64
        )
        model_present_cols[str(matrix_name)] = present_cols
        model_present_idx[str(matrix_name)] = present_idx

    LOGGER.info(
        "Streaming %s with chunk_size=%s, max_test_rows=%s",
        test_csv,
        args.chunk_size,
        "ALL" if remaining is None else remaining,
    )

    for chunk in pd.read_csv(test_csv, chunksize=int(args.chunk_size), usecols=usecols):
        if remaining is not None and remaining <= 0:
            break
        total_rows += int(len(chunk))
        if remaining is not None and len(chunk) > remaining:
            chunk = chunk.iloc[:remaining].copy()

        comp_id = chunk["comp_id"].to_numpy(dtype=np.int64, copy=False)
        y_true = chunk["rt"].to_numpy(dtype=np.float64, copy=False)
        species_cluster = chunk["species_cluster"].to_numpy(dtype=np.int64, copy=False)

        matrix_series = chunk["sampleset_id"].map(ssid_to_matrix)
        ok_matrix = matrix_series.notna().to_numpy(dtype=bool, copy=False)
        skipped_no_matrix += int((~ok_matrix).sum())

        # We'll fill in-place predictions and windows for rows we can score.
        pred = np.full((len(chunk),), np.nan, dtype=np.float64)
        win = np.full((len(chunk),), np.nan, dtype=np.float64)
        modeled = np.zeros((len(chunk),), dtype=bool)

        if ok_matrix.any():
            matrix_arr = matrix_series.to_numpy(dtype=object, copy=False)
            for matrix_name in np.unique(matrix_arr[ok_matrix]).tolist():
                m_name = str(matrix_name)
                counts = matrix_counts.setdefault(m_name, MatrixRowCounts())
                model = models_by_matrix.get(m_name)
                idx_rows = np.flatnonzero(ok_matrix & (matrix_arr == matrix_name))
                if idx_rows.size == 0:
                    continue
                counts.n_rows_mapped += int(idx_rows.size)
                if model is None:
                    skipped_no_model += int(idx_rows.size)
                    counts.n_rows_skipped_no_bundle += int(idx_rows.size)
                    continue
                if not model_supported.get(m_name, True):
                    skipped_no_model += int(idx_rows.size)
                    counts.n_rows_skipped_unsupported += int(idx_rows.size)
                    continue

                comp_rows = comp_id[idx_rows]
                idx_m = np.searchsorted(model.comp_ids_sorted, comp_rows)
                ok = (idx_m >= 0) & (idx_m < model.comp_ids_sorted.size)
                if ok.any():
                    ok[ok] &= model.comp_ids_sorted[idx_m[ok]] == comp_rows[ok]
                if not ok.any():
                    skipped_no_model += int(idx_rows.size)
                    counts.n_rows_skipped_comp_missing += int(idx_rows.size)
                    continue

                idx_ok = idx_rows[ok]
                idx_m_ok = idx_m[ok].astype(np.int64, copy=False)
                missing_comp = int(idx_rows.size - idx_ok.size)
                skipped_no_model += missing_comp
                counts.n_rows_skipped_comp_missing += missing_comp

                b = model.b[idx_m_ok].astype(np.float64, copy=False)
                w = model.window[idx_m_ok].astype(np.float64, copy=False)
                coefs = model.coefs[idx_m_ok].astype(np.float64, copy=False)

                feat_cols = model_present_cols.get(m_name, [])
                feat_idx = model_present_idx.get(m_name, np.asarray([], dtype=np.int64))
                x_all = chunk[feat_cols].to_numpy(dtype=np.float32, copy=False)
                x = x_all[idx_ok]
                np.nan_to_num(x, copy=False)
                pred_vals = b + np.sum(
                    coefs[:, feat_idx] * x.astype(np.float64, copy=False), axis=1
                )

                pred[idx_ok] = pred_vals
                win[idx_ok] = w
                modeled[idx_ok] = True
                counts.n_rows_evaluated += int(idx_ok.size)

        if modeled.any():
            err = pred[modeled] - y_true[modeled]
            sq_err = np.square(err)
            abs_err = np.abs(err)
            covered = abs_err <= win[modeled]
            interval_width = 2.0 * win[modeled]
            global_stats.update(
                sq_err=sq_err,
                abs_err=abs_err,
                covered=covered,
                interval_width=interval_width,
            )
            evaluated_rows += int(err.size)

            if support_keys is not None and support_bin_idx is not None:
                group_key = (species_cluster[modeled].astype(np.int64) << 32) + comp_id[
                    modeled
                ].astype(np.int64)
                idx = np.searchsorted(support_keys, group_key)
                ok_support = (idx >= 0) & (idx < support_keys.size)
                if ok_support.any():
                    ok_support[ok_support] &= support_keys[idx[ok_support]] == group_key[ok_support]
                skipped_missing_support_bin += int((~ok_support).sum())
                if ok_support.any():
                    bins = support_bin_idx[idx[ok_support]]
                    sq_err_b = sq_err[ok_support]
                    abs_err_b = abs_err[ok_support]
                    covered_b = covered[ok_support]
                    interval_width_b = interval_width[ok_support]
                    for b in np.unique(bins).tolist():
                        b_int = int(b)
                        mask = bins == b_int
                        support_stats[b_int].update(
                            sq_err=sq_err_b[mask],
                            abs_err=abs_err_b[mask],
                            covered=covered_b[mask],
                            interval_width=interval_width_b[mask],
                        )

            cl = species_cluster[modeled]
            for c in np.unique(cl).tolist():
                c_int = int(c)
                mask = cl == c_int
                stats = cluster_stats.setdefault(c_int, AggStats())
                stats.update(
                    sq_err=sq_err[mask],
                    abs_err=abs_err[mask],
                    covered=covered[mask],
                    interval_width=interval_width[mask],
                )

        if remaining is not None:
            remaining -= int(len(chunk))

    global_metrics = global_stats.to_metrics()
    LOGGER.info(
        "Total rows seen=%s, evaluated=%s, skipped_no_matrix=%s, skipped_no_model=%s",
        f"{total_rows:,}",
        f"{evaluated_rows:,}",
        f"{skipped_no_matrix:,}",
        f"{skipped_no_model:,}",
    )
    LOGGER.info(
        "Global: n=%s, RMSE=%.6f, MAE=%.6f, coverage=%.3f, width_mean=%.5f",
        f"{global_metrics['n_obs']:,}",
        global_metrics["rmse"],
        global_metrics["mae"],
        global_metrics["coverage_95"],
        global_metrics["interval_width_mean"],
    )

    out_json = results_dir / f"rt_eval_lasso_local{label_suffix}.json"
    out_matrix_csv = results_dir / f"rt_eval_lasso_local_by_matrix_key{label_suffix}.csv"
    out_json.write_text(
        json.dumps(
            {
                "metrics": global_metrics,
                "n_test_rows_seen": int(total_rows),
                "n_rows_evaluated": int(evaluated_rows),
                "chunk_size": int(args.chunk_size),
                "lib_id": int(args.lib_id),
                "label": str(label),
                "models_root": str(Path(args.models_root)),
                "species_mapping_csv": str(mapping_csv),
                "skipped_no_matrix": int(skipped_no_matrix),
                "skipped_no_model": int(skipped_no_model),
                "support_map_csv": support_map_csv_str,
                "skipped_missing_support_bin": int(skipped_missing_support_bin),
                "matrix_keys_loaded": sorted(models_by_matrix.keys()),
                "matrix_row_counts_csv": str(out_matrix_csv),
            },
            indent=2,
        )
    )

    matrix_rows = []
    all_keys = sorted(set(matrix_counts.keys()) | set(models_by_matrix.keys()))
    for key in all_keys:
        counts = matrix_counts.get(key, MatrixRowCounts())
        available = key in models_by_matrix
        supported = bool(model_supported.get(key, False)) if available else False
        matrix_rows.append(
            counts.to_row(matrix_key=key, model_available=available, model_supported=supported)
        )
    pd.DataFrame(matrix_rows).to_csv(out_matrix_csv, index=False)
    LOGGER.info("Wrote per-matrix row counts to %s", out_matrix_csv)

    rows_out = []
    for cluster_id in sorted(cluster_stats.keys()):
        m = cluster_stats[cluster_id].to_metrics()
        rows_out.append({"species_cluster": int(cluster_id), **m})
    cluster_df = pd.DataFrame(rows_out)
    out_csv = results_dir / f"rt_eval_lasso_local_by_species_cluster{label_suffix}.csv"
    cluster_df.to_csv(out_csv, index=False)
    LOGGER.info("Wrote per-cluster metrics to %s", out_csv)

    if support_keys is not None and support_bin_idx is not None:
        rows_support = []
        for i, name in enumerate(DEFAULT_SUPPORT_BIN_ORDER):
            m = support_stats[i].to_metrics()
            rows_support.append(
                {
                    "support_bin": str(name),
                    "n_groups": int(n_groups_by_bin[i]),
                    "n_obs_test": int(m["n_obs"]),
                    "rmse": float(m["rmse"]),
                    "mae": float(m["mae"]),
                    "coverage_95": float(m["coverage_95"]),
                    "interval_width_mean": float(m["interval_width_mean"]),
                }
            )
        support_df = pd.DataFrame(rows_support)
        out_support_csv = results_dir / f"rt_eval_lasso_local_by_support{label_suffix}.csv"
        support_df.to_csv(out_support_csv, index=False)
        LOGGER.info(
            "Wrote support-bin metrics to %s (skipped_missing_support_bin=%s)",
            out_support_csv,
            f"{skipped_missing_support_bin:,}",
        )


if __name__ == "__main__":
    main()
