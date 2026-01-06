#!/usr/bin/env python3
"""Evaluate legacy eslasso RT models as a baseline, aggregated by species_cluster.

It:
  - Loads per-supercategory eslasso regression weights from
    external_repos/sally/new_models/eslasso.
  - Streams an RT production CSV in chunks.
  - For each row with a matching lasso model, computes a point prediction and a window-based
    interval, then aggregates global and per-species-cluster metrics.
    The evaluated window matches the production baseline behavior (Hippopotamus/Sally):
      effective_half_width = window_multiplier * max(window, min_window)
  - Writes:
      results/rt_eval_lasso_by_species_cluster_<label>.csv
      results/rt_eval_lasso_<label>.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SUPPORT_BIN_ORDER = ["<= 1", "2-2", "3-5", "6-10", "11-20", "21-50", "51-100", "> 100"]

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
class LassoSupercategoryModels:
    features: List[str]
    comp_ids_sorted: np.ndarray  # (M,) int64
    coefs: np.ndarray  # (M, F) float32
    b: np.ndarray  # (M,) float32
    window: np.ndarray  # (M,) float32
    es_col_idx: np.ndarray  # (E,) int64 indices into features for ES_* columns


@dataclass
class AggStats:
    n: int = 0
    sum_sq_err: float = 0.0
    sum_abs_err: float = 0.0
    sum_covered: float = 0.0
    sum_interval_width: float = 0.0

    def update(
        self,
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
                "coverage_window": float("nan"),
                "window_width_mean": float("nan"),
            }
        rmse = float(np.sqrt(self.sum_sq_err / self.n))
        mae = float(self.sum_abs_err / self.n)
        cov = float(self.sum_covered / self.n)
        width_mean = float(self.sum_interval_width / self.n)
        return {
            "n_obs": int(self.n),
            "rmse": rmse,
            "mae": mae,
            "coverage_window": cov,
            "window_width_mean": width_mean,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate eslasso RT baseline by species_cluster on RT production CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory (writes results/*).",
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
        "--label",
        type=str,
        default=None,
        help="Optional label for this evaluation used in output filenames.",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=REPO_ROOT / "external_repos" / "sally" / "new_models" / "eslasso",
        help="Root directory containing per-supercategory eslasso regression CSVs.",
    )
    parser.add_argument(
        "--species-mapping-csv",
        type=Path,
        default=None,
        help="Optional species mapping CSV (columns: sample_set_id, species_group_raw).",
    )
    parser.add_argument(
        "--drop-es",
        action="store_true",
        help="Ignore ES_* covariates by zeroing them before prediction (what-if analysis).",
    )
    parser.add_argument(
        "--window-multiplier",
        type=float,
        default=4.0,
        help=(
            "Multiply the stored per-model window by this factor before computing coverage/width. "
            "Matches the production Sally default window_multiplier."
        ),
    )
    parser.add_argument(
        "--min-window",
        type=float,
        default=0.001,
        help=(
            "Clamp the stored per-model window to at least this value before applying "
            "--window-multiplier (Sally uses max(window, 0.001))."
        ),
    )
    parser.add_argument(
        "--support-map-csv",
        type=Path,
        default=None,
        help=(
            "Optional support-bin mapping CSV (e.g. PyMC by-group CSV for species_cluster) used to "
            "write rt_eval_lasso_by_support_<label>.csv."
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


def _discover_supercategory_dirs(models_root: Path) -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    if not models_root.is_dir():
        raise SystemExit(f"Models root not found: {models_root}")
    for d in models_root.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"supercategory_(\d+)_", d.name)
        if not m:
            continue
        mapping[int(m.group(1))] = d
    if not mapping:
        raise SystemExit(f"No supercategory_* directories found under {models_root}")
    return mapping


def _load_lasso_models(
    *, models_root: Path, lib_id: int
) -> tuple[Dict[int, LassoSupercategoryModels], List[str]]:
    super_dirs = _discover_supercategory_dirs(models_root)
    out: Dict[int, LassoSupercategoryModels] = {}
    all_features: set[str] = set()

    for super_num, d in super_dirs.items():
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

        es_col_idx = np.asarray(
            [i for i, name in enumerate(feature_cols_sorted) if name.startswith("ES_")],
            dtype=np.int64,
        )

        out[int(super_num)] = LassoSupercategoryModels(
            features=feature_cols_sorted,
            comp_ids_sorted=comp_ids,
            coefs=coefs,
            b=b,
            window=window,
            es_col_idx=es_col_idx,
        )
        all_features.update(feature_cols_sorted)

    if not out:
        raise SystemExit(f"No eslasso models found under {models_root} for lib {lib_id}")
    return out, sorted(all_features)


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


def _load_sampleset_to_supercategory(mapping_csv: Path) -> Dict[int, int]:
    if not mapping_csv.exists():
        raise SystemExit(f"Species mapping not found: {mapping_csv}")
    df = pd.read_csv(mapping_csv)
    if not {"sample_set_id", "species_group_raw"}.issubset(df.columns):
        raise SystemExit(
            f"Species mapping {mapping_csv} missing sample_set_id/species_group_raw columns"
        )
    out: Dict[int, int] = {}
    for _, row in df.iterrows():
        ssid = int(row["sample_set_id"])
        g_raw = row["species_group_raw"]
        if pd.isna(g_raw):
            continue
        m = re.match(r"\s*(\d+)", str(g_raw))
        if not m:
            continue
        out[ssid] = int(m.group(1))
    if not out:
        raise SystemExit(f"Species mapping {mapping_csv} did not yield any valid supercategories")
    return out


def main() -> None:
    args = parse_args()
    if float(args.window_multiplier) <= 0.0:
        raise SystemExit("--window-multiplier must be > 0")
    if float(args.min_window) < 0.0:
        raise SystemExit("--min-window must be >= 0")
    window_multiplier = float(args.window_multiplier)
    min_window = float(args.min_window)

    test_csv = args.test_csv
    if not test_csv.is_absolute():
        test_csv = (REPO_ROOT / test_csv).resolve()
    if not test_csv.exists():
        raise SystemExit(f"Test CSV not found: {test_csv}")

    output_dir = args.output_dir
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

    print(f"[lasso_cluster] Loading eslasso models from {args.models_root} for lib {args.lib_id}")
    models_by_super, union_features = _load_lasso_models(
        models_root=args.models_root, lib_id=int(args.lib_id)
    )
    print(
        f"[lasso_cluster] Loaded {len(models_by_super):,} supercategories; "
        f"union_features={len(union_features):,}"
    )

    mapping_csv = args.species_mapping_csv or _default_species_mapping_csv(int(args.lib_id))
    ssid_to_super = _load_sampleset_to_supercategory(mapping_csv)
    print(f"[lasso_cluster] Using species mapping: {mapping_csv} (n={len(ssid_to_super):,})")

    header = pd.read_csv(test_csv, nrows=0)
    required_cols = {"sampleset_id", "comp_id", "rt", "species_cluster"}
    missing_req = [c for c in sorted(required_cols) if c not in header.columns]
    if missing_req:
        raise SystemExit(f"Test CSV missing required columns: {missing_req}")
    missing_features = [c for c in union_features if c not in header.columns]
    if missing_features:
        raise SystemExit(
            f"RT CSV missing covariate columns required by lasso models (first 20): {missing_features[:20]}"
        )

    usecols = ["sampleset_id", "comp_id", "rt", "species_cluster", *union_features]
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
        print(
            f"[lasso_cluster] Loaded support-bin map: {support_map_csv_str} "
            f"(n_keys={len(support_keys):,})"
        )

    total_rows = 0
    evaluated_rows = 0
    skipped_no_super = 0
    skipped_no_model = 0

    print(
        f"[lasso_cluster] Streaming {test_csv} with chunk_size={args.chunk_size}, "
        f"max_test_rows={'ALL' if remaining is None else remaining}"
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

        super_num = chunk["sampleset_id"].map(ssid_to_super)
        ok_super = super_num.notna().to_numpy(dtype=bool, copy=False)
        skipped_no_super += int((~ok_super).sum())

        pred = np.full((len(chunk),), np.nan, dtype=np.float64)
        win = np.full((len(chunk),), np.nan, dtype=np.float64)
        modeled = np.zeros((len(chunk),), dtype=bool)

        if ok_super.any():
            super_num_arr = super_num.to_numpy(dtype=np.int64, copy=False)
            for s in np.unique(super_num_arr[ok_super]).tolist():
                super_id = int(s)
                sm = models_by_super.get(super_id)
                idx_rows = np.flatnonzero(ok_super & (super_num_arr == super_id))
                if idx_rows.size == 0:
                    continue
                if sm is None:
                    skipped_no_model += int(idx_rows.size)
                    continue
                comp_rows = comp_id[idx_rows]

                idx_m = np.searchsorted(sm.comp_ids_sorted, comp_rows)
                ok = (idx_m >= 0) & (idx_m < sm.comp_ids_sorted.size)
                if ok.any():
                    ok[ok] &= sm.comp_ids_sorted[idx_m[ok]] == comp_rows[ok]
                if not ok.any():
                    skipped_no_model += int(idx_rows.size)
                    continue

                idx_ok = idx_rows[ok]
                idx_m_ok = idx_m[ok].astype(np.int64, copy=False)
                skipped_no_model += int(idx_rows.size - idx_ok.size)

                x_all = chunk[sm.features].to_numpy(dtype=np.float32, copy=False)
                x = x_all[idx_ok]
                np.nan_to_num(x, copy=False)
                if args.drop_es and sm.es_col_idx.size > 0:
                    x[:, sm.es_col_idx] = 0.0

                b = sm.b[idx_m_ok].astype(np.float64, copy=False)
                w = sm.window[idx_m_ok].astype(np.float64, copy=False)
                coefs = sm.coefs[idx_m_ok].astype(np.float64, copy=False)
                pred_vals = b + np.sum(coefs * x.astype(np.float64, copy=False), axis=1)

                pred[idx_ok] = pred_vals
                win[idx_ok] = w
                modeled[idx_ok] = True

        if modeled.any():
            err = pred[modeled] - y_true[modeled]
            sq_err = np.square(err)
            abs_err = np.abs(err)
            half_width = window_multiplier * np.maximum(win[modeled], min_window)
            covered = abs_err <= half_width
            interval_width = 2.0 * half_width
            global_stats.update(sq_err, abs_err, covered, interval_width)
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
                            sq_err_b[mask],
                            abs_err_b[mask],
                            covered_b[mask],
                            interval_width_b[mask],
                        )

            cl = species_cluster[modeled]
            for c in np.unique(cl).tolist():
                c_int = int(c)
                mask = cl == c_int
                stats = cluster_stats.setdefault(c_int, AggStats())
                stats.update(sq_err[mask], abs_err[mask], covered[mask], interval_width[mask])

        if remaining is not None:
            remaining -= int(len(chunk))

    global_metrics = global_stats.to_metrics()
    print(
        f"[lasso_cluster] Total rows seen={total_rows:,}, evaluated={evaluated_rows:,}, "
        f"skipped_no_super={skipped_no_super:,}, skipped_no_model={skipped_no_model:,}"
    )
    print(
        f"[lasso_cluster] Global: n={global_metrics['n_obs']:,}, RMSE={global_metrics['rmse']:.3f}, "
        f"MAE={global_metrics['mae']:.3f}, coverage_window={global_metrics['coverage_window']:.3f}"
    )

    out_json = results_dir / f"rt_eval_lasso{label_suffix}.json"
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
                "drop_es": bool(args.drop_es),
                "window_multiplier": float(window_multiplier),
                "min_window": float(min_window),
                "skipped_no_super": int(skipped_no_super),
                "skipped_no_model": int(skipped_no_model),
                "support_map_csv": support_map_csv_str,
                "skipped_missing_support_bin": int(skipped_missing_support_bin),
            },
            indent=2,
        )
    )

    rows_out = []
    for cluster_id in sorted(cluster_stats.keys()):
        m = cluster_stats[cluster_id].to_metrics()
        rows_out.append({"species_cluster": int(cluster_id), **m})
    cluster_df = pd.DataFrame(rows_out)
    out_csv = results_dir / f"rt_eval_lasso_by_species_cluster{label_suffix}.csv"
    cluster_df.to_csv(out_csv, index=False)
    print(f"[lasso_cluster] Wrote per-cluster metrics to {out_csv}")

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
                    "coverage_window": float(m["coverage_window"]),
                    "window_width_mean": float(m["window_width_mean"]),
                }
            )
        support_df = pd.DataFrame(rows_support)
        out_support_csv = results_dir / f"rt_eval_lasso_by_support{label_suffix}.csv"
        support_df.to_csv(out_support_csv, index=False)
        print(
            f"[lasso_cluster] Wrote support-bin metrics to {out_support_csv} "
            f"(skipped_missing_support_bin={skipped_missing_support_bin:,})"
        )


if __name__ == "__main__":
    main()
