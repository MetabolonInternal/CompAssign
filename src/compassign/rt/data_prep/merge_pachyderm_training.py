#!/usr/bin/env python3
"""
Merge Pachyderm training inputs (create_training_data + combine_predictors) into
partitioned Parquet shards and a single consolidated Parquet file.

This script is standalone and does NOT call pachctl. Point it at an export
directory that already contains:
  <input_dir>/create_training_data/...
  <input_dir>/combine_predictors/...

CLI:
  python -m compassign.rt.data_prep.merge_pachyderm_training --input-dir <export-root>

Outputs:
  <output_dir>/merged_training/species_matrix_type=.../lib_id=.../*.parquet
  <output_dir>/merged_training/merged_training_all.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional nicety

    def tqdm(it: Iterable, **kwargs):
        return it


def list_parts(base: Path) -> Iterable[Tuple[str, Path]]:
    for lib_dir in base.glob("lib_id=*"):
        lib_id = lib_dir.name.split("=", 1)[-1]
        yield lib_id, lib_dir


def load_train_group(group_dir: Path) -> pd.DataFrame:
    parts = list(group_dir.glob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    dfs: list[pd.DataFrame] = []
    for part in parts:
        try:
            dfs.append(pd.read_parquet(part))
        except Exception as exc:  # pragma: no cover - defensive path
            print(
                f"[merge_pachyderm_training] Skipping corrupt file {part}: {exc}", file=sys.stderr
            )
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_predictors(lib_id: str, predictor_root: Path) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for sm_dir in predictor_root.glob(f"lib_id={lib_id}/species_matrix_type=*"):
        sm = sm_dir.name.split("=", 1)[-1]
        parts = list(sm_dir.glob("*.parquet"))
        if not parts:
            continue
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df["lib_id"] = lib_id
        df["species_matrix_type"] = sm
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def build_cases(create_root: Path, predictor_root: Path) -> List[Tuple[str, Path, pd.DataFrame]]:
    predictor_cache: dict[str, pd.DataFrame] = {}
    cases: List[Tuple[str, Path, pd.DataFrame]] = []
    for lib_id, lib_dir in list_parts(create_root):
        preds = predictor_cache.get(lib_id)
        if preds is None:
            preds = load_predictors(lib_id, predictor_root)
            predictor_cache[lib_id] = preds
        if preds.empty:
            continue
        for group_dir in lib_dir.glob("group=*"):
            cases.append((lib_id, group_dir, preds))
    return cases


def _add_in_order(target: List[str], seen: set[str], cols: Iterable[str]) -> None:
    for col in cols:
        if col not in seen:
            target.append(col)
            seen.add(col)


def scan_master_columns(cases: List[Tuple[str, Path, pd.DataFrame]]) -> List[str]:
    # Preserve column order: metadata/non-predictors on the left, predictor columns on the right.
    meta_order: List[str] = []
    pred_order: List[str] = []
    meta_seen: set[str] = set()
    pred_seen: set[str] = set()

    for lib_id, group_dir, preds in tqdm(cases, desc="Scanning groups", unit="group"):
        train_df = load_train_group(group_dir)
        if train_df.empty:
            continue
        if "cpd_status" in train_df.columns:
            train_df = train_df[train_df["cpd_status"].isin(["Complete", "Complete - NFS"])]
        if "sample_type" in train_df.columns:
            train_df = train_df[~train_df["sample_type"].eq("PRCS")]
        if train_df.empty:
            continue
        if "correct" in train_df.columns:
            train_df = train_df.rename(columns={"correct": "comp_id"})
        if "worksheet_id" in train_df.columns and "batch" not in train_df.columns:
            train_df = train_df.rename(columns={"worksheet_id": "batch"})
        required = ["sample_set_id", "task_id"]
        if any(col not in train_df.columns for col in required):
            continue
        merged_cols = train_df.merge(
            preds,
            on=["sample_set_id", "task_id", "species_matrix_type"],
            how="inner",
            suffixes=("", "_pred"),
        )
        if merged_cols.empty:
            continue
        if "fit_scores" in merged_cols.columns:
            merged_cols = merged_cols.drop(columns=["fit_scores"])

        meta_cols = [c for c in merged_cols.columns if not c.startswith(("IS", "ES", "RS"))]
        pred_cols = [c for c in merged_cols.columns if c.startswith(("IS", "ES", "RS"))]

        _add_in_order(meta_order, meta_seen, meta_cols)
        _add_in_order(pred_order, pred_seen, pred_cols)

    # Ensure derived columns are present and placed with metadata.
    for col in ("lib_id", "cpd_group"):
        if col not in meta_seen:
            meta_order.append(col)
            meta_seen.add(col)

    return meta_order + pred_order


def _promote_null_fields(schema: pa.Schema) -> pa.Schema:
    """Replace null-typed fields with a concrete type so later casts do not fail."""
    fields: list[pa.Field] = []
    for field in schema:
        if pa.types.is_null(field.type):
            # Predictor columns should stay numeric; fall back to string for others.
            f_type = pa.float64() if field.name.startswith(("IS", "ES", "RS")) else pa.string()
            fields.append(pa.field(field.name, f_type))
        else:
            fields.append(field)
    return pa.schema(fields)


def merge_and_write(
    cases: List[Tuple[str, Path, pd.DataFrame]],
    master_cols: List[str],
    out_root: Path,
    chunk_size: int = 1_000_000,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    all_path = out_root / "merged_training_all.parquet"
    writer: pq.ParquetWriter | None = None

    for lib_id, group_dir, preds in tqdm(cases, desc="Merging groups", unit="group"):
        group_name = group_dir.name.split("=", 1)[-1]
        train_df = load_train_group(group_dir)
        if train_df.empty:
            continue

        if "cpd_status" in train_df.columns:
            train_df = train_df[train_df["cpd_status"].isin(["Complete", "Complete - NFS"])]
        if "sample_type" in train_df.columns:
            train_df = train_df[~train_df["sample_type"].eq("PRCS")]
        if train_df.empty:
            continue

        if "correct" in train_df.columns:
            train_df = train_df.rename(columns={"correct": "comp_id"})
        if "worksheet_id" in train_df.columns and "batch" not in train_df.columns:
            train_df = train_df.rename(columns={"worksheet_id": "batch"})

        required = ["sample_set_id", "task_id"]
        if any(col not in train_df.columns for col in required):
            continue

        merged = train_df.merge(
            preds,
            on=["sample_set_id", "task_id", "species_matrix_type"],
            how="inner",
            suffixes=("", "_pred"),
        )
        if merged.empty:
            continue

        if "fit_scores" in merged.columns:
            merged = merged.drop(columns=["fit_scores"])

        merged["lib_id"] = int(lib_id)
        merged["cpd_group"] = group_name

        if "batch" not in merged.columns and "batch_pred" in merged.columns:
            merged = merged.rename(columns={"batch_pred": "batch"})

        # Normalize dtypes so columns with all missing values do not become Arrow null types.
        merged = merged.convert_dtypes()
        for col in merged.columns:
            if pd.api.types.is_integer_dtype(merged[col].dtype):
                merged[col] = merged[col].astype("Float64")

        drop_cols_pred = {c for c in merged.columns if c.endswith("_pred")}
        merged = merged.drop(columns=drop_cols_pred, errors="ignore")

        missing = set(master_cols) - set(merged.columns)
        for col in missing:
            merged[col] = pd.NA
        merged = merged.reindex(columns=master_cols)

        sm_types = merged["species_matrix_type"].unique()
        for sm in sm_types:
            shard = merged[merged["species_matrix_type"] == sm]
            out_dir = out_root / f"species_matrix_type={sm}" / f"lib_id={lib_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(0, len(shard), chunk_size):
                chunk = shard.iloc[i : i + chunk_size]
                chunk.to_parquet(
                    out_dir / f"{group_name}_{i // chunk_size}.parquet",
                    index=False,
                    compression="zstd",
                )
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    schema = _promote_null_fields(table.schema)
                    if schema != table.schema:
                        table = table.cast(schema)
                    writer = pq.ParquetWriter(all_path, schema, compression="zstd")
                elif table.schema != writer.schema:
                    table = table.cast(writer.schema)
                if writer is None:
                    writer = pq.ParquetWriter(all_path, table.schema, compression="zstd")
                writer.write_table(table)

    if writer is not None:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Pachyderm training exports into merged Parquet shards and a single file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        type=Path,
        help="Path to export root containing create_training_data/ and combine_predictors/",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory (defaults to <input-dir>/merged_training)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Rows per chunk/file when writing shards",
    )
    args = parser.parse_args()

    create_root = args.input_dir / "create_training_data"
    predictor_root = args.input_dir / "combine_predictors"
    if not create_root.is_dir() or not predictor_root.is_dir():
        raise SystemExit(
            f"Expected create_training_data and combine_predictors under {args.input_dir}"
        )

    out_root = args.output_dir or (args.input_dir / "merged_training")

    cases = build_cases(create_root, predictor_root)
    if not cases:
        raise SystemExit("No training cases found to merge.")

    master_cols = scan_master_columns(cases)
    if not master_cols:
        raise SystemExit("Could not determine master schema (no merged columns).")

    merge_and_write(cases, master_cols, out_root, chunk_size=args.chunk_size)
    print(f"Done. Outputs in {out_root}")


if __name__ == "__main__":
    main()
