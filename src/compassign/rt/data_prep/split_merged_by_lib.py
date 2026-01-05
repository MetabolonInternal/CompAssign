#!/usr/bin/env python3
"""
Split a merged training Parquet file into per-library files based on `lib_id`.

Example:
  python -m compassign.rt.data_prep.split_merged_by_lib repo_export/merged_training_all.parquet

Outputs:
  repo_export/merged_training_de194c2cc2114efaa1075ccf7539d0cb_lib209.parquet
  repo_export/merged_training_de194c2cc2114efaa1075ccf7539d0cb_lib305.parquet
  ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Split merged training Parquet into per-lib_id files.")
    ap.add_argument("input", type=Path, help="Merged training Parquet file")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs (default: same as input)",
    )
    return ap.parse_args()


def split_by_lib(src: Path, out_dir: Path) -> None:
    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow

    writers: Dict[float, pq.ParquetWriter] = {}
    counts: Dict[float, int] = {}

    for rg_idx in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg_idx)
        df = tbl.to_pandas()
        if "lib_id" not in df.columns:
            raise SystemExit("Column 'lib_id' not found in input; cannot split by library.")

        for lib_val, grp in df.groupby("lib_id", sort=False):
            if grp.empty:
                continue
            lib_float = float(lib_val)
            writer = writers.get(lib_float)
            if writer is None:
                out_path = out_dir / f"{src.stem}_lib{int(lib_float)}.parquet"
                print(f"[split] Creating writer for lib_id={int(lib_float)} -> {out_path}")
                writers[lib_float] = pq.ParquetWriter(out_path, schema, compression="zstd")
                counts[lib_float] = 0

            # Ensure column order matches original schema
            grp_ordered = grp[schema.names]
            table = pa.Table.from_pandas(grp_ordered, schema=schema, preserve_index=False)
            writers[lib_float].write_table(table)
            counts[lib_float] += len(grp_ordered)

        if (rg_idx + 1) % 50 == 0:
            print(f"[split] processed {rg_idx + 1}/{pf.num_row_groups} row groups")

    for lib_val, writer in writers.items():
        writer.close()
        print(
            f"[split] Closed writer for lib_id={int(lib_val)} with {counts.get(lib_val, 0):,} rows"
        )


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    split_by_lib(args.input, out_dir)


if __name__ == "__main__":
    main()
