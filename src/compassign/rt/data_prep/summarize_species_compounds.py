#!/usr/bin/env python3
"""
Summarize unique species_matrix_type × comp_id counts and estimate row counts under sampling caps.

Usage:
  python -m compassign.rt.data_prep.summarize_species_compounds <input.parquet> [<input2.parquet> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow.parquet as pq


def summarize(
    path: Path, caps: Tuple[int, ...] = (10, 20, 50, 100)
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int]]:
    """
    Compute exact species_matrix_type × comp_id combinations and row counts,
    and total rows retained under different per-combination caps.
    """
    pf = pq.ParquetFile(path)
    counts: Dict[Tuple[str, float], int] = {}

    for i in range(pf.num_row_groups):
        tbl = pf.read_row_group(i, columns=["species_matrix_type", "comp_id"])
        df = tbl.to_pandas()[["species_matrix_type", "comp_id"]]
        # Count rows per combination within this row group
        grp_sizes = df.groupby(["species_matrix_type", "comp_id"], observed=True).size()
        for (species, comp_id), n in grp_sizes.items():
            key = (str(species), float(comp_id))
            counts[key] = counts.get(key, 0) + int(n)

    # Build pair-level DataFrame
    rows: List[Dict[str, object]] = []
    for (species, comp_id), n in counts.items():
        rows.append(
            {
                "species_matrix_type": species,
                "comp_id": comp_id,
                "n_rows": n,
            }
        )
    pairs_df = pd.DataFrame(rows)

    summary = (
        pairs_df.groupby("species_matrix_type")
        .agg(n_compounds=("comp_id", "nunique"), n_rows=("n_rows", "sum"))
        .reset_index()
        .sort_values("species_matrix_type")
    )

    total_by_cap: Dict[int, int] = {}
    for cap in caps:
        capped = pairs_df["n_rows"].clip(upper=cap)
        total_by_cap[cap] = int(capped.sum())

    return pairs_df, summary, total_by_cap


def main(paths: list[str]) -> None:
    if not paths:
        raise SystemExit(
            "Usage: summarize_species_compounds.py <input.parquet> [<input2.parquet> ...]"
        )
    for p_str in paths:
        path = Path(p_str)
        if not path.exists():
            print(f"Missing {path}", file=sys.stderr)
            continue
        print(f"\n{path.name}")
        pairs_df, summary, total_by_cap = summarize(path)

        # Write out all species × compound combinations with row counts
        out_csv = path.with_name(path.stem + "_species_compounds.csv")
        pairs_df.to_csv(out_csv, index=False)
        print(f"Wrote species×compound combinations to {out_csv}")

        print("Species × compound counts:")
        print(summary.to_string(index=False))
        total_pairs = int(summary["n_compounds"].sum())
        print(f"Total unique species×compound pairs: {total_pairs:,}")
        for cap, total_rows in total_by_cap.items():
            print(f"Exact rows at cap {cap} per pair: {total_rows:,}")


if __name__ == "__main__":
    main(sys.argv[1:])
