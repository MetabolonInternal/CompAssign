#!/usr/bin/env python3
"""
Down-sample per-library merged training Parquet by capping rows per
(species_matrix_type, comp_id) pair.

This preserves all species×compound combinations while limiting the number
of repeated peaks per pair.

Usage:
  python -m compassign.rt.data_prep.sample_per_species_compound <input.parquet> \\
      --cap-per-pair 10 --seed 42

Outputs:
  <stem>_cap10.parquet by default (configurable via --output).

Optionally, when provided with lib mapping, chem-class, and species mapping
tables, this script filters rows so that only rows that will survive downstream
mapping are sampled. This ensures that any (species_matrix_type, comp_id) pair
present in the capped output has at least one valid row in the final RT CSV,
so the set of species×compound pairs does not grow with higher caps.

Note: in our merged Parquets, the column name `species_matrix_type` refers to the
label used to partition predictor tables in the upstream export (often a supercategory
identifier), not necessarily a curate SMT phrase.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


Key = Tuple[object, object]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Cap rows per (species_matrix_type, comp_id) pair in a merged training Parquet."
    )
    ap.add_argument("input", type=Path, help="Per-lib merged training Parquet file")
    ap.add_argument(
        "--cap-per-pair",
        type=int,
        default=10,
        help="Maximum rows to keep per (species_matrix_type, comp_id) pair",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for within-pair sampling")
    ap.add_argument(
        "--lib-mapping",
        type=Path,
        default=None,
        help=(
            "Optional CSV with lib_id, comp_id, chemical_id. When provided together with "
            "--classes, comp_ids without a valid compound_class are excluded before sampling."
        ),
    )
    ap.add_argument(
        "--classes",
        type=Path,
        default=None,
        help=(
            "Optional chem-classes Parquet/CSV with chem_id and compound_class. Used with "
            "--lib-mapping to determine which comp_id values have a valid class."
        ),
    )
    ap.add_argument(
        "--species-mapping",
        type=Path,
        default=None,
        help=(
            "Optional species mapping CSV with sample_set_id. When provided, rows whose "
            "sample_set_id is not present in this mapping are filtered out before sampling."
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <stem>_cap<cap-per-pair>.parquet)",
    )
    return ap.parse_args()


def _load_good_comp_ids(lib_mapping: Path, classes: Path) -> Set[int]:
    """
    Determine the set of comp_id values that have a valid compound_class.

    This mirrors the logic in attach_chem_classes_and_filter.py but operates only
    on the small mapping tables, not on the full merged Parquet.
    """
    map_df = pd.read_csv(lib_mapping)
    if "comp_id" not in map_df.columns or "chemical_id" not in map_df.columns:
        raise SystemExit(
            f"Lib mapping {lib_mapping} must contain 'comp_id' and 'chemical_id' columns"
        )

    if classes.suffix == ".parquet":
        classes_df = pd.read_parquet(classes)
    else:
        classes_df = pd.read_csv(classes)

    if not {"chem_id", "compound_class"}.issubset(classes_df.columns):
        raise SystemExit(
            f"Classes file {classes} must contain 'chem_id' and 'compound_class' columns"
        )

    map_df = map_df[["comp_id", "chemical_id"]].copy()
    map_df["comp_id"] = map_df["comp_id"].astype("Int64")
    map_df["chemical_id"] = map_df["chemical_id"].astype("Int64")
    classes_df = classes_df[["chem_id", "compound_class"]].copy()
    classes_df["chem_id"] = classes_df["chem_id"].astype("Int64")

    merged = map_df.merge(
        classes_df[["chem_id"]],
        left_on="chemical_id",
        right_on="chem_id",
        how="inner",
    )
    good = set(int(v) for v in merged["comp_id"].dropna().unique())
    if not good:
        raise SystemExit(
            f"No comp_id values with a valid compound_class found using {lib_mapping} and {classes}"
        )
    return good


def _load_species_mapping_info(species_mapping: Path) -> tuple[Set[int], Dict[int, int]]:
    """
    Determine the set of sample_set_id values that have a valid species mapping
    and build a lookup from sample_set_id -> species.
    """
    sm_df = pd.read_csv(species_mapping)
    required = {"sample_set_id", "species"}
    if not required.issubset(sm_df.columns):
        raise SystemExit(
            f"Species mapping {species_mapping} must contain columns {sorted(required)}"
        )
    sm_df = sm_df[["sample_set_id", "species"]].dropna()
    sm_df["sample_set_id"] = sm_df["sample_set_id"].astype(int)
    sm_df["species"] = sm_df["species"].astype(int)

    good_ids = set(sm_df["sample_set_id"].unique().tolist())
    if not good_ids:
        raise SystemExit(f"No sample_set_id values found in species mapping {species_mapping}")
    ssid_to_species = {
        int(row["sample_set_id"]): int(row["species"]) for _, row in sm_df.iterrows()
    }
    return good_ids, ssid_to_species


def sample_file(
    src: Path,
    dst: Path,
    cap_per_pair: int,
    seed: int,
    good_comp_ids: Optional[Set[int]] = None,
    good_sample_set_ids: Optional[Set[int]] = None,
    ssid_to_species: Optional[Dict[int, int]] = None,
) -> None:
    rng = random.Random(seed)
    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow

    if "species_matrix_type" not in schema.names or "comp_id" not in schema.names:
        raise SystemExit("Input must contain 'species_matrix_type' and 'comp_id' columns")
    if (
        good_sample_set_ids is not None or ssid_to_species is not None
    ) and "sample_set_id" not in schema.names:
        raise SystemExit(
            "Input uses species mapping filtering but Parquet is missing 'sample_set_id' column"
        )

    counts: Dict[Key, int] = {}
    writer: pq.ParquetWriter | None = None

    for rg_idx in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg_idx)
        df = tbl.to_pandas()

        # Apply optional filters so that we only sample rows that will survive
        # downstream mapping (chem classes + species mapping). This ensures that
        # any (species_matrix_type, comp_id) pair present in the capped output
        # has at least one valid row in the final RT CSV, so the set of pairs
        # does not grow with higher caps.
        if good_comp_ids is not None:
            df = df[df["comp_id"].isin(good_comp_ids)]
        if good_sample_set_ids is not None:
            # sample_set_id is often stored as a string column in the merged
            # Parquet; normalise to integers before membership testing so that
            # we align with the species mapping CSV semantics.
            ssid_series = df["sample_set_id"].astype(int)
            df = df[ssid_series.isin(good_sample_set_ids)]
        if ssid_to_species is not None:
            # Attach species per row using the mapping.
            ssid_series = df["sample_set_id"].astype(int)
            df = df.assign(species=ssid_series.map(ssid_to_species).astype(int))
        if df.empty:
            continue

        # Decide capping key: when species info is available, cap per
        # (species_matrix_type, species, comp_id); otherwise fall back to
        # (species_matrix_type, comp_id) as before.
        group_cols = ["species_matrix_type", "comp_id"]
        if "species" in df.columns:
            group_cols = ["species_matrix_type", "species", "comp_id"]

        keep_chunks = []
        for _, grp in df.groupby(group_cols, sort=False):
            mtype = grp["species_matrix_type"].iloc[0]
            comp = grp["comp_id"].iloc[0]
            if "species" in grp.columns:
                sp = grp["species"].iloc[0]
                key: Key = (mtype, sp, comp)
            else:
                key = (mtype, comp)
            seen = counts.get(key, 0)
            allowed = cap_per_pair - seen
            if allowed <= 0:
                continue

            n = len(grp)
            k = min(allowed, n)
            if k <= 0:
                continue

            idx = list(grp.index)
            rng.shuffle(idx)
            take_idx = idx[:k]
            keep_chunks.append(grp.loc[take_idx])
            counts[key] = seen + k

        if not keep_chunks:
            continue

        kept = pd.concat(keep_chunks, ignore_index=True)
        table = pa.Table.from_pandas(kept, schema=schema, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(dst, schema, compression="zstd")
        writer.write_table(table)

        if (rg_idx + 1) % 100 == 0:
            print(
                f"[sample_per_pair] processed {rg_idx + 1}/{pf.num_row_groups} row groups; "
                f"last batch kept {len(kept)} rows"
            )

    if writer is not None:
        writer.close()
    print(f"[sample_per_pair] Done. Output: {dst}")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    if args.cap_per_pair <= 0:
        raise SystemExit("cap-per-pair must be positive")

    good_comp_ids: Optional[Set[int]] = None
    good_sample_set_ids: Optional[Set[int]] = None
    ssid_to_species: Optional[Dict[int, int]] = None

    if args.lib_mapping is not None or args.classes is not None:
        if args.lib_mapping is None or args.classes is None:
            raise SystemExit(
                "Provide both --lib-mapping and --classes when enabling compound-class filtering"
            )
        if not args.lib_mapping.exists():
            raise SystemExit(f"Lib mapping not found: {args.lib_mapping}")
        if not args.classes.exists():
            raise SystemExit(f"Classes file not found: {args.classes}")
        good_comp_ids = _load_good_comp_ids(args.lib_mapping, args.classes)

    if args.species_mapping is not None:
        if not args.species_mapping.exists():
            raise SystemExit(f"Species mapping not found: {args.species_mapping}")
        good_sample_set_ids, ssid_to_species = _load_species_mapping_info(args.species_mapping)

    out = args.output or args.input.with_name(f"{args.input.stem}_cap{args.cap_per_pair}.parquet")
    sample_file(
        args.input,
        out,
        args.cap_per_pair,
        args.seed,
        good_comp_ids,
        good_sample_set_ids,
        ssid_to_species,
    )


if __name__ == "__main__":
    main()
