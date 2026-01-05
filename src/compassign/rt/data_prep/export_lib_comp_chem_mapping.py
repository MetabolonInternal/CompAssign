#!/usr/bin/env python3
"""
Export lib_id/comp_id → chemical_id mappings from Oracle for the compounds
present in our merged RT training Parquets.

This script mirrors the query used in Sally's evaluation utilities
(`_fetch_library_metadata`), but filters the results down to the comp_ids that
actually appear in the given Parquet files.

Requirements
------------
- SQLAlchemy and an Oracle DB driver (e.g. cx_Oracle) installed in the environment.
- An Oracle connection string with access to:
    - limsuser.chro_lib_entry
    - limsuser.chro_comp_entry
    - limsuser.chemicals
    - limsuser.chemical_synonym

Usage
-----
  python -m compassign.rt.data_prep.export_lib_comp_chem_mapping \\
      --oracle-conn <ORACLE_CONN_STR> \\
      repo_export/merged_training_5684639a28c04bc5af7c4fd1a75e62b5_lib208.parquet \\
      repo_export/merged_training_de194c2cc2114efaa1075ccf7539d0cb_lib209.parquet \\
      --output repo_export/lib_comp_chem_mapping_208_209.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import os

from dotenv import load_dotenv
import pandas as pd
import pyarrow.parquet as pq
import sqlalchemy
import oracledb  # type: ignore[import-untyped]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Export lib_id/comp_id → chemical_id mappings for compounds present in Parquet files.\n"
            "To be robust to flaky DB connections, libs are processed one at a time and\n"
            "each lib's mapping is written to its own CSV."
        )
    )
    ap.add_argument(
        "parquets",
        type=Path,
        nargs="+",
        help="Per-lib merged training Parquet files (must contain lib_id and comp_id columns).",
    )
    ap.add_argument(
        "--oracle-conn",
        type=str,
        default=None,
        help=(
            "Oracle connection string understood by SQLAlchemy "
            "(e.g. oracle+oracledb://...). "
            "Defaults to PROD_READ_CONN_STR or READ_CONN_STR if omitted."
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("repo_export"),
        help="Directory to write per-lib mapping CSVs into (default: repo_export).",
    )
    ap.add_argument(
        "--max-comp-per-lib",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of comp_ids to query per lib "
            "(useful for testing flaky DB connections). "
            "If provided, only the smallest N comp_ids per lib are queried."
        ),
    )
    return ap.parse_args()


def collect_comp_ids(parquets: Iterable[Path]) -> Dict[int, Set[int]]:
    """Collect unique comp_ids per lib_id from the given Parquet files."""
    result: Dict[int, Set[int]] = {}
    for path in parquets:
        if not path.exists():
            raise FileNotFoundError(f"Parquet not found: {path}")
        pf = pq.ParquetFile(path)
        libs: Set[int] = set()
        comp_ids: Set[int] = set()
        for i in range(pf.num_row_groups):
            tbl = pf.read_row_group(i, columns=["lib_id", "comp_id"])
            df = tbl.to_pandas()
            libs.update(df["lib_id"].dropna().astype(int).unique().tolist())
            comp_ids.update(df["comp_id"].dropna().astype(int).unique().tolist())
        if len(libs) != 1:
            raise ValueError(f"{path} contains multiple lib_id values: {sorted(libs)}")
        lib_id = int(next(iter(libs)))
        existing = result.setdefault(lib_id, set())
        existing.update(comp_ids)
    return result


def fetch_library_metadata_for_lib(
    lib_id: int, comp_ids: Set[int], conn_str: str, chunk_size: int = 900
) -> pd.DataFrame:
    """Fetch lib_id/comp_id/chemical_id metadata for a single lib_id and a set of comp_ids.

    The query is chunked so that each IN-list stays below Oracle's 1000-expression limit.
    """
    comp_ids = {int(c) for c in comp_ids}
    if not comp_ids:
        return pd.DataFrame(
            columns=["lib_id", "comp_id", "chemical_id", "report_name", "short_name"]
        )

    # Prefer the modern python-oracledb driver when the URL uses the legacy
    # "oracle://" scheme by swapping it to "oracle+oracledb://".
    dialect_url = conn_str
    if conn_str.startswith("oracle://"):
        dialect_url = "oracle+oracledb://" + conn_str[len("oracle://") :]

    # For legacy Oracle servers, python-oracledb may need thick mode; users
    # can configure this via environment (see python-oracledb docs).
    engine = sqlalchemy.create_engine(dialect_url)
    comp_list_sorted = sorted(comp_ids)
    frames: List[pd.DataFrame] = []
    with engine.connect() as conn:
        for start in range(0, len(comp_list_sorted), chunk_size):
            chunk = comp_list_sorted[start : start + chunk_size]
            if not chunk:
                continue
            comp_in = ",".join(str(c) for c in chunk)
            query = f"""
                SELECT
                    cle.lib_id,
                    cle.comp_id,
                    cle.chemical_id
                FROM limsuser.chro_lib_entry cle
                WHERE cle.comp_id IN ({comp_in}) AND cle.lib_id = {int(lib_id)}
            """
            df_chunk = pd.read_sql(query, con=conn)
            if not df_chunk.empty:
                frames.append(df_chunk)
    if not frames:
        return pd.DataFrame(columns=["lib_id", "comp_id", "chemical_id"])
    df = pd.concat(frames, ignore_index=True)
    return df.astype({"lib_id": "int32", "comp_id": "int32"}).drop_duplicates(
        ["lib_id", "comp_id"], keep="first"
    )


def main() -> None:
    args = parse_args()
    # Load .env for Oracle connection configuration
    load_dotenv()
    conn_str: Optional[str] = (
        args.oracle_conn or os.environ.get("PROD_READ_CONN_STR") or os.environ.get("READ_CONN_STR")
    )
    if not conn_str:
        raise SystemExit(
            "Oracle connection string not provided. "
            "Pass --oracle-conn or set PROD_READ_CONN_STR / READ_CONN_STR."
        )

    # Initialize python-oracledb in thick mode so that older database servers
    # are supported (avoids DPY-3010 in thin mode). Follow Sally's pattern:
    # call init_oracle_client(), allowing ORACLE_HOME / PATH / lib_dir to
    # control where the client is loaded from.
    if oracledb is not None:
        client_dir = os.environ.get("ORACLE_CLIENT_LIB_DIR")
        try:
            if client_dir:
                oracledb.init_oracle_client(lib_dir=client_dir)
                print(
                    f"[export] Initialized python-oracledb thick mode with client at {client_dir}"
                )
            else:
                oracledb.init_oracle_client()
                print(
                    "[export] Initialized python-oracledb thick mode with default client settings"
                )
        except Exception as exc:  # pragma: no cover - best-effort
            print(f"[export] Warning: failed to init Oracle client for thick mode ({exc})")

    comp_ids_by_lib = collect_comp_ids(args.parquets)
    libs = sorted(comp_ids_by_lib.keys())
    print(f"[export] Found libs: {libs}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for lib in libs:
        all_comp_ids = comp_ids_by_lib.get(lib, set())
        if not all_comp_ids:
            print(f"[export] lib_id={lib}: no comp_ids found in Parquet, skipping.")
            continue
        # Optionally cap the number of compounds per lib for testing
        if args.max_comp_per_lib is not None:
            comp_ids = set(sorted(all_comp_ids)[: args.max_comp_per_lib])
            print(
                f"[export] lib_id={lib}: limiting to {len(comp_ids):,} comp_ids "
                f"out of {len(all_comp_ids):,} (max-comp-per-lib={args.max_comp_per_lib})"
            )
        else:
            comp_ids = all_comp_ids
        if not comp_ids:
            print(f"[export] lib_id={lib}: no comp_ids found in Parquet, skipping.")
            continue
        print(f"[export] lib_id={lib}: fetching metadata for {len(comp_ids):,} comp_ids...")
        try:
            lib_df = fetch_library_metadata_for_lib(lib, comp_ids, conn_str)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[export] lib_id={lib}: ERROR fetching metadata ({exc}), skipping.")
            continue
        if lib_df.empty:
            print(f"[export] lib_id={lib}: no rows returned from Oracle, skipping.")
            continue

        mask_rows: List[bool] = []
        for _, row in lib_df.iterrows():
            comp = int(row["comp_id"])
            mask_rows.append(comp in comp_ids)
        filtered = lib_df[mask_rows].copy()
        if filtered.empty:
            print(
                f"[export] lib_id={lib}: no overlap between Oracle metadata and Parquet comp_ids."
            )
            continue
        filtered.sort_values(["lib_id", "comp_id"], inplace=True)

        out_path = out_dir / f"lib_comp_chem_mapping_lib{lib}.csv"
        filtered.to_csv(out_path, index=False)
        print(f"[export] lib_id={lib}: wrote {len(filtered):,} rows to {out_path}")


if __name__ == "__main__":
    main()
