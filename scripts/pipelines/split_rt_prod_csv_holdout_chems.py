#!/usr/bin/env python3
"""
Split an RT production CSV into train/test by holding out *chemicals* (chem_id).

This creates a true cold-start slice for the RT models because held-out rows will have
unseen (species_cluster, comp_id) groups in the training artifact. It is intended for
evaluating the chem_hier backoff model.

Inputs:
  - A production RT CSV with a `compound` column (chem_id).

Outputs (under --output-dir):
  - train.csv
  - test.csv
  - holdout_chem_ids.csv
  - config.json (split parameters + row counts)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Set

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split an RT production CSV by holding out chem_ids.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-csv", type=Path, required=True, help="Input RT production CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory (writes train.csv, test.csv, holdout_chem_ids.csv, config.json).",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--holdout-frac",
        type=float,
        default=0.1,
        help="Fraction of unique chem_ids to hold out (used if --holdout-n is not set).",
    )
    group.add_argument(
        "--holdout-n",
        type=int,
        default=None,
        help="Number of unique chem_ids to hold out (overrides --holdout-frac).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for selecting held-out chem_ids."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Rows per chunk when streaming the CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files under --output-dir.",
    )
    return parser.parse_args()


def _iter_unique_chems(*, data_csv: Path, chunk_size: int) -> Iterable[int]:
    for chunk in pd.read_csv(data_csv, chunksize=int(chunk_size), usecols=["compound"]):
        s = pd.to_numeric(chunk["compound"], errors="coerce").dropna()
        if s.empty:
            continue
        for cid in s.astype(np.int64).unique().tolist():
            yield int(cid)


def main() -> None:
    args = parse_args()

    data_csv = args.data_csv
    if not data_csv.is_absolute():
        data_csv = (REPO_ROOT / data_csv).resolve()
    if not data_csv.exists():
        raise SystemExit(f"Input CSV not found: {data_csv}")

    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = out_dir / "train.csv"
    test_csv = out_dir / "test.csv"
    holdout_csv = out_dir / "holdout_chem_ids.csv"
    config_json = out_dir / "config.json"

    if not args.overwrite:
        for p in [train_csv, test_csv, holdout_csv, config_json]:
            if p.exists():
                raise SystemExit(f"Refusing to overwrite existing file: {p} (use --overwrite)")

    holdout_frac = float(args.holdout_frac)
    if args.holdout_n is None and not (0.0 < holdout_frac < 1.0):
        raise SystemExit("--holdout-frac must be in (0, 1) when --holdout-n is not set")

    print(f"[split] Scanning unique chem_ids from {data_csv} ...")
    unique_chems: Set[int] = set()
    for cid in _iter_unique_chems(data_csv=data_csv, chunk_size=int(args.chunk_size)):
        unique_chems.add(int(cid))
    chem_ids = np.asarray(sorted(unique_chems), dtype=np.int64)
    if chem_ids.size == 0:
        raise SystemExit("No chem_ids found in input CSV (compound column)")

    if args.holdout_n is None:
        n_holdout = int(np.floor(holdout_frac * float(chem_ids.size)))
        n_holdout = max(1, n_holdout)
    else:
        n_holdout = int(args.holdout_n)
        if not (0 < n_holdout < chem_ids.size):
            raise SystemExit(f"--holdout-n must be in [1, {chem_ids.size - 1}] (got {n_holdout})")

    rng = np.random.default_rng(int(args.seed))
    holdout_ids = np.asarray(
        rng.choice(chem_ids, size=int(n_holdout), replace=False),
        dtype=np.int64,
    )
    holdout_ids = np.sort(holdout_ids)
    holdout_set = set(int(x) for x in holdout_ids.tolist())

    pd.DataFrame({"chem_id": holdout_ids}).to_csv(holdout_csv, index=False)
    print(f"[split] Unique chem_ids={chem_ids.size:,}; holdout chem_ids={holdout_ids.size:,}")
    print(f"[split] Wrote {holdout_csv}")

    # Stream the full CSV and write row splits.
    total_rows = 0
    train_rows = 0
    test_rows = 0
    first_train = True
    first_test = True
    print(f"[split] Writing train/test CSVs under {out_dir} ...")
    for chunk in pd.read_csv(data_csv, chunksize=int(args.chunk_size)):
        total_rows += int(len(chunk))
        chem = pd.to_numeric(chunk["compound"], errors="coerce").fillna(-1).astype(np.int64)
        is_holdout = chem.isin(holdout_set)
        chunk_test = chunk.loc[is_holdout].copy()
        chunk_train = chunk.loc[~is_holdout].copy()

        if not chunk_train.empty:
            train_rows += int(len(chunk_train))
            chunk_train.to_csv(
                train_csv, mode="w" if first_train else "a", index=False, header=first_train
            )
            first_train = False
        if not chunk_test.empty:
            test_rows += int(len(chunk_test))
            chunk_test.to_csv(
                test_csv, mode="w" if first_test else "a", index=False, header=first_test
            )
            first_test = False

    payload = {
        "data_csv": str(data_csv),
        "seed": int(args.seed),
        "holdout_frac": float(args.holdout_frac) if args.holdout_n is None else None,
        "holdout_n": int(args.holdout_n) if args.holdout_n is not None else None,
        "n_unique_chems": int(chem_ids.size),
        "n_holdout_chems": int(holdout_ids.size),
        "holdout_chem_ids_csv": str(holdout_csv),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "rows_total": int(total_rows),
        "rows_train": int(train_rows),
        "rows_test": int(test_rows),
    }
    config_json.write_text(json.dumps(payload, indent=2))
    print(
        f"[split] Done. rows_total={total_rows:,}, rows_train={train_rows:,}, rows_test={test_rows:,}"
    )
    print(f"[split] Wrote {config_json}")


if __name__ == "__main__":
    main()
