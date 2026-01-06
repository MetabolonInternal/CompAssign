#!/usr/bin/env python3
"""
Build a common training-support bin map keyed by (species_cluster, comp_id).

This map is used to produce "by support bin" plots where bins have the same meaning
across all RT models (ridge supercategory, ridge partial pooling, and lasso baseline).

Input: a production-style training CSV containing at least:
  - species_cluster
  - comp_id

Output: a CSV with columns:
  - group_key: (species_cluster << 32) + comp_id
  - species_cluster
  - comp_id
  - n_obs_train
  - support_bin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SUPPORT_EDGES = [1, 2, 5, 10, 20, 50, 100]


def _bin_labels(edges: list[int]) -> list[str]:
    edges_sorted = sorted(int(x) for x in edges)
    labels: list[str] = []
    prev = 0
    for e in edges_sorted:
        if prev == 0:
            labels.append(f"<= {e}")
        elif prev + 1 == e:
            labels.append(f"{e}-{e}")
        else:
            labels.append(f"{prev + 1}-{e}")
        prev = e
    labels.append(f"> {edges_sorted[-1]}")
    return labels


def _build_support_map(
    *, train_csv: Path, out_csv: Path, chunk_size: int, support_edges: list[int]
) -> None:
    labels = _bin_labels(support_edges)
    edges_arr = np.asarray(sorted(int(x) for x in support_edges), dtype=np.int64)

    counts_by_key: dict[int, int] = {}

    required_cols = {"species_cluster", "comp_id"}
    header = pd.read_csv(train_csv, nrows=0)
    missing = required_cols - set(header.columns)
    if missing:
        raise SystemExit(f"Train CSV missing required columns {sorted(missing)}: {train_csv}")

    usecols = ["species_cluster", "comp_id"]
    for chunk in pd.read_csv(train_csv, usecols=usecols, chunksize=int(chunk_size)):
        species_cluster = chunk["species_cluster"].astype(int).to_numpy(dtype=np.int64, copy=False)
        comp_id = chunk["comp_id"].astype(int).to_numpy(dtype=np.int64, copy=False)
        group_key = (species_cluster << np.int64(32)) + comp_id
        uniq, cnt = np.unique(group_key, return_counts=True)
        for k, c in zip(uniq.tolist(), cnt.tolist(), strict=True):
            k_int = int(k)
            counts_by_key[k_int] = counts_by_key.get(k_int, 0) + int(c)

    if not counts_by_key:
        raise SystemExit(f"No rows found in train CSV: {train_csv}")

    keys = np.asarray(sorted(counts_by_key.keys()), dtype=np.int64)
    n_obs_train = np.asarray([counts_by_key[int(k)] for k in keys.tolist()], dtype=np.int64)
    support_idx = np.digitize(n_obs_train, bins=edges_arr, right=True).astype(np.int16)
    support_bin = np.asarray([labels[int(i)] for i in support_idx.tolist()], dtype=object)

    species_cluster = (keys >> np.int64(32)).astype(np.int64)
    comp_id = (keys & np.int64(0xFFFFFFFF)).astype(np.int64)

    df = pd.DataFrame(
        {
            "group_key": keys,
            "species_cluster": species_cluster,
            "comp_id": comp_id,
            "n_obs_train": n_obs_train,
            "support_bin": support_bin,
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a (species_cluster, comp_id) training support-bin map CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--train-csv", type=Path, required=True, help="Training CSV to compute support counts from."
    )
    p.add_argument(
        "--out-csv", type=Path, required=True, help="Output CSV path for the support-bin map."
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="CSV rows per chunk when counting groups.",
    )
    p.add_argument(
        "--support-bins",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SUPPORT_EDGES),
        help="Comma-separated bin edges used to form labels (e.g. '1,2,5,10,20,50,100').",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    edges = [int(x.strip()) for x in str(args.support_bins).split(",") if x.strip()]
    if not edges:
        raise SystemExit("--support-bins must contain at least one integer edge")
    if any(x <= 0 for x in edges):
        raise SystemExit("--support-bins edges must be positive")

    _build_support_map(
        train_csv=args.train_csv,
        out_csv=args.out_csv,
        chunk_size=int(args.chunk_size),
        support_edges=edges,
    )


if __name__ == "__main__":
    main()
