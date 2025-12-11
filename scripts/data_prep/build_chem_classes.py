#!/usr/bin/env python3
"""
Build chemistry-informed compound classes for all CHEM_IDs with embeddings.

We load the global ChemBERTa PCA-20 embeddings, run k-means over the 20D
space, and assign a discrete compound_class label per chem_id.

Outputs:
  resources/metabolites/chem_classes_k<n>.parquet
  resources/metabolites/chem_classes_k<n>.csv

Usage:
  python scripts/data_prep/build_chem_classes.py --n-clusters 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

# Avoid importing the full compassign package (and its heavy deps) by reading
# the embedding artifact directly with pandas.

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_PATH = REPO_ROOT / "resources" / "metabolites" / "embeddings_chemberta_pca20.parquet"
OUT_DIR = REPO_ROOT / "resources" / "metabolites"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster ChemBERTa PCA-20 embeddings into compound classes."
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=32,
        help="Number of k-means clusters to form (compound classes).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_clusters = int(args.n_clusters)
    if n_clusters <= 1:
        raise SystemExit("--n-clusters must be >= 2")

    if not EMBEDDINGS_PATH.exists():
        raise SystemExit(
            f"Embedding artifact not found at {EMBEDDINGS_PATH}. "
            "Build it via scripts/data_prep/build_chem_library.sh"
        )
    df_emb = pd.read_parquet(EMBEDDINGS_PATH)
    required = {"chem_id", "smiles", "embedding20"}
    missing = required.difference(df_emb.columns)
    if missing:
        raise SystemExit(
            f"Embedding file {EMBEDDINGS_PATH} missing required columns: {sorted(missing)}"
        )

    chem_id = df_emb["chem_id"].to_numpy(dtype=int)
    emb_list = df_emb["embedding20"].tolist()
    try:
        import numpy as np

        X = np.asarray(emb_list, dtype="float32")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to coerce embedding20 to numeric matrix: {exc}") from exc

    print(
        f"[build_chem_classes] Clustering {X.shape[0]} chem_ids in 20D space "
        f"into {n_clusters} classes..."
    )
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X).astype(int)

    df = pd.DataFrame(
        {
            "chem_id": chem_id,
            "compound_class": labels,
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"chem_classes_k{n_clusters}"
    out_parquet = OUT_DIR / f"{stem}.parquet"
    out_csv = OUT_DIR / f"{stem}.csv"

    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print(f"[build_chem_classes] Wrote {len(df):,} rows to {out_parquet}")
    print(f"[build_chem_classes] Wrote {len(df):,} rows to {out_csv}")


if __name__ == "__main__":
    main()
