#!/usr/bin/env python3
"""
Build ChemBERTa embeddings (mean‑pooled) and reduce to 20D via PCA.

Reads resources/metabolites/metabolites.tsv and writes
resources/metabolites/embeddings_chemberta_pca20.parquet

Displays a tqdm progress bar while encoding.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCES_DIR = REPO_ROOT / "resources" / "metabolites"
DEFAULT_INPUT = RESOURCES_DIR / "metabolites.tsv"
DEFAULT_OUTPUT = RESOURCES_DIR / "embeddings_chemberta_pca20.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ChemBERTa PCA-20 embeddings from SMILES")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input TSV with SMILES")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output Parquet path")
    p.add_argument("--model-id", type=str, default="seyonec/ChemBERTa-zinc-base-v1")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def mean_pool_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input TSV not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.input, sep="\t")
    n_all = len(df_raw)
    if not {"chem_id", "smiles", "inchikey"}.issubset(df_raw.columns):
        raise ValueError("Input TSV must contain 'chem_id', 'smiles', and 'inchikey' columns")

    # Keep one row per input row with non-empty SMILES (no deduplication)
    df = df_raw[df_raw["smiles"].notna() & (df_raw["smiles"].astype(str).str.len() > 0)].copy()
    n_nonempty = len(df)
    print(f"Rows: total={n_all}, non-empty SMILES={n_nonempty}")

    smiles: List[str] = df["smiles"].astype(str).tolist()
    chem_id = df["chem_id"].astype(int).to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id)
    model.eval()

    # Encode with a progress bar in smiles units
    B = int(args.batch_size)
    n = len(smiles)
    chunks = [smiles[i : i + B] for i in range(0, n, B)]
    embs: List[np.ndarray] = []
    pbar = tqdm(total=n, desc=f"Encoding SMILES ({n} total)", unit="smiles")
    for batch in chunks:
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        pooled = mean_pool_last_hidden(last_hidden, inputs["attention_mask"])  # [B, H]
        embs.append(pooled.cpu().numpy())
        pbar.update(len(batch))
    pbar.close()

    X = np.vstack(embs).astype(np.float32)  # [n, H]

    # PCA → 20D with whitening
    pca = PCA(n_components=20, whiten=True, random_state=int(args.seed))
    Z = pca.fit_transform(X).astype(np.float32)

    out_df = pd.DataFrame(
        {
            "chem_id": chem_id,
            "smiles": smiles,
            "embedding20": [row.tolist() for row in Z],
            "model_id": args.model_id,
            "tokenizer_id": args.model_id,
            "seed": int(args.seed),
        }
    )

    out_df.to_parquet(args.output, index=False)
    print(f"Wrote embeddings: {args.output} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
