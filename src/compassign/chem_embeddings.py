"""
Utilities for loading descriptor-informed chemical embeddings.

This module provides a loader for offline ChemBERTa embeddings reduced to 20D
via PCA, persisted as a Parquet file under resources.

The expected schema (column names) follows the docs:
- chem_id: int identifier aligned to library rows
- smiles: SMILES string used to compute the embedding
- embedding20: array[20] float (per-row)
- Optional metadata columns repeated per row:
  - pca_mean: array[H]
  - pca_components: array[20, H]
  - model_id: str
  - tokenizer_id: str
  - seed: int

No heavy dependencies are imported at module import time; reading the file
requires pandas/pyarrow which are standard runtime deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChemEmbedding:
    """Container for chemical embeddings and transform metadata.

    Attributes
    ----------
    chem_id : np.ndarray
        Array of integer chemical identifiers.
    smiles : List[str]
        SMILES strings used to compute embeddings.
    features : np.ndarray
        Matrix of shape (n_compounds, 20) with float32 embeddings.
    pca_mean : Optional[np.ndarray]
        Optional PCA mean vector from the high-D space, if provided.
    pca_components : Optional[np.ndarray]
        Optional PCA components (20, H) if provided.
    model_id : Optional[str]
        Identifier of the encoder model used.
    tokenizer_id : Optional[str]
        Identifier of the tokenizer used.
    seed : Optional[int]
        Random seed used in PCA.
    """

    chem_id: np.ndarray
    smiles: List[str]
    features: np.ndarray
    pca_mean: Optional[np.ndarray] = None
    pca_components: Optional[np.ndarray] = None
    model_id: Optional[str] = None
    tokenizer_id: Optional[str] = None
    seed: Optional[int] = None


def _coerce_array(value: Any) -> Optional[np.ndarray]:
    """Coerce a DataFrame cell containing a list/array into a numpy array.

    Returns None when the value is missing.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    arr = np.asarray(value)
    return arr


def load_chemberta_pca20(
    path: str | Path = "resources/metabolites/embeddings_chemberta_pca20.parquet",
) -> ChemEmbedding:
    """Load ChemBERTa PCA-20 embeddings from a Parquet artifact.

    Parameters
    ----------
    path : str | Path
        Path to the Parquet file. Defaults to the canonical resources location.

    Returns
    -------
    ChemEmbedding
        Dataclass with identifiers, 20D features, and optional transform metadata.
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"Embedding artifact not found: {file}. Build it via the offline builder."
        )

    # Read using pyarrow-backed pandas; preserve nested arrays as Python lists
    df = pd.read_parquet(file)

    required = {"chem_id", "smiles", "embedding20"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Embedding file {file} is missing required columns: {sorted(missing)}")

    chem_id = df["chem_id"].to_numpy(dtype=int)
    smiles: List[str] = df["smiles"].astype(str).tolist()

    # Convert list/array column to (n, 20) float32 matrix
    emb_list = df["embedding20"].tolist()
    try:
        features = np.asarray(emb_list, dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        raise ValueError("Failed to coerce 'embedding20' into a numeric matrix") from e

    if features.ndim != 2 or features.shape[1] != 20:
        raise ValueError(f"Expected 'embedding20' with shape (n, 20); got {features.shape}")

    # Optional metadata (usually repeated per row); read from first row when present
    pca_mean = None
    pca_components = None
    model_id = None
    tokenizer_id = None
    seed: Optional[int] = None

    if "pca_mean" in df.columns:
        pca_mean = _coerce_array(df.iloc[0]["pca_mean"])  # type: ignore[index]
    if "pca_components" in df.columns:
        pca_components = _coerce_array(df.iloc[0]["pca_components"])  # type: ignore[index]
    if "model_id" in df.columns:
        model_id = str(df.iloc[0]["model_id"])  # type: ignore[index]
    if "tokenizer_id" in df.columns:
        tokenizer_id = str(df.iloc[0]["tokenizer_id"])  # type: ignore[index]
    if "seed" in df.columns:
        try:
            seed = int(df.iloc[0]["seed"])  # type: ignore[index]
        except Exception:
            seed = None

    return ChemEmbedding(
        chem_id=chem_id,
        smiles=smiles,
        features=features,
        pca_mean=pca_mean,
        pca_components=pca_components,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        seed=seed,
    )
