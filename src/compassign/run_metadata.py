"""Helpers for extracting run-level covariates from peak data frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


RUN_COVARIATE_PREFIX = "run_covariate_"


class RunMetadataError(ValueError):
    """Raised when run-level covariates cannot be extracted cleanly."""


@dataclass(frozen=True)
class RunMetadata:
    """Container for run-level covariates used by the RT and assignment models."""

    df: pd.DataFrame
    features: np.ndarray
    species: np.ndarray
    covariate_columns: List[str]


def _resolve_covariate_columns(
    peak_df: pd.DataFrame, explicit: Optional[Sequence[str]] = None
) -> List[str]:
    if explicit is not None:
        cols = [str(col) for col in explicit]
        if not cols:
            raise RunMetadataError("Explicit run covariate column list is empty")
        missing = set(cols) - set(peak_df.columns)
        if missing:
            raise RunMetadataError(f"Requested run covariate columns not found: {sorted(missing)}")
        return cols

    inferred = [col for col in peak_df.columns if col.startswith(RUN_COVARIATE_PREFIX)]
    if not inferred:
        raise RunMetadataError(
            "Could not infer run covariate columns. Provide explicit names or ensure "
            f"columns are prefixed with '{RUN_COVARIATE_PREFIX}'."
        )
    return inferred


def extract_run_metadata(
    peak_df: pd.DataFrame, covariate_columns: Optional[Sequence[str]] = None
) -> RunMetadata:
    """Extract unique run-level rows and convert them into matrix form."""

    if "run" not in peak_df.columns:
        raise RunMetadataError("peak_df must include a 'run' column")
    if "species" not in peak_df.columns:
        raise RunMetadataError("peak_df must include a 'species' column")

    columns = _resolve_covariate_columns(peak_df, covariate_columns)

    run_df = peak_df[["run", "species", *columns]].drop_duplicates(subset="run").copy()
    if run_df.empty:
        raise RunMetadataError("No run-level rows found when extracting covariates")

    run_df["run"] = run_df["run"].astype(int)
    run_df["species"] = run_df["species"].astype(int)

    # Validate one species per run
    duplicates = run_df.duplicated(subset="run", keep=False)
    if duplicates.any():
        dup_runs = sorted(run_df.loc[duplicates, "run"].unique())
        raise RunMetadataError(
            "Multiple covariate rows detected for runs: "
            f"{dup_runs}. Ensure run identifiers are unique."
        )

    run_ids = run_df["run"].to_numpy(dtype=int)
    if run_ids.min() < 0:
        raise RunMetadataError("Run identifiers must be non-negative integers")

    max_run = int(run_ids.max())
    n_runs = max_run + 1

    features = np.zeros((n_runs, len(columns)), dtype=float)
    species = np.zeros(n_runs, dtype=int)

    values = run_df[columns].to_numpy(dtype=float)
    features[run_ids] = values
    species[run_ids] = run_df["species"].to_numpy(dtype=int)

    ordered_df = run_df.sort_values("run").reset_index(drop=True)
    return RunMetadata(
        df=ordered_df,
        features=features,
        species=species,
        covariate_columns=columns,
    )


def ensure_run_metadata(
    run_features: Optional[np.ndarray | pd.DataFrame],
    *,
    run_species: Optional[Iterable[int]] = None,
    peak_df: Optional[pd.DataFrame] = None,
    covariate_columns: Optional[Sequence[str]] = None,
) -> RunMetadata:
    """
    Coerce different input representations into ``RunMetadata``.

    One of ``peak_df`` (preferred) or ``run_features`` must be provided. When using
    raw arrays, ``run_species`` is also required to map runs to species.
    """

    if peak_df is not None:
        return extract_run_metadata(peak_df, covariate_columns=covariate_columns)

    if run_features is None:
        raise RunMetadataError(
            "Either peak_df or run_features must be provided to construct run metadata"
        )

    if isinstance(run_features, pd.DataFrame):
        return extract_run_metadata(run_features, covariate_columns=covariate_columns)

    arr = np.asarray(run_features, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise RunMetadataError("run_features must be 1D or 2D array-like")

    if run_species is None:
        raise RunMetadataError("run_species array required when supplying raw run_features")

    run_species_arr = np.asarray(list(run_species), dtype=int)
    if run_species_arr.shape[0] != arr.shape[0]:
        raise RunMetadataError("run_species length must match number of run feature rows")
    if run_species_arr.min() < 0:
        raise RunMetadataError("run_species must contain non-negative integers")

    n_runs = arr.shape[0]
    run_ids = np.arange(n_runs, dtype=int)
    df = pd.DataFrame(arr, columns=[f"array_covariate_{i}" for i in range(arr.shape[1])])
    df.insert(0, "species", run_species_arr)
    df.insert(0, "run", run_ids)

    return RunMetadata(
        df=df,
        features=arr,
        species=run_species_arr,
        covariate_columns=list(df.columns[2:]),
    )
