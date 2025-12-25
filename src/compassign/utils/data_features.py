"""Shared data containers + feature utilities.

This module intentionally combines small, cross-cutting helpers that are used
across multiple pipelines:
- Run-level covariate extraction (`RunMetadata`, `ensure_run_metadata`)
- Synthetic dataset container (`SyntheticDataset`)
- Ion/adduct transform definitions (shared by generators and feature extraction)
- Chemical relationship feature computation for peak assignment
- Lightweight loader for ChemBERTa PCA-20 embeddings

Keeping these together reduces subpackage sprawl while maintaining a clear
"shared utilities" boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Run metadata helpers
# --------------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------------
# Synthetic dataset container
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticDataset:
    """Synthetic metabolomics dataset returned by the generator."""

    peak_df: pd.DataFrame
    compound_df: pd.DataFrame
    true_assignments: Mapping[int, Optional[int]]
    rt_uncertainties: Mapping[int, float]
    hierarchical_params: Dict[str, Any]

    def run_meta(self, covariate_columns: Optional[Sequence[str]] = None) -> RunMetadata:
        cols = covariate_columns
        if cols is None:
            cols = self.hierarchical_params.get("run_covariate_columns")
        return extract_run_metadata(self.peak_df, covariate_columns=cols)


# --------------------------------------------------------------------------------------
# Ion/adduct transforms
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IonTransform:
    name: str
    mass_shift: float
    feature_name: str
    category: str  # 'adduct', 'isotope', 'fragment'
    intensity_factor: float | None = None


ION_TRANSFORMS: Dict[str, IonTransform] = {
    "M+H": IonTransform("M+H", 0.0, "is_MH", "adduct", 0.9),
    "M+13C": IonTransform("M+13C", 1.00335, "is_M13C", "isotope", 0.35),
    "M+Na": IonTransform("M+Na", 21.9819, "is_MNa", "adduct", 0.5),
    "M+K": IonTransform("M+K", 37.9559, "is_MK", "adduct", 0.35),
    "M+NH4": IonTransform("M+NH4", 18.0338, "is_MNH4", "adduct", 0.45),
    "M-H2O+H": IonTransform("M-H2O+H", -18.0106, "is_MH2O_loss", "fragment", 0.25),
    "M-CO2+H": IonTransform("M-CO2+H", -43.9898, "is_MCO2_loss", "fragment", 0.2),
}

MATCHING_TRANSFORMS: Dict[str, float] = {
    name: transform.mass_shift for name, transform in ION_TRANSFORMS.items()
}

ADDUCT_FEATURE_MAP: List[Tuple[str, str]] = [
    (transform.name, transform.feature_name) for transform in ION_TRANSFORMS.values()
]

CHEMICAL_RELATION_MASSES: Dict[str, float] = {
    name: transform.mass_shift
    for name, transform in ION_TRANSFORMS.items()
    if transform.mass_shift != 0.0
}

ADDUCT_DEFS: List[Dict[str, float]] = [
    {
        "name": transform.name,
        "delta": transform.mass_shift,
        "intensity_factor": transform.intensity_factor,
    }
    for transform in ION_TRANSFORMS.values()
    if transform.category == "adduct" and transform.mass_shift != 0.0
]

FRAGMENT_DEFS: List[Dict[str, float]] = [
    {
        "name": transform.name,
        "delta": transform.mass_shift,
        "intensity_factor": transform.intensity_factor,
    }
    for transform in ION_TRANSFORMS.values()
    if transform.category == "fragment"
]

ISOTOPE_SHIFT = ION_TRANSFORMS["M+13C"].mass_shift
ISOTOPE_INTENSITY_FACTOR = ION_TRANSFORMS["M+13C"].intensity_factor or 0.35


# --------------------------------------------------------------------------------------
# Chemical feature computation
# --------------------------------------------------------------------------------------


ISOTOPE_MASS_DIFF = ISOTOPE_SHIFT
ADDUCT_MASSES = CHEMICAL_RELATION_MASSES
MASS_TOLERANCE = 0.01
RT_TOLERANCE = 0.1
SNR_PERCENTILE = 10


def compute_isotope_features(
    mz: float, rt: float, intensity: float, peak_df: pd.DataFrame, species: int
) -> Tuple[float, float]:
    isotope_mz = mz + ISOTOPE_MASS_DIFF
    species_peaks = peak_df[peak_df["species"] == species]
    isotope_candidates = species_peaks[
        (abs(species_peaks["mass"] - isotope_mz) < MASS_TOLERANCE)
        & (abs(species_peaks["rt"] - rt) < RT_TOLERANCE)
    ]
    has_isotope = float(len(isotope_candidates) > 0)
    if has_isotope and len(isotope_candidates) > 0:
        mass_errors = abs(isotope_candidates["mass"] - isotope_mz)
        closest_idx = mass_errors.idxmin()
        isotope_intensity = isotope_candidates.loc[closest_idx, "intensity"]
        intensity_ratio = isotope_intensity / intensity
        if 0.05 <= intensity_ratio <= 0.55:
            if 0.11 <= intensity_ratio <= 0.33:
                isotope_score = 1.0
            else:
                isotope_score = 0.7
        else:
            isotope_score = 0.3
    else:
        isotope_score = 0.0
    return has_isotope, isotope_score


def compute_adduct_features(mz: float, rt: float, peak_df: pd.DataFrame, species: int) -> float:
    species_peaks = peak_df[peak_df["species"] == species]
    if species_peaks.empty:
        return 0.0

    n_adducts = 0
    for _, delta in ADDUCT_MASSES.items():
        candidates = species_peaks[
            (abs(species_peaks["mass"] - (mz + delta)) < MASS_TOLERANCE)
            & (abs(species_peaks["rt"] - rt) < RT_TOLERANCE)
        ]
        if len(candidates) > 0:
            n_adducts += 1

        candidates_reverse = species_peaks[
            (abs(species_peaks["mass"] - (mz - delta)) < MASS_TOLERANCE)
            & (abs(species_peaks["rt"] - rt) < RT_TOLERANCE)
        ]
        if len(candidates_reverse) > 0:
            n_adducts += 1

    return float(n_adducts)


def compute_rt_cluster_features(
    mz: float, rt: float, intensity: float, peak_df: pd.DataFrame, species: int
) -> float:
    species_peaks = peak_df[peak_df["species"] == species]
    if species_peaks.empty:
        return 0.0
    coeluting = species_peaks[abs(species_peaks["rt"] - rt) < RT_TOLERANCE]
    return float(len(coeluting))


def compute_correlation_features(
    mz: float, rt: float, intensity: float, peak_df: pd.DataFrame, species: int
) -> float:
    species_peaks = peak_df[peak_df["species"] == species]
    if len(species_peaks) < 2:
        return 0.0
    rt_window = species_peaks[abs(species_peaks["rt"] - rt) < 0.5]
    if len(rt_window) < 2:
        return 0.0

    # Rough proxy: correlated peaks -> similar RT and comparable intensity
    intensity_ratios = np.abs(
        np.log1p(rt_window["intensity"].to_numpy(dtype=float)) - np.log1p(intensity)
    )
    close = intensity_ratios < 1.0
    return float(max(0, int(np.sum(close)) - 1))


def compute_snr(intensity: float, peak_df: pd.DataFrame, species: int) -> float:
    species_peaks = peak_df[peak_df["species"] == species]
    if species_peaks.empty:
        return float("nan")
    baseline = np.percentile(species_peaks["intensity"], SNR_PERCENTILE)
    return float(intensity / (baseline + 1e-9))


def compute_all_chemical_features(
    mz: float, rt: float, intensity: float, peak_df: pd.DataFrame, species: int
) -> Dict[str, float]:
    has_isotope, isotope_score = compute_isotope_features(mz, rt, intensity, peak_df, species)
    n_adducts = compute_adduct_features(mz, rt, peak_df, species)
    rt_cluster_size = compute_rt_cluster_features(mz, rt, intensity, peak_df, species)
    n_correlated = compute_correlation_features(mz, rt, intensity, peak_df, species)
    snr = compute_snr(intensity, peak_df, species)
    return {
        "has_isotope": has_isotope,
        "isotope_score": isotope_score,
        "n_adducts": n_adducts,
        "rt_cluster_size": rt_cluster_size,
        "n_correlated": n_correlated,
        "snr": snr,
    }


# --------------------------------------------------------------------------------------
# Embedding loader (ChemBERTa PCA-20)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChemEmbedding:
    chem_id: np.ndarray
    smiles: List[str]
    features: np.ndarray
    pca_mean: Optional[np.ndarray] = None
    pca_components: Optional[np.ndarray] = None
    model_id: Optional[str] = None
    tokenizer_id: Optional[str] = None
    seed: Optional[int] = None


def _coerce_array(value: Any) -> Optional[np.ndarray]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return np.asarray(value)


def load_chemberta_pca20(
    path: str | Path = "resources/metabolites/embeddings_chemberta_pca20.parquet",
) -> ChemEmbedding:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"Embedding artifact not found: {file}. Build it via the offline builder."
        )

    df = pd.read_parquet(file)
    required = {"chem_id", "smiles", "embedding20"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Embedding file {file} is missing required columns: {sorted(missing)}")

    chem_id = df["chem_id"].to_numpy(dtype=int)
    smiles: List[str] = df["smiles"].astype(str).tolist()

    emb_list = df["embedding20"].tolist()
    try:
        features = np.asarray(emb_list, dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        raise ValueError("Failed to coerce 'embedding20' into a numeric matrix") from e

    if features.ndim != 2 or features.shape[1] != 20:
        raise ValueError(f"Expected 'embedding20' with shape (n, 20); got {features.shape}")

    pca_mean = _coerce_array(df.iloc[0]["pca_mean"]) if "pca_mean" in df.columns else None
    pca_components = (
        _coerce_array(df.iloc[0]["pca_components"]) if "pca_components" in df.columns else None
    )
    model_id = str(df.iloc[0]["model_id"]) if "model_id" in df.columns else None
    tokenizer_id = str(df.iloc[0]["tokenizer_id"]) if "tokenizer_id" in df.columns else None
    seed: Optional[int]
    if "seed" in df.columns:
        try:
            seed = int(df.iloc[0]["seed"])
        except Exception:
            seed = None
    else:
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


__all__ = [
    "RUN_COVARIATE_PREFIX",
    "RunMetadata",
    "RunMetadataError",
    "SyntheticDataset",
    "ensure_run_metadata",
    "extract_run_metadata",
    "IonTransform",
    "ION_TRANSFORMS",
    "MATCHING_TRANSFORMS",
    "ADDUCT_FEATURE_MAP",
    "CHEMICAL_RELATION_MASSES",
    "ADDUCT_DEFS",
    "FRAGMENT_DEFS",
    "ISOTOPE_SHIFT",
    "ISOTOPE_INTENSITY_FACTOR",
    "compute_all_chemical_features",
    "ChemEmbedding",
    "load_chemberta_pca20",
]
