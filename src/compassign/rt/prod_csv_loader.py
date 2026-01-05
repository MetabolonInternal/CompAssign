"""Production RT CSV loader.

This normalises production-style RT CSVs (as written under `repo_export/`) into indexed arrays/dataframes
used by RT modeling code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RUN_KEY_COLS = ["sampleset_id", "worksheet_id", "task_id"]


@dataclass(frozen=True)
class LoadedData:
    rt_df: pd.DataFrame
    n_species: int
    n_compounds: int
    species_cluster: np.ndarray
    compound_class: np.ndarray
    n_clusters: int
    n_classes: int
    # Map (raw_species, raw_species_cluster) → model species index
    species_map: Dict[Tuple[int, int], int]
    # Map raw compound id → model compound index
    compound_map: Dict[int, int]
    feature_names: List[str]
    run_features: np.ndarray
    run_species: np.ndarray
    n_runs: int
    feature_group_map: Dict[str, List[str]]


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


def _detect_lib_id(df: pd.DataFrame, data_csv: Path) -> Optional[int]:
    if "lib_id" in df.columns:
        libs = sorted(df["lib_id"].dropna().unique().tolist())
        if len(libs) == 1:
            try:
                return int(libs[0])
            except Exception:
                return None
    m = re.search(r"lib(\d+)", data_csv.name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _find_species_mapping_for_lib(lib_id: int) -> Optional[Path]:
    repo_root = _repo_root()
    patterns = [
        repo_root / f"repo_export/lib{lib_id}/species_mapping",
        repo_root / "repo_export",
    ]
    for base in patterns:
        if base.is_dir():
            files = sorted(base.glob(f"**/*_lib{lib_id}_species_mapping.csv"))
            if files:
                return files[0]
    return None


def load_production_csv(data_csv: Path, include_es_group1: bool = False) -> LoadedData:
    """Load a production-style RT CSV.

    Required columns:
      - sampleset_id, worksheet_id, task_id
      - species, compound, rt
      - species_cluster, compound_class
      - IS* numeric columns (internal standards)

    Optional:
      - ES_* columns can be included via include_es_group1. When enabled, we zero-mask ES_* to
        Group 1 runs ("1 human blood") to avoid leaking group-specific surrogates.
    """
    df = pd.read_csv(data_csv)

    required = [
        *RUN_KEY_COLS,
        "species",
        "compound",
        "rt",
        "species_cluster",
        "compound_class",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["species"] = df["species"].astype(int)
    df["species_cluster"] = df["species_cluster"].astype(int)
    df["compound"] = df["compound"].astype(int)
    df["rt"] = df["rt"].astype(float)

    compound_ids = pd.Index(sorted(df["compound"].unique()))
    compound_map: Dict[int, int] = {int(c): i for i, c in enumerate(compound_ids)}

    tuples = list(map(tuple, df[RUN_KEY_COLS].astype(int).to_numpy()))
    unique_keys = pd.Index(sorted(set(tuples)))
    run_map: Dict[tuple, int] = {k: i for i, k in enumerate(unique_keys)}

    species_pairs = (
        df[["species", "species_cluster"]]
        .drop_duplicates()
        .sort_values(["species", "species_cluster"])
        .reset_index(drop=True)
    )
    species_pairs["species_model"] = np.arange(len(species_pairs), dtype=int)
    species_map: Dict[Tuple[int, int], int] = {
        (int(row["species"]), int(row["species_cluster"])): int(row["species_model"])
        for _, row in species_pairs.iterrows()
    }

    df_m = df.merge(species_pairs, on=["species", "species_cluster"], how="left")
    if df_m["species_model"].isna().any():
        bad_rows = (
            df_m.loc[df_m["species_model"].isna(), ["species", "species_cluster"]]
            .drop_duplicates()
            .head()
            .to_dict(orient="records")
        )
        raise ValueError(
            "Failed to map some (species, species_cluster) pairs to model indices; "
            f"examples: {bad_rows}"
        )

    df_m["species"] = df_m["species_model"].astype(int)
    df_m["compound"] = df_m["compound"].map(lambda x: compound_map[int(x)])
    df_m["run"] = df[RUN_KEY_COLS].astype(int).apply(lambda r: run_map[tuple(r)], axis=1)
    df_m["rt"] = df_m["rt"].astype(float)

    is_cols: List[str] = [c for c in df.columns if str(c).startswith("IS")]
    if not is_cols:
        raise ValueError("No IS* covariate columns detected in CSV")
    non_numeric_is = [c for c in is_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric_is:
        raise ValueError(f"IS covariate columns must be numeric: {non_numeric_is}")
    if df[is_cols].isna().any().any():
        bad = df[is_cols].isna().sum()
        bad = bad[bad > 0]
        raise ValueError(
            "IS covariate columns contain missing values; fix upstream data prep: "
            f"{list(bad.index)}"
        )

    feature_group_map: Dict[str, List[str]] = {}
    es_cols: List[str] = []
    if include_es_group1:
        lib_id = _detect_lib_id(df, data_csv)
        if lib_id is not None:
            group1_es_by_lib: Dict[int, List[str]] = {
                208: ["ES_15506", "ES_1649", "ES_27718", "ES_32198", "ES_33941", "ES_33955"],
                209: [
                    "ES_1299",
                    "ES_1564",
                    "ES_1604",
                    "ES_32425",
                    "ES_32553",
                    "ES_33955",
                    "ES_34419",
                    "ES_36103",
                    "ES_42398",
                    "ES_606",
                ],
            }
            es_candidates = group1_es_by_lib.get(lib_id, [])
            es_cols = [c for c in es_candidates if c in df.columns]
            if es_cols:
                mapping_path = _find_species_mapping_for_lib(lib_id)
                group1_sample_sets: List[int] = []
                if mapping_path is not None:
                    sm_df = pd.read_csv(mapping_path)
                    if {"sample_set_id", "species_group_raw"}.issubset(sm_df.columns):
                        group1_sample_sets = (
                            sm_df.loc[
                                sm_df["species_group_raw"] == "1 human blood", "sample_set_id"
                            ]
                            .dropna()
                            .astype(int)
                            .tolist()
                        )
                mask_group1 = (
                    df["sampleset_id"].astype(int).isin(group1_sample_sets)
                    if group1_sample_sets
                    else pd.Series(False, index=df.index)
                )
                df[es_cols] = df[es_cols].fillna(0.0)
                df.loc[~mask_group1, es_cols] = 0.0
                for col in es_cols:
                    feature_group_map[col] = ["1 human blood"]

    covariate_cols: List[str] = is_cols + es_cols

    feat_df = df_m[["run"]].join(df[covariate_cols])
    run_index_sorted = sorted(df_m["run"].unique())
    feat_first = feat_df.groupby("run").mean(numeric_only=True).reindex(run_index_sorted)
    run_features = feat_first[covariate_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(run_features)):
        raise ValueError("Run-level feature matrix must be finite")

    run_species_counts = df_m.groupby("run")["species"].nunique()
    if (run_species_counts > 1).any():
        problematic = run_species_counts[run_species_counts > 1].index.tolist()
        raise ValueError(
            "Each run must correspond to a single species; duplicate species detected for runs "
            f"{problematic}"
        )
    run_species_series = df_m.groupby("run")["species"].first()
    run_species_aligned = run_species_series.reindex(run_index_sorted)
    if run_species_aligned.isna().any():
        missing_runs = run_species_aligned[run_species_aligned.isna()].index.tolist()
        raise ValueError(f"Missing species assignments for runs {missing_runs}")
    run_species = run_species_aligned.to_numpy(dtype=int)

    n_species = int(len(species_pairs))
    species_cluster_raw = species_pairs["species_cluster"].to_numpy(dtype=int)
    unique_clusters, inv = np.unique(species_cluster_raw, return_inverse=True)
    species_cluster = inv.astype(int)
    n_clusters = int(len(unique_clusters))

    cc_df = (
        df[["compound", "compound_class"]]
        .drop_duplicates(subset=["compound"])
        .set_index("compound")
    )
    if not pd.api.types.is_integer_dtype(cc_df["compound_class"]):
        cc_df["compound_class"] = cc_df["compound_class"].astype("category").cat.codes
    cc_df = cc_df.reindex(compound_ids.astype(int))
    compound_class = cc_df["compound_class"].to_numpy(dtype=int)
    _, invc = np.unique(compound_class, return_inverse=True)
    compound_class = invc.astype(int)
    n_classes = int(compound_class.max() + 1) if len(compound_class) else 0

    rt_df = df_m[["species", "compound", "run", "rt"]].copy()
    if not np.all(np.isfinite(rt_df["rt"].to_numpy(dtype=float))):
        raise ValueError("Observed RT values must be finite")

    return LoadedData(
        rt_df=rt_df,
        n_species=n_species,
        n_compounds=len(compound_ids),
        species_cluster=species_cluster,
        compound_class=compound_class,
        n_clusters=n_clusters,
        n_classes=n_classes,
        species_map=species_map,
        compound_map=compound_map,
        feature_names=covariate_cols,
        run_features=run_features,
        run_species=run_species,
        n_runs=len(run_index_sorted),
        feature_group_map=feature_group_map,
    )
