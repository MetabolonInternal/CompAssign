#!/usr/bin/env python3
"""
Load production-style RT data (CSV), fit the hierarchical RT model, and emit diagnostics.

Inputs expected:
- single CSV with columns:
  - species|species_id
  - task_id|run|run_id (per measurement identifier)
  - compound|compound_id|comp_id
  - compound_rt|rt|retention_time|rt_min
  - optional species_cluster|cluster and compound_class|class columns
  - remaining numeric columns treated as run-level internal-standard measurements

Usage example:
  python scripts/train_rt_prod.py \
    --data-csv path/rt.csv \
    --quick
  python scripts/train_rt_prod.py \
    --data-csv path/rt.csv \
    --quick

Notes:
- "--smoke" uses small sampler settings to check end-to-end wiring.
- Add "--no-fit" to validate loading without MCMC if your environment lacks PyMC.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import re

# Ensure the repository root (containing src/) is on sys.path before importing compassign.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compassign.rt.hierarchical import HierarchicalRTModel
from src.compassign.utils import load_chemberta_pca20


# ----------------------------
# CLI
# ----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load production-style RT CSVs, fit hierarchical RT model, and report diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data source (single-file input)
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=None,
        help="CSV with species, sampleset_id, worksheet_id, task_id, compound, clusters/classes, IS_* columns, and rt",
    )

    # Output and sampling
    parser.add_argument("--output-dir", type=Path, default=Path("output/rt_prod"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Worker processes for sampling (default: PyMC picks min(chains, CPU cores, 4)). "
        "Set to chains to force full parallelism when enough cores are available.",
    )
    parser.add_argument("--draws", type=int, default=1000, help="Samples per chain")
    parser.add_argument("--tune", type=int, default=1000, help="Tuning steps")
    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.8,
        help="NUTS target acceptance rate (PyMC default ≈ 0.8; higher = smaller step size, deeper trees)",
    )
    parser.add_argument(
        "--max-treedepth",
        type=int,
        default=10,
        help="Maximum tree depth for NUTS (controls worst-case gradient evaluations per draw)",
    )
    parser.add_argument(
        "--ppc-draws",
        type=int,
        default=200,
        help="Posterior predictive draws to use for diagnostics (reduces memory). Set to 0 to use all draws.",
    )
    parser.add_argument("--no-fit", action="store_true", help="Only load/validate, skip MCMC")
    parser.add_argument("--quick", action="store_true", help="Use quicker sampler settings")
    parser.add_argument("--smoke", action="store_true", help="Use tiny sampler settings for E2E")
    parser.add_argument(
        "--include-es-group1",
        action="store_true",
        help="Include ES_* covariates for Group 1 (human blood) with zero-fill elsewhere.",
    )
    parser.add_argument(
        "--no-clusters",
        action="store_true",
        help="Disable cluster hierarchy; species effects are independent of clusters.",
    )
    parser.add_argument(
        "--no-species",
        action="store_true",
        help="Disable species-level intercepts entirely.",
    )
    parser.add_argument(
        "--class-only-gamma",
        action="store_true",
        help="Use class-level run covariate slopes only (no per-compound random slopes).",
    )
    parser.add_argument(
        "--sigma-gamma-class-scale",
        type=float,
        default=None,
        help="Override HalfNormal scale for gamma_class (default 0.5).",
    )
    parser.add_argument(
        "--sigma-y-loc",
        type=float,
        default=None,
        help="Override prior mean for the log noise scale (log sigma_y) used for group-specific observation noise (default log(0.02)).",
    )
    parser.add_argument(
        "--sigma-y-scale",
        type=float,
        default=None,
        help="Override prior sigma for the log noise scale (default 0.4).",
    )
    parser.add_argument(
        "--sigma-species-scale",
        type=float,
        default=None,
        help="Override HalfNormal scale for species effects (default 0.5).",
    )
    parser.add_argument(
        "--species-compound-intercept",
        action="store_true",
        help="Add a shrunken (species, compound) interaction intercept (delta_sc).",
    )
    parser.add_argument(
        "--sigma-sc-scale",
        type=float,
        default=None,
        help="Override HalfNormal scale for delta_sc (default 0.1).",
    )

    # Holdout evaluation
    parser.add_argument(
        "--holdout-by",
        choices=["none", "run"],
        default="none",
        help="Create a holdout set. 'run' holds out entire runs",
    )
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.2,
        help="Fraction of units to hold out (applies to --holdout-by)",
    )
    parser.add_argument("--holdout-seed", type=int, default=123, help="Holdout random seed")

    return parser.parse_args()


# ----------------------------
# Utilities
# ----------------------------


def _first_present(d: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in d.columns:
            return c
    return None


def _ensure_int_series(s: pd.Series, name: str) -> pd.Series:
    try:
        arr = s.astype(int)
    except Exception as e:  # pragma: no cover - helpful error message
        raise ValueError(f"Column '{name}' must be integer-convertible") from e
    return arr


def _categorical_to_int(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    if pd.api.types.is_integer_dtype(series):
        return series.astype(int), {}
    cat = series.astype("category")
    codes = cat.cat.codes.astype(int)
    mapping = {str(cat.cat.categories[i]): int(i) for i in range(len(cat.cat.categories))}
    return codes, mapping


@dataclass
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
    compound_map: Dict[int, int]
    feature_names: List[str]
    run_features: np.ndarray
    run_species: np.ndarray
    n_runs: int
    feature_group_map: Dict[str, List[str]]


def _detect_lib_id(df: pd.DataFrame, data_csv: Path) -> Optional[int]:
    if "lib_id" in df.columns:
        libs = sorted(df["lib_id"].dropna().unique().tolist())
        if len(libs) == 1:
            try:
                return int(libs[0])
            except Exception:
                pass
    m = re.search(r"lib(\d+)", data_csv.name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _find_species_mapping_for_lib(lib_id: int) -> Optional[Path]:
    patterns = [
        REPO_ROOT / f"repo_export/lib{lib_id}/species_mapping",
        REPO_ROOT / "repo_export",
    ]
    # Look for any *_species_mapping.csv under the lib-specific folder first, then legacy root.
    for base in patterns:
        if base.is_dir():
            files = sorted(base.glob(f"**/*_lib{lib_id}_species_mapping.csv"))
            if files:
                return files[0]
    return None


def load_production_csv(data_csv: Path, include_es_group1: bool = False) -> LoadedData:
    """Load a CSV with explicit identifiers and IS_* covariates.

    Required columns:
      - sampleset_id, worksheet_id, task_id
      - species, compound, rt
      - species_cluster, compound_class
      - IS_* numeric columns (10+ recommended)
    Optional:
      - ES_* covariates can be included (when include_es_group1=True). These are
        zero-filled outside the target group to avoid leaking group-specific
        surrogates into other matrices.
    """
    df = pd.read_csv(data_csv)

    required = [
        "sampleset_id",
        "worksheet_id",
        "task_id",
        "species",
        "compound",
        "rt",
        "species_cluster",
        "compound_class",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Normalise core identifier dtypes.
    df["species"] = df["species"].astype(int)
    df["species_cluster"] = df["species_cluster"].astype(int)
    df["compound"] = df["compound"].astype(int)
    df["rt"] = df["rt"].astype(float)

    # Build compound id map (single global index per chem_id).
    compound_ids = pd.Index(sorted(df["compound"].unique()))
    compound_map: Dict[int, int] = {int(c): i for i, c in enumerate(compound_ids)}

    # Composite run key: (sampleset_id, worksheet_id, task_id) → run
    key_cols = ["sampleset_id", "worksheet_id", "task_id"]
    tuples = list(map(tuple, df[key_cols].astype(int).to_numpy()))
    unique_keys = pd.Index(sorted(set(tuples)))
    run_map: Dict[tuple, int] = {k: i for i, k in enumerate(unique_keys)}

    # Species indices at the (species, species_cluster) level so that the same
    # organism appearing in multiple matrix groups is treated as distinct.
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

    # Remap identifiers to internal indices.
    df_m = df.merge(species_pairs, on=["species", "species_cluster"], how="left")
    if df_m["species_model"].isna().any():
        bad_rows = (
            df_m[df_m["species_model"].isna()][["species", "species_cluster"]]
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
    df_m["run"] = df[key_cols].astype(int).apply(lambda r: run_map[tuple(r)], axis=1)
    df_m["rt"] = df_m["rt"].astype(float)

    # Determine covariate columns: use only IS* (internal standards) as run-level covariates.
    is_cols: List[str] = [c for c in df.columns if c.startswith("IS")]
    if not is_cols:
        raise ValueError("No IS* covariate columns detected in CSV")
    non_numeric_is = [c for c in is_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric_is:
        raise ValueError(f"IS covariate columns must be numeric: {non_numeric_is}")
    # Enforce no missing values in IS covariates; upstream data prep should guarantee this.
    na_counts = df[is_cols].isna().sum()
    bad_na = na_counts[na_counts > 0]
    if not bad_na.empty:
        raise ValueError(
            "IS covariate columns contain missing values; fix upstream data prep: "
            f"{list(bad_na.index)}"
        )

    # Optionally include ES covariates for Group 1 ("1 human blood") only.
    feature_group_map: Dict[str, List[str]] = {}
    es_cols: List[str] = []
    if include_es_group1:
        lib_id = _detect_lib_id(df, data_csv)
        if lib_id is None:
            print("Warning: unable to detect lib_id; skipping ES inclusion")
        else:
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
            if es_candidates and not es_cols:
                print(
                    f"Warning: expected Group 1 ES covariates for lib {lib_id} "
                    f"but none were found in the CSV; continuing without ES"
                )
                es_cols = []
            if es_cols:
                mapping_path = _find_species_mapping_for_lib(lib_id)
                if mapping_path is None:
                    print(
                        f"Warning: no species mapping found for lib {lib_id}; "
                        "cannot mask ES to Group 1"
                    )
                    group1_sample_sets: List[int] = []
                else:
                    sm_df = pd.read_csv(mapping_path)
                    if not {"sample_set_id", "species_group_raw"}.issubset(sm_df.columns):
                        print(
                            f"Warning: species mapping {mapping_path} missing sample_set_id/species_group_raw; "
                            "cannot mask ES to Group 1"
                        )
                        group1_sample_sets = []
                    else:
                        group1_sample_sets = (
                            sm_df.loc[sm_df["species_group_raw"] == "1 human blood", "sample_set_id"]
                            .dropna()
                            .astype(int)
                            .tolist()
                        )
                if group1_sample_sets:
                    mask_group1 = df["sampleset_id"].astype(int).isin(group1_sample_sets)
                else:
                    mask_group1 = pd.Series(False, index=df.index)

                df[es_cols] = df[es_cols].fillna(0.0)
                df.loc[~mask_group1, es_cols] = 0.0
                for col in es_cols:
                    feature_group_map[col] = ["1 human blood"]

    covariate_cols: List[str] = is_cols + es_cols

    # Build run_features by aggregating IS covariates per run (mean, though each run
    # should typically have a single row per IS panel).
    feat_df = df_m[["run"]].join(df[covariate_cols])
    run_index_sorted = sorted(df_m["run"].unique())
    feat_first = feat_df.groupby("run").mean(numeric_only=True).reindex(run_index_sorted)
    run_features = feat_first[covariate_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(run_features)):
        raise ValueError("Run-level feature matrix must be finite")

    n_runs = len(run_index_sorted)

    run_species_counts = df_m.groupby("run")["species"].nunique()
    if (run_species_counts > 1).any():
        problematic = run_species_counts[run_species_counts > 1].index.tolist()
        raise ValueError(
            "Each run must correspond to a single species; duplicate species detected for runs "
            f"{problematic}"
        )
    run_species_series = df_m.groupby("run")["species"].first()
    # Align to canonical run order
    run_species_aligned = run_species_series.reindex(run_index_sorted)
    if run_species_aligned.isna().any():
        missing_runs = run_species_aligned[run_species_aligned.isna()].index.tolist()
        raise ValueError(f"Missing species assignments for runs {missing_runs}")
    run_species = run_species_aligned.to_numpy(dtype=int)

    # Species clusters: one cluster per matrix group, with species-level
    # units indexed at the (species, species_cluster) level above.
    n_species = int(len(species_pairs))
    species_cluster_raw = species_pairs["species_cluster"].to_numpy(dtype=int)
    unique_clusters, inv = np.unique(species_cluster_raw, return_inverse=True)
    species_cluster = inv.astype(int)
    n_clusters = int(len(unique_clusters))

    # Compound classes
    n_compounds = len(compound_ids)
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

    # Final observation dataframe
    rt_df = df_m[["species", "compound", "run", "rt"]].copy()
    if not np.all(np.isfinite(rt_df["rt"].to_numpy(dtype=float))):
        raise ValueError("Observed RT values must be finite")

    return LoadedData(
        rt_df=rt_df,
        n_species=n_species,
        n_compounds=n_compounds,
        species_cluster=species_cluster,
        compound_class=compound_class,
        n_clusters=n_clusters,
        n_classes=n_classes,
        run_features=run_features,
        run_species=run_species,
        n_runs=n_runs,
        species_map=species_map,
        compound_map=compound_map,
        feature_names=covariate_cols,
        feature_group_map=feature_group_map,
    )


@dataclass
class Metrics:
    rmse: float
    mae: float
    coverage_95: float


def compute_ppc_metrics(ppc: Dict[str, np.ndarray], y_true: np.ndarray) -> Metrics:
    y_pred = ppc["pred_mean"]
    y_std = ppc["pred_std"]
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    lower = y_pred - 1.96 * y_std
    upper = y_pred + 1.96 * y_std
    cov95 = float(np.mean((y_true >= lower) & (y_true <= upper)))
    return Metrics(rmse=rmse, mae=mae, coverage_95=cov95)


def compute_pred_metrics(
    pred_mean: np.ndarray, pred_std: np.ndarray, y_true: np.ndarray
) -> Metrics:
    rmse = float(np.sqrt(np.mean((pred_mean - y_true) ** 2)))
    mae = float(np.mean(np.abs(pred_mean - y_true)))
    lower = pred_mean - 1.96 * pred_std
    upper = pred_mean + 1.96 * pred_std
    cov95 = float(np.mean((y_true >= lower) & (y_true <= upper)))
    return Metrics(rmse=rmse, mae=mae, coverage_95=cov95)


def build_compound_features(data_csv: Path, loaded: LoadedData) -> np.ndarray:
    """
    Build a descriptor matrix aligned to loaded.compound_map for the production RT model.

    The production RT CSV encodes compounds by their global chem_id in the
    ``compound`` column. ``load_production_csv`` preserves this in
    ``loaded.compound_map`` (chem_id → internal index). We therefore join
    directly on chem_id to the ChemBERTa PCA embeddings.
    """
    emb = load_chemberta_pca20()
    chem_to_row = {int(c): i for i, c in enumerate(emb.chem_id)}
    features = emb.features
    d = int(features.shape[1])

    compound_features = np.zeros((loaded.n_compounds, d), dtype=features.dtype)
    missing_chem_ids: List[int] = []

    # loaded.compound_map maps chem_id (from RT CSV 'compound' column) -> 0..n_compounds-1
    for chem_id, idx in loaded.compound_map.items():
        row = chem_to_row.get(int(chem_id))
        if row is None:
            missing_chem_ids.append(int(chem_id))
            continue
        compound_features[idx, :] = features[row]

    if missing_chem_ids:
        raise ValueError(
            "Failed to build compound_features for all compounds in the RT CSV: "
            f"{len(missing_chem_ids)} chem_ids lack embeddings "
            f"(examples: {sorted(missing_chem_ids)[:5]})"
        )

    return compound_features


def main() -> None:
    args = parse_args()

    # Sampler presets
    if args.smoke:
        # Ultra-fast sanity check: single chain, very few draws.
        args.chains = 1
        args.draws = min(args.draws, 20)
        args.tune = min(args.tune, 20)
    elif args.quick:
        # Use fewer chains and moderate draws for quicker iteration.
        args.chains = min(max(args.chains, 2), 2)
        args.draws = min(args.draws, 500)
        args.tune = min(args.tune, 500)

    # Resolve data source
    data_csv = args.data_csv
    if data_csv is None:
        raise SystemExit("Provide --data-csv pointing to a production RT CSV")

    # Load CSV
    print("Loading data...")
    loaded = load_production_csv(data_csv, include_es_group1=args.include_es_group1)

    # Optionally construct train/test split
    full_df = loaded.rt_df.copy()
    train_df = full_df
    test_df = None
    if args.holdout_by == "run":
        if "run" not in full_df.columns:
            raise SystemExit("--holdout-by run requires a 'run' column (use --data-csv)")
        rng = np.random.default_rng(args.holdout_seed)
        unique_runs = np.unique(full_df["run"].to_numpy(dtype=int))
        n_hold = max(1, int(np.ceil(len(unique_runs) * float(args.holdout_frac))))
        holdout_runs = set(rng.choice(unique_runs, size=n_hold, replace=False).tolist())
        mask = full_df["run"].isin(holdout_runs)
        test_df = full_df[mask].reset_index(drop=True)
        train_df = full_df[~mask].reset_index(drop=True)

    # Build descriptor matrix aligned to loaded.compound_map for the production model.
    compound_features = build_compound_features(data_csv, loaded)

    # Prepare model params
    rt_model = HierarchicalRTModel(
        n_clusters=loaded.n_clusters,
        n_species=loaded.n_species,
        n_classes=loaded.n_classes,
        n_compounds=loaded.n_compounds,
        species_cluster=loaded.species_cluster,
        compound_class=loaded.compound_class,
        run_features=loaded.run_features,
        run_species=loaded.run_species,
        run_covariate_columns=loaded.feature_names,
        compound_features=compound_features,
        use_clusters=not args.no_clusters,
        use_species=not args.no_species,
        class_only_gamma=args.class_only_gamma,
        sigma_y_loc=args.sigma_y_loc,
        sigma_y_scale=args.sigma_y_scale,
        sigma_gamma_class_scale=args.sigma_gamma_class_scale,
        sigma_species_scale=args.sigma_species_scale,
        species_compound_intercept=args.species_compound_intercept,
        sigma_sc_scale=args.sigma_sc_scale,
    )

    # Output dirs
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    (out_dir / "results").mkdir(exist_ok=True)

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "n_species": loaded.n_species,
                "n_compounds": loaded.n_compounds,
                "n_clusters": loaded.n_clusters,
                "n_classes": loaded.n_classes,
                "chains": args.chains,
                "draws": args.draws,
                "tune": args.tune,
                "target_accept": args.target_accept,
                "max_treedepth": args.max_treedepth,
                "data_csv": str(data_csv),
                "feature_names": loaded.feature_names,
                "feature_group_map": loaded.feature_group_map,
                "n_runs": loaded.n_runs,
                "holdout_by": args.holdout_by,
                "holdout_frac": args.holdout_frac,
                "include_es_group1": args.include_es_group1,
                "use_clusters": not args.no_clusters,
                "use_species": not args.no_species,
                "class_only_gamma": args.class_only_gamma,
                "sigma_gamma_class_scale": args.sigma_gamma_class_scale,
                "sigma_y_loc": args.sigma_y_loc,
                "sigma_y_scale": args.sigma_y_scale,
                "sigma_species_scale": args.sigma_species_scale,
                "species_compound_intercept": args.species_compound_intercept,
                "sigma_sc_scale": args.sigma_sc_scale,
            },
            indent=2,
        )
    )

    # Fit or only validate
    if args.no_fit:
        print("Loaded data successfully. Skipping MCMC (--no-fit).")
        print(
            f"n_species={loaded.n_species}, n_compounds={loaded.n_compounds}, "
            f"n_clusters={loaded.n_clusters}, n_classes={loaded.n_classes}, "
            f"n_runs={loaded.n_runs}, n_features={loaded.run_features.shape[1]}"
        )
        return

    # Build + sample
    print("> Building model")
    rt_model.build_model(train_df)
    use_jax = HierarchicalRTModel._jax_available()
    sampler_name = "numpyro" if use_jax else "pymc"
    chain_method = "vectorized" if use_jax else None
    print(
        "> Sampling: chains=%d, draws=%d, tune=%d, target_accept=%.3f, max_treedepth=%d"
        % (args.chains, args.draws, args.tune, args.target_accept, args.max_treedepth)
    )
    print(f"> Using {'NumPyro/JAX' if use_jax else 'PyMC'} sampler")
    cores = args.cores if args.cores is not None else args.chains

    try:
        trace = rt_model.sample(
            n_samples=args.draws,
            n_tune=args.tune,
            n_chains=args.chains,
            cores=cores,
            target_accept=args.target_accept,
            max_treedepth=args.max_treedepth,
            random_seed=args.seed,
            verbose=True,
            nuts_sampler=sampler_name,
            chain_method=chain_method,
        )
    except Exception as e:  # pragma: no cover - environment may lack PyMC dependencies
        print(f"Sampling failed: {e}")
        print("Data loaded correctly; to fit the model, ensure PyMC stack is installed.")
        return

    # Save trace and PPC diagnostics
    try:
        import arviz as az  # type: ignore

        trace.to_netcdf(out_dir / "models" / "rt_trace.nc")
        summary = az.summary(trace)
        summary.to_csv(out_dir / "results" / "rt_summary.csv")
    except Exception:
        pass

    print("> Posterior predictive checks")
    max_ppc_draws = None if args.ppc_draws == 0 else args.ppc_draws
    ppc = rt_model.posterior_predictive_check(train_df, max_ppc_draws=max_ppc_draws)
    metrics = compute_ppc_metrics(ppc, train_df["rt"].to_numpy())
    (out_dir / "results" / "rt_ppc.json").write_text(
        json.dumps({"metrics": asdict(metrics)}, indent=2)
    )
    print(f"RMSE={metrics.rmse:.3f}, MAE={metrics.mae:.3f}, coverage95={metrics.coverage_95:.3f}")

    # Holdout evaluation
    if test_df is not None and len(test_df) > 0:
        print("> Holdout evaluation (by run)")
        species_idx = test_df["species"].to_numpy(dtype=int)
        compound_idx = test_df["compound"].to_numpy(dtype=int)
        run_idx = test_df["run"].to_numpy(dtype=int)
        pred_mean, pred_std = rt_model.predict_new(
            species_idx=species_idx, compound_idx=compound_idx, run_idx=run_idx
        )
        h_metrics = compute_pred_metrics(pred_mean, pred_std, test_df["rt"].to_numpy(dtype=float))
        (out_dir / "results" / "rt_holdout.json").write_text(
            json.dumps({"metrics": asdict(h_metrics), "n_test": int(len(test_df))}, indent=2)
        )
        print(
            f"Holdout: n={len(test_df)}, RMSE={h_metrics.rmse:.3f}, MAE={h_metrics.mae:.3f}, "
            f"coverage95={h_metrics.coverage_95:.3f}"
        )


if __name__ == "__main__":
    main()
