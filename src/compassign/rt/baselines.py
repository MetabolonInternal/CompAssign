"""Baseline RT prediction models used for comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SpeciesCompoundLassoBaseline:
    """Per (species × compound) Lasso on run covariates.

    - Fits an independent model for each (species, compound) key using per-run covariates.
    - Requires at least `min_samples` distinct runs per key; otherwise the key is skipped.
    """

    def __init__(
        self,
        species_cluster: np.ndarray | None = None,  # kept for interface parity; unused here
        *,
        alpha: float | None = None,
        cv: int = 5,
        min_samples: int = 3,
        random_state: int = 42,
    ) -> None:
        self.alpha = alpha
        self.cv = max(2, int(cv))
        self.min_samples = max(1, int(min_samples))
        self.random_state = int(random_state)
        self.models: Dict[tuple[int, int], ClusterModel] = {}
        self._run_features: np.ndarray | None = None
        self._covariate_columns: list[str] | None = None

    def _resolve_run_matrix(
        self, df: pd.DataFrame, run_df: pd.DataFrame | None, covariate_columns: Sequence[str] | None
    ) -> tuple[np.ndarray, list[str]]:
        if run_df is not None:
            if covariate_columns is None:
                covs = [c for c in run_df.columns if c.startswith("run_covariate_")]
            else:
                covs = [str(c) for c in covariate_columns]
            rdf = run_df.copy()
        else:
            if covariate_columns is None:
                covs = [c for c in df.columns if c.startswith("run_covariate_")]
            else:
                covs = [str(c) for c in covariate_columns]
            required = {"run", *covs}
            missing = required.difference(df.columns)
            if missing:
                raise ValueError(
                    "run covariate columns required for baseline are missing from training data: "
                    f"{sorted(missing)}"
                )
            rdf = df[["run", *covs]].drop_duplicates(subset=["run"]).copy()  # type: ignore[arg-type]

        if "run" not in rdf.columns:
            raise ValueError("run_df must include a 'run' column")
        max_run = int(pd.to_numeric(rdf["run"]).max())
        X = np.zeros((max_run + 1, len(covs)), dtype=float)
        X[pd.to_numeric(rdf["run"]).to_numpy(dtype=int)] = rdf[covs].to_numpy(dtype=float)
        return X, covs

    def _make_pipeline(self, n_samples: int) -> Pipeline:
        scaler = StandardScaler()
        if n_samples >= 5:
            reg = LassoCV(
                cv=min(self.cv, n_samples),
                random_state=self.random_state,
                max_iter=20000,
            )
        else:
            reg = Lasso(
                alpha=self.alpha if self.alpha is not None else 0.01,
                max_iter=50000,
            )
        return Pipeline([("scaler", scaler), ("regressor", reg)])

    def fit(
        self,
        train_df: pd.DataFrame,
        *,
        run_df: pd.DataFrame | None = None,
        covariate_columns: Sequence[str] | None = None,
    ) -> "SpeciesCompoundLassoBaseline":
        df = train_df.copy()
        df["species"] = df["species"].astype(int)
        df["compound"] = df["compound"].astype(int)
        df["run"] = df["run"].astype(int)

        run_features, covs = self._resolve_run_matrix(df, run_df, covariate_columns)
        self._run_features = run_features
        self._covariate_columns = covs

        for (sp, cid), sub in df.groupby(["species", "compound"], sort=False):
            X = run_features[sub["run"].to_numpy(dtype=int)]
            y = sub["rt"].to_numpy(dtype=float)
            pl = self._make_pipeline(len(sub))
            pl.fit(X, y)
            self.models[(int(sp), int(cid))] = ClusterModel(pipeline=pl)

        return self

    def predict(
        self,
        species_idx: Sequence[int],
        compound_idx: Sequence[int],
        run_idx: Sequence[int],
        *,
        run_features_override: np.ndarray | None = None,
    ) -> np.ndarray:
        features = (
            run_features_override if run_features_override is not None else self._run_features
        )
        if features is None:
            raise RuntimeError("Baseline must be fit before predict")
        species_arr = np.asarray(species_idx, dtype=int)
        comp_arr = np.asarray(compound_idx, dtype=int)
        run_arr = np.asarray(run_idx, dtype=int)
        if not (species_arr.shape == comp_arr.shape == run_arr.shape):
            raise ValueError("species_idx, compound_idx, and run_idx must have the same shape")
        max_run = features.shape[0]
        if np.any(run_arr < 0) or np.any(run_arr >= max_run):
            raise ValueError("run_idx out of bounds for fitted run feature matrix")

        preds = np.full_like(run_arr, np.nan, dtype=float)
        for i in range(len(run_arr)):
            key = (int(species_arr[i]), int(comp_arr[i]))
            model = self.models.get(key)
            if model is None:
                continue
            x = features[int(run_arr[i])].reshape(1, -1)
            preds[i] = float(model.pipeline.predict(x)[0])
        return preds


@dataclass
class ClusterModel:
    """Container for a fitted per-cluster regressor."""

    pipeline: Pipeline


class ClusterLassoBaseline:
    """Fits independent Lasso regressions for each species cluster."""

    def __init__(
        self,
        species_cluster: np.ndarray,
        *,
        alpha: float | None = None,
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        self.species_cluster = np.asarray(species_cluster, dtype=int)
        self.alpha = alpha
        self.cv = max(2, int(cv))
        self.random_state = int(random_state)
        self.cluster_models: Dict[int, ClusterModel] = {}
        self._species_features: np.ndarray | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        *,
        run_df: pd.DataFrame | None = None,
        covariate_columns: Sequence[str] | None = None,
    ) -> "ClusterLassoBaseline":
        df = train_df.copy()
        df["species"] = df["species"].astype(int)
        df["compound"] = df["compound"].astype(int)
        df["cluster"] = self.species_cluster[df["species"].to_numpy()]

        # Determine run-level covariate columns and construct a run_df if not provided
        run_cov_cols: list[str]
        if run_df is not None:
            if covariate_columns is None:
                # Infer from prefixed columns
                run_cov_cols = [c for c in run_df.columns if c.startswith("run_covariate_")]
            else:
                run_cov_cols = [str(c) for c in covariate_columns]
            rdf = run_df.copy()
        else:
            # Extract run-level data from the training dataframe
            candidate_cols = [c for c in df.columns if c.startswith("run_covariate_")]
            if covariate_columns is not None:
                run_cov_cols = [str(c) for c in covariate_columns]
            else:
                run_cov_cols = candidate_cols
            required = {"run", "species", *run_cov_cols}
            missing = required.difference(df.columns)
            if missing:
                raise ValueError(
                    "run covariate columns required for baseline are missing from training data: "
                    f"{sorted(missing)}"
                )
            rdf = df[["run", "species", *run_cov_cols]].drop_duplicates(subset=["run"])  # type: ignore[arg-type]

        # Compute species-level feature summaries (mean over runs)
        n_species = int(self.species_cluster.shape[0])
        if not run_cov_cols:
            raise ValueError("No run covariate columns available to build baseline features")
        species_features = np.zeros((n_species, len(run_cov_cols)), dtype=float)
        for s in range(n_species):
            mask = rdf["species"].to_numpy(dtype=int) == s
            if not np.any(mask):
                # Leave zeros if unseen in training; predictions for unseen species will fail below
                continue
            species_features[s] = rdf.loc[mask, run_cov_cols].to_numpy(dtype=float).mean(axis=0)
        self._species_features = species_features

        for cluster_id in np.unique(self.species_cluster):
            cluster_rows = df[df["cluster"] == cluster_id]
            if cluster_rows.empty:
                continue
            X = self._build_features(
                cluster_rows["species"].to_numpy(),
                cluster_rows["compound"].to_numpy(),
            )
            y = cluster_rows["rt"].to_numpy(dtype=float)

            pipeline = self._build_pipeline(len(cluster_rows))
            pipeline.fit(X, y)

            self.cluster_models[int(cluster_id)] = ClusterModel(pipeline=pipeline)

        if not self.cluster_models:
            raise RuntimeError("No clusters had training data; cannot fit baseline model.")

        return self

    def predict(self, species_idx: Sequence[int], compound_idx: Sequence[int]) -> np.ndarray:
        species_arr = np.asarray(species_idx, dtype=int)
        compound_arr = np.asarray(compound_idx, dtype=int)
        if species_arr.shape != compound_arr.shape:
            raise ValueError("species_idx and compound_idx must have the same shape")

        cluster_assignments = self.species_cluster[species_arr]
        preds = np.zeros_like(species_arr, dtype=float)

        for cluster_id in np.unique(cluster_assignments):
            model = self.cluster_models.get(int(cluster_id))
            if model is None:
                raise ValueError(
                    f"Cluster {cluster_id} was not seen during training; cannot predict for it."
                )

            mask = cluster_assignments == cluster_id
            cluster_species = species_arr[mask]
            cluster_compounds = compound_arr[mask]
            X = self._build_features(cluster_species, cluster_compounds)
            preds[mask] = model.pipeline.predict(X)

        return preds

    def _build_pipeline(self, n_samples: int) -> Pipeline:
        scaler = StandardScaler()
        if n_samples >= 5:
            reg = LassoCV(
                cv=min(self.cv, n_samples),
                random_state=self.random_state,
                max_iter=20000,
            )
        else:
            reg = Lasso(
                alpha=self.alpha if self.alpha is not None else 0.01,
                max_iter=50000,
            )
        return Pipeline([("scaler", scaler), ("regressor", reg)])

    def _build_features(
        self,
        species_idx: np.ndarray,
        compound_idx: np.ndarray,
    ) -> np.ndarray:
        if self._species_features is None:
            raise RuntimeError("Baseline must be fit before calling predict/build_features")
        return self._species_features[species_idx]


class ClusterCompoundLassoBaseline:
    """Per (species cluster × compound) Lasso on run covariates.

    - Fits an independent model for each (cluster, compound) using per-run covariates.
    - Rows without a fitted per-compound model remain NaN so downstream plots reflect coverage.
    """

    def __init__(
        self,
        species_cluster: np.ndarray,
        *,
        alpha: float | None = None,
        cv: int = 5,
        min_samples: int = 3,
        random_state: int = 42,
    ) -> None:
        self.species_cluster = np.asarray(species_cluster, dtype=int)
        self.alpha = alpha
        self.cv = max(2, int(cv))
        self.min_samples = max(1, int(min_samples))
        self.random_state = int(random_state)
        self.models: Dict[tuple[int, int], ClusterModel] = {}
        self._run_features: np.ndarray | None = None
        self._covariate_columns: list[str] | None = None

    def _resolve_run_matrix(
        self, df: pd.DataFrame, run_df: pd.DataFrame | None, covariate_columns: Sequence[str] | None
    ) -> tuple[np.ndarray, list[str]]:
        if run_df is not None:
            if covariate_columns is None:
                covs = [c for c in run_df.columns if c.startswith("run_covariate_")]
            else:
                covs = [str(c) for c in covariate_columns]
            rdf = run_df.copy()
        else:
            if covariate_columns is None:
                covs = [c for c in df.columns if c.startswith("run_covariate_")]
            else:
                covs = [str(c) for c in covariate_columns]
            required = {"run", *covs}
            missing = required.difference(df.columns)
            if missing:
                raise ValueError(
                    "run covariate columns required for baseline are missing from training data: "
                    f"{sorted(missing)}"
                )
            rdf = df[["run", *covs]].drop_duplicates(subset=["run"]).copy()  # type: ignore[arg-type]

        if "run" not in rdf.columns:
            raise ValueError("run_df must include a 'run' column")
        max_run = int(pd.to_numeric(rdf["run"]).max())
        X = np.zeros((max_run + 1, len(covs)), dtype=float)
        X[pd.to_numeric(rdf["run"]).to_numpy(dtype=int)] = rdf[covs].to_numpy(dtype=float)
        return X, covs

    def _make_pipeline(self, n_samples: int) -> Pipeline:
        scaler = StandardScaler()
        if n_samples >= 5:
            reg = LassoCV(
                cv=min(self.cv, n_samples),
                random_state=self.random_state,
                max_iter=20000,
            )
        else:
            reg = Lasso(
                alpha=self.alpha if self.alpha is not None else 0.01,
                max_iter=50000,
            )
        return Pipeline([("scaler", scaler), ("regressor", reg)])

    def fit(
        self,
        train_df: pd.DataFrame,
        *,
        run_df: pd.DataFrame | None = None,
        covariate_columns: Sequence[str] | None = None,
    ) -> "ClusterCompoundLassoBaseline":
        df = train_df.copy()
        df["species"] = df["species"].astype(int)
        df["compound"] = df["compound"].astype(int)
        df["run"] = df["run"].astype(int)
        df["cluster"] = self.species_cluster[df["species"].to_numpy()]

        run_features, covs = self._resolve_run_matrix(df, run_df, covariate_columns)
        self._run_features = run_features
        self._covariate_columns = covs

        # Fit per (cluster, compound) models
        for (cl, cid), sub in df.groupby(["cluster", "compound"], sort=False):
            X = run_features[sub["run"].to_numpy(dtype=int)]
            y = sub["rt"].to_numpy(dtype=float)
            pl = self._make_pipeline(len(sub))
            pl.fit(X, y)
            self.models[(int(cl), int(cid))] = ClusterModel(pipeline=pl)

        return self

    def predict(
        self,
        species_idx: Sequence[int],
        compound_idx: Sequence[int],
        run_idx: Sequence[int],
    ) -> np.ndarray:
        if self._run_features is None:
            raise RuntimeError("Baseline must be fit before predict")
        species_arr = np.asarray(species_idx, dtype=int)
        comp_arr = np.asarray(compound_idx, dtype=int)
        run_arr = np.asarray(run_idx, dtype=int)
        if not (species_arr.shape == comp_arr.shape == run_arr.shape):
            raise ValueError("species_idx, compound_idx, and run_idx must have the same shape")
        max_run = self._run_features.shape[0]
        if np.any(run_arr < 0) or np.any(run_arr >= max_run):
            raise ValueError("run_idx out of bounds for fitted run feature matrix")

        preds = np.zeros_like(run_arr, dtype=float)
        cl_assign = self.species_cluster[species_arr]
        for cl in np.unique(cl_assign):
            mask_cl = cl_assign == cl
            idxs = np.where(mask_cl)[0]
            for i in idxs:
                cid = int(comp_arr[i])
                key = (int(cl), cid)
                model = self.models.get(key)
                if model is None:
                    # No per-(cluster×compound) model: emit NaN rather than fallback to avoid skew
                    preds[i] = np.nan
                    continue
                x = self._run_features[int(run_arr[i])].reshape(1, -1)
                preds[i] = float(model.pipeline.predict(x)[0])

        return preds
