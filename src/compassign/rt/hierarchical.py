"""
Hierarchical Bayesian retention‑time prediction in PyMC (production profile).

This module provides the production configuration of the hierarchical RT model
used by the RT production pipeline (for example, ``train_rt_prod.py`` and
``run_rt_prod.sh``). It fixes the gamma structure and class hierarchy and does
not depend on the experimental model, so changes to experiment code cannot
silently affect production behaviour.
"""

from __future__ import annotations

import logging
import importlib.util
from typing import Dict, Any, List, Optional, Sequence, Tuple

try:  # Imported lazily where possible to avoid hard dependency at import time.
    import arviz as az
except Exception:  # pragma: no cover - ArviZ/numba may be unavailable in some environments
    az = None  # type: ignore
import numpy as np
import pandas as pd
import pymc as pm

from ..utils import ensure_run_metadata

logger = logging.getLogger(__name__)


class HierarchicalRTModel:
    """
    Production hierarchical Bayesian model for retention‑time prediction.

    This class implements a hierarchical configuration used by the RT pipeline:
    - cluster → species hierarchy,
    - optional descriptor‑based compound baseline (``compound_features``),
    - optional class hierarchy for compound baselines (``include_class_hierarchy``),
    - class‑pooled (default) or global run‑covariate coefficients γ (``global_gamma``),
    - non‑centred parameterisation with sum‑to‑zero constraints.
    """

    def __init__(
        self,
        n_clusters: int,
        n_species: int,
        n_classes: int,
        n_compounds: int,
        species_cluster: np.ndarray,
        compound_class: np.ndarray,
        *,
        run_features: Optional[np.ndarray | pd.DataFrame] = None,
        run_species: Optional[Sequence[int]] = None,
        run_metadata: Optional[pd.DataFrame] = None,
        run_covariate_columns: Optional[Sequence[str]] = None,
        compound_features: Optional[np.ndarray | pd.DataFrame] = None,
        include_class_hierarchy: bool = False,
        global_gamma: bool = False,
        use_clusters: bool = True,
        use_species: bool = True,
        class_only_gamma: bool = False,
        sigma_y_loc: Optional[float] = None,
        sigma_y_scale: Optional[float] = None,
        sigma_gamma_class_scale: Optional[float] = None,
        sigma_species_scale: Optional[float] = None,
        species_compound_intercept: bool = False,
        sigma_sc_scale: Optional[float] = None,
    ):
        """
        Initialize the production RT model.

        Parameters
        ----------
        n_clusters : int
            Number of species clusters.
        n_species : int
            Number of species.
        n_classes : int
            Number of compound classes.
        n_compounds : int
            Number of compounds.
        species_cluster : np.ndarray
            Mapping of species to clusters (shape (n_species,)).
        compound_class : np.ndarray
            Mapping of compounds to classes (shape (n_compounds,)).
        run_features : array-like or DataFrame, optional
            Run-level covariates of shape (n_runs × n_features). May also be provided as a
            DataFrame containing ``run``, ``species``, and covariate columns.
        run_species : Sequence[int], optional
            Species mapping for each run when ``run_features`` is an array.
        run_metadata : pd.DataFrame, optional
            Pre-extracted run metadata with ``run``, ``species``, and covariate columns. Takes
            precedence over ``run_features`` when provided.
        run_covariate_columns : Sequence[str], optional
            Explicit covariate column names to use. When omitted, columns prefixed with
            ``run_covariate_`` are inferred.
        compound_features : array-like or DataFrame, optional
            Compound descriptor matrix of shape (n_compounds × d). When omitted, the
            model falls back to a descriptor-free compound baseline.
        include_class_hierarchy : bool, optional
            If True, include an explicit class-pooled component in the compound baseline.
        global_gamma : bool, optional
            If True, use a single global γ vector for run covariates (no class pooling).
        use_clusters : bool, optional
            Whether to include the cluster → species hierarchy.
        use_species : bool, optional
            Whether to include species-level intercepts.
        class_only_gamma : bool, optional
            If True, use class-level γ slopes only (no per-compound random slopes).
        sigma_y_loc : float, optional
            Override prior mean for the log noise scale (log sigma_y) used for
            group-specific observation noise. When omitted, a data-informed
            default close to the typical cap20 posterior scale is used.
        sigma_y_scale : float, optional
            Override prior sigma for the log noise scale. When omitted, a
            moderately tight default is used to encourage realistic noise
            levels while allowing groups to deviate.
        sigma_gamma_class_scale : float, optional
            Override HalfNormal scale for gamma_class.
        sigma_species_scale : float, optional
            Override HalfNormal scale for species effects.
        species_compound_intercept : bool, optional
            If True, add a shrunken (species, compound) interaction intercept δ_sc.
        sigma_sc_scale : float, optional
            Override HalfNormal scale for δ_sc.
        """
        self.n_clusters = n_clusters
        self.n_species = n_species
        self.n_classes = n_classes
        self.n_compounds = n_compounds
        self.species_cluster = np.asarray(species_cluster, dtype=int)
        self.compound_class = np.asarray(compound_class, dtype=int)
        self.use_clusters = bool(use_clusters)
        self.use_species = bool(use_species)
        self.class_only_gamma = bool(class_only_gamma)
        self.include_class_hierarchy = bool(include_class_hierarchy)
        self.global_gamma = bool(global_gamma)
        self.sigma_y_loc = sigma_y_loc
        self.sigma_y_scale = sigma_y_scale
        self.sigma_gamma_class_scale = sigma_gamma_class_scale
        self.sigma_species_scale = sigma_species_scale
        self.species_compound_intercept = bool(species_compound_intercept)
        self.sigma_sc_scale = sigma_sc_scale

        run_meta_source = run_metadata if run_metadata is not None else None
        run_meta = ensure_run_metadata(
            run_features,
            run_species=run_species,
            peak_df=run_meta_source,
            covariate_columns=run_covariate_columns,
        )

        self.run_features: np.ndarray = run_meta.features
        self.run_species: np.ndarray = run_meta.species
        self.run_covariate_columns: List[str] = run_meta.covariate_columns
        self.n_runs: int = int(self.run_features.shape[0])
        self.n_covariates: int = int(self.run_features.shape[1])

        if compound_features is None:
            self.compound_features = np.zeros((n_compounds, 0), dtype=float)
        else:
            if isinstance(compound_features, pd.DataFrame):
                z_arr = compound_features.to_numpy(dtype=float)
            else:
                z_arr = np.asarray(compound_features, dtype=float)
            if z_arr.shape[0] != n_compounds:
                raise ValueError(
                    "compound_features must have shape (n_compounds, d)"
                    f" (got {z_arr.shape}, expected ({n_compounds}, d))"
                )
            self.compound_features = z_arr.astype(float, copy=False)

        # Species-compound interaction lookup (filled when the model is built).
        self._sc_lookup: Optional[np.ndarray] = None

        self.model: Optional[pm.Model] = None
        self.trace: Optional[az.InferenceData] = None
        self.ppc: Optional[Dict[str, Any]] = None

        self._cov_mean: Optional[np.ndarray] = None
        self._cov_std: Optional[np.ndarray] = None

        # Track which run-level covariates correspond to ES_* features so that
        # we can apply slightly tighter priors to their coefficients while
        # keeping IS_* coefficients on the original prior scale.
        self._es_feature_mask: Optional[np.ndarray] = None

        self._validate_core_dimensions()

    def _validate_core_dimensions(self) -> None:
        """Validate core array dimensions and index alignment."""
        if self.species_cluster.shape[0] != self.n_species:
            raise ValueError(
                "species_cluster length does not match n_species"
                f" ({self.species_cluster.shape[0]} != {self.n_species})"
            )
        if self.compound_class.shape[0] != self.n_compounds:
            raise ValueError(
                "compound_class length does not match n_compounds"
                f" ({self.compound_class.shape[0]} != {self.n_compounds})"
            )
        if self.run_species.shape[0] != self.n_runs:
            raise ValueError(
                "run_species length must match number of runs"
                f" ({self.run_species.shape[0]} != {self.n_runs})"
            )

    def _prepare_indices(self, obs_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate and extract integer indices from an RT observation DataFrame."""
        if obs_df is None or len(obs_df) == 0:
            raise ValueError(
                "HierarchicalRTModel received an empty observation DataFrame; "
                "no labeled compound RTs available to fit the model."
            )
        required_columns = {"species", "compound", "rt", "run"}
        missing_cols = required_columns.difference(obs_df.columns)
        if missing_cols:
            raise ValueError(
                "HierarchicalRTModel requires columns 'species', 'compound', 'rt', 'run'; "
                f"missing {sorted(missing_cols)}"
            )

        species_idx = obs_df["species"].to_numpy(dtype=int)
        compound_idx = obs_df["compound"].to_numpy(dtype=int)
        run_idx = obs_df["run"].to_numpy(dtype=int)

        if np.any(species_idx < 0) or np.any(species_idx >= self.n_species):
            raise ValueError("species indices must be in [0, n_species)")
        if np.any(compound_idx < 0) or np.any(compound_idx >= self.n_compounds):
            raise ValueError("compound indices must be in [0, n_compounds)")
        if np.any(run_idx < 0) or np.any(run_idx >= self.n_runs):
            raise ValueError("run indices must be in [0, n_runs)")

        if not np.all(np.isfinite(obs_df["rt"].to_numpy(dtype=float))):
            raise ValueError("Observed RT values must be finite")

        return species_idx, compound_idx, run_idx

    def build_model(self, obs_df: pd.DataFrame) -> pm.Model:
        """
        Build the production hierarchical RT model in PyMC.

        The model uses a non‑centred parameterisation with:
        - species effects pooled via clusters,
        - compound baselines pooled via chemical classes (no descriptors),
        - run‑level covariates standardised on the training runs and shared via
          class‑pooled compound‑specific coefficients ``gamma``.
        """
        species_idx, compound_idx, run_idx = self._prepare_indices(obs_df)

        # Map each observation to its species cluster (matrix group). Even when
        # the cluster → species mean hierarchy is disabled, we retain the
        # cluster indices as a natural grouping for the observation noise.
        cluster_idx = self.species_cluster[species_idx]

        unique_runs = np.unique(run_idx)
        cov_train = self.run_features[unique_runs]
        self._cov_mean = cov_train.mean(axis=0)
        self._cov_std = cov_train.std(axis=0) + 1e-8
        cov_std_all = (self.run_features - self._cov_mean) / self._cov_std

        # Identify ES_* features (if any) among the run-level covariates so we
        # can apply slightly tighter priors to their coefficients. This keeps
        # the baseline IS_* effects on their original prior scale while
        # regularising ES_* coefficients a bit more strongly.
        if self._es_feature_mask is None and self.run_covariate_columns:
            es_mask = np.array(
                [str(col).startswith("ES_") for col in self.run_covariate_columns],
                dtype=bool,
            )
            self._es_feature_mask = es_mask

        with pm.Model() as model:
            # Observation noise: group-specific (per cluster) scales on the log
            # scale. The prior is centred near the typical cap20 posterior
            # (~0.02 minutes) with moderate spread, and can be overridden via
            # sigma_y_loc / sigma_y_scale in log space.
            mu_log_sigma = np.log(0.02) if self.sigma_y_loc is None else float(self.sigma_y_loc)
            sigma_log_sigma = 0.4 if self.sigma_y_scale is None else float(self.sigma_y_scale)

            log_sigma_y_group = pm.Normal(
                "log_sigma_y_group",
                mu=mu_log_sigma,
                sigma=sigma_log_sigma,
                shape=self.n_clusters,
            )
            sigma_y_group = pm.Deterministic("sigma_y_group", pm.math.exp(log_sigma_y_group))
            # Scalar summary of the typical observation noise scale across
            # groups (used by downstream diagnostics that expect `sigma_y`).
            # Root-mean-square of group-specific noise scales as a scalar summary.
            pm.Deterministic("sigma_y", pm.math.sqrt(pm.math.mean(pm.math.sqr(sigma_y_group))))

            # Species/cluster effects (optionally disabled or simplified).
            sigma_species = (
                pm.HalfNormal(
                    "sigma_species",
                    sigma=0.5
                    if self.sigma_species_scale is None
                    else float(self.sigma_species_scale),
                )
                if self.use_species
                else None
            )
            if self.use_clusters:
                sigma_cluster = pm.HalfNormal("sigma_cluster", sigma=0.5)
                cluster_raw = pm.Normal("cluster_raw", 0.0, 1.0, shape=self.n_clusters)
                cluster_eff = pm.Deterministic(
                    "cluster_eff",
                    cluster_raw * sigma_cluster - pm.math.mean(cluster_raw * sigma_cluster),
                )
            else:
                cluster_eff = None

            if self.use_species:
                species_raw = pm.Normal("species_raw", 0.0, 1.0, shape=self.n_species)
                if cluster_eff is not None:
                    species_base = cluster_eff[self.species_cluster] + species_raw * sigma_species
                else:
                    species_base = species_raw * sigma_species
                species_eff = pm.Deterministic(
                    "species_eff", species_base - pm.math.mean(species_base)
                )
            else:
                species_eff = pm.Deterministic(
                    "species_eff", pm.math.zeros((self.n_species,), dtype="float64")
                )

            # Compound baselines (optional descriptors + optional class hierarchy)
            sigma_compound = pm.HalfNormal("sigma_compound", sigma=0.5)
            Z = np.asarray(self.compound_features, dtype=float)
            if Z.shape[1] > 0:
                z_mean = Z.mean(axis=0)
                z_std = Z.std(axis=0) + 1e-8
                Zs = (Z - z_mean) / z_std
                d = int(Zs.shape[1])
                tau_beta = pm.HalfNormal("tau_beta", sigma=0.6)
                theta_beta = pm.Normal("theta_beta", 0.0, tau_beta, shape=d)
                desc_term = pm.math.dot(Zs, theta_beta)
            else:
                desc_term = 0.0

            if self.include_class_hierarchy:
                sigma_class = pm.HalfNormal("sigma_class", sigma=0.5)
                class_raw = pm.Normal("class_raw", 0.0, 1.0, shape=self.n_classes)
                class_eff = pm.Deterministic(
                    "class_eff",
                    class_raw * sigma_class - pm.math.mean(class_raw * sigma_class),
                )
                delta_c_raw = pm.Normal("delta_c_raw", 0.0, 1.0, shape=self.n_compounds)
                delta_c = pm.Deterministic("delta_c", delta_c_raw * sigma_compound)
                beta_base = desc_term + class_eff[self.compound_class] + delta_c
            else:
                delta_c = pm.Normal("delta_c", mu=0.0, sigma=sigma_compound, shape=self.n_compounds)
                beta_base = desc_term + delta_c

            compound_eff = pm.Deterministic("compound_eff", beta_base - pm.math.mean(beta_base))

            mu0 = pm.Normal("mu0", float(obs_df["rt"].mean()), 5.0)

            # Run-level covariates
            if self.n_covariates > 0:
                cov_obs = cov_std_all[run_idx]

                # Per-feature scaling: ES_* covariates receive a slightly tighter prior.
                scale_vec = np.ones(self.n_covariates, dtype=float)
                if self._es_feature_mask is not None and self._es_feature_mask.any():
                    scale_vec[self._es_feature_mask] = 0.7

                if self.global_gamma:
                    sigma_gamma_global = pm.HalfNormal("sigma_gamma_global", sigma=0.5)
                    gamma_global_raw = pm.Normal(
                        "gamma_global_raw",
                        0.0,
                        1.0,
                        shape=(self.n_covariates,),
                    )
                    gamma_global = pm.Deterministic(
                        "gamma_global", gamma_global_raw * sigma_gamma_global * scale_vec
                    )
                    gamma = pm.Deterministic(
                        "gamma",
                        pm.math.tile(gamma_global, (self.n_compounds, 1)),
                    )
                else:
                    sigma_gamma_class = pm.HalfNormal(
                        "sigma_gamma_class",
                        sigma=0.5
                        if self.sigma_gamma_class_scale is None
                        else float(self.sigma_gamma_class_scale),
                    )
                    gamma_class_raw = pm.Normal(
                        "gamma_class_raw",
                        0.0,
                        1.0,
                        shape=(self.n_classes, self.n_covariates),
                    )
                    gamma_class = pm.Deterministic(
                        "gamma_class",
                        gamma_class_raw * sigma_gamma_class * scale_vec,
                    )
                    if self.class_only_gamma:
                        gamma = pm.Deterministic("gamma", gamma_class[self.compound_class])
                    else:
                        sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=0.5)
                        gamma_compound_raw = pm.Normal(
                            "gamma_compound_raw",
                            0.0,
                            1.0,
                            shape=(self.n_compounds, self.n_covariates),
                        )
                        gamma_mean = gamma_class[self.compound_class]
                        gamma = pm.Deterministic(
                            "gamma", gamma_mean + gamma_compound_raw * sigma_gamma
                        )

                cov_term = pm.math.sum(cov_obs * gamma[compound_idx], axis=1)
            else:
                cov_term = 0.0
                gamma = None  # type: ignore[assignment]

            # Species-compound interaction (optional)
            if self.species_compound_intercept:
                # Map observed (species, compound) pairs to a compact index.
                pairs = np.stack((species_idx, compound_idx), axis=1)
                pairs_struct = np.core.records.fromarrays([pairs[:, 0], pairs[:, 1]], names="s,c")
                unique_pairs, inv = np.unique(pairs_struct, return_inverse=True)
                sc_sigma = self.sigma_sc_scale if self.sigma_sc_scale is not None else 0.1
                sigma_sc = pm.HalfNormal("sigma_sc", sigma=float(sc_sigma))
                delta_sc_raw = pm.Normal("delta_sc_raw", 0.0, 1.0, shape=int(len(unique_pairs)))
                delta_sc = pm.Deterministic("delta_sc", delta_sc_raw * sigma_sc)
                sc_term = delta_sc[inv]

                # Build a lookup for predict_new (species × compound -> idx or -1).
                lookup = np.full((self.n_species, self.n_compounds), -1, dtype=int)
                for idx, (s_val, c_val) in enumerate(zip(unique_pairs["s"], unique_pairs["c"])):
                    lookup[int(s_val), int(c_val)] = idx
                self._sc_lookup = lookup
            else:
                sc_term = 0.0
                self._sc_lookup = None

            rt_mean = (
                mu0 + species_eff[species_idx] + compound_eff[compound_idx] + cov_term + sc_term
            )
            pm.Normal(
                "y_obs",
                mu=rt_mean,
                sigma=sigma_y_group[cluster_idx],
                observed=obs_df["rt"].values,
            )

        self.model = model
        return model

    def sample(
        self,
        n_samples: int = 1000,
        n_tune: int = 1000,
        n_chains: Optional[int] = None,
        cores: Optional[int] = None,
        target_accept: float = 0.8,
        max_treedepth: int = 10,
        random_seed: int = 42,
        init: str = "adapt_diag",
        verbose: bool = False,
        nuts_sampler: Optional[str] = None,
        chain_method: Optional[str] = None,
    ) -> az.InferenceData:
        """Sample from the posterior using NUTS."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        if verbose:
            chains_str = f"{n_chains} chains" if n_chains is not None else "default chains"
            logger.info(
                "Sampling hierarchical RT model with %s, %d draws, target_accept %.2f, max_treedepth %d",
                chains_str,
                n_samples,
                target_accept,
                max_treedepth,
            )

        selected_sampler = nuts_sampler
        selected_chain_method = chain_method
        if selected_sampler is None and self._jax_available():
            selected_sampler = "numpyro"
            if selected_chain_method is None:
                selected_chain_method = "vectorized"
            if verbose:
                logger.info("Using NumPyro NUTS via JAX backend.")
        elif verbose and selected_sampler:
            logger.info("Using explicit sampler override: %s", selected_sampler)

        sample_kwargs: Dict[str, Any] = {
            "draws": n_samples,
            "tune": n_tune,
            "target_accept": target_accept,
            "random_seed": random_seed,
            "init": init,
            "progressbar": True,
            "return_inferencedata": True,
            "nuts": {"max_treedepth": max_treedepth},
        }
        if n_chains is not None:
            sample_kwargs["chains"] = n_chains
        if cores is not None:
            sample_kwargs["cores"] = cores
        if selected_sampler:
            sample_kwargs["nuts_sampler"] = selected_sampler
        if selected_chain_method:
            sample_kwargs["chain_method"] = selected_chain_method

        with self.model:
            self.trace = pm.sample(**sample_kwargs)

        return self.trace

    @staticmethod
    def _jax_available() -> bool:
        """Detect whether JAX/NumPyro are importable without raising."""
        try:
            return bool(importlib.util.find_spec("jax") and importlib.util.find_spec("numpyro"))
        except Exception:  # pragma: no cover
            return False

    def posterior_predictive_check(
        self,
        obs_df: pd.DataFrame,
        random_seed: int = 42,
        max_ppc_draws: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform posterior predictive checks and compute summary diagnostics.
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model and trace must be available for PPC.")
        import pymc.sampling

        trace = self.trace

        if max_ppc_draws is not None and hasattr(trace, "posterior"):
            posterior = trace.posterior
            # ArviZ InferenceData uses (chain, draw) for posterior samples.
            n_draws = int(posterior.dims.get("draw", 0))
            n_chains = int(posterior.dims.get("chain", 1))
            total_draws = n_draws * n_chains

            if n_draws > 0 and max_ppc_draws < total_draws:
                rng = np.random.default_rng(int(random_seed))
                # Approximate max_ppc_draws total samples by selecting a subset of draws
                # per chain. Ensure at least one draw per chain.
                draws_per_chain = max(max_ppc_draws // max(n_chains, 1), 1)
                draws_per_chain = min(draws_per_chain, n_draws)
                draw_idx = np.sort(rng.choice(n_draws, size=draws_per_chain, replace=False))
                trace = trace.isel(draw=draw_idx)

        ppc = pymc.sampling.sample_posterior_predictive(
            trace,
            var_names=["y_obs"],
            random_seed=random_seed,
            model=self.model,
        )

        y_true = obs_df["rt"].to_numpy(dtype=float)
        y_pred = ppc.posterior_predictive["y_obs"].values
        y_sample = y_pred.reshape(-1, y_pred.shape[-1])

        pred_mean = y_sample.mean(axis=0)
        pred_std = y_sample.std(axis=0)
        lower = np.percentile(y_sample, 2.5, axis=0)
        upper = np.percentile(y_sample, 97.5, axis=0)

        residuals = pred_mean - y_true
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        mae = float(np.mean(np.abs(residuals)))
        coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))

        coverage_curve: Dict[str, float] = {}
        for level in (50, 80, 90, 95):
            lo = np.percentile(y_sample, (100 - level) / 2, axis=0)
            hi = np.percentile(y_sample, 100 - (100 - level) / 2, axis=0)
            coverage_curve[str(level)] = float(np.mean((y_true >= lo) & (y_true <= hi)))

        self.ppc = {
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "pred_lower_95": lower,
            "pred_upper_95": upper,
            "rmse": rmse,
            "mae": mae,
            "coverage_95": coverage,
            "residuals": residuals,
            "y_true": y_true,
            "coverage_curve": coverage_curve,
        }

        logger.info("RT prediction RMSE: %.3f", rmse)
        logger.info("RT prediction MAE: %.3f", mae)
        logger.info("95%% predictive interval coverage: %.1f%%", coverage * 100.0)
        logger.info("Posterior predictive checks for hierarchical RT model completed.")

        return self.ppc

    def predict_new(
        self,
        species_idx: np.ndarray,
        compound_idx: np.ndarray,
        run_idx: Optional[np.ndarray] = None,
        run_features: Optional[np.ndarray] = None,
        n_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict RT for new species-compound combinations.
        """
        if self.trace is None:
            raise ValueError("No trace available. Run sample() first.")

        if run_idx is not None:
            run_idx = np.asarray(run_idx, dtype=int)
            if np.any(run_idx < 0) or np.any(run_idx >= self.n_runs):
                raise ValueError("run_idx must be within [0, n_runs)")

        posterior = self.trace.posterior

        mu0 = posterior["mu0"].values.flatten()
        species_eff = posterior["species_eff"].values.reshape(-1, self.n_species)
        compound_eff = posterior["compound_eff"].values.reshape(-1, self.n_compounds)
        # Group-specific noise scales (per cluster). For older traces that only
        # have a single observation noise parameter (sigma_y), broadcast it
        # across clusters for backwards compatibility.
        if "sigma_y_group" in posterior:
            sigma_y_group = posterior["sigma_y_group"].values.reshape(-1, self.n_clusters)
        else:
            # Backwards compatibility for traces generated before the
            # introduction of group-specific noise.
            sigma_y = posterior["sigma_y"].values.flatten()
            sigma_y_group = np.repeat(sigma_y[:, None], self.n_clusters, axis=1)
        has_covariates = self.n_covariates > 0 and "gamma" in posterior
        has_sc = (
            self.species_compound_intercept
            and self._sc_lookup is not None
            and "delta_sc" in posterior
        )
        if has_covariates:
            gamma = posterior["gamma"].values.reshape(-1, self.n_compounds, self.n_covariates)

        if n_samples is not None:
            idx = np.random.choice(len(mu0), n_samples, replace=False)
            mu0 = mu0[idx]
            species_eff = species_eff[idx]
            compound_eff = compound_eff[idx]
            sigma_y_group = sigma_y_group[idx]
            if has_covariates:
                gamma = gamma[idx]
            if has_sc:
                delta_sc = posterior["delta_sc"].values.reshape(-1, posterior["delta_sc"].shape[-1])
                delta_sc = delta_sc[idx]
        else:
            if has_sc:
                delta_sc = posterior["delta_sc"].values.reshape(-1, posterior["delta_sc"].shape[-1])

        if has_covariates:
            if run_features is not None:
                cov_matrix = np.asarray(run_features, dtype=float)
                cov_std_all = (cov_matrix - self._cov_mean) / self._cov_std  # type: ignore[operator]
                if run_idx is not None and cov_std_all.shape[0] == self.n_runs:
                    cov_std = cov_std_all[run_idx]
                else:
                    cov_std = cov_std_all
            else:
                if run_idx is None:
                    raise ValueError(
                        "Provide either run_idx or run_features when covariates are enabled"
                    )
                cov_std = (self.run_features[run_idx] - self._cov_mean) / self._cov_std  # type: ignore[index,operator]

        n_pred = len(species_idx)
        n_samples_used = len(mu0)
        predictions = np.zeros((n_samples_used, n_pred))

        if has_sc:
            sc_idx = self._sc_lookup[species_idx, compound_idx]  # type: ignore[index]
            sc_mask = sc_idx >= 0

        # Map predictions to clusters for group-specific observation noise.
        cluster_idx = self.species_cluster[species_idx]

        for i in range(n_samples_used):
            pred = mu0[i] + species_eff[i, species_idx] + compound_eff[i, compound_idx]
            if has_covariates:
                pred = pred + np.sum(cov_std * gamma[i][compound_idx], axis=1)  # type: ignore[name-defined]
            if has_sc:
                sc_term = np.zeros_like(pred)
                if sc_mask.any():
                    sc_term[sc_mask] = delta_sc[i, sc_idx[sc_mask]]  # type: ignore[index]
                pred = pred + sc_term
            predictions[i] = pred

        pred_mean = predictions.mean(axis=0)
        param_var = predictions.var(axis=0)

        # Observation noise variance per prediction, averaged across posterior draws.
        obs_var = np.zeros(n_pred, dtype=float)
        for i in range(n_samples_used):
            sigma_obs = sigma_y_group[i, cluster_idx]
            obs_var += np.square(sigma_obs)
        obs_var /= float(n_samples_used)

        total_var = param_var + obs_var
        pred_std = np.sqrt(np.maximum(total_var, 0.0))

        return pred_mean, pred_std
