"""
Hierarchical Bayesian retention‑time prediction in PyMC (experimental variant).

This module contains the fully parameterised hierarchical RT model used in
descriptor and covariate‑shift experiments. It supports:

- optional descriptor‑based compound baselines via ``compound_features``,
- class‑based compound baselines when descriptors are omitted,
- optional class pooling for compound baselines (``include_class_hierarchy``),
- either global or class‑pooled run‑covariate effects (``global_gamma``).

Production code should import ``HierarchicalRTModel`` from
``rt_hierarchical`` instead, which wraps this implementation with fixed
defaults suitable for the RT production pipeline.
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

from .run_metadata import ensure_run_metadata, RunMetadataError

logger = logging.getLogger(__name__)


class HierarchicalRTModel:
    """
    Hierarchical Bayesian model for retention‑time prediction with run covariates.

    This class pools information across species and compound groups and incorporates
    run‑level covariates to explain systematic shifts. It uses a non‑centred NUTS‑friendly
    parameterisation with sum‑to‑zero constraints on group effects so that the global
    intercept (``mu0``) remains interpretable as a grand mean.
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
        global_gamma: bool = False,
        include_class_hierarchy: bool = True,
    ):
        """
        Initialize the RT model.

        Parameters
        ----------
        n_clusters : int
            Number of species clusters
        n_species : int
            Number of species
        n_classes : int
            Number of compound classes
        n_compounds : int
            Number of compounds
        species_cluster : np.ndarray
            Mapping of species to clusters
        compound_class : np.ndarray
            Mapping of compounds to classes
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
            Optional compound descriptor matrix of shape (n_compounds × d). When provided,
            the compound baseline is modelled as β_c = Z θ_β + residual (non‑centred),
            with Z standardised internally. When omitted, the class‑based hierarchy is used.
        global_gamma : bool, default False
            When True, model γ with a global prior (γ = γ0 + ε), i.e., no chemistry
            structure. When False (default), pool γ within chemistry clusters using
            ``compound_class``: γ[c] ~ N(γ_class[class(c)], σ_γ).
        include_class_hierarchy : bool, default True
            When descriptors are omitted (``compound_features`` is None), controls whether
            to include the class-level hierarchy for compounds. If False, compounds have
            only a residual deviation (no class intercept). Ignored when descriptors are provided.
        """
        self.n_clusters = n_clusters
        self.n_species = n_species
        self.n_classes = n_classes
        self.n_compounds = n_compounds
        self.species_cluster = np.asarray(species_cluster, dtype=int)
        self.compound_class = np.asarray(compound_class, dtype=int)

        run_meta_source = run_metadata if run_metadata is not None else None
        try:
            run_meta = ensure_run_metadata(
                run_features,
                run_species=run_species,
                peak_df=run_meta_source,
                covariate_columns=run_covariate_columns,
            )
        except RunMetadataError as exc:  # pragma: no cover - defensive logging
            msg = f"Failed to construct run metadata: {exc}"
            logger.error(msg)
            raise

        self.run_features: np.ndarray = run_meta.features
        self.run_species: np.ndarray = run_meta.species
        self.run_covariate_columns: List[str] = run_meta.covariate_columns
        self.n_runs: int = int(self.run_features.shape[0])
        self.n_covariates: int = int(self.run_features.shape[1])
        # Optional compound descriptors
        if compound_features is not None:
            if isinstance(compound_features, pd.DataFrame):
                z_arr = compound_features.to_numpy(dtype=float)
            else:
                z_arr = np.asarray(compound_features, dtype=float)
            if z_arr.shape[0] != n_compounds:
                raise ValueError(
                    "compound_features must have shape (n_compounds, d)"
                    f" (got {z_arr.shape}, expected ({n_compounds}, d))"
                )
            self.compound_features: Optional[np.ndarray] = z_arr
        else:
            self.compound_features = None

        self.global_gamma: bool = bool(global_gamma)
        self.include_class_hierarchy: bool = bool(include_class_hierarchy)

        self.model: Optional[pm.Model] = None
        self.trace: Optional[az.InferenceData] = None
        self.ppc: Optional[Dict[str, Any]] = None
        self._cov_mean: Optional[np.ndarray] = None
        self._cov_std: Optional[np.ndarray] = None

        self._validate_core_dimensions()

    def _validate_core_dimensions(self) -> None:
        """Validate core dimensions and index mappings."""
        if self.species_cluster.shape[0] != self.n_species:
            raise ValueError(
                "species_cluster length must match n_species"
                f" ({self.species_cluster.shape[0]} != {self.n_species})"
            )
        if self.compound_class.shape[0] != self.n_compounds:
            raise ValueError(
                "compound_class length must match n_compounds"
                f" ({self.compound_class.shape[0]} != {self.n_compounds})"
            )
        if self.run_species.shape[0] != self.n_runs:
            raise ValueError(
                "run_species length must match number of runs"
                f" ({self.run_species.shape[0]} != {self.n_runs})"
            )

    def _prepare_indices(self, obs_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Validate and extract integer index arrays from an observation DataFrame.

        This helper enforces the presence of the required columns, checks index
        bounds against the model dimensions, and verifies finite observed RTs.
        It returns numpy arrays suitable for advanced indexing within PyMC.
        """
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
        Build the hierarchical RT model in PyMC (non‑centred parameterisation).

        Notation and data
        ------------------
        Observations are retention times `y_i` for rows `i = 1..N` with indices
        `(species[i], compound[i], run[i])`. Each run has a covariate vector
        `x[run[i], :]` (e.g., internal‑standard panel), which we standardise
        using only the runs present in `obs_df`.

        Mean structure
        --------------
        The linear predictor decomposes into a global intercept, species and
        compound effects, and compound‑specific covariate coefficients applied to
        the standardised run covariates:

            y_i ~ Normal(mu_i, sigma_y)
            mu_i = mu0
                   + species_eff[species[i]]
                   + compound_eff[compound[i]]
                   + < x_std[run[i], :], gamma[compound[i], :] >

        Hierarchical structure
        ----------------------
        Species share information via clusters and compounds via classes. We place
        hierarchical priors so that sparse groups borrow strength from their
        parent levels:

            cluster_eff ~ Normal(0, sigma_cluster) (centred to sum to 0)
            class_eff   ~ Normal(0, sigma_class)   (centred to sum to 0)

            species_base  ~ Normal(cluster_eff[species_cluster], sigma_species)
            compound_base ~ Normal(class_eff[compound_class],   sigma_compound)

        Centred vs non‑centred
        ----------------------
        A strictly centred implementation samples `species_base` and
        `compound_base` directly from the Normal distributions above. When the
        group‑level scales (`sigma_species`, `sigma_compound`) are small, the
        posterior can exhibit funnel geometry, yielding slow exploration and
        divergent transitions.

        We therefore use a non‑centred parameterisation: draw standard normals
        `species_raw` and `compound_raw` and transform them as

            species_base  = cluster_eff[species_cluster] + species_raw  * sigma_species
            compound_base = class_eff[compound_class]    + compound_raw * sigma_compound

        This preserves the prior while improving sampler geometry and mixing.
        We subsequently centre the realised effects to zero so that the
        intercept `mu0` remains interpretable as the grand mean across all
        species and compounds.

        Covariate effects and shrinkage
        --------------------------------
        The covariate coefficients `gamma` are specific to each compound but
        shrink towards a class‑level prior `gamma_class[class, :]`. This achieves
        within‑class sharing while allowing compound‑level flexibility. Covariates
        are standardised to zero mean and unit scale on the training runs to
        improve numerical stability and interpretability.

        (Run random effects are intentionally omitted to keep the model compact
        and better aligned with generalisation to unseen runs.)
        """
        species_idx, compound_idx, run_idx = self._prepare_indices(obs_df)

        # Standardise run‑level covariates using only the runs present in obs_df.
        # The same mean/std are reused at prediction time to avoid data leakage.
        unique_runs = np.unique(run_idx)
        cov_train = self.run_features[unique_runs]
        self._cov_mean = cov_train.mean(axis=0)
        self._cov_std = cov_train.std(axis=0) + 1e-8
        cov_std_all = (self.run_features - self._cov_mean) / self._cov_std
        self._is_mean = self._cov_mean
        self._is_std = self._cov_std

        with pm.Model() as model:
            # Noise scales and global observation noise. The observation noise is
            # parameterised on the log scale (log_sigma_y) for numerical stability.
            sigma_cluster = pm.HalfNormal("sigma_cluster", sigma=0.5)
            sigma_species = pm.HalfNormal("sigma_species", sigma=0.5)
            sigma_compound = pm.HalfNormal("sigma_compound", sigma=0.5)
            log_sigma_y = pm.Normal("log_sigma_y", mu=np.log(0.1), sigma=0.5)
            sigma_y = pm.Deterministic("sigma_y", pm.math.exp(log_sigma_y))

            # Group‑level effects (non‑centred): clusters → species, classes → compounds.
            # We draw standard normals ("*_raw"), scale by the group‑level sigma, and
            # subtract the mean to impose a sum‑to‑zero constraint. This identifies the
            # intercept (mu0) and prevents arbitrary shifts in group means.
            cluster_raw = pm.Normal("cluster_raw", 0.0, 1.0, shape=self.n_clusters)
            cluster_eff = pm.Deterministic(
                "cluster_eff",
                cluster_raw * sigma_cluster - pm.math.mean(cluster_raw * sigma_cluster),
            )

            # Map from groups to leaves with non‑centred effects. The species effect is
            # the sum of the species' cluster effect and a species‑specific deviation.
            # We subtract the mean again at the leaf level to ensure the combined effect
            # is centred across species, keeping mu0 interpretable.
            species_raw = pm.Normal("species_raw", 0.0, 1.0, shape=self.n_species)
            species_base = cluster_eff[self.species_cluster] + species_raw * sigma_species
            species_eff = pm.Deterministic(
                "species_eff", species_base - pm.math.mean(species_base)
            )

            # Compound baseline: descriptor-informed if features provided; otherwise class-based
            if self.compound_features is not None:
                Z = np.asarray(self.compound_features, dtype=float)
                z_mean = Z.mean(axis=0)
                z_std = Z.std(axis=0) + 1e-8
                Zs = (Z - z_mean) / z_std
                d = int(Zs.shape[1])
                tau_beta = pm.HalfNormal("tau_beta", sigma=0.6)
                theta_beta = pm.Normal("theta_beta", 0.0, tau_beta, shape=d)
                # Simple normal residual around Z·θβ (no optional gating or heavy tails)
                delta_c = pm.Normal(
                    "delta_c", mu=0.0, sigma=sigma_compound, shape=self.n_compounds
                )
                beta_base = pm.math.dot(Zs, theta_beta) + delta_c
                compound_eff = pm.Deterministic(
                    "compound_eff", beta_base - pm.math.mean(beta_base)
                )
            else:
                compound_raw = pm.Normal("compound_raw", 0.0, 1.0, shape=self.n_compounds)
                if self.include_class_hierarchy:
                    sigma_class = pm.HalfNormal("sigma_class", sigma=0.5)
                    class_raw = pm.Normal("class_raw", 0.0, 1.0, shape=self.n_classes)
                    class_eff = pm.Deterministic(
                        "class_eff",
                        class_raw * sigma_class - pm.math.mean(class_raw * sigma_class),
                    )
                    compound_base = class_eff[self.compound_class] + compound_raw * sigma_compound
                else:
                    compound_base = compound_raw * sigma_compound
                compound_eff = pm.Deterministic(
                    "compound_eff", compound_base - pm.math.mean(compound_base)
                )

            # Intercept and (optional) covariate effects. When there are no
            # covariates (n_covariates==0), we skip gamma-related variables and
            # contributions entirely to allow covariate ablations.
            mu0 = pm.Normal("mu0", float(obs_df["rt"].mean()), 5.0)

            if self.n_covariates > 0:
                sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=0.5)
                gamma_compound_raw = pm.Normal(
                    "gamma_compound_raw",
                    0.0,
                    1.0,
                    shape=(self.n_compounds, self.n_covariates),
                )
                cov_obs = cov_std_all[run_idx]

                if self.global_gamma:
                    # γ = γ0 + ε (no chemistry structure)
                    sigma_gamma0 = pm.HalfNormal("sigma_gamma0", sigma=0.5)
                    gamma0 = pm.Normal(
                        "gamma0", 0.0, sigma_gamma0, shape=self.n_covariates
                    )
                    gamma = pm.Deterministic(
                        "gamma", gamma0 + gamma_compound_raw * sigma_gamma
                    )
                else:
                    # Pool γ within chemistry classes derived from embeddings
                    sigma_gamma_class = pm.HalfNormal("sigma_gamma_class", sigma=0.5)
                    gamma_class = pm.Normal(
                        "gamma_class",
                        0.0,
                        sigma_gamma_class,
                        shape=(self.n_classes, self.n_covariates),
                    )
                    gamma_mean = gamma_class[self.compound_class]
                    gamma = pm.Deterministic(
                        "gamma", gamma_mean + gamma_compound_raw * sigma_gamma
                    )

                cov_term = pm.math.sum(cov_obs * gamma[compound_idx], axis=1)
            else:
                cov_term = 0.0

            # Observation model: linear predictor + Gaussian noise.
            rt_mean = mu0 + species_eff[species_idx] + compound_eff[compound_idx] + cov_term
            pm.Normal("y_obs", mu=rt_mean, sigma=sigma_y, observed=obs_df["rt"].values)

        self.model = model
        return model

    def sample(
        self,
        n_samples: int = 1000,
        n_tune: int = 1000,
        n_chains: int | None = None,
        cores: int | None = None,
        target_accept: float = 0.8,
        max_treedepth: int = 10,
        random_seed: int = 42,
        init: str = "adapt_diag",
        verbose: bool = False,
        nuts_sampler: Optional[str] = None,
        chain_method: Optional[str] = None,
    ) -> az.InferenceData:
        """
        Sample from the posterior using NUTS.

        Parameters
        ----------
        n_samples : int
            Number of samples per chain
        n_tune : int
            Number of tuning steps
        n_chains : int, optional
            Number of MCMC chains (default: PyMC default)
        cores : int, optional
            Number of worker processes for parallel chains. If None, PyMC defaults
            to min(chains, CPU cores, 4). Set cores=n_chains to force full parallelism.
        target_accept : float
            Target acceptance probability (higher = fewer divergences but slower)
        max_treedepth : int
            Maximum tree depth for NUTS
        random_seed : int
            Random seed for reproducibility
        init : str
            Initialization method ('adapt_diag', 'jitter+adapt_diag', etc.)
        nuts_sampler : str, optional
            NUTS backend to use. When omitted, attempts to use NumPyro via JAX if available,
            otherwise falls back to PyMC's default sampler.
        chain_method : str, optional
            Chain execution strategy ('parallel', 'vectorized', etc.). When using NumPyro,
            defaults to 'vectorized' if not provided.

        Returns
        -------
        az.InferenceData
            ArviZ InferenceData object with posterior samples
        """
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

        # Prefer JAX/NumPyro when available unless explicitly overridden.
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

        # Build sampling kwargs
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

        # Only add chains if explicitly specified
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
        except Exception:  # pragma: no cover - defensive
            return False

    def posterior_predictive_check(
        self,
        obs_df: pd.DataFrame,
        random_seed: int = 42,
        max_ppc_draws: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform posterior predictive checks and compute summary diagnostics.

        Parameters
        ----------
        obs_df : pd.DataFrame
            Original observations (used only as a container for the observed RTs)
        random_seed : int
            Random seed for reproducibility
        max_ppc_draws : int, optional
            Maximum posterior predictive draws to use. If provided, a subset of draws is
            sampled to reduce memory and compute.
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model and trace must be available for PPC.")
        import pymc.sampling

        trace = self.trace

        if max_ppc_draws is not None and hasattr(trace, "posterior"):
            posterior = trace.posterior
            n_draws = int(posterior.dims.get("draw", 0))
            n_chains = int(posterior.dims.get("chain", 1))
            total_draws = n_draws * n_chains

            if n_draws > 0 and max_ppc_draws < total_draws:
                rng = np.random.default_rng(int(random_seed))
                draws_per_chain = max(max_ppc_draws // max(n_chains, 1), 1)
                draws_per_chain = min(draws_per_chain, n_draws)
                draw_idx = np.sort(
                    rng.choice(n_draws, size=draws_per_chain, replace=False)
                )
                trace = trace.isel(draw=draw_idx)

        ppc = pymc.sampling.sample_posterior_predictive(
            trace,
            var_names=["y_obs"],
            random_seed=random_seed,
        )

        y_true = obs_df["rt"].to_numpy(dtype=float)
        y_pred = ppc.posterior_predictive["y_obs"].values  # (chain, draw, obs)
        y_sample = y_pred.reshape(-1, y_pred.shape[-1])  # (samples, obs)

        pred_mean = y_sample.mean(axis=0)
        lower = np.percentile(y_sample, 2.5, axis=0)
        upper = np.percentile(y_sample, 97.5, axis=0)

        residuals = pred_mean - y_true
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        mae = float(np.mean(np.abs(residuals)))

        coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
        coverage_curve = {}
        for level in (50, 80, 90, 95):
            lo = np.percentile(y_sample, (100 - level) / 2, axis=0)
            hi = np.percentile(y_sample, 100 - (100 - level) / 2, axis=0)
            coverage_curve[str(level)] = float(np.mean((y_true >= lo) & (y_true <= hi)))

        self.ppc = {
            "pred_mean": pred_mean,
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

        Parameters
        ----------
        species_idx : np.ndarray
            Species indices for predictions
        compound_idx : np.ndarray
            Compound indices for predictions
        run_idx : np.ndarray, optional
            Run indices matching predictions (required when run-level features are enabled)
        run_features : np.ndarray, optional
            Optional override for the run-level feature matrix used at prediction time
        n_samples : int, optional
            Number of posterior samples to use (None for all)

        Returns
        -------
        pred_mean : np.ndarray
            Mean predictions
        pred_std : np.ndarray
            Standard deviation of predictions
        """
        if self.trace is None:
            raise ValueError("No trace available. Run sample() first.")

        # When predicting for known runs, `run_idx` provides per‑prediction run indices.
        # For truly new runs, pass `run_features` with one row per prediction; no
        # run‑indexed random effect is used in this model.
        if run_idx is not None:
            run_idx = np.asarray(run_idx, dtype=int)
            if np.any(run_idx < 0) or np.any(run_idx >= self.n_runs):
                raise ValueError("run_idx must be within [0, n_runs)")

        # Extract posterior samples
        posterior = self.trace.posterior

        # Get parameter samples (combine chains)
        mu0 = posterior["mu0"].values.flatten()
        species_eff = posterior["species_eff"].values.reshape(-1, self.n_species)
        compound_eff = posterior["compound_eff"].values.reshape(-1, self.n_compounds)
        sigma_y = posterior["sigma_y"].values.flatten()
        has_covariates = self.n_covariates > 0 and "gamma" in posterior
        if has_covariates:
            gamma = posterior["gamma"].values.reshape(-1, self.n_compounds, self.n_covariates)

        if n_samples is not None:
            idx = np.random.choice(len(mu0), n_samples, replace=False)
            mu0 = mu0[idx]
            species_eff = species_eff[idx]
            compound_eff = compound_eff[idx]
            sigma_y = sigma_y[idx]
            if has_covariates:
                gamma = gamma[idx]

        # Standardize covariates using the same parameters from build_model (if enabled)
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

        for i in range(n_samples_used):
            pred = mu0[i] + species_eff[i, species_idx] + compound_eff[i, compound_idx]
            if has_covariates:
                pred = pred + np.sum(cov_std * gamma[i][compound_idx], axis=1)  # type: ignore[name-defined]
            predictions[i] = pred

        pred_mean = predictions.mean(axis=0)
        param_var = predictions.var(axis=0)
        obs_var = float(np.mean(np.square(sigma_y)))
        total_var = param_var + obs_var
        pred_std = np.sqrt(np.maximum(total_var, 0.0))

        return pred_mean, pred_std
