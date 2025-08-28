"""
Hierarchical Bayesian RT (Retention Time) prediction model using PyMC.

This module implements a hierarchical model for predicting retention times
with species clusters, compound classes, molecular descriptors, and internal standards.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, Any, Tuple, Optional


class HierarchicalRTModel:
    """
    Hierarchical Bayesian model for RT prediction.
    
    The model structure:
    - Species are grouped into clusters
    - Compounds are grouped into classes
    - RT depends on species, compound, molecular descriptors, and internal standard
    - Uses non-centered parameterization for efficient sampling
    """
    
    def __init__(self, 
                 n_clusters: int,
                 n_species: int,
                 n_classes: int,
                 n_compounds: int,
                 species_cluster: np.ndarray,
                 compound_class: np.ndarray,
                 descriptors: np.ndarray,
                 internal_std: np.ndarray):
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
        descriptors : np.ndarray
            Molecular descriptors matrix (n_compounds x n_features)
        internal_std : np.ndarray
            Internal standard RT for each species
        """
        self.n_clusters = n_clusters
        self.n_species = n_species
        self.n_classes = n_classes
        self.n_compounds = n_compounds
        self.species_cluster = species_cluster
        self.compound_class = compound_class
        self.descriptors = descriptors
        self.internal_std = internal_std
        self.model = None
        self.trace = None
        self.ppc = None
        
    def build_model(self, obs_df: pd.DataFrame, use_non_centered: bool = True) -> pm.Model:
        """
        Build the PyMC model.
        
        Parameters
        ----------
        obs_df : pd.DataFrame
            Observations with columns: species, compound, rt
        use_non_centered : bool
            If True, use non-centered parameterization for better sampling
            
        Returns
        -------
        pm.Model
            The constructed PyMC model
        """
        # Prepare indices and data for PyMC
        species_idx = obs_df['species'].values.astype(int)
        compound_idx = obs_df['compound'].values.astype(int)
        
        with pm.Model() as model:
            # Hyperpriors for random effect standard deviations
            # Use more informative priors based on typical RT ranges
            sigma_cluster = pm.Exponential('sigma_cluster', 1.0)
            sigma_species = pm.Exponential('sigma_species', 2.0)
            sigma_class = pm.Exponential('sigma_class', 1.0)
            sigma_compound = pm.Exponential('sigma_compound', 3.0)
            sigma_y = pm.Exponential('sigma_y', 2.0)
            
            if use_non_centered:
                # Non-centered parameterization (better for sampling)
                # Cluster effects
                cluster_raw = pm.Normal('cluster_raw', 0.0, 1.0, shape=self.n_clusters)
                cluster_eff = pm.Deterministic('cluster_eff', cluster_raw * sigma_cluster)
                
                # Class effects
                class_raw = pm.Normal('class_raw', 0.0, 1.0, shape=self.n_classes)
                class_eff = pm.Deterministic('class_eff', class_raw * sigma_class)
                
                # Species effects (hierarchical)
                species_raw = pm.Normal('species_raw', 0.0, 1.0, shape=self.n_species)
                species_eff = pm.Deterministic('species_eff', 
                                              cluster_eff[self.species_cluster] + species_raw * sigma_species)
                
                # Compound effects (hierarchical)
                compound_raw = pm.Normal('compound_raw', 0.0, 1.0, shape=self.n_compounds)
                compound_eff = pm.Deterministic('compound_eff',
                                               class_eff[self.compound_class] + compound_raw * sigma_compound)
            else:
                # Centered parameterization (original)
                cluster_eff = pm.Normal('cluster_eff', 0.0, sigma_cluster, shape=self.n_clusters)
                class_eff = pm.Normal('class_eff', 0.0, sigma_class, shape=self.n_classes)
                
                species_eff = pm.Normal('species_eff', 
                                      mu=cluster_eff[self.species_cluster], 
                                      sigma=sigma_species, 
                                      shape=self.n_species)
                compound_eff = pm.Normal('compound_eff', 
                                       mu=class_eff[self.compound_class], 
                                       sigma=sigma_compound, 
                                       shape=self.n_compounds)
            
            # Global intercept and regression coefficients
            mu0 = pm.Normal('mu0', 5.0, 2.0)  # Prior centered at typical RT value
            beta = pm.Normal('beta', 0.0, 2.0, shape=self.descriptors.shape[1])
            gamma = pm.Normal('gamma', 1.0, 0.5)  # coefficient for internal standard (tighter prior)
            
            # Expected retention time for each observation
            rt_mean = (mu0 
                      + species_eff[species_idx] 
                      + compound_eff[compound_idx] 
                      + pm.math.dot(self.descriptors[compound_idx], beta) 
                      + gamma * self.internal_std[species_idx])
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=rt_mean, sigma=sigma_y, observed=obs_df['rt'].values)
            
        self.model = model
        return model
    
    def sample(self, 
               n_samples: int = 1000, 
               n_tune: int = 1000, 
               n_chains: int = None,
               target_accept: float = 0.95,
               max_treedepth: int = 12,
               random_seed: int = 42,
               init: str = 'adapt_diag') -> az.InferenceData:
        """
        Sample from the posterior using NUTS.
        
        Parameters
        ----------
        n_samples : int
            Number of samples per chain
        n_tune : int
            Number of tuning steps
        n_chains : int, optional
            Number of MCMC chains (default: None, uses PyMC default)
        target_accept : float
            Target acceptance probability (higher = fewer divergences but slower)
        max_treedepth : int
            Maximum tree depth for NUTS
        random_seed : int
            Random seed for reproducibility
        init : str
            Initialization method ('adapt_diag', 'jitter+adapt_diag', etc.)
            
        Returns
        -------
        az.InferenceData
            ArviZ InferenceData object with posterior samples
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        chains_str = f"{n_chains} chains" if n_chains is not None else "default chains"
        print(f"Sampling with {chains_str}, {n_samples} samples each...")
        print(f"Target accept: {target_accept}, Max treedepth: {max_treedepth}")
        
        # Build sampling kwargs
        sample_kwargs = {
            'draws': n_samples,
            'tune': n_tune,
            'target_accept': target_accept,
            'max_treedepth': max_treedepth,
            'random_seed': random_seed,
            'init': init,
            'progressbar': True,
            'return_inferencedata': True  # Explicitly request InferenceData
        }
        
        # Only add chains if explicitly specified
        if n_chains is not None:
            sample_kwargs['chains'] = n_chains
        
        with self.model:
            self.trace = pm.sample(**sample_kwargs)
        
        # TODO: Re-enable divergence and convergence checks
        # Currently disabled due to hanging issues when accessing sample_stats
        # Need to investigate proper ArviZ data access patterns
        return self.trace
    
    def posterior_predictive_check(self, 
                                  obs_df: pd.DataFrame,
                                  random_seed: int = 42) -> Dict[str, Any]:
        """
        Perform posterior predictive checks.
        
        Parameters
        ----------
        obs_df : pd.DataFrame
            Original observations
        random_seed : int
            Random seed for reproducibility
            
        Returns
        -------
        dict
            Dictionary with PPC results including RMSE, coverage, predictions
        """
        if self.trace is None:
            raise ValueError("No trace available. Run sample() first.")
        
        print("Performing posterior predictive checks...")
        with self.model:
            self.ppc = pm.sample_posterior_predictive(
                self.trace, 
                var_names=['y_obs'], 
                random_seed=random_seed
            )
        
        # Extract predictions
        y_pred = self.ppc.posterior_predictive['y_obs'].values
        # Reshape if needed (chains, draws, observations) -> (draws, observations)
        if len(y_pred.shape) == 3:
            n_chains, n_draws, n_obs = y_pred.shape
            y_pred = y_pred.reshape(n_chains * n_draws, n_obs)
        
        # Compute predictive statistics
        pred_mean = y_pred.mean(axis=0)
        pred_std = y_pred.std(axis=0)
        pred_lower = np.percentile(y_pred, 2.5, axis=0)
        pred_upper = np.percentile(y_pred, 97.5, axis=0)
        
        # Compare to true observed RT
        y_true = obs_df['rt'].values
        rmse = np.sqrt(np.mean((pred_mean - y_true)**2))
        mae = np.mean(np.abs(pred_mean - y_true))
        coverage = np.mean((y_true >= pred_lower) & (y_true <= pred_upper))
        
        # Residuals
        residuals = y_true - pred_mean
        
        results = {
            'rmse': rmse,
            'mae': mae,
            'coverage_95': coverage,
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'pred_lower_95': pred_lower,
            'pred_upper_95': pred_upper,
            'residuals': residuals,
            'y_true': y_true
        }
        
        print(f"RT prediction RMSE: {rmse:.3f}")
        print(f"RT prediction MAE: {mae:.3f}")
        print(f"95% predictive interval coverage: {coverage*100:.1f}%")
        
        return results
    
    def get_parameter_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for model parameters.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics for all parameters
        """
        if self.trace is None:
            raise ValueError("No trace available. Run sample() first.")
        
        return az.summary(self.trace)
    
    def predict_new(self, 
                   species_idx: np.ndarray,
                   compound_idx: np.ndarray,
                   n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict RT for new species-compound combinations.
        
        Parameters
        ----------
        species_idx : np.ndarray
            Species indices for predictions
        compound_idx : np.ndarray
            Compound indices for predictions
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
        
        # Extract posterior samples
        posterior = self.trace.posterior
        
        # Get parameter samples (combine chains)
        mu0 = posterior['mu0'].values.flatten()
        beta = posterior['beta'].values.reshape(-1, self.descriptors.shape[1])
        gamma = posterior['gamma'].values.flatten()
        species_eff = posterior['species_eff'].values.reshape(-1, self.n_species)
        compound_eff = posterior['compound_eff'].values.reshape(-1, self.n_compounds)
        
        if n_samples is not None:
            # Subsample if requested
            idx = np.random.choice(len(mu0), n_samples, replace=False)
            mu0 = mu0[idx]
            beta = beta[idx]
            gamma = gamma[idx]
            species_eff = species_eff[idx]
            compound_eff = compound_eff[idx]
        
        # Compute predictions for each posterior sample
        n_pred = len(species_idx)
        n_samples_used = len(mu0)
        predictions = np.zeros((n_samples_used, n_pred))
        
        for i in range(n_samples_used):
            pred = (mu0[i] 
                   + species_eff[i, species_idx]
                   + compound_eff[i, compound_idx]
                   + np.dot(self.descriptors[compound_idx], beta[i])
                   + gamma[i] * self.internal_std[species_idx])
            predictions[i] = pred
        
        pred_mean = predictions.mean(axis=0)
        pred_std = predictions.std(axis=0)
        
        return pred_mean, pred_std