"""
Diagnostic plots for RT regression model using ArviZ.

This module provides functions to create various diagnostic plots
for assessing MCMC convergence and model fit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


def create_all_diagnostic_plots(trace: az.InferenceData, 
                               ppc_results: Dict[str, Any],
                               output_path: Path,
                               params_true: Optional[Dict[str, Any]] = None):
    """
    Create all diagnostic plots and save them.
    
    Parameters
    ----------
    trace : az.InferenceData
        Posterior samples from MCMC
    ppc_results : dict
        Results from posterior predictive checks
    output_path : Path
        Directory to save plots
    params_true : dict, optional
        True parameter values for comparison
    """
    plots_path = output_path / "plots" / "rt_model"
    plots_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating diagnostic plots...")
    
    # 1. Trace plots for key parameters
    plot_trace(trace, plots_path)
    
    # 2. Energy plot
    plot_energy(trace, plots_path)
    
    # 3. Posterior distributions vs true values
    if params_true:
        plot_posterior_vs_true(trace, params_true, plots_path)
    
    # 4. Pair plot for regression coefficients
    plot_pairs(trace, plots_path)
    
    # 5. Forest plot for hierarchical effects
    plot_forest(trace, plots_path)
    
    # 6. Posterior predictive checks
    plot_ppc(ppc_results, plots_path)
    
    # 7. Residual diagnostics
    plot_residuals(ppc_results, plots_path)
    
    # 8. R-hat and ESS diagnostics
    plot_rhat_ess(trace, plots_path)
    
    print(f"Diagnostic plots saved to: {plots_path}")


def plot_trace(trace: az.InferenceData, plots_path: Path):
    """Create trace plots for key parameters."""
    # Select key parameters to plot
    var_names = ['mu0', 'gamma', 'beta', 'sigma_y', 'sigma_cluster', 
                 'sigma_species', 'sigma_class', 'sigma_compound']
    
    # Filter to only existing variables
    available_vars = [v for v in var_names if v in trace.posterior]
    
    if available_vars:
        fig = az.plot_trace(
            trace,
            var_names=available_vars,
            compact=True,
            figsize=(12, len(available_vars) * 2)
        )
        plt.suptitle("Trace Plots - Key Parameters", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(plots_path / "trace_plots.png", dpi=100, bbox_inches='tight')
        plt.close()


def plot_energy(trace: az.InferenceData, plots_path: Path):
    """Create energy plot to diagnose sampling issues."""
    fig = az.plot_energy(trace, figsize=(10, 6))
    plt.suptitle("Energy Plot - Sampling Diagnostics", fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "energy_plot.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_posterior_vs_true(trace: az.InferenceData, params_true: Dict[str, Any], plots_path: Path):
    """Compare posterior distributions with true values."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()
    
    # Parameters to compare
    param_map = {
        'mu0': ('mu_true', 'Intercept (μ₀)'),
        'gamma': ('eta_true', 'Internal Std Coef (γ)'),
        'beta[0]': ('beta_true', 'β₁ (Descriptor 1)'),
        'beta[1]': ('beta_true', 'β₂ (Descriptor 2)'),
        'sigma_y': ('sigma_y_true', 'Observation Noise (σy)'),
        'sigma_cluster': ('sigma_cluster_true', 'Cluster SD (σ_cluster)'),
        'sigma_species': ('sigma_species_true', 'Species SD (σ_species)'),
        'sigma_compound': ('sigma_compound_true', 'Compound SD (σ_compound)')
    }
    
    for idx, (param_name, (true_key, label)) in enumerate(param_map.items()):
        ax = axes[idx]
        
        # Get posterior samples
        if 'beta' in param_name:
            beta_idx = int(param_name.split('[')[1].split(']')[0])
            if 'beta' in trace.posterior:
                samples = trace.posterior['beta'].values[:, :, beta_idx].flatten()
                true_val = params_true[true_key][beta_idx] if true_key in params_true else None
            else:
                continue
        else:
            if param_name in trace.posterior:
                samples = trace.posterior[param_name].values.flatten()
                true_val = params_true.get(true_key)
            else:
                continue
        
        # Plot posterior
        ax.hist(samples, bins=30, alpha=0.7, density=True, color='blue', edgecolor='black')
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        
        # Add true value line
        if true_val is not None:
            ax.axvline(true_val, color='red', linestyle='--', linewidth=2, label=f'True: {true_val:.2f}')
            ax.axvline(np.mean(samples), color='green', linestyle='-', linewidth=2, label=f'Post: {np.mean(samples):.2f}')
            ax.legend()
    
    plt.suptitle("Posterior Distributions vs True Values", fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "posterior_vs_true.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_pairs(trace: az.InferenceData, plots_path: Path):
    """Create pair plots for regression coefficients."""
    var_names = ['mu0', 'gamma']
    if 'beta' in trace.posterior:
        var_names.append('beta')
    
    fig = az.plot_pair(
        trace,
        var_names=var_names,
        divergences=True,
        figsize=(10, 10)
    )
    plt.suptitle("Pairs Plot - Fixed Effects", fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "pairs_plot.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_forest(trace: az.InferenceData, plots_path: Path):
    """Create forest plot for variance components."""
    var_names = ['sigma_y', 'sigma_cluster', 'sigma_species', 'sigma_class', 'sigma_compound']
    available_vars = [v for v in var_names if v in trace.posterior]
    
    if available_vars:
        fig = az.plot_forest(
            trace,
            var_names=available_vars,
            combined=True,
            figsize=(10, 6),
            r_hat=True,
            ess=True
        )
        plt.title("Forest Plot - Variance Components", fontsize=14)
        plt.tight_layout()
        plt.savefig(plots_path / "forest_plot.png", dpi=100, bbox_inches='tight')
        plt.close()


def plot_ppc(ppc_results: Dict[str, Any], plots_path: Path):
    """Plot posterior predictive checks."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Observed vs Predicted
    ax = axes[0, 0]
    ax.scatter(ppc_results['y_true'], ppc_results['pred_mean'], alpha=0.5)
    ax.plot([ppc_results['y_true'].min(), ppc_results['y_true'].max()], 
            [ppc_results['y_true'].min(), ppc_results['y_true'].max()], 
            'r--', label='Perfect prediction')
    ax.set_xlabel('Observed RT')
    ax.set_ylabel('Predicted RT')
    ax.set_title(f"Observed vs Predicted (RMSE: {ppc_results['rmse']:.3f})")
    ax.legend()
    
    # 2. Prediction intervals
    ax = axes[0, 1]
    sorted_idx = np.argsort(ppc_results['y_true'])
    ax.plot(range(len(sorted_idx)), ppc_results['y_true'][sorted_idx], 'ko', label='Observed', markersize=3)
    ax.fill_between(range(len(sorted_idx)), 
                    ppc_results['pred_lower_95'][sorted_idx],
                    ppc_results['pred_upper_95'][sorted_idx],
                    alpha=0.3, label='95% PI')
    ax.plot(range(len(sorted_idx)), ppc_results['pred_mean'][sorted_idx], 'r-', label='Predicted', linewidth=1)
    ax.set_xlabel('Observation (sorted)')
    ax.set_ylabel('RT')
    ax.set_title(f"95% Prediction Intervals (Coverage: {ppc_results['coverage_95']*100:.1f}%)")
    ax.legend()
    
    # 3. Residuals histogram
    ax = axes[1, 0]
    ax.hist(ppc_results['residuals'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    ax.set_title(f"Residual Distribution (Mean: {np.mean(ppc_results['residuals']):.3f})")
    
    # 4. Q-Q plot
    ax = axes[1, 1]
    from scipy import stats
    stats.probplot(ppc_results['residuals'], dist="norm", plot=ax)
    ax.set_title("Q-Q Plot of Residuals")
    
    plt.suptitle("Posterior Predictive Checks", fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "ppc_plots.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_residuals(ppc_results: Dict[str, Any], plots_path: Path):
    """Detailed residual diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(ppc_results['pred_mean'], ppc_results['residuals'], alpha=0.5)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted')
    
    # 2. Scale-Location plot
    ax = axes[0, 1]
    standardized_residuals = ppc_results['residuals'] / np.std(ppc_results['residuals'])
    ax.scatter(ppc_results['pred_mean'], np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('√|Standardized Residuals|')
    ax.set_title('Scale-Location Plot')
    
    # 3. Residuals by index
    ax = axes[1, 0]
    ax.plot(ppc_results['residuals'], 'o-', alpha=0.5, markersize=3)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Observation Index')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals by Order')
    
    # 4. ACF of residuals
    ax = axes[1, 1]
    from statsmodels.tsa.stattools import acf
    acf_values = acf(ppc_results['residuals'], nlags=20)
    ax.bar(range(len(acf_values)), acf_values)
    ax.axhline(0, color='black')
    ax.axhline(1.96/np.sqrt(len(ppc_results['residuals'])), color='red', linestyle='--', alpha=0.5)
    ax.axhline(-1.96/np.sqrt(len(ppc_results['residuals'])), color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.set_title('Residual Autocorrelation')
    
    plt.suptitle("Residual Diagnostics", fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "residual_diagnostics.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_rhat_ess(trace: az.InferenceData, plots_path: Path):
    """Plot R-hat and ESS diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get summary statistics
    summary = az.summary(trace)
    
    # 1. R-hat values
    ax = axes[0]
    rhat_vals = summary['r_hat'].dropna().sort_values(ascending=False)
    colors = ['red' if r > 1.01 else 'green' for r in rhat_vals]
    ax.barh(range(min(20, len(rhat_vals))), rhat_vals[:20], color=colors[:20])
    ax.axvline(1.01, color='red', linestyle='--', label='R-hat = 1.01')
    ax.set_yticks(range(min(20, len(rhat_vals))))
    ax.set_yticklabels(rhat_vals.index[:20])
    ax.set_xlabel('R-hat')
    ax.set_title('R-hat Values (Top 20)')
    ax.legend()
    
    # 2. ESS values
    ax = axes[1]
    ess_bulk = summary['ess_bulk'].dropna().sort_values()
    ax.barh(range(min(20, len(ess_bulk))), ess_bulk[:20])
    ax.axvline(100, color='red', linestyle='--', label='ESS = 100')
    ax.set_yticks(range(min(20, len(ess_bulk))))
    ax.set_yticklabels(ess_bulk.index[:20])
    ax.set_xlabel('ESS (bulk)')
    ax.set_title('Effective Sample Size (Bottom 20)')
    ax.legend()
    
    plt.suptitle("Convergence Diagnostics", fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "convergence_diagnostics.png", dpi=100, bbox_inches='tight')
    plt.close()