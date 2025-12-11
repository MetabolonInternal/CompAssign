"""
Diagnostic plots for RT regression model using ArviZ.

This module provides functions to create various diagnostic plots
for assessing MCMC convergence and model fit.
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from pathlib import Path
from typing import Dict, Any, Optional

from scipy import stats

try:  # Optional dependency for residual autocorrelation
    from statsmodels.tsa.stattools import acf
except ImportError:  # pragma: no cover
    acf = None

warnings.filterwarnings("ignore")


def create_all_diagnostic_plots(
    trace: az.InferenceData,
    ppc_results: Dict[str, Any],
    output_path: Path,
    params_true: Optional[Dict[str, Any]] = None,
    obs_df: Optional[pd.DataFrame] = None,
):
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
    if ppc_results and "y_true" in ppc_results:
        plot_ppc(ppc_results, plots_path)
    else:
        print("  Skipping PPC plots (no PPC results available)")

    # 7. Residual diagnostics
    if ppc_results and "residuals" in ppc_results:
        plot_residuals(ppc_results, plots_path)
    else:
        print("  Skipping residual plots (no PPC results available)")

    # 8. R-hat and ESS diagnostics
    plot_rhat_ess(trace, plots_path)

    # 9. Coverage calibration curve (paper-quality)
    if ppc_results and "y_true" in ppc_results:
        plot_coverage_calibration(ppc_results, plots_path)

    # 10. Hierarchical pooling diagnostics (effect uncertainty vs data density)
    if obs_df is not None:
        try:
            plot_effect_uncertainty_vs_counts(trace, obs_df, plots_path)
            plot_effects_forest_subset(trace, obs_df, plots_path)
            plot_error_vs_prevalence(ppc_results, obs_df, plots_path)
        except Exception as e:  # pragma: no cover - plotting best-effort
            print(f"  Warning: pooling diagnostic plots skipped: {e}")

    print(f"Diagnostic plots saved to: {plots_path}")


def plot_trace(trace: az.InferenceData, plots_path: Path):
    """Create trace plots for key parameters."""
    # Select key parameters to plot
    var_names = [
        "mu0",
        "gamma",
        "beta",
        "sigma_y",
        "sigma_y_group",
        "sigma_cluster",
        "sigma_species",
        "sigma_class",
        "sigma_compound",
    ]

    # Filter to only existing variables
    available_vars = [v for v in var_names if v in trace.posterior]

    if available_vars:
        az.plot_trace(
            trace, var_names=available_vars, compact=True, figsize=(12, len(available_vars) * 2)
        )
        plt.suptitle("Trace Plots - Key Parameters", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(plots_path / "trace_plots.png", dpi=100, bbox_inches="tight")
        plt.close()


def plot_energy(trace: az.InferenceData, plots_path: Path):
    """Create energy plot to diagnose sampling issues."""
    az.plot_energy(trace, figsize=(10, 6))
    plt.suptitle("Energy Plot - Sampling Diagnostics", fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "energy_plot.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_posterior_vs_true(trace: az.InferenceData, params_true: Dict[str, Any], plots_path: Path):
    """Compare posterior distributions with true values."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()

    # Parameters to compare
    param_map = {
        "mu0": ("mu_true", "Intercept (μ₀)"),
        "gamma": ("eta_true", "Internal Std Coef (γ)"),
        "beta[0]": ("beta_true", "β₁ (Descriptor 1)"),
        "beta[1]": ("beta_true", "β₂ (Descriptor 2)"),
        "sigma_y": ("sigma_y_true", "Global Noise (σy)"),
        "sigma_cluster": ("sigma_cluster_true", "Cluster SD (σ_cluster)"),
        "sigma_species": ("sigma_species_true", "Species SD (σ_species)"),
        "sigma_compound": ("sigma_compound_true", "Compound SD (σ_compound)"),
    }

    for idx, (param_name, (true_key, label)) in enumerate(param_map.items()):
        ax = axes[idx]

        # Get posterior samples
        if "beta" in param_name:
            beta_idx = int(param_name.split("[")[1].split("]")[0])
            if "beta" in trace.posterior:
                samples = trace.posterior["beta"].values[:, :, beta_idx].flatten()
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
        ax.hist(samples, bins=30, alpha=0.7, density=True, color="blue", edgecolor="black")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")

        # Add true value line
        if true_val is not None:
            ax.axvline(
                true_val, color="red", linestyle="--", linewidth=2, label=f"True: {true_val:.2f}"
            )
            ax.axvline(
                np.mean(samples),
                color="green",
                linestyle="-",
                linewidth=2,
                label=f"Post: {np.mean(samples):.2f}",
            )
            ax.legend()

    plt.suptitle("Posterior Distributions vs True Values", fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "posterior_vs_true.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_pairs(trace: az.InferenceData, plots_path: Path):
    """Create pair plots for regression coefficients."""
    var_names = ["mu0", "gamma"]
    if "beta" in trace.posterior:
        var_names.append("beta")

    az.plot_pair(trace, var_names=var_names, divergences=True, figsize=(10, 10))
    plt.suptitle("Pairs Plot - Fixed Effects", fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "pairs_plot.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_forest(trace: az.InferenceData, plots_path: Path):
    """Create forest plot for variance components."""
    var_names = [
        "sigma_y",
        "sigma_y_group",
        "sigma_cluster",
        "sigma_species",
        "sigma_class",
        "sigma_compound",
    ]
    available_vars = [v for v in var_names if v in trace.posterior]

    if available_vars:
        az.plot_forest(
            trace,
            var_names=available_vars,
            combined=True,
            figsize=(10, 6),
            r_hat=True,
            ess=True,
        )
        plt.title("Forest Plot - Variance Components", fontsize=14)
        plt.tight_layout()
        plt.savefig(plots_path / "forest_plot.png", dpi=100, bbox_inches="tight")
        plt.close()


def plot_ppc(ppc_results: Dict[str, Any], plots_path: Path):
    """Plot posterior predictive checks."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Observed vs Predicted
    ax = axes[0, 0]
    ax.scatter(ppc_results["y_true"], ppc_results["pred_mean"], alpha=0.5)
    ax.plot(
        [ppc_results["y_true"].min(), ppc_results["y_true"].max()],
        [ppc_results["y_true"].min(), ppc_results["y_true"].max()],
        "r--",
        label="Perfect prediction",
    )
    ax.set_xlabel("Observed RT")
    ax.set_ylabel("Predicted RT")
    ax.set_title(f"Observed vs Predicted (RMSE: {ppc_results['rmse']:.3f})")
    ax.legend()

    # 2. Prediction intervals
    ax = axes[0, 1]
    sorted_idx = np.argsort(ppc_results["y_true"])
    ax.plot(
        range(len(sorted_idx)),
        ppc_results["y_true"][sorted_idx],
        "ko",
        label="Observed",
        markersize=3,
    )
    ax.fill_between(
        range(len(sorted_idx)),
        ppc_results["pred_lower_95"][sorted_idx],
        ppc_results["pred_upper_95"][sorted_idx],
        alpha=0.3,
        label="95% PI",
    )
    ax.plot(
        range(len(sorted_idx)),
        ppc_results["pred_mean"][sorted_idx],
        "r-",
        label="Predicted",
        linewidth=1,
    )
    ax.set_xlabel("Observation (sorted)")
    ax.set_ylabel("RT")
    ax.set_title(f"95% Prediction Intervals (Coverage: {ppc_results['coverage_95']*100:.1f}%)")
    ax.legend()

    # 3. Residuals histogram
    ax = axes[1, 0]
    ax.hist(ppc_results["residuals"], bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Residual Distribution (Mean: {np.mean(ppc_results['residuals']):.3f})")

    # 4. Q-Q plot
    ax = axes[1, 1]
    if stats is not None:
        stats.probplot(ppc_results["residuals"], dist="norm", plot=ax)
        ax.set_title("Q-Q Plot of Residuals")
    else:  # pragma: no cover - optional dependency
        ax.text(
            0.5,
            0.5,
            "scipy not installed\nQ-Q plot unavailable",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()

    plt.suptitle("Posterior Predictive Checks", fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "ppc_plots.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_residuals(ppc_results: Dict[str, Any], plots_path: Path):
    """Detailed residual diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(ppc_results["pred_mean"], ppc_results["residuals"], alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")

    # 2. Scale-Location plot
    ax = axes[0, 1]
    standardized_residuals = ppc_results["residuals"] / np.std(ppc_results["residuals"])
    ax.scatter(ppc_results["pred_mean"], np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("√|Standardized Residuals|")
    ax.set_title("Scale-Location Plot")

    # 3. Residuals by index
    ax = axes[1, 0]
    ax.plot(ppc_results["residuals"], "o-", alpha=0.5, markersize=3)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Observation Index")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals by Order")

    # 4. ACF of residuals
    ax = axes[1, 1]
    if acf is not None:
        acf_values = acf(ppc_results["residuals"], nlags=20)
        ax.bar(range(len(acf_values)), acf_values)
        ax.axhline(0, color="black")
        threshold = 1.96 / np.sqrt(len(ppc_results["residuals"]))
        ax.axhline(threshold, color="red", linestyle="--", alpha=0.5)
        ax.axhline(-threshold, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title("Residual Autocorrelation")
    else:  # pragma: no cover - optional dependency
        ax.text(
            0.5,
            0.5,
            "statsmodels not installed\nACF plot unavailable",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()

    plt.suptitle("Residual Diagnostics", fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "residual_diagnostics.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_rhat_ess(trace: az.InferenceData, plots_path: Path):
    """Plot R-hat and ESS diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Get summary statistics
    summary = az.summary(trace)

    # 1. R-hat values
    ax = axes[0]
    rhat_vals = summary["r_hat"].dropna().sort_values(ascending=False)
    colors = ["red" if r > 1.01 else "green" for r in rhat_vals]
    ax.barh(range(min(20, len(rhat_vals))), rhat_vals[:20], color=colors[:20])
    ax.axvline(1.01, color="red", linestyle="--", label="R-hat = 1.01")
    ax.set_yticks(range(min(20, len(rhat_vals))))
    ax.set_yticklabels(rhat_vals.index[:20])
    ax.set_xlabel("R-hat")
    ax.set_title("R-hat Values (Top 20)")
    ax.legend()

    # 2. ESS values
    ax = axes[1]
    ess_bulk = summary["ess_bulk"].dropna().sort_values()
    ax.barh(range(min(20, len(ess_bulk))), ess_bulk[:20])
    ax.axvline(100, color="red", linestyle="--", label="ESS = 100")
    ax.set_yticks(range(min(20, len(ess_bulk))))
    ax.set_yticklabels(ess_bulk.index[:20])
    ax.set_xlabel("ESS (bulk)")
    ax.set_title("Effective Sample Size (Bottom 20)")
    ax.legend()

    plt.suptitle("Convergence Diagnostics", fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "convergence_diagnostics.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_coverage_calibration(ppc_results: Dict[str, Any], plots_path: Path) -> None:
    """Plot empirical coverage vs nominal predictive interval levels."""
    levels = []
    coverages = []
    # Prefer exact curve if provided; else approximate using Normal quantiles
    if "coverage_curve" in ppc_results and isinstance(ppc_results["coverage_curve"], dict):
        for k, v in sorted(ppc_results["coverage_curve"].items(), key=lambda kv: float(kv[0])):
            levels.append(float(k))
            coverages.append(float(v))
    else:
        # Approximate using predictive mean/sd
        y_true = ppc_results["y_true"]
        mu = ppc_results["pred_mean"]
        sd = ppc_results["pred_std"]
        for lvl in [0.5, 0.8, 0.9, 0.95]:
            z = stats.norm.ppf((1.0 + lvl) / 2.0)
            lo = mu - z * sd
            hi = mu + z * sd
            cov = np.mean((y_true >= lo) & (y_true <= hi))
            levels.append(lvl)
            coverages.append(float(cov))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Ideal")
    ax.plot(levels, coverages, marker="o", label="Empirical")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Predictive Coverage Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_path / "coverage_calibration.png", dpi=120, bbox_inches="tight")
    plt.close()


def _posterior_mean_sd(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = array.mean(axis=(0, 1)) if array.ndim == 3 else array.mean(axis=0)
    sd = array.std(axis=(0, 1)) if array.ndim == 3 else array.std(axis=0)
    return mean, sd


def plot_effect_uncertainty_vs_counts(
    trace: az.InferenceData, obs_df: pd.DataFrame, plots_path: Path
) -> None:
    """Scatter: presence counts (unique partners) vs posterior SD of effects.

    - Species count = unique compounds observed in that species
    - Compound count = unique species where that compound is observed
    """
    # Presence counts
    sp_counts = obs_df.groupby("species")["compound"].nunique().sort_index()
    co_counts = obs_df.groupby("compound")["species"].nunique().sort_index()

    # Extract posterior samples
    if "species_eff" not in trace.posterior or "compound_eff" not in trace.posterior:
        return
    sp_eff = trace.posterior["species_eff"].values  # (chains, draws, n_species)
    co_eff = trace.posterior["compound_eff"].values  # (chains, draws, n_compounds)
    _, sp_sd = _posterior_mean_sd(sp_eff)
    _, co_sd = _posterior_mean_sd(co_eff)

    # Align lengths
    n_species = len(sp_sd)
    n_compounds = len(co_sd)
    sp_counts = sp_counts.reindex(range(n_species), fill_value=0).to_numpy()
    co_counts = co_counts.reindex(range(n_compounds), fill_value=0).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Species panel
    ax = axes[0]
    ax.scatter(sp_counts, sp_sd, alpha=0.6, s=20)
    ax.set_xlabel("Unique compounds per species")
    ax.set_ylabel("Posterior SD of species effect")
    ax.set_title("Pooling: species uncertainty vs data")
    ax.grid(True, alpha=0.3)

    # Compound panel
    ax = axes[1]
    ax.scatter(co_counts, co_sd, alpha=0.6, s=20)
    ax.set_xlabel("Unique species per compound")
    ax.set_ylabel("Posterior SD of compound effect")
    ax.set_title("Pooling: compound uncertainty vs data")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_path / "effect_uncertainty_vs_counts.png", dpi=120, bbox_inches="tight")
    plt.close()


def plot_effects_forest_subset(
    trace: az.InferenceData, obs_df: pd.DataFrame, plots_path: Path, k: int = 20
) -> None:
    """Forest/dot-whisker plots for a subset of species/compounds with lowest counts."""
    if "species_eff" not in trace.posterior or "compound_eff" not in trace.posterior:
        return

    sp_counts = obs_df["species"].value_counts().sort_index()
    co_counts = obs_df["compound"].value_counts().sort_index()
    n_species = trace.posterior["species_eff"].shape[-1]
    n_compounds = trace.posterior["compound_eff"].shape[-1]
    sp_counts = sp_counts.reindex(range(n_species), fill_value=0)
    co_counts = co_counts.reindex(range(n_compounds), fill_value=0)

    # Select k lowest-count indices for each
    sp_idx = sp_counts.sort_values().index.to_numpy()[: min(k, n_species)]
    co_idx = co_counts.sort_values().index.to_numpy()[: min(k, n_compounds)]

    # Compute posterior mean and 95% HDI
    # Work with NumPy arrays to avoid xarray Dataset pitfalls
    sp_arr = trace.posterior["species_eff"].values  # shape (chain, draw, n_species)
    co_arr = trace.posterior["compound_eff"].values  # shape (chain, draw, n_compounds)
    sp_arr = sp_arr[:, :, sp_idx]
    co_arr = co_arr[:, :, co_idx]
    sp_mean = sp_arr.mean(axis=(0, 1))
    co_mean = co_arr.mean(axis=(0, 1))
    # Compute 95% intervals via percentiles
    sp_flat = sp_arr.reshape(-1, sp_arr.shape[-1])
    co_flat = co_arr.reshape(-1, co_arr.shape[-1])
    sp_low = np.percentile(sp_flat, 2.5, axis=0)
    sp_high = np.percentile(sp_flat, 97.5, axis=0)
    co_low = np.percentile(co_flat, 2.5, axis=0)
    co_high = np.percentile(co_flat, 97.5, axis=0)
    sp_hdi = np.stack([sp_low, sp_high], axis=1)  # (n, 2)
    co_hdi = np.stack([co_low, co_high], axis=1)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    # Species subset
    ax = axes[0]
    y = np.arange(len(sp_idx))
    ax.errorbar(
        sp_mean, y, xerr=[sp_mean - sp_hdi[:, 0], sp_hdi[:, 1] - sp_mean], fmt="o", capsize=3
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"s{int(i)} (n={int(sp_counts[i])})" for i in sp_idx])
    ax.set_xlabel("Species effect (mean ±95% HDI)")
    ax.set_title("Low-data species effects")

    # Compound subset
    ax = axes[1]
    y = np.arange(len(co_idx))
    ax.errorbar(
        co_mean, y, xerr=[co_mean - co_hdi[:, 0], co_hdi[:, 1] - co_mean], fmt="o", capsize=3
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"c{int(i)} (n={int(co_counts[i])})" for i in co_idx])
    ax.set_xlabel("Compound effect (mean ±95% HDI)")
    ax.set_title("Low-data compound effects")

    plt.tight_layout()
    plt.savefig(plots_path / "effects_forest_subset.png", dpi=120, bbox_inches="tight")
    plt.close()


def plot_error_vs_prevalence(
    ppc_results: Dict[str, Any], obs_df: pd.DataFrame, plots_path: Path
) -> None:
    """Binned absolute error vs presence counts (unique partners).

    - Species prevalence = unique compounds in species
    - Compound prevalence = unique species of compound
    Error aggregated at the species–compound pair level to avoid overweighting replicates.
    """
    y_true = np.asarray(ppc_results.get("y_true"))
    y_pred = np.asarray(ppc_results.get("pred_mean"))
    if y_true is None or y_pred is None or len(y_true) != len(obs_df):
        return
    abs_err_obs = np.abs(y_true - y_pred)

    df = obs_df[["species", "compound"]].copy().reset_index(drop=True)
    df["abs_err_obs"] = abs_err_obs

    # Collapse to pair-level mean abs error
    pair_err = df.groupby(["species", "compound"], as_index=False)["abs_err_obs"].mean()

    # Prevalence counts (unique partners)
    sp_prev = obs_df.groupby("species")["compound"].nunique()
    co_prev = obs_df.groupby("compound")["species"].nunique()

    # Species view
    sp_pair = pair_err.groupby("species", as_index=False)["abs_err_obs"].mean()
    sp_pair["prev"] = sp_pair["species"].map(sp_prev)

    # Compound view
    co_pair = pair_err.groupby("compound", as_index=False)["abs_err_obs"].mean()
    co_pair["prev"] = co_pair["compound"].map(co_prev)

    def bin_and_summarize(df_level: pd.DataFrame, label: str):
        bins = [-1, 2, 5, 10, 1000]
        labels = ["≤2", "3–5", "6–10", "11+"]
        df_level[label] = pd.cut(df_level["prev"], bins=bins, labels=labels)
        summ = df_level.groupby(label)["abs_err_obs"].agg(["mean", "count"]).reindex(labels)
        return summ.fillna({"mean": 0.0, "count": 0}), labels

    sp_summary, labels = bin_and_summarize(sp_pair, "sp_bin")
    co_summary, _ = bin_and_summarize(co_pair, "co_bin")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, summary, title in [
        (axes[0], sp_summary, "By species prevalence"),
        (axes[1], co_summary, "By compound prevalence"),
    ]:
        x = np.arange(len(summary))
        ax.bar(x, summary["mean"].values, color="#4E79A7", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(summary.index.tolist())
        ax.set_xlabel("Presence count bin")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for xi, (m, n) in enumerate(zip(summary["mean"].values, summary["count"].values)):
            ax.text(
                xi,
                (m if np.isfinite(m) else 0) + 0.02,
                f"n={int(n)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axes[0].set_ylabel("Mean |residual| (min)")
    fig.suptitle("Prediction error vs prevalence (pair-aggregated)", fontsize=13)
    plt.tight_layout()
    plt.savefig(plots_path / "error_vs_prevalence.png", dpi=120, bbox_inches="tight")
    plt.close()


def create_combined_dashboard(
    rt_metrics: Dict[str, Any],
    assignment_metrics: Dict[str, Any],
    ppc_results: Dict[str, Any],
    output_path: Path,
):
    """
    Create a combined diagnostic dashboard showing both RT and assignment performance.

    Parameters
    ----------
    rt_metrics : dict
        RT model metrics including RMSE, MAE, coverage
    assignment_metrics : dict
        Assignment model metrics including precision, recall, F1
    ppc_results : dict
        RT posterior predictive check results
    output_path : Path
        Directory to save the dashboard
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle("Two-Stage Model Performance Dashboard", fontsize=16, fontweight="bold")

    # === RT Model Performance (Top Row) ===

    # RT Scatter Plot
    ax1 = fig.add_subplot(gs[0, :2])
    if ppc_results and "y_true" in ppc_results:
        y_true = ppc_results["y_true"]
        y_pred = ppc_results["pred_mean"]
        y_lower = ppc_results.get("pred_lower_95", y_pred - 2 * ppc_results.get("pred_std", 0.5))
        y_upper = ppc_results.get("pred_upper_95", y_pred + 2 * ppc_results.get("pred_std", 0.5))

        ax1.scatter(y_true, y_pred, alpha=0.4, s=20, label="Predictions")
        # Plot confidence bands
        sort_idx = np.argsort(y_true)
        ax1.fill_between(
            y_true[sort_idx],
            y_lower[sort_idx],
            y_upper[sort_idx],
            alpha=0.2,
            color="blue",
            label="95% CI",
        )
        # Perfect prediction line
        ax1.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r--",
            label="Perfect prediction",
        )

        ax1.set_xlabel("True RT (min)")
        ax1.set_ylabel("Predicted RT (min)")
        rmse = rt_metrics.get("rmse", 0)
        coverage = rt_metrics.get("coverage_95", 0) * 100
        ax1.set_title(f"RT Predictions (RMSE: {rmse:.3f}, Coverage: {coverage:.1f}%)")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)

    # RT Residual Histogram
    ax2 = fig.add_subplot(gs[0, 2])
    if ppc_results and "residuals" in ppc_results:
        residuals = ppc_results["residuals"]
        ax2.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        ax2.axvline(0, color="red", linestyle="--", linewidth=2)
        ax2.set_xlabel("Residual (min)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("RT Residual Distribution")
        ax2.grid(True, alpha=0.3)

    # RT Metrics Summary
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis("off")
    metrics_text = "RT Model Metrics\n" + "=" * 20 + "\n"
    metrics_text += f"RMSE: {rt_metrics.get('rmse', 0):.3f} min\n"
    metrics_text += f"MAE: {rt_metrics.get('mae', 0):.3f} min\n"
    metrics_text += f"95% Coverage: {rt_metrics.get('coverage_95', 0)*100:.1f}%\n"
    metrics_text += f"R-hat (max): {rt_metrics.get('rhat_max', 0):.3f}\n"
    metrics_text += f"Divergences: {rt_metrics.get('n_divergences', 0)}\n"
    metrics_text += f"RT Recall Ceiling: {rt_metrics.get('rt_recall_ceiling', 0)*100:.1f}%"

    ax3.text(
        0.1,
        0.9,
        metrics_text,
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    # === Assignment Model Performance (Middle Row) ===

    # Confusion Matrix
    ax4 = fig.add_subplot(gs[1, :2])
    if "confusion_matrix" in assignment_metrics:
        cm = assignment_metrics["confusion_matrix"]
        tp = cm.get("TP", 0)
        fp = cm.get("FP", 0)
        fn = cm.get("FN", 0)
        tn = cm.get("TN", 0) if "TN" in cm else 0

        # Create 2x2 matrix
        conf_matrix = np.array([[tp, fp], [fn, tn]])

        # Plot heatmap
        im = ax4.imshow(conf_matrix, cmap="Blues")

        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax4.text(
                    j,
                    i,
                    f"{conf_matrix[i, j]:d}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                )

        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xticklabels(["Predicted\nPositive", "Predicted\nNegative"])
        ax4.set_yticklabels(["True\nPositive", "True\nNegative"])
        ax4.set_title("Assignment Confusion Matrix")
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    # Calibration Plot (placeholder for now)
    ax5 = fig.add_subplot(gs[1, 2])
    if "calibration_error" in assignment_metrics:
        # Simple placeholder - ideally would show calibration curve
        ax5.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
        ax5.set_xlabel("Mean Predicted Probability")
        ax5.set_ylabel("Fraction of Positives")
        ax5.set_title(f'Calibration (ECE: {assignment_metrics.get("calibration_error", 0):.3f})')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # Assignment Metrics Summary
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.axis("off")
    assign_text = "Assignment Metrics\n" + "=" * 20 + "\n"
    assign_text += f"Precision: {assignment_metrics.get('precision', 0)*100:.1f}%\n"
    assign_text += f"Recall: {assignment_metrics.get('recall', 0)*100:.1f}%\n"
    assign_text += f"F1 Score: {assignment_metrics.get('f1', 0)*100:.1f}%\n"
    assign_text += f"ECE: {assignment_metrics.get('calibration_error', 0):.3f}\n"
    assign_text += f"Brier Score: {assignment_metrics.get('brier_ovr', 0):.3f}"

    ax6.text(
        0.1,
        0.9,
        assign_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
    )

    # === Stage Attribution Analysis (Bottom Row) ===

    # Waterfall chart showing where recall is lost
    ax7 = fig.add_subplot(gs[2, :3])
    rt_recall_ceiling = rt_metrics.get("rt_recall_ceiling", 1.0)
    assignment_recall = assignment_metrics.get("recall", 0.0)

    stages = ["Initial\nPeaks", "After RT\nFiltering", "After\nAssignment"]
    values = [1.0, rt_recall_ceiling, assignment_recall]
    colors = ["blue", "orange", "green"]

    bars = ax7.bar(stages, values, color=colors, alpha=0.7, edgecolor="black")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax7.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{val*100:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add loss annotations
    if rt_recall_ceiling < 1.0:
        rt_loss = 1.0 - rt_recall_ceiling
        ax7.annotate(
            f"RT Loss: {rt_loss*100:.1f}%",
            xy=(0.5, (1.0 + rt_recall_ceiling) / 2),
            xytext=(0.5, (1.0 + rt_recall_ceiling) / 2 + 0.1),
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red",
            fontweight="bold",
            ha="center",
        )

    if assignment_recall < rt_recall_ceiling:
        assign_loss = rt_recall_ceiling - assignment_recall
        ax7.annotate(
            f"Assignment Loss: {assign_loss*100:.1f}%",
            xy=(1.5, (rt_recall_ceiling + assignment_recall) / 2),
            xytext=(1.5, (rt_recall_ceiling + assignment_recall) / 2 + 0.1),
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red",
            fontweight="bold",
            ha="center",
        )

    ax7.set_ylim(0, 1.1)
    ax7.set_ylabel("Recall")
    ax7.set_title("Stage-wise Recall Attribution")
    ax7.grid(True, alpha=0.3, axis="y")

    # Stage Analysis Summary
    ax8 = fig.add_subplot(gs[2, 3])
    ax8.axis("off")

    # Determine which stage is the bottleneck
    rt_loss_pct = (1.0 - rt_recall_ceiling) * 100
    assign_loss_pct = (rt_recall_ceiling - assignment_recall) * 100

    if rt_loss_pct > 5:
        if assign_loss_pct > 5:
            stage_analysis = "Both"
            analysis_color = "orange"
            recommendation = "Both RT model and\nassignment need\nimprovement"
        else:
            stage_analysis = "RT"
            analysis_color = "red"
            recommendation = "RT model is the\nprimary bottleneck"
    else:
        if assign_loss_pct > 10:
            stage_analysis = "Assignment"
            analysis_color = "yellow"
            recommendation = "Assignment model\nneeds improvement"
        else:
            stage_analysis = "Good"
            analysis_color = "green"
            recommendation = "Both stages\nperforming well"

    analysis_text = "Stage Analysis\n" + "=" * 20 + "\n"
    analysis_text += f"Bottleneck: {stage_analysis}\n\n"
    analysis_text += f"RT Loss: {rt_loss_pct:.1f}%\n"
    analysis_text += f"Assignment Loss: {assign_loss_pct:.1f}%\n\n"
    analysis_text += recommendation

    ax8.text(
        0.1,
        0.9,
        analysis_text,
        transform=ax8.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor=analysis_color, alpha=0.2),
    )

    # Save the dashboard
    plt.tight_layout()
    dashboard_path = output_path / "plots" / "combined_dashboard.png"
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Combined dashboard saved to: {dashboard_path}")
