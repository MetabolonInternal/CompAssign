#!/usr/bin/env python3
"""
Test and visualize chemical features for noise/signal discrimination.

This script generates synthetic data, computes chemical features for all peaks,
and visualizes the distributions to verify that features effectively discriminate
between real metabolite signals and noise.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_prep.create_synthetic_data import create_synthetic_dataset  # noqa: E402
from src.compassign.utils import compute_all_chemical_features  # noqa: E402


def analyze_chemical_features(n_compounds=10, n_species=5, n_noise_peaks=100):
    """
    Generate data and analyze chemical feature distributions.
    """
    print("Generating synthetic data...")
    dataset = create_synthetic_dataset(
        n_compounds=n_compounds,
        n_species=n_species,
        n_noise_peaks=n_noise_peaks,
        n_peaks_per_compound=3,
    )
    peak_df = dataset.peak_df

    print(f"Generated {len(peak_df)} peaks ({n_noise_peaks} noise)")

    # Compute chemical features for all peaks
    print("Computing chemical features...")
    features_list = []

    for _, peak in peak_df.iterrows():
        features = compute_all_chemical_features(
            peak["mass"], peak["rt"], peak["intensity"], peak_df, peak["species"]
        )

        # Add peak type (real vs noise)
        features["is_real"] = peak["true_compound"] is not None and not pd.isna(
            peak["true_compound"]
        )
        features["peak_id"] = peak["peak_id"]
        features["species"] = peak["species"]

        features_list.append(features)

    features_df = pd.DataFrame(features_list)

    # Separate real and noise peaks
    real_features = features_df[features_df["is_real"]]
    noise_features = features_df[~features_df["is_real"]]

    print(f"Real peaks: {len(real_features)}, Noise peaks: {len(noise_features)}")

    return features_df, real_features, noise_features


def plot_feature_distributions(real_features, noise_features):
    """
    Create visualization of feature distributions for real vs noise peaks.
    """
    feature_names = ["has_isotope", "isotope_score", "n_adducts", "rt_cluster_size", "n_correlated"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feature in enumerate(feature_names):
        ax = axes[i]

        # Get feature values
        real_vals = real_features[feature].values
        noise_vals = noise_features[feature].values

        # Plot distributions
        if feature == "has_isotope":  # Binary feature
            # Bar plot for binary
            x_labels = ["No isotope", "Has isotope"]
            real_counts = [sum(real_vals == 0), sum(real_vals == 1)]
            noise_counts = [sum(noise_vals == 0), sum(noise_vals == 1)]

            x = np.arange(len(x_labels))
            width = 0.35

            ax.bar(
                x - width / 2,
                np.array(real_counts) / len(real_vals),
                width,
                label="Real",
                alpha=0.7,
                color="green",
            )
            ax.bar(
                x + width / 2,
                np.array(noise_counts) / len(noise_vals),
                width,
                label="Noise",
                alpha=0.7,
                color="red",
            )

            ax.set_ylabel("Fraction")
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
        else:
            # Histogram for continuous
            bins = np.linspace(
                min(real_vals.min(), noise_vals.min()), max(real_vals.max(), noise_vals.max()), 20
            )

            ax.hist(real_vals, bins=bins, alpha=0.5, label="Real", color="green", density=True)
            ax.hist(noise_vals, bins=bins, alpha=0.5, label="Noise", color="red", density=True)

            ax.set_ylabel("Density")

        ax.set_title(f"{feature}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Summary statistics in last subplot
    ax = axes[5]
    ax.axis("off")

    stats_text = "Feature Discrimination Power\n" + "=" * 30 + "\n\n"

    for feature in feature_names:
        real_mean = real_features[feature].mean()
        noise_mean = noise_features[feature].mean()
        separation = abs(real_mean - noise_mean) / (
            (real_features[feature].std() + noise_features[feature].std()) / 2 + 1e-6
        )

        stats_text += f"{feature:15s}: \n"
        stats_text += f"  Real mean:  {real_mean:.2f}\n"
        stats_text += f"  Noise mean: {noise_mean:.2f}\n"
        stats_text += f"  Separation: {separation:.2f}σ\n\n"

    ax.text(
        0.1,
        0.9,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
    )

    # Hide any remaining unused axes
    axes[-1].axis("off")

    plt.suptitle("Chemical Feature Distributions: Real vs Noise Peaks", fontsize=14)
    plt.tight_layout()
    return fig


def create_feature_correlation_plot(features_df):
    """
    Create correlation matrix of features.
    """
    feature_cols = ["has_isotope", "isotope_score", "n_adducts", "rt_cluster_size", "n_correlated"]

    corr_matrix = features_df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax
    )
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    return fig


def main():
    """
    Main function to test chemical features.
    """
    np.random.seed(42)

    # Generate and analyze features
    features_df, real_features, noise_features = analyze_chemical_features(
        n_compounds=10, n_species=5, n_noise_peaks=100
    )

    # Create visualizations
    print("\nCreating visualizations...")

    # Feature distributions
    dist_fig = plot_feature_distributions(real_features, noise_features)

    # Correlation matrix
    corr_fig = create_feature_correlation_plot(features_df)

    # Save plots
    output_dir = Path("output/feature_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    dist_fig.savefig(output_dir / "feature_distributions.png", dpi=150, bbox_inches="tight")
    corr_fig.savefig(output_dir / "feature_correlations.png", dpi=150, bbox_inches="tight")

    print(f"Plots saved to {output_dir}")

    # Print summary statistics
    print("\n" + "=" * 50)
    print("FEATURE EFFECTIVENESS SUMMARY")
    print("=" * 50)

    feature_names = ["has_isotope", "isotope_score", "n_adducts", "rt_cluster_size", "n_correlated"]

    for feature in feature_names:
        real_mean = real_features[feature].mean()
        noise_mean = noise_features[feature].mean()

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((real_features[feature].var() + noise_features[feature].var()) / 2)
        if pooled_std > 0:
            cohens_d = abs(real_mean - noise_mean) / pooled_std
        else:
            cohens_d = 0

        print(f"\n{feature}:")
        print(f"  Real peaks:  mean={real_mean:.3f}, std={real_features[feature].std():.3f}")
        print(f"  Noise peaks: mean={noise_mean:.3f}, std={noise_features[feature].std():.3f}")
        if cohens_d > 0.8:
            strength = "strong"
        elif cohens_d > 0.5:
            strength = "moderate"
        else:
            strength = "weak"

        print(f"  Cohen's d:   {cohens_d:.3f} ({strength})")

    # Overall assessment
    print("\n" + "=" * 50)
    print("OVERALL ASSESSMENT")
    print("=" * 50)

    strong_features = []
    moderate_features = []
    weak_features = []

    for feature in feature_names:
        real_mean = real_features[feature].mean()
        noise_mean = noise_features[feature].mean()
        pooled_std = np.sqrt((real_features[feature].var() + noise_features[feature].var()) / 2)
        if pooled_std > 0:
            cohens_d = abs(real_mean - noise_mean) / pooled_std
            if cohens_d > 0.8:
                strong_features.append(feature)
            elif cohens_d > 0.5:
                moderate_features.append(feature)
            else:
                weak_features.append(feature)

    print(f"Strong discriminators:   {', '.join(strong_features) if strong_features else 'None'}")
    print(
        f"Moderate discriminators: {', '.join(moderate_features) if moderate_features else 'None'}"
    )
    print(f"Weak discriminators:     {', '.join(weak_features) if weak_features else 'None'}")

    if len(strong_features) >= 2:
        print("\n✅ Chemical features show good discrimination power!")
        print("   The model should improve significantly with these features.")
    else:
        print("\n⚠️  Limited discrimination power in features.")
        print("   May need to adjust feature computation or thresholds.")


if __name__ == "__main__":
    main()
