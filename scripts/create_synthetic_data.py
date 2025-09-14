#!/usr/bin/env python3
"""
Synthetic data creation for CompAssign training with hierarchical structure.
Generates metabolomics data with isomers, near-isobars, and realistic noise.

DAT-004 alignment: This generator now creates causal RT covariates and RTs that
follow the hierarchical model used in training. Specifically, we generate:
- Molecular descriptors (p=10) with class-wise centroids + per-compound jitter
- Internal standard measurements per species correlated with species effects
- RT observations: y_sc = mu0 + alpha_s + beta_c + D_c·theta + gamma*IS_s + eps

This replaces the previous ad hoc "predicted_rt" pathway and ensures that the
RT model can learn meaningful coefficients and predictive variance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_metabolomics_data(
    n_compounds: int = 60,
    n_peaks_per_compound: int = 3,
    n_noise_peaks: int = 100,
    n_species: int = 3,
    isomer_fraction: float = 0.3,
    near_isobar_fraction: float = 0.2,
    mass_error_std: float = 0.002,
    rt_uncertainty_range: Tuple[float, float] = (0.05, 0.5),
    decoy_fraction: float = 0.5  # Fraction of compounds that are decoys (never present)
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict, Dict, np.ndarray, np.ndarray]:
    """
    Create synthetic metabolomics data with hierarchical structure.
    
    Parameters
    ----------
    n_compounds : int
        Number of compounds in the library
    n_peaks_per_compound : int
        Average peaks per compound
    n_noise_peaks : int
        Number of noise peaks to add
    n_species : int
        Number of species/samples
    isomer_fraction : float
        Fraction of compounds that are isomers
    near_isobar_fraction : float
        Fraction of compounds that are near-isobars
    mass_error_std : float
        Standard deviation of mass measurement error
    rt_uncertainty_range : tuple
        Range of RT prediction uncertainties
    
    Returns
    -------
    peak_df : pd.DataFrame
        Peak data with mass, RT, intensity, species, and true assignments
    compound_df : pd.DataFrame
        Compound library with masses and properties
    true_assignments : dict
        Mapping of peak_id to true compound_id (or None for noise)
    rt_uncertainties : dict
        Legacy field retained for compatibility; set to homoscedastic sigma_y
        for all compounds in this generator.
    hierarchical_params : dict
        Hierarchical model parameters (clusters, classes)
    descriptors : np.ndarray
        Molecular descriptor matrix of shape (n_compounds, 10)
    internal_std : np.ndarray
        Internal-standard proxy per species of shape (n_species,)
    """
    np.random.seed(42)
    
    # Create hierarchical structure
    n_clusters = min(8, n_species // 5 + 1)  # Reasonable number of clusters
    n_classes = min(10, n_compounds // 6 + 1)  # Chemical classes
    
    # Assign species to clusters
    species_cluster = np.random.choice(n_clusters, size=n_species)
    
    # Assign compounds to chemical classes
    compound_class = np.random.choice(n_classes, size=n_compounds)
    
    # Generate compound library with clustered masses for more overlaps
    compounds = []
    # Create mass clusters to increase density and overlaps
    n_mass_clusters = max(3, n_compounds // 10)
    cluster_centers = np.random.uniform(200, 600, n_mass_clusters)
    base_masses = []
    for i in range(n_compounds):
        cluster = np.random.choice(n_mass_clusters)
        # Masses clustered within ±50 Da of cluster centers
        mass = cluster_centers[cluster] + np.random.uniform(-50, 50)
        base_masses.append(mass)
    base_masses = np.array(base_masses)
    # Base RT scale reference (used only for reporting compatibility)
    base_rts = np.random.uniform(1, 15, n_compounds)
    
    # Create isomers (same mass, different RT)
    n_isomers = int(n_compounds * isomer_fraction)
    isomer_groups = {}
    
    for i in range(n_isomers):
        if i < n_isomers // 2:
            # Create new isomer group
            group_id = len(isomer_groups)
            isomer_groups[group_id] = base_masses[i]
        else:
            # Add to existing group
            if len(isomer_groups) > 0:
                group_id = np.random.choice(len(isomer_groups))
                base_masses[i] = isomer_groups[group_id]
                # MUCH closer RTs for isomers - often hard to separate
                base_rts[i] = base_rts[group_id % n_compounds] + np.random.normal(0, 0.3)
    
    # Create near-isobars (similar mass AND RT to create maximum confusion)
    n_near_isobars = int(n_compounds * near_isobar_fraction)
    for i in range(n_isomers, n_isomers + n_near_isobars):
        # Find a nearby compound
        ref_idx = np.random.choice(i)
        # Very close mass - often within instrument precision
        base_masses[i] = base_masses[ref_idx] + np.random.uniform(-0.005, 0.005)
        # Also similar RT! (near-isobars often have similar chemistry)
        base_rts[i] = base_rts[ref_idx] + np.random.normal(0, 0.5)
    
    # Build compound dataframe (predicted_rt kept for compatibility; will be
    # set later to a species-agnostic baseline RT mean)
    compound_df = pd.DataFrame({
        'compound_id': range(n_compounds),
        'true_mass': base_masses,
        'predicted_rt': base_rts,  # temporary; overwritten below
        'type': ['isomer' if i < n_isomers else 
                'near_isobar' if i < n_isomers + n_near_isobars else 
                'normal' for i in range(n_compounds)],
        'class': compound_class
    })

    # --- New causal covariates and hierarchical RT generative model ---
    p = 10  # descriptor dimensionality (per user instruction)

    # Class-wise descriptor centroids + per-compound jitter
    class_centroids = np.random.normal(0.0, 1.0, size=(n_classes, p))
    descriptors = np.zeros((n_compounds, p), dtype=float)
    for j in range(n_compounds):
        c = compound_class[j]
        descriptors[j] = class_centroids[c] + np.random.normal(0.0, 0.5, size=p)

    # Hierarchical effects (with sum-to-zero centering)
    # Variance scales chosen to yield realistic minutes-scale variability
    sigma_cluster = 1.0
    sigma_species = 0.7
    sigma_class = 1.0
    sigma_compound = 0.7
    sigma_y = 0.4  # observation noise (homoscedastic)

    # Cluster, class base effects
    cluster_raw = np.random.normal(0.0, 1.0, size=n_clusters) * sigma_cluster
    cluster_eff = cluster_raw - cluster_raw.mean()

    class_raw = np.random.normal(0.0, 1.0, size=n_classes) * sigma_class
    class_eff = class_raw - class_raw.mean()

    # Species effects conditional on clusters
    species_raw = np.random.normal(0.0, 1.0, size=n_species) * sigma_species
    species_base = cluster_eff[species_cluster] + species_raw
    alpha_species = species_base - species_base.mean()

    # Compound effects conditional on classes
    compound_raw = np.random.normal(0.0, 1.0, size=n_compounds) * sigma_compound
    compound_base = class_eff[compound_class] + compound_raw
    beta_compound = compound_base - compound_base.mean()

    # Coefficients for descriptors and internal standard
    theta = np.random.normal(0.0, 0.3, size=p)  # moderate signal per feature
    gamma = np.random.normal(0.0, 0.7)          # moderate effect on IS

    # Internal-standard proxy per species, correlated with alpha_species
    internal_std = alpha_species + np.random.normal(0.0, 0.5, size=n_species)

    # Global intercept near center of chromatogram
    mu0 = np.random.uniform(5.0, 11.0)

    # Species-agnostic baseline RT per compound (for compatibility column)
    baseline_rt = mu0 + beta_compound + descriptors @ theta
    compound_df['predicted_rt'] = baseline_rt
    
    # Generate peaks using the hierarchical RT generative story
    peaks = []
    peak_id = 0
    true_assignments = {}
    
    # Mark decoy compounds (will NEVER appear in samples)
    n_decoys = int(n_compounds * decoy_fraction)
    decoy_compounds = set(np.random.choice(n_compounds, size=n_decoys, replace=False))
    real_compounds = set(range(n_compounds)) - decoy_compounds
    
    # Add decoy flag to compound dataframe
    compound_df['is_decoy'] = [i in decoy_compounds for i in range(n_compounds)]
    
    # Generate real peaks
    for species in range(n_species):
        # Only select from non-decoy compounds
        presence_prob = 0.4  # Back to reasonable presence for non-decoys
        present_compounds = np.random.choice(
            list(real_compounds), 
            size=min(len(real_compounds), int(len(real_compounds) * presence_prob)),
            replace=False
        )
        
        for compound_id in present_compounds:
            compound = compound_df.iloc[compound_id]
            # Generate multiple peaks per compound (isotopes, fragments)
            n_peaks = np.random.poisson(n_peaks_per_compound)
            for _ in range(n_peaks):
                # Mass measurement error
                measured_mass = compound['true_mass'] + np.random.normal(0, mass_error_std)

                # RT mean per species–compound from hierarchical story
                mu_sc = (
                    mu0
                    + alpha_species[species]
                    + beta_compound[compound_id]
                    + float(descriptors[compound_id] @ theta)
                    + gamma * internal_std[species]
                )

                measured_rt = np.random.normal(mu_sc, sigma_y)
                
                peaks.append({
                    'peak_id': peak_id,
                    'species': species,
                    'mass': measured_mass,
                    'rt': measured_rt,
                    'intensity': np.exp(np.random.normal(12, 1.5)),  # Log-normal intensity
                    'true_compound': compound_id
                })
                true_assignments[peak_id] = compound_id
                peak_id += 1
    
    # Add noise peaks
    for _ in range(n_noise_peaks):
        species = np.random.choice(n_species)
        peaks.append({
            'peak_id': peak_id,
            'species': species,
            'mass': np.random.uniform(50, 850),
            'rt': np.random.uniform(0, 16),
            'intensity': np.exp(np.random.normal(10, 2.0)),  # Lower intensity for noise
            'true_compound': None
        })
        true_assignments[peak_id] = None
        peak_id += 1
    
    peak_df = pd.DataFrame(peaks)
    
    # Create hierarchical parameters dict
    hierarchical_params = {
        'n_clusters': n_clusters,
        'n_classes': n_classes,
        'species_cluster': species_cluster,
        'compound_class': compound_class
    }

    # Legacy uncertainties dict retained (homoscedastic sigma_y)
    rt_uncertainties = {i: sigma_y for i in range(n_compounds)}

    return peak_df, compound_df, true_assignments, rt_uncertainties, hierarchical_params, descriptors, internal_std


if __name__ == "__main__":
    # Test the function
    peaks, compounds, assignments, rt_uncert, hier_params, desc, is_vec = create_metabolomics_data()
    print(f"Generated {len(peaks)} peaks with {len(compounds)} compounds")
    print(f"True assignments: {sum(1 for v in assignments.values() if v is not None)}")
    print(f"Noise peaks: {sum(1 for v in assignments.values() if v is None)}")
    print(f"Hierarchical params: {hier_params['n_clusters']} clusters, {hier_params['n_classes']} classes")
    print(f"Descriptors shape: {desc.shape}, Internal std shape: {is_vec.shape}")
