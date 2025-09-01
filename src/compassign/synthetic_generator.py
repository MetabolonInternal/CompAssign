"""
Synthetic data generator for testing RT (Retention Time) regression models.

This module generates synthetic data that simulates:
- Hierarchical structure: species clusters and compound classes
- Molecular descriptors for compounds
- RT measurements with various sources of variation
- Peak data including decoy peaks for testing
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def generate_synthetic_data(
    n_clusters: int = 5,
    n_species: int = 80,
    n_classes: int = 4,
    n_compounds: int = 60,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generate synthetic data for RT regression modeling.
    
    Parameters
    ----------
    n_clusters : int
        Number of species clusters
    n_species : int
        Number of species
    n_classes : int
        Number of compound classes
    n_compounds : int
        Number of unique compounds
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    obs_df : pd.DataFrame
        RT observations with columns: species, compound, rt
    peak_df : pd.DataFrame
        Peak data with columns: species, peak_id, true_compound, mass, rt, intensity
    params : dict
        Dictionary containing true parameter values and other metadata
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Assign each species to a cluster, each compound to a class
    species_cluster = np.random.choice(n_clusters, size=n_species)
    compound_class = np.random.choice(n_classes, size=n_compounds)

    # Simulate molecular descriptors for each compound (2 features for simplicity)
    desc1 = np.random.normal(0.0, 1.0, size=n_compounds)  # e.g. hydrophobicity
    desc2 = np.random.normal(0.0, 1.0, size=n_compounds)  # e.g. polarity
    descriptors = np.column_stack((desc1, desc2))

    # True regression coefficients for descriptors and intercept
    beta_true = np.array([1.5, -0.8])   # assume RT increases with desc1, decreases with desc2
    mu_true = 5.0                      # global intercept (baseline RT)
    eta_true = 1.0                     # coefficient for internal standard (ideal ~1)

    # Hyperparameters for simulation (true values of variances)
    sigma_cluster_true = 1.0
    sigma_species_true = 0.5
    sigma_class_true = 0.8
    sigma_compound_true = 0.3
    sigma_y_true = 0.5

    # Simulate random effects
    alpha = np.random.normal(0, sigma_cluster_true, size=n_clusters)             # cluster effects
    gamma = np.random.normal(0, sigma_class_true, size=n_classes)               # class effects
    # Species effect = cluster effect + species deviation
    u = np.array([alpha[species_cluster[s]] + np.random.normal(0, sigma_species_true) 
                  for s in range(n_species)])
    # Compound effect = class effect + compound deviation
    v = np.array([gamma[compound_class[m]] + np.random.normal(0, sigma_compound_true) 
                  for m in range(n_compounds)])

    # Simulate sample-level offsets and internal standard measurements
    sample_offset = np.random.normal(0, 0.3, size=n_species)  # one sample per species
    internal_std = sample_offset + np.random.normal(0, 0.1, size=n_species)    # measured internal standard RT for each sample

    # Determine which compounds are present in each species (imbalanced presence matrix)
    presence = np.zeros((n_species, n_compounds), dtype=bool)
    # Assign each compound a base prevalence (some compounds common, some rare)
    comp_prevalence = np.random.beta(2, 5, size=n_compounds)  # Beta(2,5) gives many low-probability (rare) compounds
    for m in range(n_compounds):
        for s in range(n_species):
            if np.random.rand() < comp_prevalence[m]:
                presence[s, m] = True

    # Ensure every compound appears at least once
    for m in range(n_compounds):
        if not presence[:, m].any():
            s = np.random.choice(n_species)
            presence[s, m] = True

    # Ensure one compound is extremely rare (e.g., present in only one species)
    rare_comp = 0
    present_species = np.where(presence[:, rare_comp])[0]
    if len(present_species) == 0:
        presence[0, rare_comp] = True
    elif len(present_species) > 1:
        # Keep only the first occurrence
        for s in present_species[1:]:
            presence[s, rare_comp] = False

    # Generate true molecular masses for compounds (some masses deliberately similar to cause ambiguity)
    np.random.seed(101)
    class_mass_base = np.random.uniform(100, 500, size=n_classes)
    compound_mass = np.zeros(n_compounds)
    for m in range(n_compounds):
        # Compound mass ~ base of its class + some random offset
        compound_mass[m] = class_mass_base[compound_class[m]] + np.random.normal(0, 2.0)
    # Make two compounds have nearly identical mass (to simulate isomers causing confusion)
    if n_compounds >= 2:
        compound_mass[0] = 250.00
        compound_mass[1] = 250.01

    # Generate observations and peaks
    obs_records = []   # for RT model (species, compound, observed RT)
    peak_records = []  # for peaks (species, peak_id, true_compound, measured mass, RT, intensity)
    peak_id = 0
    for s in range(n_species):
        # Generate peaks for each compound present in species s
        for m in range(n_compounds):
            if presence[s, m]:
                # True mean RT for this species-compound
                true_mean_rt = (mu_true + u[s] + v[m] 
                                + beta_true.dot(descriptors[m]) + eta_true * internal_std[s])
                # Observed RT with noise
                obs_rt = np.random.normal(true_mean_rt, sigma_y_true)
                # Peak intensity (log-normal, higher for true compounds)
                intensity = np.exp(np.random.normal(np.log(1e5), 0.5))
                # Measured mass = true mass + small instrument error
                measured_mz = np.random.normal(compound_mass[m], 0.001)
                obs_records.append((s, m, obs_rt))
                peak_records.append((s, peak_id, m, measured_mz, obs_rt, intensity))
                peak_id += 1
        # Add some decoy peaks (peaks from compounds not in the known list or not actually present)
        # Decoy type 1: use a known compound's mass that is NOT present in this species (false candidate)
        absent_comps = [m for m in range(n_compounds) if not presence[s, m]]
        if absent_comps:
            m_decoy = np.random.choice(absent_comps)
            decoy_mass = np.random.normal(compound_mass[m_decoy], 0.001)   # mass matching a known compound
            # RT that does not match the predicted RT of that compound (offset by ~1 minute)
            pred_rt = (mu_true + u[s] + v[m_decoy] + beta_true.dot(descriptors[m_decoy]) + eta_true * internal_std[s])
            decoy_rt = pred_rt + np.random.normal(1.0, 0.2)
            intensity = np.exp(np.random.normal(np.log(5e4), 0.5))        # slightly lower intensity
            peak_records.append((s, peak_id, None, decoy_mass, decoy_rt, intensity))
            peak_id += 1
        # Decoy type 2: random mass that likely doesn't match any known compound
        random_mass = np.random.uniform(100, 500)
        decoy_rt = np.random.normal(mu_true + np.mean(u), 5.0)
        intensity = np.exp(np.random.normal(np.log(2e4), 0.5))
        peak_records.append((s, peak_id, None, random_mass, decoy_rt, intensity))
        peak_id += 1

    # Convert to DataFrame for convenience
    obs_df = pd.DataFrame(obs_records, columns=["species", "compound", "rt"])
    peak_df = pd.DataFrame(peak_records, columns=["species", "peak_id", "true_compound", "mass", "rt", "intensity"])
    
    # Store parameters and metadata
    params = {
        "n_clusters": n_clusters,
        "n_species": n_species,
        "n_classes": n_classes,
        "n_compounds": n_compounds,
        "species_cluster": species_cluster,
        "compound_class": compound_class,
        "descriptors": descriptors,
        "beta_true": beta_true,
        "mu_true": mu_true,
        "eta_true": eta_true,
        "sigma_cluster_true": sigma_cluster_true,
        "sigma_species_true": sigma_species_true,
        "sigma_class_true": sigma_class_true,
        "sigma_compound_true": sigma_compound_true,
        "sigma_y_true": sigma_y_true,
        "alpha": alpha,
        "gamma": gamma,
        "u": u,
        "v": v,
        "sample_offset": sample_offset,
        "internal_std": internal_std,
        "presence": presence,
        "compound_mass": compound_mass,
        "comp_prevalence": comp_prevalence
    }
    
    print("RT observations (training) count:", len(obs_df))
    print("Unique species in RT data:", obs_df['species'].nunique(), "| Unique compounds in RT data:", obs_df['compound'].nunique())
    print("Total peaks (including decoys):", len(peak_df))
    
    return obs_df, peak_df, params


if __name__ == "__main__":
    # Test the generator
    obs_df, peak_df, params = generate_synthetic_data()