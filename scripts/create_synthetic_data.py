#!/usr/bin/env python3
"""
Synthetic data creation for CompAssign training with hierarchical structure.
Generates metabolomics data with isomers, near-isobars, and realistic noise.
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
    rt_uncertainty_range: Tuple[float, float] = (0.05, 0.5)
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict, Dict]:
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
        RT prediction uncertainties for each compound
    hierarchical_params : dict
        Hierarchical model parameters (clusters, classes)
    """
    np.random.seed(42)
    
    # Create hierarchical structure
    n_clusters = min(8, n_species // 5 + 1)  # Reasonable number of clusters
    n_classes = min(10, n_compounds // 6 + 1)  # Chemical classes
    
    # Assign species to clusters
    species_cluster = np.random.choice(n_clusters, size=n_species)
    
    # Assign compounds to chemical classes
    compound_class = np.random.choice(n_classes, size=n_compounds)
    
    # Generate compound library
    compounds = []
    base_masses = np.random.uniform(100, 800, n_compounds)
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
                base_rts[i] = base_rts[group_id % n_compounds] + np.random.normal(0, 2.0)
    
    # Create near-isobars (similar mass within tolerance)
    n_near_isobars = int(n_compounds * near_isobar_fraction)
    for i in range(n_isomers, n_isomers + n_near_isobars):
        # Find a nearby compound
        ref_idx = np.random.choice(i)
        base_masses[i] = base_masses[ref_idx] + np.random.uniform(-0.05, 0.05)
    
    # Build compound dataframe
    compound_df = pd.DataFrame({
        'compound_id': range(n_compounds),
        'true_mass': base_masses,
        'predicted_rt': base_rts,
        'type': ['isomer' if i < n_isomers else 
                'near_isobar' if i < n_isomers + n_near_isobars else 
                'normal' for i in range(n_compounds)],
        'class': compound_class
    })
    
    # Generate RT uncertainties
    rt_uncertainties = {
        i: np.random.uniform(*rt_uncertainty_range) 
        for i in range(n_compounds)
    }
    
    # Generate peaks
    peaks = []
    peak_id = 0
    true_assignments = {}
    
    # Generate real peaks
    for species in range(n_species):
        # Compounds present in this species (with some missing)
        presence_prob = 0.65  # 65% of compounds are present
        present_compounds = np.random.choice(
            n_compounds, 
            size=int(n_compounds * presence_prob),
            replace=False
        )
        
        for compound_id in present_compounds:
            compound = compound_df.iloc[compound_id]
            
            # Generate multiple peaks per compound (isotopes, fragments)
            n_peaks = np.random.poisson(n_peaks_per_compound)
            for _ in range(n_peaks):
                # Add measurement error
                measured_mass = compound['true_mass'] + np.random.normal(0, mass_error_std)
                measured_rt = compound['predicted_rt'] + np.random.normal(0, rt_uncertainties[compound_id])
                
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
    
    return peak_df, compound_df, true_assignments, rt_uncertainties, hierarchical_params


if __name__ == "__main__":
    # Test the function
    peaks, compounds, assignments, rt_uncert, hier_params = create_metabolomics_data()
    print(f"Generated {len(peaks)} peaks with {len(compounds)} compounds")
    print(f"True assignments: {sum(1 for v in assignments.values() if v is not None)}")
    print(f"Noise peaks: {sum(1 for v in assignments.values() if v is None)}")
    print(f"Hierarchical params: {hier_params['n_clusters']} clusters, {hier_params['n_classes']} classes")