#!/usr/bin/env python
"""
Create more challenging synthetic data for k parameter optimization.

This script generates data with:
1. Isomers (same mass, different RT)
2. Near-isobars (very similar masses)
3. RT drift and varying uncertainty
4. Realistic mass/RT distributions from metabolomics
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import json
from pathlib import Path


def create_metabolomics_data(
    n_compounds: int = 100,
    n_peaks_per_compound: int = 3,
    n_noise_peaks: int = 100,
    n_species: int = 5,
    isomer_fraction: float = 0.3,
    near_isobar_fraction: float = 0.2,
    mass_error_std: float = 0.002,  # 2 ppm at m/z 1000
    rt_uncertainty_range: Tuple[float, float] = (0.05, 0.5)
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """
    Create synthetic metabolomics data with isomers and near-isobars.
    
    Parameters
    ----------
    n_compounds : int
        Number of unique compounds
    n_peaks_per_compound : int
        Average peaks per compound across species
    n_noise_peaks : int
        Number of noise/contaminant peaks
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
    tuple
        (peak_df, compound_df, true_assignments, rt_uncertainties)
    """
    np.random.seed(42)
    
    print("Creating synthetic metabolomics dataset...")
    print(f"  Compounds: {n_compounds}")
    print(f"  Isomer fraction: {isomer_fraction:.1%}")
    print(f"  Near-isobar fraction: {near_isobar_fraction:.1%}")
    
    # Step 1: Create compound library with realistic challenges
    compounds = []
    compound_groups = {}  # Track isomer/isobar groups
    
    # Create base compounds
    base_masses = np.random.uniform(100, 800, int(n_compounds * 0.5))
    base_rts = np.random.uniform(0.5, 15, int(n_compounds * 0.5))
    
    compound_id = 0
    
    # Add normal compounds
    for mass, rt in zip(base_masses, base_rts):
        compounds.append({
            'compound_id': compound_id,
            'true_mass': mass,
            'true_rt': rt,
            'rt_uncertainty': np.random.uniform(*rt_uncertainty_range),
            'type': 'normal',
            'group_id': compound_id
        })
        compound_id += 1
    
    # Add isomers (same mass, different RT)
    n_isomers = int(n_compounds * isomer_fraction)
    isomer_base = np.random.choice(len(compounds), n_isomers // 2, replace=False)
    
    for base_idx in isomer_base:
        base = compounds[base_idx]
        n_isomers_in_group = np.random.randint(2, 4)
        
        for i in range(n_isomers_in_group):
            # Isomers have identical mass but different RT
            rt_separation = np.random.uniform(0.2, 2.0)  # RT difference
            compounds.append({
                'compound_id': compound_id,
                'true_mass': base['true_mass'],  # EXACT same mass
                'true_rt': base['true_rt'] + (i + 1) * rt_separation,
                'rt_uncertainty': np.random.uniform(*rt_uncertainty_range),
                'type': 'isomer',
                'group_id': base['compound_id']
            })
            compound_id += 1
    
    # Add near-isobars (very similar mass, different RT)
    n_isobars = int(n_compounds * near_isobar_fraction)
    isobar_base = np.random.choice(len(compounds), n_isobars // 2, replace=False)
    
    for base_idx in isobar_base:
        base = compounds[base_idx]
        
        # Near-isobars have mass within 0.01 Da
        mass_diff = np.random.uniform(0.003, 0.010)  # Just outside mass tolerance
        rt_diff = np.random.uniform(0.5, 3.0)
        
        compounds.append({
            'compound_id': compound_id,
            'true_mass': base['true_mass'] + mass_diff,
            'true_rt': base['true_rt'] + rt_diff,
            'rt_uncertainty': np.random.uniform(*rt_uncertainty_range),
            'type': 'near_isobar',
            'group_id': base['compound_id']
        })
        compound_id += 1
    
    # Trim to desired number
    compounds = compounds[:n_compounds]
    compound_df = pd.DataFrame(compounds)
    
    # Step 2: Create peaks with realistic variations
    # Generate hierarchical structure matching Bayesian model assumptions
    n_clusters = min(5, max(2, n_species // 2))  # Reasonable number of clusters
    n_classes = min(4, max(2, n_compounds // 5))  # Reasonable number of classes
    
    # Hierarchical RT effects
    # Level 1: Cluster effects (larger scale structure)
    cluster_rt_effects = np.random.normal(0, 0.3, n_clusters)
    
    # Level 2: Class effects for compounds
    class_rt_effects = np.random.normal(0, 0.2, n_classes)
    
    # Assign species to clusters
    species_cluster = np.random.choice(n_clusters, n_species)
    
    # Assign compounds to classes (keeping isomers in same class for realism)
    compound_class = np.zeros(len(compound_df), dtype=int)
    for i in range(len(compound_df)):
        compound = compound_df.iloc[i]
        if compound['type'] == 'isomer' and 'isomer_group' in compound and pd.notna(compound['isomer_group']):
            # Find if any compound in this isomer group already has a class
            same_group = compound_df[compound_df.get('isomer_group', -1) == compound['isomer_group']]
            if len(same_group) > 1 and same_group.index[0] < i:
                # Use same class as first compound in group
                compound_class[i] = compound_class[same_group.index[0]]
            else:
                compound_class[i] = np.random.choice(n_classes)
        else:
            compound_class[i] = np.random.choice(n_classes)
    
    # Generate species-specific RT effects (hierarchical: cluster + individual)
    species_rt_effects = np.zeros(n_species)
    for s in range(n_species):
        cluster = species_cluster[s]
        # Species effect combines cluster effect with individual variation
        species_rt_effects[s] = cluster_rt_effects[cluster] + np.random.normal(0, 0.1)
    
    # Update compound RTs to include class effects
    for c in range(len(compound_df)):
        class_idx = int(compound_class[c])
        # Add class effect to compound's base RT
        compound_df.loc[c, 'true_rt'] += class_rt_effects[class_idx]
    
    peaks = []
    true_assignments = {}
    peak_id = 0
    
    # Generate peaks for real compounds
    for species in range(n_species):
        # Use hierarchical species RT effect
        species_rt_shift = species_rt_effects[species]
        
        # Sample which compounds are present in this species
        present_compounds = np.random.choice(
            n_compounds, 
            size=int(n_compounds * np.random.uniform(0.5, 0.8)),
            replace=False
        )
        
        for compound_idx in present_compounds:
            if compound_idx >= len(compound_df):
                continue
            compound = compound_df.iloc[compound_idx]
            
            # Observed mass with measurement error
            mass_error_ppm = np.random.normal(0, 2)  # 2 ppm error
            observed_mass = compound['true_mass'] * (1 + mass_error_ppm / 1e6)
            
            # Observed RT with prediction error and species shift
            rt_error = np.random.normal(0, compound['rt_uncertainty'])
            observed_rt = compound['true_rt'] + rt_error + species_rt_shift
            
            # Intensity with variation - TRUE peaks have HIGH intensity
            # Mean log intensity = 14, std = 1.2 → median ≈ 1.2M, range 100K-10M
            intensity = np.random.lognormal(14, 1.2)
            
            peaks.append({
                'peak_id': peak_id,
                'species': species,
                'mass': observed_mass,
                'rt': observed_rt,
                'intensity': intensity,
                'true_compound': compound['compound_id']
            })
            true_assignments[peak_id] = compound['compound_id']
            peak_id += 1
    
    # Add noise peaks (false positives)
    # Make 80% of noise peaks targeted to pass mass filter (within tolerance of real compounds)
    for i in range(n_noise_peaks):
        species = np.random.randint(0, n_species)
        
        if i < int(n_noise_peaks * 0.8):
            # Targeted noise: very close in mass to real compounds (will pass mass filter)
            # but wrong RT and low intensity
            target_compound = compound_df.sample(1).iloc[0]
            
            # Add small mass error within typical tolerance (0.001-0.004 Da)
            mass_error = np.random.uniform(-0.004, 0.004)
            observed_mass = target_compound['true_mass'] + mass_error
            
            # RT is deliberately wrong (outside typical RT window)
            rt_offset = np.random.choice([-1, 1]) * np.random.uniform(2.0, 5.0)
            observed_rt = max(0.5, min(15, target_compound['true_rt'] + rt_offset))
        else:
            # Random noise (20% of noise peaks)
            observed_mass = np.random.uniform(100, 800)
            observed_rt = np.random.uniform(0.5, 15)
        
        # Noise peaks have MUCH LOWER intensity than true peaks
        # Mean log intensity = 8, std = 1.5 → median ≈ 3K, range 200-50K
        # This creates ~400x difference from true peaks on average
        noise_intensity = np.random.lognormal(8, 1.5)
        
        peaks.append({
            'peak_id': peak_id,
            'species': species,
            'mass': observed_mass,
            'rt': observed_rt,
            'intensity': noise_intensity,
            'true_compound': None
        })
        true_assignments[peak_id] = None
        peak_id += 1
    
    peak_df = pd.DataFrame(peaks)
    
    # Step 3: Create RT uncertainty dictionary
    rt_uncertainties = {}
    for species in range(n_species):
        for _, compound in compound_df.iterrows():
            rt_uncertainties[(species, compound['compound_id'])] = compound['rt_uncertainty']
    
    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  Total peaks: {len(peak_df)}")
    print(f"  True assignments: {sum(1 for v in true_assignments.values() if v is not None)}")
    print(f"  Noise peaks: {sum(1 for v in true_assignments.values() if v is None)}")
    
    # Analyze challenges
    isomers = compound_df[compound_df['type'] == 'isomer']
    if len(isomers) > 0:
        print(f"  Isomer groups: {isomers['group_id'].nunique()}")
        
    near_isobars = compound_df[compound_df['type'] == 'near_isobar']
    if len(near_isobars) > 0:
        print(f"  Near-isobar pairs: {len(near_isobars)}")
    
    # Skip expensive peak pair calculation - not needed for training
    # (This was just for diagnostics and caused performance issues)
    # If needed later, this could be vectorized using numpy broadcasting
    
    # Package hierarchical structure for the RT model
    hierarchical_params = {
        'n_clusters': n_clusters,
        'n_classes': n_classes,
        'species_cluster': species_cluster,
        'compound_class': compound_class,
        'cluster_rt_effects': cluster_rt_effects,
        'class_rt_effects': class_rt_effects
    }
    
    return peak_df, compound_df, true_assignments, rt_uncertainties, hierarchical_params


def test_k_with_challenging_data(
    k_value: float,
    mass_tolerance: float = 0.005,
    save_results: bool = True
) -> Dict:
    """
    Test a k value with challenging data.
    
    Parameters
    ----------
    k_value : float
        RT window k parameter
    mass_tolerance : float
        Mass tolerance in Da
    save_results : bool
        Whether to save results to file
        
    Returns
    -------
    dict
        Test results
    """
    # Create challenging data
    peak_df, compound_df, true_assignments, rt_uncertainties = \
        create_challenging_metabolomics_data()
    
    # Simulate RT predictions with uncertainties
    rt_predictions = {}
    for species in range(peak_df['species'].nunique()):
        for _, compound in compound_df.iterrows():
            # Add prediction error to true RT
            pred_error = np.random.normal(0, compound['rt_uncertainty'] * 0.5)
            rt_pred = compound['true_rt'] + pred_error
            rt_std = compound['rt_uncertainty']
            rt_predictions[(species, compound['compound_id'])] = (rt_pred, rt_std)
    
    # Filter candidates using mass tolerance
    candidates_before_mass = len(peak_df) * len(compound_df)
    candidates_after_mass = 0
    candidates_after_rt = 0
    
    true_positives_lost_mass = 0
    true_positives_lost_rt = 0
    false_positives_kept = 0
    
    for _, peak in peak_df.iterrows():
        species = peak['species']
        true_compound = peak['true_compound']
        
        for _, compound in compound_df.iterrows():
            # Mass filter
            mass_diff = abs(peak['mass'] - compound['true_mass'])
            if mass_diff > mass_tolerance:
                if true_compound == compound['compound_id']:
                    true_positives_lost_mass += 1
                continue
            
            candidates_after_mass += 1
            
            # RT filter (k*sigma window)
            rt_pred, rt_std = rt_predictions[(species, compound['compound_id'])]
            rt_z = abs(peak['rt'] - rt_pred) / rt_std
            
            if rt_z > k_value:
                if true_compound == compound['compound_id']:
                    true_positives_lost_rt += 1
                continue
                
            candidates_after_rt += 1
            
            # Count false positives that made it through
            if true_compound != compound['compound_id']:
                false_positives_kept += 1
    
    # Calculate metrics
    mass_filter_rate = 1 - (candidates_after_mass / candidates_before_mass)
    rt_filter_rate = 1 - (candidates_after_rt / candidates_after_mass) if candidates_after_mass > 0 else 0
    total_filter_rate = 1 - (candidates_after_rt / candidates_before_mass)
    
    # Estimate precision (simplified)
    true_positives_kept = sum(1 for v in true_assignments.values() if v is not None) - \
                         true_positives_lost_mass - true_positives_lost_rt
    
    precision = true_positives_kept / (true_positives_kept + false_positives_kept) \
                if (true_positives_kept + false_positives_kept) > 0 else 0
    
    recall = true_positives_kept / sum(1 for v in true_assignments.values() if v is not None) \
             if sum(1 for v in true_assignments.values() if v is not None) > 0 else 0
    
    results = {
        'k_value': k_value,
        'mass_tolerance': mass_tolerance,
        'filtering': {
            'mass_filter_rate': mass_filter_rate,
            'rt_filter_rate': rt_filter_rate,
            'total_filter_rate': total_filter_rate,
            'candidates_before': candidates_before_mass,
            'after_mass': candidates_after_mass,
            'after_rt': candidates_after_rt
        },
        'losses': {
            'true_positives_lost_mass': true_positives_lost_mass,
            'true_positives_lost_rt': true_positives_lost_rt,
            'false_positives_kept': false_positives_kept
        },
        'performance': {
            'precision_estimate': precision,
            'recall_estimate': recall,
            'f1_estimate': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        },
        'data_stats': {
            'n_compounds': len(compound_df),
            'n_isomers': len(compound_df[compound_df['type'] == 'isomer']),
            'n_near_isobars': len(compound_df[compound_df['type'] == 'near_isobar']),
            'n_peaks': len(peak_df),
            'n_true_assignments': sum(1 for v in true_assignments.values() if v is not None)
        }
    }
    
    if save_results:
        output_dir = Path('output/challenging_k_tests')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / f'k_{k_value:.2f}_challenging.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def compare_mass_vs_rt_filtering():
    """
    Analyze the relative importance of mass vs RT filtering.
    """
    print("\n" + "="*60)
    print("MASS vs RT FILTERING ANALYSIS")
    print("="*60)
    
    mass_tolerances = [0.001, 0.003, 0.005, 0.010, 0.020]
    k_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    results = []
    
    for mass_tol in mass_tolerances:
        for k in k_values:
            print(f"\nTesting mass_tol={mass_tol:.3f} Da, k={k:.1f}")
            
            result = test_k_with_challenging_data(k, mass_tol, save_results=False)
            result['mass_tolerance'] = mass_tol
            results.append(result)
            
            print(f"  Precision: {result['performance']['precision_estimate']:.3f}")
            print(f"  Recall: {result['performance']['recall_estimate']:.3f}")
            print(f"  Mass filters: {result['filtering']['mass_filter_rate']:.1%}")
            print(f"  RT filters additional: {result['filtering']['rt_filter_rate']:.1%}")
    
    # Analyze results
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    # Group by mass tolerance
    for mass_tol in mass_tolerances:
        subset = results_df[results_df['mass_tolerance'] == mass_tol]
        
        print(f"\nMass tolerance = {mass_tol:.3f} Da:")
        print(f"  Mean precision: {subset['performance'].apply(lambda x: x['precision_estimate']).mean():.3f}")
        print(f"  Precision range: {subset['performance'].apply(lambda x: x['precision_estimate']).min():.3f} - "
              f"{subset['performance'].apply(lambda x: x['precision_estimate']).max():.3f}")
        print(f"  Mass filter rate: {subset['filtering'].apply(lambda x: x['mass_filter_rate']).mean():.1%}")
    
    # Find when RT filtering matters most
    print("\n" + "-"*60)
    print("When does RT filtering matter?")
    print("-"*60)
    
    for _, row in results_df.iterrows():
        rt_impact = row['losses']['true_positives_lost_rt']
        if rt_impact > 0:
            print(f"  mass_tol={row['mass_tolerance']:.3f}, k={row['k_value']:.1f}: "
                  f"Lost {rt_impact} true positives to RT filter")
    
    return results_df


def main():
    """Main test function."""
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="Test k optimization with challenging data")
    parser.add_argument('--test-k', type=float, nargs='+',
                       help='Specific k values to test')
    parser.add_argument('--compare-filtering', action='store_true',
                       help='Compare mass vs RT filtering importance')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of results')
    
    args = parser.parse_args()
    
    if args.compare_filtering:
        results_df = compare_mass_vs_rt_filtering()
        
        if args.visualize:
            # Create heatmap of precision vs mass_tol and k
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Reshape data for heatmap
            mass_tols = results_df['mass_tolerance'].unique()
            k_vals = results_df['k_value'].unique()
            
            precision_matrix = np.zeros((len(mass_tols), len(k_vals)))
            recall_matrix = np.zeros((len(mass_tols), len(k_vals)))
            
            for i, mass_tol in enumerate(mass_tols):
                for j, k in enumerate(k_vals):
                    subset = results_df[(results_df['mass_tolerance'] == mass_tol) & 
                                      (results_df['k_value'] == k)]
                    if len(subset) > 0:
                        precision_matrix[i, j] = subset.iloc[0]['performance']['precision_estimate']
                        recall_matrix[i, j] = subset.iloc[0]['performance']['recall_estimate']
            
            # Precision heatmap
            im1 = axes[0].imshow(precision_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            axes[0].set_xticks(range(len(k_vals)))
            axes[0].set_xticklabels([f'{k:.1f}' for k in k_vals])
            axes[0].set_yticks(range(len(mass_tols)))
            axes[0].set_yticklabels([f'{m:.3f}' for m in mass_tols])
            axes[0].set_xlabel('k (RT window)')
            axes[0].set_ylabel('Mass tolerance (Da)')
            axes[0].set_title('Precision')
            plt.colorbar(im1, ax=axes[0])
            
            # Recall heatmap
            im2 = axes[1].imshow(recall_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            axes[1].set_xticks(range(len(k_vals)))
            axes[1].set_xticklabels([f'{k:.1f}' for k in k_vals])
            axes[1].set_yticks(range(len(mass_tols)))
            axes[1].set_yticklabels([f'{m:.3f}' for m in mass_tols])
            axes[1].set_xlabel('k (RT window)')
            axes[1].set_ylabel('Mass tolerance (Da)')
            axes[1].set_title('Recall')
            plt.colorbar(im2, ax=axes[1])
            
            plt.suptitle('Impact of Mass Tolerance and k on Performance\n(Challenging Data)', fontsize=14)
            plt.tight_layout()
            
            output_dir = Path('output/plots')
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'mass_vs_rt_filtering.png', dpi=150)
            plt.show()
            
    elif args.test_k:
        print("Testing specific k values with challenging data...")
        for k in args.test_k:
            print(f"\n{'='*40}")
            print(f"Testing k={k:.2f}")
            print('='*40)
            
            results = test_k_with_challenging_data(k)
            
            print(f"\nResults:")
            print(f"  Precision: {results['performance']['precision_estimate']:.3f}")
            print(f"  Recall: {results['performance']['recall_estimate']:.3f}")
            print(f"  F1: {results['performance']['f1_estimate']:.3f}")
            print(f"\nFiltering breakdown:")
            print(f"  Mass filter: {results['filtering']['mass_filter_rate']:.1%} of candidates")
            print(f"  RT filter: {results['filtering']['rt_filter_rate']:.1%} of remaining")
            print(f"  Total filtered: {results['filtering']['total_filter_rate']:.1%}")
            print(f"\nLosses:")
            print(f"  True positives lost to mass filter: {results['losses']['true_positives_lost_mass']}")
            print(f"  True positives lost to RT filter: {results['losses']['true_positives_lost_rt']}")
            print(f"  False positives kept: {results['losses']['false_positives_kept']}")
    
    else:
        # Default: test a range of k values
        k_values = [0.5, 1.0, 1.5, 2.0, 3.0]
        print(f"Testing k values: {k_values}")
        
        all_results = []
        for k in k_values:
            results = test_k_with_challenging_data(k)
            all_results.append(results)
            
            print(f"\nk={k:.1f}: P={results['performance']['precision_estimate']:.3f}, "
                  f"R={results['performance']['recall_estimate']:.3f}, "
                  f"Mass filters {results['filtering']['mass_filter_rate']:.1%}, "
                  f"RT filters {results['filtering']['rt_filter_rate']:.1%}")
        
        # Find best k
        best_result = max(all_results, 
                         key=lambda x: x['performance']['precision_estimate'])
        
        print(f"\n{'='*60}")
        print(f"Best k for challenging data: {best_result['k_value']:.1f}")
        print(f"  Precision: {best_result['performance']['precision_estimate']:.3f}")
        print(f"  Recall: {best_result['performance']['recall_estimate']:.3f}")
        print('='*60)


if __name__ == "__main__":
    main()