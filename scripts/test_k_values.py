#!/usr/bin/env python
"""
Individual k value testing script for serial execution.

Designed to be called by subagents with specific k values.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compassign.peak_assignment import PeakAssignmentModel
from scripts.create_challenging_test_data import (
    create_challenging_metabolomics_data,
    test_k_with_challenging_data
)


def test_single_k_value(k_value: float, 
                       mass_tolerance: float = 0.005,
                       target_precision: float = 0.99,
                       use_challenging_data: bool = True,
                       output_file: str = None):
    """
    Test a single k value and save results.
    
    Parameters
    ----------
    k_value : float
        The k value to test
    mass_tolerance : float
        Mass tolerance in Da
    target_precision : float
        Target precision for threshold tuning
    use_challenging_data : bool
        Use challenging data with isomers/near-isobars (default: True)
    output_file : str
        Where to save results (JSON format)
    """
    print(f"Testing k={k_value:.2f}")
    print("-" * 40)
    
    if use_challenging_data:
        print("Using CHALLENGING data (with isomers and near-isobars)")
        # Use the more realistic challenging data
        result = test_k_with_challenging_data(k_value, mass_tolerance, save_results=False)
        
        # Format results for compatibility
        results = {
            'k_value': k_value,
            'mass_tolerance': mass_tolerance,
            'optimal_threshold': 0.9,  # Default for challenging data
            'metrics': {
                'precision': result['performance']['precision_estimate'],
                'recall': result['performance']['recall_estimate'],
                'f1_score': result['performance']['f1_estimate'],
                'true_positives': int(result['data_stats']['n_true_assignments'] * result['performance']['recall_estimate']),
                'false_positives': int(result['losses']['false_positives_kept']),
                'false_negatives': int(result['losses']['true_positives_lost_rt']),
                'true_negatives': 0  # Not tracked in simplified version
            },
            'filtering': result['filtering'],
            'data_stats': result['data_stats'],
            'data_type': 'challenging'
        }
        
        print(f"\nResults:")
        print(f"  Precision: {results['metrics']['precision']:.3f}")
        print(f"  Recall: {results['metrics']['recall']:.3f}")
        print(f"  F1 Score: {results['metrics']['f1_score']:.3f}")
        print(f"  Filter Rate: {results['filtering']['total_filter_rate']:.1%}")
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to {output_path}")
        
        return results
        
    # Original simple data code (kept for backwards compatibility)
    print("Using SIMPLE data (random, no isomers)")
    
    # Generate simple synthetic data
    from scripts.test_uncertainty_propagation import create_synthetic_test_data
    peak_data, compound_data, true_assignments, rt_trace = create_synthetic_test_data(
        200, 50, n_species=5
    )
    
    # Add true_compound column to peak_data
    peak_data['true_compound'] = peak_data['peak_id'].map(true_assignments)
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    train_peaks, test_peaks = train_test_split(
        peak_data, test_size=0.3, random_state=42
    )
    
    # Create assignment model with this k
    model = PeakAssignmentModel(
        mass_tolerance=mass_tolerance,
        rt_window_k=k_value,
        matching="greedy"
    )
    
    # Compute RT predictions (simplified for testing)
    rt_predictions = {}
    for s in range(peak_data['species'].nunique()):
        for c in range(n_compounds):
            rt_mean = compound_data.iloc[c]['true_rt_mean'] + np.random.normal(0, 0.1)
            rt_std = compound_data.iloc[c]['true_rt_std']
            rt_predictions[(s, c)] = (rt_mean, rt_std)
    
    model.rt_predictions = rt_predictions
    
    # Generate training data
    train_logit = model.generate_training_data(
        train_peaks,
        compound_data['mass'].values,
        n_compounds
    )
    
    # Find optimal threshold using binary search
    print("Finding optimal threshold...")
    thresholds = np.linspace(0.1, 0.99, 20)
    best_threshold = 0.5
    best_diff = float('inf')
    
    for threshold in thresholds:
        # Simple scoring based on RT z-score
        train_logit['prob'] = 1 / (1 + np.exp(-train_logit['rt_z']))
        train_logit['predicted'] = train_logit['prob'] > threshold
        
        tp = ((train_logit['predicted'] == 1) & (train_logit['label'] == 1)).sum()
        fp = ((train_logit['predicted'] == 1) & (train_logit['label'] == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Find threshold closest to target precision
        diff = abs(precision - target_precision)
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold
            if precision >= target_precision:
                break  # Stop if we meet target
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    
    # Evaluate on test set
    test_logit = model.generate_training_data(
        test_peaks,
        compound_data['mass'].values,
        n_compounds
    )
    
    # Apply threshold
    test_logit['prob'] = 1 / (1 + np.exp(-test_logit['rt_z']))
    test_logit['predicted'] = test_logit['prob'] > best_threshold
    
    # Calculate metrics
    tp = ((test_logit['predicted'] == 1) & (test_logit['label'] == 1)).sum()
    fp = ((test_logit['predicted'] == 1) & (test_logit['label'] == 0)).sum()
    fn = ((test_logit['predicted'] == 0) & (test_logit['label'] == 1)).sum()
    tn = ((test_logit['predicted'] == 0) & (test_logit['label'] == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate filtering statistics
    n_total_candidates = len(test_peaks) * n_compounds
    n_candidates_after_filter = len(test_logit)
    n_filtered = n_total_candidates - n_candidates_after_filter
    filter_rate = n_filtered / n_total_candidates
    
    # Count true positives filtered out
    true_pairs = set()
    for peak_id, compound_id in true_assignments.items():
        if compound_id is not None and peak_id in test_peaks['peak_id'].values:
            species = test_peaks[test_peaks['peak_id'] == peak_id]['species'].iloc[0]
            true_pairs.add((peak_id, species, compound_id))
    
    retained_pairs = set(zip(test_logit['peak_id'], 
                            test_logit['species'], 
                            test_logit['compound']))
    
    true_filtered = len(true_pairs - retained_pairs)
    
    results = {
        'k_value': k_value,
        'mass_tolerance': mass_tolerance,
        'optimal_threshold': best_threshold,
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn)
        },
        'filtering': {
            'n_total_candidates': n_total_candidates,
            'n_filtered': n_filtered,
            'filter_rate': filter_rate,
            'n_true_filtered': true_filtered,
            'n_candidates_retained': n_candidates_after_filter
        },
        'data_stats': {
            'n_train_peaks': len(train_peaks),
            'n_test_peaks': len(test_peaks),
            'n_compounds': n_compounds,
            'n_true_assignments': sum(1 for v in true_assignments.values() if v is not None)
        }
    }
    
    # Print results
    print(f"\nResults for k={k_value:.2f}:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  Filter Rate: {filter_rate:.1%}")
    print(f"  True positives filtered: {true_filtered}")
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test a single k value")
    parser.add_argument('k_value', type=float, help='k value to test')
    parser.add_argument('--mass-tolerance', type=float, default=0.005,
                       help='Mass tolerance in Da')
    parser.add_argument('--target-precision', type=float, default=0.99,
                       help='Target precision for threshold tuning')
    parser.add_argument('--n-peaks', type=int, default=200,
                       help='Number of peaks in synthetic data')
    parser.add_argument('--n-compounds', type=int, default=50,
                       help='Number of compounds in synthetic data')
    parser.add_argument('--output', type=str, 
                       help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    results = test_single_k_value(
        k_value=args.k_value,
        mass_tolerance=args.mass_tolerance,
        target_precision=args.target_precision,
        use_challenging_data=True,  # Use challenging data by default
        output_file=args.output
    )
    
    # Return exit code based on whether target precision was met
    if results['metrics']['precision'] >= args.target_precision:
        print(f"\n✅ Target precision {args.target_precision:.1%} achieved!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Target precision {args.target_precision:.1%} not met")
        sys.exit(1)


if __name__ == "__main__":
    main()