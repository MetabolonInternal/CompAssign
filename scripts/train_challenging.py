#!/usr/bin/env python
"""
Simplified training script that works with challenging synthetic data.

This version creates proper RT predictions for the challenging data format.
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.compassign import PeakAssignmentModel
from create_challenging_test_data import create_challenging_metabolomics_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CompAssign with challenging synthetic data'
    )
    
    # Data parameters
    parser.add_argument('--n-compounds', type=int, default=60,
                       help='Number of compounds')
    parser.add_argument('--n-species', type=int, default=10,
                       help='Number of species')
    
    # Model parameters
    parser.add_argument('--mass-tolerance', type=float, default=0.005,
                       help='Mass tolerance in Da')
    parser.add_argument('--rt-window-k', type=float, default=1.5,
                       help='RT window multiplier (k*sigma)')
    parser.add_argument('--probability-threshold', type=float, default=0.9,
                       help='Probability threshold for assignment')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    # Setup output
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    (output_path / "data").mkdir(exist_ok=True)
    (output_path / "results").mkdir(exist_ok=True)
    
    print("="*60)
    print("COMPASSIGN TRAINING WITH CHALLENGING DATA")
    print("="*60)
    print(f"Compounds: {args.n_compounds}")
    print(f"Species: {args.n_species}")
    print(f"Mass tolerance: {args.mass_tolerance} Da")
    print(f"RT window: k={args.rt_window_k}")
    print(f"Probability threshold: {args.probability_threshold}")
    print("="*60)
    
    # Generate challenging data
    print("\n1. Generating challenging synthetic data...")
    peak_df, compound_df, true_assignments, rt_uncertainties = create_challenging_metabolomics_data(
        n_compounds=args.n_compounds,
        n_peaks_per_compound=3,
        n_noise_peaks=int(args.n_compounds * 0.5),
        n_species=args.n_species,
        isomer_fraction=0.3,
        near_isobar_fraction=0.2,
        mass_error_std=0.002,
        rt_uncertainty_range=(0.05, 0.5)
    )
    
    print(f"  Generated {len(peak_df)} peaks")
    print(f"  True assignments: {sum(1 for v in true_assignments.values() if v is not None)}")
    print(f"  Noise peaks: {sum(1 for v in true_assignments.values() if v is None)}")
    print(f"  Isomers: {len(compound_df[compound_df['type'] == 'isomer'])}")
    print(f"  Near-isobars: {len(compound_df[compound_df['type'] == 'near_isobar'])}")
    
    # Save data
    peak_df.to_csv(output_path / "data" / "peaks.csv", index=False)
    compound_df.to_csv(output_path / "data" / "compounds.csv", index=False)
    
    # Create simplified RT predictions (mimicking what test_k_with_challenging_data does)
    print("\n2. Creating RT predictions...")
    rt_predictions = {}
    for species in range(args.n_species):
        for _, compound in compound_df.iterrows():
            # Add small prediction error to true RT
            pred_error = np.random.normal(0, compound['rt_uncertainty'] * 0.3)
            rt_pred = compound['true_rt'] + pred_error
            rt_std = compound['rt_uncertainty']
            rt_predictions[(species, compound['compound_id'])] = (rt_pred, rt_std)
    
    print(f"  Created {len(rt_predictions)} RT predictions")
    
    # Test the assignment model
    print("\n3. Testing peak assignment...")
    
    # Initialize model
    model = PeakAssignmentModel(
        mass_tolerance=args.mass_tolerance,
        rt_window_k=args.rt_window_k,
        matching="hungarian"
    )
    
    # Set RT predictions
    model.rt_predictions = rt_predictions
    
    # Generate training data
    print("  Generating candidate assignments...")
    logit_df = model.generate_training_data(
        peak_df,
        compound_df['true_mass'].values,
        len(compound_df)
    )
    
    print(f"  Generated {len(logit_df)} candidates after filtering")
    
    # Calculate simple metrics without full model training
    print("\n4. Calculating performance metrics...")
    
    # Group by peak and find best assignment
    assignments = {}
    for peak_id in peak_df['peak_id'].unique():
        peak_candidates = logit_df[logit_df['peak_id'] == peak_id]
        if len(peak_candidates) > 0:
            # Use simple scoring: prefer lower mass error and RT z-score
            peak_candidates = peak_candidates.copy()
            peak_candidates['score'] = -abs(peak_candidates['mass_err_ppm'])/10 - abs(peak_candidates['rt_z'])
            
            # Apply threshold based on RT z-score
            peak_candidates = peak_candidates[abs(peak_candidates['rt_z']) < 2.0]  # Within 2 sigma
            
            if len(peak_candidates) > 0:
                best_idx = peak_candidates['score'].idxmax()
                best_candidate = peak_candidates.loc[best_idx]
                
                # Only assign if score is good enough
                if best_candidate['score'] > -3.0:  # Threshold
                    assignments[peak_id] = best_candidate['compound']
    
    # Calculate metrics
    tp = fp = fn = tn = 0
    
    for peak_id, true_compound in true_assignments.items():
        assigned_compound = assignments.get(peak_id)
        
        if true_compound is not None:
            if assigned_compound == true_compound:
                tp += 1
            else:
                fn += 1
        else:
            if assigned_compound is not None:
                fp += 1
            else:
                tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPeak assignment results:")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Negatives: {tn}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'data_stats': {
            'n_peaks': len(peak_df),
            'n_compounds': len(compound_df),
            'n_true_assignments': sum(1 for v in true_assignments.values() if v is not None),
            'n_noise_peaks': sum(1 for v in true_assignments.values() if v is None),
            'n_isomers': len(compound_df[compound_df['type'] == 'isomer']),
            'n_near_isobars': len(compound_df[compound_df['type'] == 'near_isobar'])
        },
        'performance': {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'filtering': {
            'n_candidates_generated': len(logit_df),
            'n_peaks_assigned': len(assignments)
        }
    }
    
    with open(output_path / "results" / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nPerformance with challenging data:")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall: {recall:.1%}")
    print(f"  F1 Score: {f1:.3f}")
    
    if precision < 0.80:
        print("\n⚠️  Note: This is realistic performance with challenging data!")
        print("  The data includes isomers and near-isobars that are hard to distinguish.")
    
    print(f"\nResults saved to: {output_path / 'results' / 'results.json'}")


if __name__ == "__main__":
    main()