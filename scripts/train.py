#!/usr/bin/env python
"""
CompAssign training pipeline for RT regression and softmax compound assignment.

This script trains both the hierarchical RT model and the softmax peak assignment model.
Optimized for high precision metabolomics compound assignment.
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path
from datetime import datetime

from src.compassign.rt_hierarchical import HierarchicalRTModel
from src.compassign.peak_assignment_softmax import PeakAssignmentSoftmaxModel
from src.compassign.diagnostic_plots import create_all_diagnostic_plots
from src.compassign.assignment_plots import create_assignment_plots

# Use synthetic data generator
sys.path.insert(0, str(Path(__file__).parent))
from create_synthetic_data import create_metabolomics_data


def print_flush(msg):
    """Print with immediate flush for real-time logging"""
    print(msg, flush=True)
    sys.stdout.flush()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CompAssign softmax model for metabolomics compound assignment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Synthetic data includes isomers and near-isobars. See docs for details.'
    )
    
    # Data generation parameters
    parser.add_argument('--n-clusters', type=int, default=8,
                       help='Number of species clusters')
    parser.add_argument('--n-species', type=int, default=40,
                       help='Number of species (~5 per cluster)')
    parser.add_argument('--n-classes', type=int, default=4,
                       help='Number of compound classes')
    parser.add_argument('--n-compounds', type=int, default=10,
                       help='Number of compounds')
    
    # Sampling parameters
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of MCMC samples per chain')
    parser.add_argument('--n-chains', type=int, default=None,
                       help='Number of MCMC chains (default: 4)')
    parser.add_argument('--n-tune', type=int, default=1000,
                       help='Number of tuning steps')
    parser.add_argument('--target-accept', type=float, default=0.95,
                       help='Target acceptance rate for NUTS')
    
    # Model parameters
    parser.add_argument('--mass-tolerance', type=float, default=0.005,
                       help='Mass tolerance in Da')
    parser.add_argument('--rt-window-k', type=float, default=1.5,
                       help='RT window multiplier k*sigma')
    parser.add_argument('--probability-threshold', type=float, default=0.7,
                       help='Probability threshold for assignment')
    parser.add_argument('--matching', type=str, default='greedy',
                       choices=['hungarian', 'greedy', 'none'],
                       help='One-to-one matching algorithm')
    parser.add_argument('--test-thresholds', action='store_true',
                       help='Test multiple probability thresholds')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def test_threshold_impact(softmax_model, output_path):
    """Test different thresholds to show precision-recall tradeoff."""
    
    print_flush("\n" + "="*60)
    print_flush("THRESHOLD IMPACT ANALYSIS")
    print_flush("="*60)
    
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    results = []
    
    for thresh in thresholds:
        result = softmax_model.assign(prob_threshold=thresh)
        results.append({
            'threshold': thresh,
            'precision': result.precision,
            'recall': result.recall,
            'f1': result.f1,
            'n_assigned': len([v for v in result.assignments.values() if v is not None])
        })
        
        print_flush(f"\nThreshold: {thresh:.2f}")
        print_flush(f"  Precision: {result.precision:.3f}")
        print_flush(f"  Recall:    {result.recall:.3f}")
        print_flush(f"  F1:        {result.f1:.3f}")
        print_flush(f"  Assigned:  {results[-1]['n_assigned']}")
    
    # Save threshold analysis
    pd.DataFrame(results).to_csv(output_path / 'threshold_analysis.csv', index=False)
    
    print_flush("\n" + "="*60)
    print_flush("Higher thresholds increase precision at the cost of recall")
    print_flush("="*60)


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Setup output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "models").mkdir(exist_ok=True)
    (output_path / "plots").mkdir(exist_ok=True)
    (output_path / "results").mkdir(exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print_flush("="*60)
    print_flush("COMPASSIGN TRAINING PIPELINE")
    print_flush(f"Compounds: {args.n_compounds}, Species: {args.n_species}")
    print_flush(f"Mass tolerance: {args.mass_tolerance} Da, RT window: ±{args.rt_window_k}σ")
    print_flush(f"Matching: {args.matching}, Threshold: {args.probability_threshold}")
    print_flush("="*60)
    
    # 1. Generate synthetic data
    print_flush("\n1. Generating synthetic data...")
    # create_metabolomics_data returns: peak_df, compound_df, true_assignments, rt_uncertainties, hierarchical_params
    peak_df, compound_df, true_assignments, rt_uncertainties, hierarchical_params = create_metabolomics_data(
        n_compounds=args.n_compounds,
        n_species=args.n_species,
        n_peaks_per_compound=3,
        n_noise_peaks=100
    )
    
    # Adapt to expected format
    compound_info = compound_df
    # Create RT observations from peaks (for RT model training)
    rt_df = peak_df[peak_df['true_compound'].notna()].copy()
    rt_df = rt_df.rename(columns={'true_compound': 'compound'})
    
    # Print data summary
    true_assignments = peak_df['true_compound'].notna().sum()
    noise_peaks = peak_df['true_compound'].isna().sum()
    print_flush(f"  Observations: {len(rt_df)}")
    print_flush(f"  Peaks: {len(peak_df)}")
    print_flush(f"  True assignments: {true_assignments}")
    print_flush(f"  Noise peaks: {noise_peaks}")
    print_flush(f"  Compounds: {args.n_compounds}")
    
    # Count isomers and near-isobars
    isomer_count = 0
    near_isobar_count = 0
    for i in range(len(compound_info)):
        for j in range(i + 1, len(compound_info)):
            mass_diff = abs(compound_info.iloc[i]['true_mass'] - compound_info.iloc[j]['true_mass'])
            if mass_diff < 0.001:  # Same formula (isomers)
                isomer_count += 1
            elif mass_diff < 0.01:  # Near-isobars
                near_isobar_count += 1
    
    print_flush(f"  - Isomers: {isomer_count}")
    print_flush(f"  - Near-isobars: {near_isobar_count}")
    
    # 2. Train RT model
    print_flush("\n2. Training hierarchical RT model...")
    
    # Generate some dummy descriptors and internal standards for now
    n_descriptors = 5
    descriptors = np.random.randn(args.n_compounds, n_descriptors)
    internal_std = np.random.rand(args.n_species)
    
    rt_model = HierarchicalRTModel(
        n_clusters=hierarchical_params['n_clusters'],
        n_species=args.n_species,
        n_classes=hierarchical_params['n_classes'],
        n_compounds=args.n_compounds,
        species_cluster=hierarchical_params['species_cluster'],
        compound_class=hierarchical_params['compound_class'],
        descriptors=descriptors,
        internal_std=internal_std
    )
    
    # Build model with RT observations
    rt_model.build_model(rt_df)
    
    print_flush(f"Sampling with {args.n_chains or 4} chains, {args.n_samples} samples each...")
    print_flush(f"Target accept: {args.target_accept}, Max treedepth: 15")
    
    sample_kwargs = {
        'n_samples': args.n_samples,
        'n_tune': args.n_tune,
        'target_accept': args.target_accept,
        'max_treedepth': 15,
        'random_seed': args.seed
    }
    if args.n_chains is not None:
        sample_kwargs['n_chains'] = args.n_chains
    
    trace_rt = rt_model.sample(**sample_kwargs)
    
    # Save RT trace
    trace_rt.to_netcdf(output_path / "models" / "rt_trace.nc")
    
    # 3. Create RT model diagnostic plots
    print_flush("\n3. Creating RT model diagnostic plots...")
    # For now, skip PPC and just pass empty dict
    create_all_diagnostic_plots(
        trace_rt, 
        {},  # No PPC results for now
        output_path / "plots" / "rt_model"
    )
    print_flush(f"Diagnostic plots saved to: {output_path / 'plots' / 'rt_model'}")
    
    # 4. Train softmax assignment model
    print_flush("\n4. Training peak assignment model...")
    print_flush(f"   Model: softmax")
    print_flush(f"   Mass tolerance: {args.mass_tolerance} Da")
    print_flush(f"   RT window: ±{args.rt_window_k}σ")
    print_flush(f"   Matching algorithm: {args.matching}")
    print_flush(f"   Probability threshold: {args.probability_threshold}")
    
    # Initialize softmax model
    softmax_model = PeakAssignmentSoftmaxModel(
        mass_tolerance=args.mass_tolerance,
        rt_window_k=args.rt_window_k,
        random_seed=args.seed
    )
    
    # Compute RT predictions
    rt_predictions = softmax_model.compute_rt_predictions(
        trace_rt=trace_rt,
        n_species=args.n_species,
        n_compounds=args.n_compounds,
        descriptors=descriptors,
        internal_std=internal_std,
        rt_model=rt_model
    )
    
    # Generate training data
    train_pack = softmax_model.generate_training_data(
        peak_df=peak_df,
        compound_mass=compound_info['true_mass'].values,
        n_compounds=args.n_compounds,
        species_cluster=hierarchical_params['species_cluster'],
        initial_labeled_fraction=0.8  # 80% labeled for training
    )
    
    # Build and sample
    softmax_model.build_model()
    trace_assignment = softmax_model.sample(
        draws=args.n_samples,
        tune=args.n_tune,
        chains=args.n_chains or 2,
        target_accept=args.target_accept
    )
    
    # Get predictions
    results = softmax_model.assign(prob_threshold=args.probability_threshold)
    
    # Save traces and results
    trace_assignment.to_netcdf(output_path / "models" / "assignment_trace.nc")
    
    # Save peak assignments
    peak_id_array = softmax_model.train_pack['peak_ids']
    assignment_df = pd.DataFrame([
        {
            'peak_id': int(pid),
            'assigned_compound': results.assignments.get(int(pid)),
            'probability': results.top_prob.get(int(pid), 0.0)
        }
        for pid in peak_id_array
    ])
    assignment_df.to_csv(output_path / "results" / "peak_assignments.csv", index=False)
    
    # Save metrics
    metrics = {
        'precision': float(results.precision),
        'recall': float(results.recall),
        'f1': float(results.f1),
        'ece': float(results.ece),
        'mce': float(results.mce),
        'true_positives': int(results.confusion_matrix.get('TP', 0)),
        'false_positives': int(results.confusion_matrix.get('FP', 0)),
        'false_negatives': int(results.confusion_matrix.get('FN', 0)),
        'n_assigned': len([v for v in results.assignments.values() if v is not None]),
        'n_peaks': len(results.assignments),
        'timestamp': datetime.now().isoformat(),
        'model': 'softmax',
        'mass_tolerance': args.mass_tolerance,
        'rt_window_k': args.rt_window_k,
        'probability_threshold': args.probability_threshold
    }
    
    with open(output_path / "results" / "assignment_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 5. Print results
    print_flush("\n5. Making predictions...")
    print_flush(f"\nResults:")
    print_flush(f"  Assignments made: {metrics['n_assigned']}")
    print_flush(f"  Null assignments: {metrics['n_peaks'] - metrics['n_assigned']}")
    print_flush(f"  Precision: {metrics['precision']:.3f}")
    print_flush(f"  Recall: {metrics['recall']:.3f}")
    print_flush(f"  F1: {metrics['f1']:.3f}")
    print_flush(f"  ECE: {metrics['ece']:.3f}")
    print_flush(f"  MCE: {metrics['mce']:.3f}")
    
    # Test threshold impact if requested
    if args.test_thresholds:
        test_threshold_impact(softmax_model, output_path / "results")
    
    # Create assignment plots (skip for now - needs fixing)
    # print_flush("\n6. Creating assignment plots...")
    # create_assignment_plots(
    #     results=results,
    #     peak_df=peak_df,
    #     compound_info=compound_info,
    #     output_dir=output_path / "plots" / "assignments"
    # )
    
    # Print final summary
    print_flush("\n" + "="*60)
    print_flush("TRAINING COMPLETE")
    print_flush("="*60)
    
    # Calculate dataset statistics
    N, K = softmax_model.train_pack['mask'].shape
    n_valid_slots = softmax_model.train_pack['mask'].sum()
    train_idx = np.where(softmax_model.train_pack['labels'] >= 0)[0]
    n_train = len(train_idx)
    n_test = N - n_train
    
    print_flush(f"\nDataset summary:")
    print_flush(f"  Total peaks: {N}")
    print_flush(f"  Max candidates: {K-1} (+ null)")
    print_flush(f"  Valid slots: {n_valid_slots}")
    print_flush(f"  Train peaks: {n_train} ({n_train/N*100:.1f}%)")
    print_flush(f"  Test peaks: {n_test} ({n_test/N*100:.1f}%)")
    
    print_flush(f"\nTest-set performance:")
    print_flush(f"  Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print_flush(f"  Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print_flush(f"  F1:        {metrics['f1']:.3f}")
    print_flush(f"  ECE:       {metrics['ece']:.3f}")
    print_flush(f"  MCE:       {metrics['mce']:.3f}")
    print_flush(f"  False Positives: {metrics['false_positives']}")
    print_flush(f"  True Positives:  {metrics['true_positives']}")
    
    print_flush(f"\nParameters used:")
    print_flush(f"  Mass tolerance: {args.mass_tolerance} Da")
    print_flush(f"  RT window: ±{args.rt_window_k}σ")
    print_flush(f"  Probability threshold: {args.probability_threshold}")
    print_flush(f"  Matching: {args.matching}")
    
    print_flush(f"\nOutput directory: {output_path}/")


if __name__ == "__main__":
    main()