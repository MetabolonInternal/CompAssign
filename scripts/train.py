#!/usr/bin/env python
"""
CompAssign training pipeline for RT regression and hierarchical Bayesian assignment.

This script trains both the hierarchical RT model and the hierarchical Bayesian assignment model.
Optimized for high precision metabolomics compound assignment with minimal features.
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
from src.compassign.peak_assignment import PeakAssignment
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
        description='Train CompAssign hierarchical Bayesian model for metabolomics compound assignment',
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
    parser.add_argument('--n-compounds', type=int, default=20,
                       help='Number of compounds')
    parser.add_argument('--decoy-fraction', type=float, default=0.5,
                       help='Fraction of compounds that are decoys (never appear in samples)')
    
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
    parser.add_argument('--mass-tolerance', type=float, default=0.01,
                       help='Mass tolerance in Da (default: 0.01 for harder data)')
    parser.add_argument('--rt-window-k', type=float, default=2.0,
                       help='RT window multiplier k*sigma (default: 2.0 for harder data)')
    parser.add_argument('--probability-threshold', type=float, default=0.7,
                       help='Probability threshold for assignment')
    parser.add_argument('--test-thresholds', action='store_true',
                       help='Test multiple probability thresholds')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def test_threshold_impact(assignment_model, output_path):
    """Test different thresholds to show precision-recall tradeoff."""
    
    print_flush("\n" + "="*60)
    print_flush("THRESHOLD IMPACT ANALYSIS")
    print_flush("="*60)
    
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    results = []
    
    for thresh in thresholds:
        result = assignment_model.assign(prob_threshold=thresh)
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
    
    # Start training
    
    # 1. Generate synthetic data
    print_flush("\n1. Generating synthetic data...")
    # create_metabolomics_data returns: peak_df, compound_df, true_assignments, rt_uncertainties, hierarchical_params
    peak_df, compound_df, true_assignments, rt_uncertainties, hierarchical_params = create_metabolomics_data(
        n_compounds=args.n_compounds,
        n_species=args.n_species,
        n_peaks_per_compound=3,
        n_noise_peaks=max(200, args.n_compounds * 10),  # More noise peaks
        isomer_fraction=0.4,  # 40% of compounds are isomers (increased)
        near_isobar_fraction=0.3,  # 30% are near-isobars (increased)
        mass_error_std=0.004,  # Higher mass error for more overlap
        rt_uncertainty_range=(0.2, 0.8),  # Higher RT uncertainty
        decoy_fraction=args.decoy_fraction  # Use command-line parameter
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
    used_target_accept = max(args.target_accept, 0.99)
    print_flush(f"Target accept: {used_target_accept}, Max treedepth: 15")
    
    sample_kwargs = {
        'n_samples': args.n_samples,
        'n_tune': args.n_tune,
        'target_accept': used_target_accept,
        'max_treedepth': 15,
        'random_seed': args.seed
    }
    # Always pass an explicit chain count to keep logs consistent
    sample_kwargs['n_chains'] = args.n_chains or 4
    
    trace_rt = rt_model.sample(**sample_kwargs)
    
    # Save RT trace
    trace_rt.to_netcdf(output_path / "models" / "rt_trace.nc")
    
    # 3. Create RT model diagnostic plots
    print_flush("\n3. Creating RT model diagnostic plots...")
    # For now, skip PPC and just pass empty dict
    create_all_diagnostic_plots(
        trace_rt,
        {},  # No PPC results for now
        output_path
    )
    
    # 4. Train softmax assignment model
    print_flush("\n4. Training peak assignment model...")
    
    # Initialize hierarchical Bayesian model
    assignment_model = PeakAssignment(
        mass_tolerance=args.mass_tolerance,
        rt_window_k=args.rt_window_k,
        random_seed=args.seed
    )
    
    # Compute RT predictions
    rt_predictions = assignment_model.compute_rt_predictions(
        trace_rt=trace_rt,
        n_species=args.n_species,
        n_compounds=args.n_compounds,
        descriptors=descriptors,
        internal_std=internal_std,
        rt_model=rt_model
    )
    
    # Generate training data
    train_pack = assignment_model.generate_training_data(
        peak_df=peak_df,
        compound_mass=compound_info['true_mass'].values,
        n_compounds=args.n_compounds,
        compound_info=compound_info,  # Pass compound_info to skip decoys during training
        initial_labeled_fraction=0.8  # 80% labeled for training
    )
    
    # Build and sample
    assignment_model.build_model()
    trace_assignment = assignment_model.sample(
        draws=args.n_samples,
        tune=args.n_tune,
        chains=args.n_chains or 4,  # Use 4 chains by default like RT model
        target_accept=args.target_accept
    )
    
    # Get predictions (pass compound_info to properly handle decoys)
    results = assignment_model.assign(
        prob_threshold=args.probability_threshold,
        compound_info=compound_info
    )
    
    # Save traces and results
    trace_assignment.to_netcdf(output_path / "models" / "assignment_trace.nc")
    
    # Save peak assignments
    peak_id_array = assignment_model.train_pack['peak_ids']
    assignment_df = pd.DataFrame([
        {
            'peak_id': int(pid),
            'assigned_compound': results.assignments.get(int(pid)),
            'probability': results.top_prob.get(int(pid), 0.0)
        }
        for pid in peak_id_array
    ])
    assignment_df.to_csv(output_path / "results" / "peak_assignments.csv", index=False)
    
    # Save compound information (including decoy status)
    compound_info.to_csv(output_path / "results" / "compound_info.csv", index=False)
    
    # Analyze confidence distributions for real vs decoy assignments
    if 'is_decoy' in compound_info.columns and results.peaks_by_compound:
        decoy_ids = set(compound_info[compound_info['is_decoy']]['compound_id'].values)
        real_ids = set(compound_info[~compound_info['is_decoy']]['compound_id'].values)
        
        decoy_probs = []
        real_probs = []
        
        for cid, peak_ids in results.peaks_by_compound.items():
            for pid in peak_ids:
                prob = results.top_prob.get(pid, 0.0)
                if cid in decoy_ids:
                    decoy_probs.append(prob)
                elif cid in real_ids:
                    real_probs.append(prob)
        
        # Save confidence analysis
        confidence_stats = {
            'real_assignments': {
                'count': len(real_probs),
                'mean_confidence': np.mean(real_probs) if real_probs else 0,
                'std_confidence': np.std(real_probs) if real_probs else 0,
                'min_confidence': min(real_probs) if real_probs else 0,
                'max_confidence': max(real_probs) if real_probs else 0
            },
            'decoy_assignments': {
                'count': len(decoy_probs),
                'mean_confidence': np.mean(decoy_probs) if decoy_probs else 0,
                'std_confidence': np.std(decoy_probs) if decoy_probs else 0,
                'min_confidence': min(decoy_probs) if decoy_probs else 0,
                'max_confidence': max(decoy_probs) if decoy_probs else 0
            }
        }
        
        with open(output_path / "results" / "confidence_analysis.json", 'w') as f:
            json.dump(confidence_stats, f, indent=2)
    
    # Save metrics
    metrics = {
        # Peak-level metrics
        'precision': float(results.precision),
        'recall': float(results.recall),
        'f1': float(results.f1),
        # Compound-level metrics (PRIMARY)
        'compound_precision': float(results.compound_precision),
        'compound_recall': float(results.compound_recall),
        'compound_f1': float(results.compound_f1),
        'compounds_identified': results.compound_metrics['identified'],
        'compounds_total': results.compound_metrics['total'],
        'compounds_false_positives': results.compound_metrics.get('false_positives', 0),
        'compounds_decoys_assigned': results.compound_metrics.get('decoys_assigned', 0),
        # Coverage
        'mean_coverage': float(results.mean_coverage),
        # Other metrics
        'ece': float(results.ece),
        'true_positives': int(results.confusion_matrix.get('TP', 0)),
        'false_positives': int(results.confusion_matrix.get('FP', 0)),
        'false_negatives': int(results.confusion_matrix.get('FN', 0)),
        'n_assigned': len([v for v in results.assignments.values() if v is not None]),
        'n_peaks': len(results.assignments),
        'timestamp': datetime.now().isoformat(),
        'model': 'hierarchical_bayesian',
        'mass_tolerance': args.mass_tolerance,
        'rt_window_k': args.rt_window_k,
        'probability_threshold': args.probability_threshold
    }
    
    with open(output_path / "results" / "assignment_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 5. Print results (concise version)
    print_flush("\n5. Results:")
    print_flush(f"  Compounds: {metrics['compounds_identified']}/{metrics['compounds_total']} identified (F1={metrics['compound_f1']:.3f})")
    print_flush(f"  Coverage: {metrics['mean_coverage']:.1%} of peaks per compound")
    print_flush(f"  Peaks: {metrics['n_assigned']}/{metrics['n_peaks']} assigned (F1={metrics['f1']:.3f})")
    
    # Test threshold impact if requested
    if args.test_thresholds:
        test_threshold_impact(assignment_model, output_path / "results")
    
    # Create assignment plots (skip for now - needs fixing)
    # print_flush("\n6. Creating assignment plots...")
    # create_assignment_plots(
    #     results=results,
    #     peak_df=peak_df,
    #     compound_info=compound_info,
    #     output_dir=output_path / "plots" / "assignments"
    # )
    
    # Print final summary
    print_flush("\n5. Training complete")
    
    # Calculate dataset statistics
    N, K = assignment_model.train_pack['mask'].shape
    n_valid_slots = assignment_model.train_pack['mask'].sum()
    train_idx = np.where(assignment_model.train_pack['labels'] >= 0)[0]
    n_train = len(train_idx)
    n_test = N - n_train
    
    print_flush(f"\nDataset summary:")
    print_flush(f"  Total peaks: {N}")
    print_flush(f"  Max candidates: {K-1} (+ null)")
    print_flush(f"  Valid slots: {n_valid_slots}")
    print_flush(f"  Train peaks: {n_train} ({n_train/N*100:.1f}%)")
    print_flush(f"  Test peaks: {n_test} ({n_test/N*100:.1f}%)")
    
    print_flush(f"\nCompound-level performance:")
    print_flush(f"  Compound Precision: {metrics['compound_precision']:.3f} ({metrics['compound_precision']*100:.1f}%)")
    print_flush(f"  Compound Recall:    {metrics['compound_recall']:.3f} ({metrics['compound_recall']*100:.1f}%)")
    print_flush(f"  Compound F1:        {metrics['compound_f1']:.3f}")
    print_flush(f"  Mean Coverage:      {metrics['mean_coverage']:.3f}")
    
    # Show decoy statistics if available
    n_decoys = len(compound_info[compound_info['is_decoy']]) if 'is_decoy' in compound_info.columns else 0
    n_real = len(compound_info[~compound_info['is_decoy']]) if 'is_decoy' in compound_info.columns else len(compound_info)
    decoys_assigned = results.compound_metrics.get('decoys_assigned', 0)
    
    if n_decoys > 0:
        print_flush(f"\nDecoy detection:")
        print_flush(f"  Library composition: {n_real} real, {n_decoys} decoys")
        print_flush(f"  Decoys incorrectly assigned: {decoys_assigned}/{n_decoys} ({decoys_assigned/n_decoys*100:.1f}%)")
        print_flush(f"  False positives from decoys: {decoys_assigned}")
        print_flush(f"  Compound false positives: {results.compound_metrics.get('false_positives', 0)}")
        
        # Show which decoys were assigned if any
        if decoys_assigned > 0 and hasattr(results, 'peaks_by_compound'):
            decoy_ids = set(compound_info[compound_info['is_decoy']]['compound_id'].values)
            assigned_decoys = [cid for cid in results.peaks_by_compound.keys() if cid in decoy_ids]
            if assigned_decoys:
                print_flush(f"  Decoy IDs assigned: {sorted(assigned_decoys)[:5]}{'...' if len(assigned_decoys) > 5 else ''}")
        
        # Show confidence analysis if available
        if Path(output_path / "results" / "confidence_analysis.json").exists():
            with open(output_path / "results" / "confidence_analysis.json") as f:
                conf_stats = json.load(f)
            
            if conf_stats['real_assignments']['count'] > 0:
                print_flush(f"\nConfidence analysis:")
                print_flush(f"  Real compound assignments:")
                print_flush(f"    Mean confidence: {conf_stats['real_assignments']['mean_confidence']:.3f}")
                print_flush(f"    Range: [{conf_stats['real_assignments']['min_confidence']:.3f}, {conf_stats['real_assignments']['max_confidence']:.3f}]")
            
            if conf_stats['decoy_assignments']['count'] > 0:
                print_flush(f"  Decoy compound assignments:")
                print_flush(f"    Mean confidence: {conf_stats['decoy_assignments']['mean_confidence']:.3f}")
                print_flush(f"    Range: [{conf_stats['decoy_assignments']['min_confidence']:.3f}, {conf_stats['decoy_assignments']['max_confidence']:.3f}]")
    
    print_flush(f"\nPeak-level performance:")
    print_flush(f"  Peak Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print_flush(f"  Peak Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print_flush(f"  Peak F1:        {metrics['f1']:.3f}")
    print_flush(f"  ECE:            {metrics['ece']:.3f}")
    
    print_flush(f"\nResults saved to: {output_path}/")


if __name__ == "__main__":
    main()
