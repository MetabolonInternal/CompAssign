#!/usr/bin/env python
"""
CompAssign training pipeline for RT regression and compound assignment.

This script trains both the hierarchical RT model and the peak assignment model.
Use --enhanced flag for ultra-high precision (>95%) required for production.
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path
from datetime import datetime

from src.compassign import (
    generate_synthetic_data, 
    HierarchicalRTModel, 
    PeakAssignmentModel,
    EnhancedPeakAssignmentModel
)
from src.compassign.diagnostic_plots import create_all_diagnostic_plots
from src.compassign.assignment_plots import create_assignment_plots


def print_flush(msg):
    """Print with immediate flush for real-time logging"""
    print(msg, flush=True)
    sys.stdout.flush()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CompAssign models for metabolomics compound assignment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument('--model', type=str, default='standard', choices=['standard', 'enhanced'],
                       help='Model type: "standard" for baseline (84%% precision) or "enhanced" for production (>95%% precision)')
    
    # Data generation parameters
    parser.add_argument('--n-clusters', type=int, default=5,
                       help='Number of species clusters')
    parser.add_argument('--n-species', type=int, default=80,
                       help='Number of species')
    parser.add_argument('--n-classes', type=int, default=4,
                       help='Number of compound classes')
    parser.add_argument('--n-compounds', type=int, default=60,
                       help='Number of compounds')
    
    # Sampling parameters
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of MCMC samples per chain')
    parser.add_argument('--n-chains', type=int, default=2,
                       help='Number of MCMC chains')
    parser.add_argument('--n-tune', type=int, default=1000,
                       help='Number of tuning steps')
    parser.add_argument('--target-accept', type=float, default=0.95,
                       help='Target acceptance rate for NUTS')
    
    # Enhanced model specific parameters
    parser.add_argument('--mass-tolerance', type=float, default=0.01,
                       help='Mass tolerance in Da (use 0.005 for enhanced)')
    parser.add_argument('--fp-penalty', type=float, default=5.0,
                       help='False positive penalty weight (enhanced model only)')
    parser.add_argument('--probability-threshold', type=float, default=0.5,
                       help='Probability threshold for assignment (use 0.9 for enhanced)')
    parser.add_argument('--high-precision-threshold', type=float, default=0.9,
                       help='High confidence threshold (enhanced model only)')
    parser.add_argument('--review-threshold', type=float, default=0.7,
                       help='Review queue threshold (enhanced model only)')
    parser.add_argument('--test-thresholds', action='store_true',
                       help='Test multiple probability thresholds')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Auto-adjust parameters for enhanced mode
    if args.model == 'enhanced':
        if args.mass_tolerance == 0.01:
            args.mass_tolerance = 0.005
            print_flush("Auto-adjusted mass_tolerance to 0.005 for enhanced mode")
        if args.probability_threshold == 0.5:
            args.probability_threshold = 0.9
            print_flush("Auto-adjusted probability_threshold to 0.9 for enhanced mode")
    
    return args


def test_threshold_impact(assignment_model, peak_df, output_path, is_enhanced=False):
    """Test different thresholds to show precision-recall tradeoff."""
    
    print_flush("\n" + "="*60)
    print_flush("THRESHOLD IMPACT ANALYSIS")
    print_flush("="*60)
    
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    results = []
    
    for thresh in thresholds:
        if is_enhanced:
            result = assignment_model.predict_assignments_staged(
                peak_df, 
                high_precision_threshold=thresh,
                review_threshold=thresh * 0.8
            )
        else:
            result = assignment_model.predict_assignments(
                peak_df,
                probability_threshold=thresh
            )
        
        results.append({
            'threshold': thresh,
            'precision': result.precision,
            'recall': result.recall,
            'f1': result.f1_score,
            'false_positives': result.confusion_matrix['FP']
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    print_flush("\nThreshold Impact on Performance:")
    print_flush(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv(output_path / "results" / "threshold_analysis.csv", index=False)
    
    # Find optimal threshold for >95% precision
    high_precision = results_df[results_df['precision'] >= 0.95]
    if not high_precision.empty:
        optimal = high_precision.iloc[0]
        print_flush(f"\n✓ Optimal threshold for >95% precision: {optimal['threshold']}")
    else:
        best = results_df.loc[results_df['precision'].idxmax()]
        print_flush(f"\n⚠ Best achievable precision: {best['precision']:.1%} at threshold {best['threshold']}")


def main():
    args = parse_args()
    
    # Setup output directories
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    (output_path / "data").mkdir(exist_ok=True)
    (output_path / "models").mkdir(exist_ok=True)
    (output_path / "results").mkdir(exist_ok=True)
    (output_path / "plots").mkdir(exist_ok=True)
    (output_path / "plots" / "rt_model").mkdir(exist_ok=True)
    (output_path / "plots" / "assignment_model").mkdir(exist_ok=True)
    
    print_flush("="*60)
    if args.model == 'enhanced':
        print_flush("COMPASSIGN ENHANCED TRAINING PIPELINE")
        print_flush("Target: >95% Precision for Production Use")
    else:
        print_flush("COMPASSIGN STANDARD TRAINING PIPELINE")
        print_flush("Baseline Model for Comparison")
    print_flush("="*60)
    
    # Save configuration
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    config['model_type'] = args.model
    with open(output_path / "results" / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Generate synthetic data
    print_flush("\n1. Generating synthetic data...")
    obs_df, peak_df, params = generate_synthetic_data(
        n_clusters=args.n_clusters,
        n_species=args.n_species,
        n_classes=args.n_classes,
        n_compounds=args.n_compounds,
        random_seed=args.seed
    )
    
    # Save data
    obs_df.to_csv(output_path / "data" / "observations.csv", index=False)
    peak_df.to_csv(output_path / "data" / "peaks.csv", index=False)
    with open(output_path / "data" / "true_parameters.json", 'w') as f:
        params_json = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in params.items()}
        json.dump(params_json, f, indent=2)
    
    print_flush(f"  Observations: {len(obs_df)}")
    print_flush(f"  Peaks: {len(peak_df)} (including {(peak_df['true_compound'].isna()).sum()} decoys)")
    
    # Train RT model (same for both standard and enhanced)
    print_flush("\n2. Training hierarchical RT model...")
    rt_model = HierarchicalRTModel(
        n_clusters=args.n_clusters,
        n_species=args.n_species,
        n_classes=args.n_classes,
        n_compounds=args.n_compounds,
        species_cluster=params['species_cluster'],
        compound_class=params['compound_class'],
        descriptors=params['descriptors'],
        internal_std=params['internal_std']
    )
    
    rt_model.build_model(obs_df, use_non_centered=True)
    
    trace_rt = rt_model.sample(
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        n_tune=args.n_tune,
        target_accept=args.target_accept,
        random_seed=args.seed
    )
    
    # Save RT trace
    trace_rt.to_netcdf(output_path / "models" / "rt_trace.nc")
    
    # Create RT diagnostic plots
    print_flush("\n3. Creating RT model diagnostic plots...")
    try:
        # For now, skip PPC results as we need to implement that separately
        # Just pass an empty dict for ppc_results
        create_all_diagnostic_plots(trace_rt, {}, output_path, params)
    except Exception as e:
        print_flush(f"WARNING: Could not create diagnostic plots: {e}")
        print_flush("Continuing with training...")
    
    # Train assignment model (standard or enhanced)
    if args.model == 'enhanced':
        print_flush("\n4. Training ENHANCED peak assignment model...")
        print_flush(f"   Mass tolerance: {args.mass_tolerance} Da")
        print_flush(f"   FP penalty weight: {args.fp_penalty}x")
        
        assignment_model = EnhancedPeakAssignmentModel(
            mass_tolerance=args.mass_tolerance,
            fp_penalty=args.fp_penalty
        )
        
        # Compute RT predictions with uncertainty
        assignment_model.compute_rt_predictions(
            trace_rt,
            params['n_species'],
            params['n_compounds'],
            params['descriptors'],
            params['internal_std']
        )
        
        # Generate enhanced training data
        logit_df = assignment_model.generate_training_data(
            peak_df,
            params['compound_mass'],
            params['n_compounds']
        )
        
        # Build and sample enhanced model
        assignment_model.build_model()
        trace_assignment = assignment_model.sample(
            n_samples=args.n_samples * 2,  # More samples for calibration
            n_chains=4,  # More chains for enhanced
            random_seed=args.seed
        )
        
        # Calibrate probabilities
        print_flush("\n5. Calibrating probabilities...")
        assignment_model.calibrate_probabilities()
        
        # Make staged predictions
        print_flush("\n6. Making staged predictions...")
        results = assignment_model.predict_assignments_staged(
            peak_df,
            high_precision_threshold=args.high_precision_threshold,
            review_threshold=args.review_threshold
        )
        
    else:
        print_flush("\n4. Training STANDARD peak assignment model...")
        print_flush(f"   Mass tolerance: {args.mass_tolerance} Da")
        
        assignment_model = PeakAssignmentModel(
            mass_tolerance=args.mass_tolerance
        )
        
        # Compute RT predictions
        assignment_model.compute_rt_predictions(
            trace_rt,
            params['n_species'],
            params['n_compounds'],
            params['descriptors'],
            params['internal_std']
        )
        
        # Generate training data
        logit_df = assignment_model.generate_training_data(
            peak_df,
            params['compound_mass'],
            params['n_compounds']
        )
        
        # Build and sample model
        assignment_model.build_model()
        trace_assignment = assignment_model.sample(
            n_samples=args.n_samples,
            n_chains=args.n_chains,
            n_tune=args.n_tune,
            target_accept=args.target_accept,
            random_seed=args.seed
        )
        
        # Make predictions
        print_flush("\n5. Making predictions...")
        results = assignment_model.predict_assignments(
            peak_df,
            probability_threshold=args.probability_threshold
        )
    
    # Save assignment results
    logit_df.to_csv(output_path / "data" / "logit_training_data.csv", index=False)
    trace_assignment.to_netcdf(output_path / "models" / "assignment_trace.nc")
    
    # Save predictions
    assignment_df = pd.DataFrame([
        {
            'peak_id': peak_id,
            'assigned_compound': results.assignments[peak_id],
            'probability': results.probabilities[peak_id]
        }
        for peak_id in results.assignments.keys()
    ])
    assignment_df.to_csv(output_path / "results" / "peak_assignments.csv", index=False)
    
    # Save metrics
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'model_type': args.model,
        'precision': results.precision,
        'recall': results.recall,
        'f1_score': results.f1_score,
        'confusion_matrix': results.confusion_matrix
    }
    
    with open(output_path / "results" / "assignment_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create assignment plots
    print_flush("\n7. Creating assignment plots...")
    # Pass the correct parameters: logit_df, trace, results dict, output_path
    assignment_results_dict = {
        'precision': results.precision,
        'recall': results.recall,
        'f1_score': results.f1_score,
        'confusion_matrix': results.confusion_matrix
    }
    create_assignment_plots(logit_df, trace_assignment, assignment_results_dict, output_path)
    
    # Test threshold impact if requested
    if args.test_thresholds:
        test_threshold_impact(assignment_model, peak_df, output_path, args.model == 'enhanced')
    
    # Final summary
    print_flush("\n" + "="*60)
    print_flush("TRAINING COMPLETE")
    print_flush("="*60)
    print_flush(f"\nModel Type: {args.model.upper()}")
    print_flush(f"Performance:")
    print_flush(f"  Precision: {results.precision:.1%} {'✓ MEETS TARGET' if results.precision >= 0.95 else ''}")
    print_flush(f"  Recall: {results.recall:.1%}")
    print_flush(f"  False Positives: {results.confusion_matrix['FP']}")
    
    if args.model == 'enhanced' and hasattr(results, 'confidence_levels'):
        print_flush(f"\nConfidence Levels:")
        print_flush(f"  Confident: {len(results.confidence_levels['confident'])} peaks")
        print_flush(f"  Review: {len(results.confidence_levels['review'])} peaks")
        print_flush(f"  Rejected: {len(results.confidence_levels['rejected'])} peaks")
    
    print_flush(f"\nResults saved to: {output_path}/")
    
    if args.model == 'standard':
        print_flush("\n💡 Tip: Use --model enhanced for production (>95% precision)")
    elif results.precision < 0.95:
        print_flush("\n⚠ To achieve >95% precision, try:")
        print_flush("  --mass-tolerance 0.003 --fp-penalty 10 --high-precision-threshold 0.95")


if __name__ == "__main__":
    main()