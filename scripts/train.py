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

from src.compassign.rt_hierarchical import HierarchicalRTModel
from src.compassign.peak_assignment import PeakAssignmentModel
from src.compassign.peak_assignment_softmax import PeakAssignmentSoftmaxModel
from src.compassign.pymc_generative_assignment import GenerativeAssignmentModel
from src.compassign.diagnostic_plots import create_all_diagnostic_plots
from src.compassign.assignment_plots import create_assignment_plots

# Use synthetic data generator with isomers and near-isobars
sys.path.insert(0, str(Path(__file__).parent))
from create_synthetic_data import create_metabolomics_data


def print_flush(msg):
    """Print with immediate flush for real-time logging"""
    print(msg, flush=True)
    sys.stdout.flush()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CompAssign models for metabolomics compound assignment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Synthetic data includes isomers and near-isobars. See docs/synthetic_data_generation.md for details.'
    )
    
    # Model parameters (optimized for >95% precision)
    
    # Data generation parameters
    parser.add_argument('--n-clusters', type=int, default=8,
                       help='Number of species clusters (default: 8)')
    parser.add_argument('--n-species', type=int, default=40,
                       help='Number of species (default: 40, ~5 per cluster)')
    parser.add_argument('--n-classes', type=int, default=4,
                       help='Number of compound classes')
    parser.add_argument('--n-compounds', type=int, default=10,
                       help='Number of compounds (default: 10 for faster testing)')
    
    # Sampling parameters
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of MCMC samples per chain')
    parser.add_argument('--n-chains', type=int, default=None,
                       help='Number of MCMC chains (default: uses PyMC default of 4)')
    parser.add_argument('--n-tune', type=int, default=1000,
                       help='Number of tuning steps')
    parser.add_argument('--target-accept', type=float, default=0.95,
                       help='Target acceptance rate for NUTS')
    
    # Model parameters (tested for 99.5% precision)
    parser.add_argument('--mass-tolerance', type=float, default=0.005,
                       help='Mass tolerance in Da (default: 0.005 Da)')
    parser.add_argument('--rt-window-k', type=float, default=1.5,
                       help='RT window multiplier k*sigma (default: 1.5)')
    parser.add_argument('--probability-threshold', type=float, default=0.7,
                       help='Probability threshold for assignment (default: 0.7 for calibrated probabilities)')
    parser.add_argument('--matching', type=str, default='greedy',
                       choices=['hungarian', 'greedy', 'none'],
                       help='One-to-one matching algorithm (default: greedy)')
    parser.add_argument('--test-thresholds', action='store_true',
                       help='Test multiple probability thresholds')
    parser.add_argument('--model', type=str, default='calibrated', choices=['calibrated', 'softmax', 'generative'],
                       help='Assignment model to run')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def test_threshold_impact(assignment_model, peak_df, output_path):
    """Test different thresholds to show precision-recall tradeoff."""
    
    print_flush("\n" + "="*60)
    print_flush("THRESHOLD IMPACT ANALYSIS")
    print_flush("="*60)
    
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    results = []
    
    for thresh in thresholds:
        # Test predictions at each threshold
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
    
    # Find best precision threshold
    best = results_df.loc[results_df['precision'].idxmax()]
    print_flush(f"\nBest precision: {best['precision']:.1%} at threshold {best['threshold']}")


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
    print_flush("COMPASSIGN TRAINING PIPELINE")
    print_flush(f"Compounds: {args.n_compounds}, Species: {args.n_species}")
    print_flush(f"Mass tolerance: {args.mass_tolerance} Da, RT window: ±{args.rt_window_k}σ")
    print_flush(f"Matching: {args.matching}, Threshold: {args.probability_threshold}")
    print_flush("="*60)
    
    # Save configuration
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    with open(output_path / "results" / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Generate synthetic data with isomers and near-isobars
    print_flush("\n1. Generating synthetic data...")
    np.random.seed(args.seed)
    
    # Create data with hierarchical structure matching the Bayesian model
    # Calculate expected number of true peaks to set realistic noise ratio
    expected_true_peaks = args.n_compounds * 3 * args.n_species * 0.65  # ~65% compounds observed per species
    # Scale noise adaptively: more for small datasets, capped for large ones
    if expected_true_peaks < 500:
        noise_ratio = 1.5  # 150% for small datasets
    elif expected_true_peaks < 1000:
        noise_ratio = 1.0  # 100% for medium datasets
    else:
        noise_ratio = 0.5  # 50% for large datasets
    n_noise_peaks = min(int(expected_true_peaks * noise_ratio), 2000)  # Cap at 2000 to prevent memory issues
    
    peak_df_raw, compound_df, true_assignments, rt_uncertainties, hierarchical_params = create_metabolomics_data(
        n_compounds=args.n_compounds,
        n_peaks_per_compound=3,
        n_noise_peaks=n_noise_peaks,  # Realistic noise ratio
        n_species=args.n_species,
        isomer_fraction=0.3,  # 30% isomers
        near_isobar_fraction=0.2,  # 20% near-isobars
        mass_error_std=0.002,
        rt_uncertainty_range=(0.05, 0.5)
    )
    
    # Convert to training format
    # Create obs_df for RT model training
    obs_records = []
    for _, peak in peak_df_raw.iterrows():
        if pd.notna(peak['true_compound']) and peak['true_compound'] is not None:
            obs_records.append({
                'species': int(peak['species']),
                'compound': int(peak['true_compound']),
                'rt': float(peak['rt'])
            })
    obs_df = pd.DataFrame(obs_records)
    
    # Ensure peak_df has required columns
    peak_df = peak_df_raw.copy()
    if 'peak_id' not in peak_df.columns:
        peak_df['peak_id'] = range(len(peak_df))
    
    # Create params dict using hierarchical structure from data generation
    params = {
        'n_clusters': hierarchical_params['n_clusters'],
        'n_species': args.n_species,
        'n_classes': hierarchical_params['n_classes'],
        'n_compounds': len(compound_df),
        'species_cluster': hierarchical_params['species_cluster'],
        'compound_class': hierarchical_params['compound_class'].astype(int),
        'descriptors': np.random.randn(len(compound_df), 5),  # Random descriptors for now
        'internal_std': np.random.randn(args.n_species),  # Random internal standards
        'compound_mass': compound_df['true_mass'].values,
        'rt_uncertainties': rt_uncertainties
    }
    
    # Save data
    obs_df.to_csv(output_path / "data" / "observations.csv", index=False)
    peak_df.to_csv(output_path / "data" / "peaks.csv", index=False)
    compound_df.to_csv(output_path / "data" / "compounds.csv", index=False)
    with open(output_path / "data" / "true_parameters.json", 'w') as f:
        params_json = {k: v.tolist() if hasattr(v, 'tolist') else v 
                      for k, v in params.items() 
                      if k != 'rt_uncertainties'}  # Skip dict
        json.dump(params_json, f, indent=2)
    
    print_flush(f"  Observations: {len(obs_df)}")
    print_flush(f"  Peaks: {len(peak_df)}")
    print_flush(f"  True assignments: {sum(1 for v in true_assignments.values() if v is not None)}")
    print_flush(f"  Noise peaks: {sum(1 for v in true_assignments.values() if v is None)}")
    print_flush(f"  Compounds: {len(compound_df)}")
    print_flush(f"  - Isomers: {len(compound_df[compound_df['type'] == 'isomer'])}")
    print_flush(f"  - Near-isobars: {len(compound_df[compound_df['type'] == 'near_isobar'])}")
    
    # Train the Bayesian RT model
    print_flush("\n2. Training hierarchical RT model...")
    
    # Initialize the RT model with hierarchical structure from data
    rt_model = HierarchicalRTModel(
        n_clusters=params['n_clusters'],
        n_species=params['n_species'],
        n_classes=params['n_classes'],
        n_compounds=params['n_compounds'],
        species_cluster=params['species_cluster'],
        compound_class=params['compound_class'],
        descriptors=params['descriptors'],
        internal_std=params['internal_std']
    )
    
    # Build the model with correct parameter
    rt_model.build_model(obs_df, use_non_centered=True)
    
    # Build sampling kwargs with improved NUTS settings
    sample_kwargs = {
        'n_samples': args.n_samples,
        'n_tune': args.n_tune,
        'target_accept': 0.99,  # Higher target_accept to reduce divergences
        'max_treedepth': 15,  # Increased tree depth
        'random_seed': args.seed
    }
    if args.n_chains is not None:
        sample_kwargs['n_chains'] = args.n_chains
    
    trace_rt = rt_model.sample(**sample_kwargs)
    
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
    
    # Train peak assignment model
    print_flush("\n4. Training peak assignment model...")
    print_flush(f"   Model: {args.model}")
    print_flush(f"   Mass tolerance: {args.mass_tolerance} Da")
    print_flush(f"   RT window: ±{args.rt_window_k}σ")
    print_flush(f"   Matching algorithm: {args.matching}")
    print_flush(f"   Probability threshold: {args.probability_threshold}")

    if args.model == 'generative':
        # Generative model pipeline
        gen_model = GenerativeAssignmentModel(
            mass_tolerance=args.mass_tolerance,
            rt_window_k=args.rt_window_k,
            random_seed=args.seed,
        )

        # Compute RT predictions (mean, std) for species-compound pairs
        gen_model.compute_rt_predictions(
            trace_rt,
            params['n_species'],
            len(compound_df),
            params['descriptors'],
            params['internal_std'],
            rt_model=rt_model,
        )

        # Generate padded tensors and labels
        train_pack = gen_model.generate_training_data(
            peak_df=peak_df,
            compound_mass=params['compound_mass'],
            n_compounds=len(compound_df),
        )

        # Split by peak_id (train/calib/test) for evaluation parity
        np.random.seed(args.seed)
        peak_ids = np.array(sorted(train_pack['peak_ids']))
        np.random.shuffle(peak_ids)
        n_peaks = len(peak_ids)
        train_p = int(0.6 * n_peaks)
        calib_p = int(0.2 * n_peaks)
        train_peaks = set(map(int, peak_ids[:train_p]))
        calib_peaks = set(map(int, peak_ids[train_p:train_p + calib_p]))
        test_peaks = set(map(int, peak_ids[train_p + calib_p:]))

        # Use labels only for train set
        labels = gen_model.train_pack['labels']
        true_labels = gen_model.train_pack['true_labels']
        peak_id_array = gen_model.train_pack['peak_ids']
        labels[:] = -1
        for i, pid in enumerate(peak_id_array):
            if int(pid) in train_peaks:
                labels[i] = true_labels[i]

        # Build and sample
        gen_model.build_model()
        sample_kwargs_assign = {
            'draws': args.n_samples,
            'tune': args.n_tune,
            'target_accept': args.target_accept,
            'random_seed': args.seed,
        }
        if args.n_chains is not None:
            sample_kwargs_assign['chains'] = args.n_chains
        trace_assignment = gen_model.sample(**sample_kwargs_assign)

        # Predictions and metrics (test set only)
        print_flush("\n5. Making predictions (generative)...")
        results = gen_model.assign(prob_threshold=args.probability_threshold, eval_peak_ids=test_peaks)

        # Save outputs
        trace_assignment.to_netcdf(output_path / "models" / "assignment_trace.nc")
        probs_dict = gen_model.predict_probs()
        assignment_df = pd.DataFrame([
            {
                'peak_id': pid,
                'assigned_compound': results.assignments.get(int(pid)),
                'probability': results.top_prob.get(int(pid), 0.0)
            }
            for pid in peak_id_array
        ])
        assignment_df.to_csv(output_path / "results" / "peak_assignments.csv", index=False)

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model': 'generative',
            'mass_tolerance': args.mass_tolerance,
            'rt_window_k': args.rt_window_k,
            'matching': args.matching,
            'probability_threshold': args.probability_threshold,
            'precision': results.precision,
            'recall': results.recall,
            'f1_score': results.f1,
            'confusion_matrix': results.confusion_matrix,
        }
        with open(output_path / "results" / "assignment_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

    elif args.model == 'softmax':
        # Keep path available for softmax model if desired
        softmax_model = PeakAssignmentSoftmaxModel(
            mass_tolerance=args.mass_tolerance,
            rt_window_k=args.rt_window_k,
            use_temperature=True,
            standardize_features=True,
            random_seed=args.seed,
        )
        softmax_model.compute_rt_predictions(
            trace_rt,
            params['n_species'],
            len(compound_df),
            params['descriptors'],
            params['internal_std'],
            rt_model=rt_model,
        )
        n_species = len(np.unique(peak_df['species']))
        softmax_model.generate_training_data(
            peak_df=peak_df,
            compound_mass=params['compound_mass'],
            n_compounds=len(compound_df),
            species_cluster=peak_df['species'].values,
        )
        softmax_model.build_model()
        trace_assignment = softmax_model.sample(draws=args.n_samples, tune=args.n_tune, chains=args.n_chains or 2,
                                               target_accept=args.target_accept)
        results = softmax_model.assign(prob_threshold=args.probability_threshold)
        trace_assignment.to_netcdf(output_path / "models" / "assignment_trace.nc")

    else:
        # Original calibrated logistic pipeline
        assignment_model = PeakAssignmentModel(
            mass_tolerance=args.mass_tolerance,
            rt_window_k=args.rt_window_k,
            use_class_weights=True,
            standardize_features=True,
            calibration_method='temperature'
        )
        assignment_model.compute_rt_predictions(
            trace_rt,
            params['n_species'],
            len(compound_df),
            params['descriptors'],
            params['internal_std'],
            rt_model=rt_model
        )
        logit_df = assignment_model.generate_training_data(
            peak_df,
            params['compound_mass'],
            len(compound_df),
            apply_rt_filter=False
        )
        np.random.seed(args.seed)
        peak_ids = np.array(sorted(logit_df['peak_id'].unique()))
        np.random.shuffle(peak_ids)
        n_peaks = len(peak_ids)
        train_p = int(0.6 * n_peaks)
        calib_p = int(0.2 * n_peaks)
        train_peaks = set(peak_ids[:train_p])
        calib_peaks = set(peak_ids[train_p:train_p + calib_p])
        test_peaks = set(peak_ids[train_p + calib_p:])

        idx = np.arange(len(logit_df))
        train_idx = idx[logit_df['peak_id'].isin(train_peaks)]
        calib_idx = idx[logit_df['peak_id'].isin(calib_peaks)]
        test_idx = idx[logit_df['peak_id'].isin(test_peaks)]
        print_flush(f"\nData split by peak_id:")
        print_flush(f"  Train: {len(train_peaks)} peaks, {len(train_idx)} examples")
        print_flush(f"  Calib: {len(calib_peaks)} peaks, {len(calib_idx)} examples")
        print_flush(f"  Test:  {len(test_peaks)} peaks, {len(test_idx)} examples")

        assignment_model.build_model(train_idx=train_idx)
        sample_kwargs_assign = {
            'n_samples': args.n_samples,
            'n_tune': args.n_tune,
            'target_accept': args.target_accept,
            'random_seed': args.seed
        }
        if args.n_chains is not None:
            sample_kwargs_assign['n_chains'] = args.n_chains
        trace_assignment = assignment_model.sample(**sample_kwargs_assign)
        if assignment_model.calibration_method != 'none':
            assignment_model._calibrate_model(calib_idx=calib_idx)
        print_flush("\n5. Making predictions...")
        results = assignment_model.predict_assignments(
            peak_df,
            probability_threshold=args.probability_threshold,
            eval_peak_ids=test_peaks
        )
        logit_df.to_csv(output_path / "data" / "logit_training_data.csv", index=False)
        trace_assignment.to_netcdf(output_path / "models" / "assignment_trace.nc")
        assignment_df = pd.DataFrame([
            {
                'peak_id': peak_id,
                'assigned_compound': results.assignments[peak_id],
                'probability': results.probabilities[peak_id]
            }
            for peak_id in results.assignments.keys()
        ])
        assignment_df.to_csv(output_path / "results" / "peak_assignments.csv", index=False)
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'mass_tolerance': args.mass_tolerance,
            'rt_window_k': args.rt_window_k,
            'matching': args.matching,
            'probability_threshold': args.probability_threshold,
            'precision': results.precision,
            'recall': results.recall,
            'f1_score': results.f1_score,
            'confusion_matrix': results.confusion_matrix
        }
        with open(output_path / "results" / "assignment_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        print_flush("\n7. Creating assignment plots...")
        assignment_results_dict = {
            'precision': results.precision,
            'recall': results.recall,
            'f1_score': results.f1_score,
            'confusion_matrix': results.confusion_matrix
        }
        create_assignment_plots(logit_df, trace_assignment, assignment_results_dict, output_path, test_peaks, args.probability_threshold)
        if args.test_thresholds:
            test_threshold_impact(assignment_model, peak_df, output_path)
    
    # Final summary with sanity checks
    print_flush("\n" + "="*60)
    print_flush("TRAINING COMPLETE")
    print_flush("="*60)
    
    # Sanity check: Print split sizes again
    print_flush("\nFinal splits summary:")
    print_flush(f"  Train examples: {len(train_idx)} ({len(train_peaks)} peaks)")
    print_flush(f"  Calib examples: {len(calib_idx)} ({len(calib_peaks)} peaks)")
    print_flush(f"  Test examples: {len(test_idx)} ({len(test_peaks)} peaks)")
    print_flush(f"  Total candidates: {len(logit_df)}")
    
    # Performance summary (test set only)
    print_flush(f"\nTest-set performance:")
    print_flush(f"  Precision: {results.precision:.3f} ({results.precision:.1%})")
    print_flush(f"  Recall:    {results.recall:.3f} ({results.recall:.1%})")
    print_flush(f"  F1:        {results.f1_score:.3f}")
    print_flush(f"  ECE:       {results.calibration_metrics['ece']:.3f}")
    print_flush(f"  MCE:       {results.calibration_metrics['mce']:.3f}")
    print_flush(f"  False Positives: {results.confusion_matrix['FP']}")
    print_flush(f"  True Positives:  {results.confusion_matrix['TP']}")
    
    # Parameters summary
    print_flush(f"\nParameters used:")
    print_flush(f"  Mass tolerance: {args.mass_tolerance} Da")
    print_flush(f"  RT window: ±{args.rt_window_k}σ (relaxed during training)")
    print_flush(f"  Probability threshold: {args.probability_threshold}")
    print_flush(f"  Calibration method: {assignment_model.calibration_method}")
    
    print_flush(f"\nOutput directory: {output_path}/")


if __name__ == "__main__":
    main()
