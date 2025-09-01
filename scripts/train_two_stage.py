#!/usr/bin/env python
"""
Two-stage training script for CompAssign.
Allows saving and loading the first stage RT model separately for debugging.
"""

import numpy as np
import pandas as pd
import pickle
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compassign import HierarchicalRTModel, PeakAssignmentModel
from src.compassign.assignment_plots import create_assignment_plots
from create_synthetic_data import create_metabolomics_data


def train_stage1(obs_df, compound_df, params, output_dir: Path, verbose: bool = True):
    """Train and save the RT hierarchical model (Stage 1)."""
    
    if verbose:
        print("\n" + "="*60)
        print("STAGE 1: RT HIERARCHICAL MODEL")
        print("="*60)
    
    # Train RT model
    rt_model = HierarchicalRTModel(
        n_species=params['n_species'],
        n_compounds=params['n_compounds'],
        n_classes=params['n_classes'],
        n_clusters=params['n_clusters']
    )
    
    # Fit the model
    trace_rt = rt_model.fit(
        obs_df,
        compound_df[['descriptors']].values,
        internal_std=params['internal_std'],
        n_samples=500,  # Quick for testing
        n_tune=500,
        target_accept=0.95,
        seed=42
    )
    
    # Save RT model and trace
    rt_model_path = output_dir / 'rt_model.pkl'
    with open(rt_model_path, 'wb') as f:
        pickle.dump({'model': rt_model, 'trace': trace_rt}, f)
    
    if verbose:
        print(f"✅ RT model saved to: {rt_model_path}")
    
    return rt_model, trace_rt


def train_stage2(rt_model, trace_rt, peak_df, compound_df, params, 
                 output_dir: Path,
                 mass_tolerance: float = 0.005,
                 rt_window_k: float = 1.5,
                 probability_threshold: float = 0.7,
                 verbose: bool = True):
    """Train and save the peak assignment model (Stage 2)."""
    
    if verbose:
        print("\n" + "="*60)
        print("STAGE 2: PEAK ASSIGNMENT MODEL")
        print("="*60)
        print(f"  Using 9 essential features only")
    
    # Create peak assigner
    assigner = PeakAssignmentModel(
        mass_tolerance=mass_tolerance,
        rt_window_k=rt_window_k,
        use_class_weights=True,
        standardize_features=True,
        calibration_method='temperature'
    )
    
    # Compute RT predictions
    assigner.compute_rt_predictions(
        trace_rt,
        params['n_species'],
        params['n_compounds'],
        compound_df[['descriptors']].values,
        params['internal_std']
    )
    
    # Generate training data with peaks
    assigner.generate_training_data(
        peak_df,
        compound_df['mass'].values,
        params['n_species']
    )
    
    # Build and sample model
    model = assigner.build_model()
    trace = assigner.sample_model(
        n_chains=2,
        n_samples=500,
        n_tune=500,
        target_accept=0.95
    )
    
    # Save assignment model
    assigner_path = output_dir / 'assigner_model.pkl'
    with open(assigner_path, 'wb') as f:
        pickle.dump(assigner, f)
    
    if verbose:
        print(f"✅ Assignment model saved to: {assigner_path}")
    
    # Make predictions for evaluation
    assignments, probs = assigner.predict(
        peak_df,
        compound_df['mass'].values,
        params['n_species'],
        probability_threshold=probability_threshold
    )
    
    # Calculate metrics
    tp = fp = fn = 0
    for _, peak in peak_df.iterrows():
        peak_id = peak['peak_id']
        true_comp = peak.get('true_compound', None)
        pred_comp = assignments.get(peak_id, None)
        
        if pd.notna(true_comp) and true_comp is not None:
            if pred_comp == true_comp:
                tp += 1
            else:
                fn += 1
                if pred_comp is not None:
                    fp += 1
        elif pred_comp is not None:
            fp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'n_features': 9,  # Fixed to essential features
        'probability_threshold': probability_threshold,
        'calibration_method': assigner.calibration_method
    }
    
    # Save metrics
    metrics_path = output_dir / 'stage2_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    if verbose:
        print(f"\n📊 Stage 2 Performance:")
        print(f"  F1 Score: {metrics['f1']:.1f}%")
        print(f"  Precision: {metrics['precision']:.1f}%")
        print(f"  Recall: {metrics['recall']:.1f}%")
    
    return assigner, metrics


def main():
    parser = argparse.ArgumentParser(description='Two-stage CompAssign training')
    
    # Data generation parameters
    parser.add_argument('--n-compounds', type=int, default=10)
    parser.add_argument('--n-species', type=int, default=40)
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    
    # Model parameters
    parser.add_argument('--mass-tolerance', type=float, default=0.005)
    parser.add_argument('--rt-window-k', type=float, default=1.5)
    parser.add_argument('--probability-threshold', type=float, default=0.7)
    
    # Stage control
    parser.add_argument('--stage', choices=['both', '1', '2'], default='both',
                       help='Which stage(s) to run')
    parser.add_argument('--load-rt-model', type=str, 
                       help='Path to saved RT model (for stage 2 only)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='two_stage_output')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Setup
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    verbose = not args.quiet
    
    # Generate synthetic data
    if verbose:
        print("Generating synthetic data...")
    
    # Calculate noise peaks
    expected_true_peaks = args.n_compounds * 3 * args.n_species * 0.65
    n_noise_peaks = min(int(expected_true_peaks * 1.0), 2000)
    
    peak_df_raw, compound_df, true_assignments, rt_uncertainties, hierarchical_params = create_metabolomics_data(
        n_compounds=args.n_compounds,
        n_peaks_per_compound=3,
        n_noise_peaks=n_noise_peaks,
        n_species=args.n_species,
        isomer_fraction=0.3,
        near_isobar_fraction=0.2,
        mass_error_std=0.002,
        rt_uncertainty_range=(0.05, 0.5)
    )
    
    # Convert to training format
    obs_records = []
    for _, peak in peak_df_raw.iterrows():
        if pd.notna(peak['true_compound']) and peak['true_compound'] is not None:
            obs_records.append({
                'species': int(peak['species']),
                'compound': int(peak['true_compound']),
                'rt': float(peak['rt'])
            })
    obs_df = pd.DataFrame(obs_records)
    
    # Prepare peak_df
    peak_df = peak_df_raw.copy()
    if 'peak_id' not in peak_df.columns:
        peak_df['peak_id'] = range(len(peak_df))
    
    # Create params dict
    params = {
        'n_clusters': hierarchical_params['n_clusters'],
        'n_species': args.n_species,
        'n_classes': hierarchical_params['n_classes'],
        'n_compounds': len(compound_df),
        'internal_std': hierarchical_params['internal_std']
    }
    
    # Stage 1: RT Model
    if args.stage in ['both', '1']:
        rt_model, trace_rt = train_stage1(obs_df, compound_df, params, output_dir, verbose)
    elif args.stage == '2':
        if not args.load_rt_model:
            rt_model_path = output_dir / 'rt_model.pkl'
            if rt_model_path.exists():
                args.load_rt_model = str(rt_model_path)
            else:
                print("ERROR: Stage 2 requires a saved RT model.")
                print("Run stage 1 first or provide --load-rt-model")
                sys.exit(1)
        
        if verbose:
            print(f"Loading RT model from: {args.load_rt_model}")
        with open(args.load_rt_model, 'rb') as f:
            rt_data = pickle.load(f)
            rt_model = rt_data['model']
            trace_rt = rt_data['trace']
    
    # Stage 2: Peak Assignment Model
    if args.stage in ['both', '2']:
        assigner, metrics = train_stage2(
            rt_model, trace_rt, peak_df, compound_df, params, output_dir,
            mass_tolerance=args.mass_tolerance,
            rt_window_k=args.rt_window_k,
            probability_threshold=args.probability_threshold,
            verbose=verbose
        )
    
    if verbose:
        print("\n" + "="*60)
        print("✨ TWO-STAGE TRAINING COMPLETE")
        print(f"📁 Results saved to: {output_dir}")
        print("="*60)


if __name__ == '__main__':
    main()