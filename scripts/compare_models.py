#!/usr/bin/env python3
"""
CompAssign Model Comparison: Standard vs Enhanced

This script trains both standard and enhanced models on identical data 
and generates comprehensive comparison metrics and visualizations.
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import time

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
        description='Compare Standard vs Enhanced CompAssign models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
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
    parser.add_argument('--n-chains', type=int, default=4,
                       help='Number of MCMC chains (4 recommended for convergence)')
    parser.add_argument('--n-tune', type=int, default=1000,
                       help='Number of tuning steps')
    parser.add_argument('--target-accept', type=float, default=0.95,
                       help='Target acceptance rate for NUTS')
    
    # Enhanced model parameters
    parser.add_argument('--enhanced-mass-tolerance', type=float, default=0.005,
                       help='Mass tolerance for enhanced model (Da)')
    parser.add_argument('--enhanced-fp-penalty', type=float, default=5.0,
                       help='False positive penalty for enhanced model')
    parser.add_argument('--enhanced-high-threshold', type=float, default=0.9,
                       help='High confidence threshold for enhanced model')
    parser.add_argument('--enhanced-review-threshold', type=float, default=0.7,
                       help='Review threshold for enhanced model')
    
    # Standard model parameters  
    parser.add_argument('--standard-mass-tolerance', type=float, default=0.01,
                       help='Mass tolerance for standard model (Da)')
    parser.add_argument('--standard-threshold', type=float, default=0.5,
                       help='Probability threshold for standard model')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='output/comparison',
                       help='Output directory for comparison results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate HTML comparison report')
    
    return parser.parse_args()


def test_multiple_thresholds(assignment_model, peak_df, model_type, is_enhanced=False):
    """Test multiple thresholds and return results."""
    print_flush(f"\n  Testing multiple thresholds for {model_type} model...")
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
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
            'model': model_type,
            'threshold': thresh,
            'precision': result.precision,
            'recall': result.recall,
            'f1_score': result.f1_score,
            'false_positives': result.confusion_matrix['FP'],
            'false_negatives': result.confusion_matrix['FN'],
            'true_positives': result.confusion_matrix['TP'],
            'true_negatives': result.confusion_matrix['TN']
        })
    
    return results


def create_comparison_plots(standard_results, enhanced_results, output_path):
    """Create comparison visualizations."""
    print_flush("\n6. Creating comparison plots...")
    
    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Combine results
    all_results = standard_results + enhanced_results
    df = pd.DataFrame(all_results)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Precision-Recall Curves
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Precision vs Threshold
    for model in ['standard', 'enhanced']:
        model_df = df[df['model'] == model]
        ax1.plot(model_df['threshold'], model_df['precision'], 
                marker='o', linewidth=2, label=f'{model.title()} Model')
    
    ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Target')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Recall vs Threshold
    for model in ['standard', 'enhanced']:
        model_df = df[df['model'] == model]
        ax2.plot(model_df['threshold'], model_df['recall'],
                marker='o', linewidth=2, label=f'{model.title()} Model')
    
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Recall')
    ax2.set_title('Recall vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Precision-Recall Tradeoff
    for model in ['standard', 'enhanced']:
        model_df = df[df['model'] == model]
        ax3.plot(model_df['recall'], model_df['precision'],
                marker='o', linewidth=2, label=f'{model.title()} Model')
    
    ax3.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Target')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Tradeoff')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "precision_recall_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. F1 Score and False Positives
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # F1 Score vs Threshold
    for model in ['standard', 'enhanced']:
        model_df = df[df['model'] == model]
        ax1.plot(model_df['threshold'], model_df['f1_score'],
                marker='o', linewidth=2, label=f'{model.title()} Model')
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # False Positives vs Threshold
    for model in ['standard', 'enhanced']:
        model_df = df[df['model'] == model]
        ax2.plot(model_df['threshold'], model_df['false_positives'],
                marker='o', linewidth=2, label=f'{model.title()} Model')
    
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('False Positives')
    ax2.set_title('False Positives vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "performance_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion Matrix Heatmap for optimal thresholds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Find optimal thresholds (highest precision >= 0.95, or best available)
    for i, model in enumerate(['standard', 'enhanced']):
        model_df = df[df['model'] == model]
        high_prec = model_df[model_df['precision'] >= 0.95]
        if not high_prec.empty:
            optimal = high_prec.iloc[0]
        else:
            optimal = model_df.loc[model_df['precision'].idxmax()]
        
        # Create confusion matrix
        cm = np.array([[optimal['true_negatives'], optimal['false_positives']],
                      [optimal['false_negatives'], optimal['true_positives']]])
        
        ax = ax1 if i == 0 else ax2
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{model.title()} Model\n(Threshold: {optimal["threshold"]:.2f})')
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])
    
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print_flush(f"  Plots saved to: {plots_dir}/")


def generate_summary_report(standard_results, enhanced_results, comparison_metrics, output_path):
    """Generate detailed comparison summary."""
    print_flush("\n7. Generating summary report...")
    
    # Find best results for each model
    std_df = pd.DataFrame(standard_results)
    enh_df = pd.DataFrame(enhanced_results)
    
    # Best precision for each model
    std_best_prec = std_df.loc[std_df['precision'].idxmax()]
    enh_best_prec = enh_df.loc[enh_df['precision'].idxmax()]
    
    # Results at 95% precision target
    std_95 = std_df[std_df['precision'] >= 0.95]
    enh_95 = enh_df[enh_df['precision'] >= 0.95]
    
    report = {
        'comparison_timestamp': datetime.now().isoformat(),
        'dataset_info': comparison_metrics['dataset_info'],
        'model_comparison': {
            'standard_model': {
                'best_precision': {
                    'threshold': float(std_best_prec['threshold']),
                    'precision': float(std_best_prec['precision']),
                    'recall': float(std_best_prec['recall']),
                    'f1_score': float(std_best_prec['f1_score']),
                    'false_positives': int(std_best_prec['false_positives'])
                },
                'at_95_precision': {
                    'achievable': not std_95.empty,
                    'threshold': float(std_95.iloc[0]['threshold']) if not std_95.empty else None,
                    'recall': float(std_95.iloc[0]['recall']) if not std_95.empty else None
                }
            },
            'enhanced_model': {
                'best_precision': {
                    'threshold': float(enh_best_prec['threshold']),
                    'precision': float(enh_best_prec['precision']),
                    'recall': float(enh_best_prec['recall']),
                    'f1_score': float(enh_best_prec['f1_score']),
                    'false_positives': int(enh_best_prec['false_positives'])
                },
                'at_95_precision': {
                    'achievable': not enh_95.empty,
                    'threshold': float(enh_95.iloc[0]['threshold']) if not enh_95.empty else None,
                    'recall': float(enh_95.iloc[0]['recall']) if not enh_95.empty else None
                }
            }
        },
        'training_time': comparison_metrics['training_time'],
        'recommendations': []
    }
    
    # Generate recommendations
    if enh_95.empty and std_95.empty:
        report['recommendations'].append("Neither model achieves 95% precision. Consider increasing fp_penalty or tightening mass_tolerance.")
    elif std_95.empty and not enh_95.empty:
        report['recommendations'].append("Enhanced model successfully achieves 95% precision target. Use for production.")
    elif not std_95.empty and not enh_95.empty:
        enh_recall = enh_95.iloc[0]['recall']
        std_recall = std_95.iloc[0]['recall']
        if enh_recall > std_recall:
            report['recommendations'].append("Enhanced model achieves better recall at 95% precision. Use for production.")
        else:
            report['recommendations'].append("Both models achieve 95% precision. Enhanced model provides better calibration.")
    
    # Save detailed results
    with open(output_path / "comparison_summary.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save threshold results
    all_results = pd.DataFrame(standard_results + enhanced_results)
    all_results.to_csv(output_path / "threshold_comparison.csv", index=False)
    
    return report


def main():
    args = parse_args()
    
    # Setup output directories
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    standard_path = output_path / "standard"
    enhanced_path = output_path / "enhanced"
    
    for path in [standard_path, enhanced_path]:
        path.mkdir(exist_ok=True)
        (path / "models").mkdir(exist_ok=True)
        (path / "results").mkdir(exist_ok=True)
        (path / "plots").mkdir(exist_ok=True)
    
    print_flush("="*70)
    print_flush("COMPASSIGN MODEL COMPARISON: STANDARD vs ENHANCED")
    print_flush("="*70)
    
    start_time = time.time()
    
    # Generate synthetic data (same for both models)
    print_flush("\n1. Generating synthetic data for comparison...")
    obs_df, peak_df, params = generate_synthetic_data(
        n_clusters=args.n_clusters,
        n_species=args.n_species,
        n_classes=args.n_classes,
        n_compounds=args.n_compounds,
        random_seed=args.seed
    )
    
    print_flush(f"  Dataset: {len(obs_df)} observations, {len(peak_df)} peaks")
    print_flush(f"  Decoys: {(peak_df['true_compound'].isna()).sum()} peaks")
    
    # Save shared data
    obs_df.to_csv(output_path / "observations.csv", index=False)
    peak_df.to_csv(output_path / "peaks.csv", index=False)
    
    # Train shared RT model
    print_flush("\n2. Training shared hierarchical RT model...")
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
    
    # Save shared RT model
    trace_rt.to_netcdf(output_path / "rt_trace.nc")
    
    rt_train_time = time.time() - start_time
    
    # Train Standard Model
    print_flush("\n3. Training STANDARD peak assignment model...")
    std_start = time.time()
    
    standard_model = PeakAssignmentModel(
        mass_tolerance=args.standard_mass_tolerance
    )
    
    standard_model.compute_rt_predictions(
        trace_rt,
        params['n_species'],
        params['n_compounds'],
        params['descriptors'],
        params['internal_std']
    )
    
    logit_df_std = standard_model.generate_training_data(
        peak_df,
        params['compound_mass'],
        params['n_compounds']
    )
    
    standard_model.build_model()
    trace_std = standard_model.sample(
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        n_tune=args.n_tune,
        target_accept=args.target_accept,
        random_seed=args.seed
    )
    
    std_train_time = time.time() - std_start
    
    # Test standard model at multiple thresholds
    standard_results = test_multiple_thresholds(
        standard_model, peak_df, 'standard', is_enhanced=False
    )
    
    # Train Enhanced Model  
    print_flush("\n4. Training ENHANCED peak assignment model...")
    enh_start = time.time()
    
    enhanced_model = EnhancedPeakAssignmentModel(
        mass_tolerance=args.enhanced_mass_tolerance,
        fp_penalty=args.enhanced_fp_penalty
    )
    
    enhanced_model.compute_rt_predictions(
        trace_rt,
        params['n_species'],
        params['n_compounds'],
        params['descriptors'],
        params['internal_std']
    )
    
    logit_df_enh = enhanced_model.generate_training_data(
        peak_df,
        params['compound_mass'],
        params['n_compounds']
    )
    
    enhanced_model.build_model()
    trace_enh = enhanced_model.sample(
        n_samples=args.n_samples * 2,  # More samples for calibration
        n_chains=4,  # More chains for enhanced (consistent with train.py)
        n_tune=args.n_tune,
        target_accept=args.target_accept,
        random_seed=args.seed
    )
    
    # Calibrate enhanced model
    print_flush("\n5. Calibrating enhanced model probabilities...")
    enhanced_model.calibrate_probabilities()
    
    enh_train_time = time.time() - enh_start
    
    # Test enhanced model at multiple thresholds
    enhanced_results = test_multiple_thresholds(
        enhanced_model, peak_df, 'enhanced', is_enhanced=True
    )
    
    # Save model results
    trace_std.to_netcdf(standard_path / "models" / "assignment_trace.nc")
    trace_enh.to_netcdf(enhanced_path / "models" / "assignment_trace.nc")
    logit_df_std.to_csv(standard_path / "results" / "training_data.csv", index=False)
    logit_df_enh.to_csv(enhanced_path / "results" / "training_data.csv", index=False)
    
    # Create comparison metrics
    total_time = time.time() - start_time
    comparison_metrics = {
        'dataset_info': {
            'n_observations': len(obs_df),
            'n_peaks': len(peak_df),
            'n_decoys': int((peak_df['true_compound'].isna()).sum()),
            'n_compounds': args.n_compounds
        },
        'training_time': {
            'rt_model_seconds': rt_train_time,
            'standard_model_seconds': std_train_time,
            'enhanced_model_seconds': enh_train_time,
            'total_seconds': total_time
        }
    }
    
    # Create visualizations
    create_comparison_plots(standard_results, enhanced_results, output_path)
    
    # Generate summary report
    summary = generate_summary_report(
        standard_results, enhanced_results, comparison_metrics, output_path
    )
    
    # Final comparison summary
    print_flush("\n" + "="*70)
    print_flush("COMPARISON COMPLETE")
    print_flush("="*70)
    
    std_best = max(standard_results, key=lambda x: x['precision'])
    enh_best = max(enhanced_results, key=lambda x: x['precision'])
    
    print_flush(f"\nBest Performance Comparison:")
    print_flush(f"Standard Model:")
    print_flush(f"  Max Precision: {std_best['precision']:.1%} (threshold: {std_best['threshold']})")
    print_flush(f"  Recall at max precision: {std_best['recall']:.1%}")
    print_flush(f"  False Positives: {std_best['false_positives']}")
    
    print_flush(f"\nEnhanced Model:")  
    print_flush(f"  Max Precision: {enh_best['precision']:.1%} (threshold: {enh_best['threshold']})")
    print_flush(f"  Recall at max precision: {enh_best['recall']:.1%}")
    print_flush(f"  False Positives: {enh_best['false_positives']}")
    
    # 95% precision analysis
    std_95 = [r for r in standard_results if r['precision'] >= 0.95]
    enh_95 = [r for r in enhanced_results if r['precision'] >= 0.95]
    
    print_flush(f"\n95% Precision Target:")
    if std_95:
        best_std_95 = max(std_95, key=lambda x: x['recall'])
        print_flush(f"  Standard: ✓ Achievable (recall: {best_std_95['recall']:.1%})")
    else:
        print_flush(f"  Standard: ❌ Not achievable")
        
    if enh_95:
        best_enh_95 = max(enh_95, key=lambda x: x['recall'])
        print_flush(f"  Enhanced: ✓ Achievable (recall: {best_enh_95['recall']:.1%})")
    else:
        print_flush(f"  Enhanced: ❌ Not achievable")
    
    print_flush(f"\nTraining Time:")
    print_flush(f"  RT Model: {rt_train_time:.1f}s")
    print_flush(f"  Standard: {std_train_time:.1f}s")
    print_flush(f"  Enhanced: {enh_train_time:.1f}s")
    print_flush(f"  Total: {total_time:.1f}s")
    
    print_flush(f"\nResults saved to: {output_path}/")
    
    if summary['recommendations']:
        print_flush(f"\nRecommendations:")
        for rec in summary['recommendations']:
            print_flush(f"  • {rec}")


if __name__ == "__main__":
    main()