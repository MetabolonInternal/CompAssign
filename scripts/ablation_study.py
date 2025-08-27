#!/usr/bin/env python
"""
Ablation Study for CompAssign Model Components

This script systematically tests different model configurations to isolate
the impact of each change on precision and recall.
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.compassign.data_generator import generate_synthetic_data
from src.compassign.rt_hierarchical import HierarchicalRTModel
from src.compassign.peak_assignment import PeakAssignmentModel
from src.compassign.peak_assignment_enhanced import EnhancedPeakAssignmentModel


def print_flush(msg):
    """Print with immediate flush."""
    print(msg, flush=True)
    sys.stdout.flush()


class AblationConfig:
    """Configuration for a single ablation experiment."""
    
    def __init__(self, name: str, description: str, model_type: str = 'standard',
                 mass_tolerance: float = 0.01, threshold: float = 0.5,
                 fp_penalty: float = 1.0, use_abs_features: bool = False,
                 use_rt_uncertainty: bool = False, use_rt_abs_diff: bool = False):
        self.name = name
        self.description = description
        self.model_type = model_type
        self.mass_tolerance = mass_tolerance
        self.threshold = threshold
        self.fp_penalty = fp_penalty
        self.use_abs_features = use_abs_features
        self.use_rt_uncertainty = use_rt_uncertainty
        self.use_rt_abs_diff = use_rt_abs_diff
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'name': self.name,
            'description': self.description,
            'model_type': self.model_type,
            'mass_tolerance': self.mass_tolerance,
            'threshold': self.threshold,
            'fp_penalty': self.fp_penalty,
            'use_abs_features': self.use_abs_features,
            'use_rt_uncertainty': self.use_rt_uncertainty,
            'use_rt_abs_diff': self.use_rt_abs_diff
        }


def define_configurations() -> List[AblationConfig]:
    """Define all ablation study configurations."""
    configs = []
    
    # Phase 1: Baseline Isolation
    configs.append(AblationConfig(
        "S-Base",
        "Standard baseline",
        model_type='standard',
        mass_tolerance=0.01,
        threshold=0.5
    ))
    
    configs.append(AblationConfig(
        "S-Threshold",
        "Only change threshold",
        model_type='standard',
        mass_tolerance=0.01,
        threshold=0.9
    ))
    
    configs.append(AblationConfig(
        "S-MassTol",
        "Only change mass tolerance",
        model_type='standard',
        mass_tolerance=0.005,
        threshold=0.5
    ))
    
    configs.append(AblationConfig(
        "S-Both",
        "Change threshold + mass tolerance",
        model_type='standard',
        mass_tolerance=0.005,
        threshold=0.9
    ))
    
    # Phase 2: Feature Engineering
    configs.append(AblationConfig(
        "S-Abs",
        "Use absolute value features",
        model_type='standard_abs',  # Special variant
        mass_tolerance=0.01,
        threshold=0.5,
        use_abs_features=True
    ))
    
    configs.append(AblationConfig(
        "S-AbsOpt",
        "Absolute features + optimized settings",
        model_type='standard_abs',
        mass_tolerance=0.005,
        threshold=0.9,
        use_abs_features=True
    ))
    
    # Phase 3: Enhanced Features
    configs.append(AblationConfig(
        "E-NoWeight",
        "Enhanced features without FP penalty",
        model_type='enhanced',
        mass_tolerance=0.005,
        threshold=0.9,
        fp_penalty=1.0,
        use_rt_uncertainty=True,
        use_rt_abs_diff=True
    ))
    
    configs.append(AblationConfig(
        "E-Weight3",
        "Enhanced with moderate penalty",
        model_type='enhanced',
        mass_tolerance=0.005,
        threshold=0.9,
        fp_penalty=3.0,
        use_rt_uncertainty=True,
        use_rt_abs_diff=True
    ))
    
    configs.append(AblationConfig(
        "E-Weight5",
        "Full enhanced model",
        model_type='enhanced',
        mass_tolerance=0.005,
        threshold=0.9,
        fp_penalty=5.0,
        use_rt_uncertainty=True,
        use_rt_abs_diff=True
    ))
    
    # Phase 4: Feature Importance
    configs.append(AblationConfig(
        "E-NoUncertainty",
        "Enhanced without RT uncertainty",
        model_type='enhanced',
        mass_tolerance=0.005,
        threshold=0.9,
        fp_penalty=5.0,
        use_rt_uncertainty=False,
        use_rt_abs_diff=True
    ))
    
    configs.append(AblationConfig(
        "E-NoAbsDiff",
        "Enhanced without RT abs diff",
        model_type='enhanced',
        mass_tolerance=0.005,
        threshold=0.9,
        fp_penalty=5.0,
        use_rt_uncertainty=True,
        use_rt_abs_diff=False
    ))
    
    configs.append(AblationConfig(
        "E-BasicFeatures",
        "Only basic features with FP penalty",
        model_type='enhanced',
        mass_tolerance=0.005,
        threshold=0.9,
        fp_penalty=5.0,
        use_rt_uncertainty=False,
        use_rt_abs_diff=False
    ))
    
    return configs


def run_single_configuration(config: AblationConfig, 
                            obs_df: pd.DataFrame,
                            peak_df: pd.DataFrame,
                            params: Dict[str, Any],
                            trace_rt: Any,
                            n_samples: int = 1000) -> Dict[str, Any]:
    """Run a single configuration and return metrics."""
    
    print_flush(f"\n{'='*60}")
    print_flush(f"Running: {config.name}")
    print_flush(f"Description: {config.description}")
    print_flush(f"{'='*60}")
    
    # Print configuration details
    print_flush(f"Configuration:")
    print_flush(f"  Model type: {config.model_type}")
    print_flush(f"  Mass tolerance: {config.mass_tolerance} Da")
    print_flush(f"  Threshold: {config.threshold}")
    print_flush(f"  FP penalty: {config.fp_penalty}")
    print_flush(f"  Use absolute features: {config.use_abs_features}")
    print_flush(f"  Use RT uncertainty: {config.use_rt_uncertainty}")
    print_flush(f"  Use RT abs diff: {config.use_rt_abs_diff}")
    
    try:
        # Create appropriate model based on configuration
        if config.model_type in ['standard', 'standard_abs']:
            # Use standard model with potential modifications
            model = PeakAssignmentModel(mass_tolerance=config.mass_tolerance)
            
            # Compute RT predictions
            model.compute_rt_predictions(
                trace_rt,
                params['n_species'],
                params['n_compounds'],
                params['descriptors'],
                params['internal_std']
            )
            
            # Generate training data
            logit_df = model.generate_training_data(
                peak_df,
                params['compound_mass'],
                params['n_compounds']
            )
            
            # Modify features if using absolute values
            if config.use_abs_features:
                print_flush("  Modifying to use absolute value features...")
                logit_df['mass_err_ppm'] = np.abs(logit_df['mass_err_ppm'])
                logit_df['rt_z'] = np.abs(logit_df['rt_z'])
                model.logit_df = logit_df
            
            # Build and sample model
            model.build_model()
            trace = model.sample(n_samples=n_samples, n_chains=2, n_tune=500)
            
            # Make predictions
            results = model.predict_assignments(peak_df, probability_threshold=config.threshold)
            
        else:  # enhanced
            # Use enhanced model
            model = EnhancedPeakAssignmentModel(
                mass_tolerance=config.mass_tolerance,
                fp_penalty=config.fp_penalty
            )
            
            # Compute RT predictions with uncertainty
            model.compute_rt_predictions(
                trace_rt,
                params['n_species'],
                params['n_compounds'],
                params['descriptors'],
                params['internal_std']
            )
            
            # Generate training data
            logit_df = model.generate_training_data(
                peak_df,
                params['compound_mass'],
                params['n_compounds']
            )
            
            # Modify features based on configuration
            if not config.use_rt_uncertainty:
                print_flush("  Removing RT uncertainty feature...")
                logit_df['rt_uncertainty'] = 0
                
            if not config.use_rt_abs_diff:
                print_flush("  Removing RT abs diff feature...")
                logit_df['rt_abs_diff'] = 0
            
            model.logit_df = logit_df
            
            # Build and sample model
            model.build_model()
            trace = model.sample(n_samples=n_samples * 2, n_chains=2)
            
            # Calibrate if using enhanced model with penalty
            if config.fp_penalty > 1.0:
                model.calibrate_probabilities()
            
            # Make predictions
            results = model.predict_assignments_staged(
                peak_df,
                high_precision_threshold=config.threshold,
                review_threshold=config.threshold * 0.8
            )
        
        # Collect metrics
        metrics = {
            'config': config.to_dict(),
            'precision': results.precision,
            'recall': results.recall,
            'f1_score': results.f1_score,
            'false_positives': results.confusion_matrix['FP'],
            'false_negatives': results.confusion_matrix['FN'],
            'true_positives': results.confusion_matrix['TP'],
            'true_negatives': results.confusion_matrix['TN']
        }
        
        print_flush(f"\nResults:")
        print_flush(f"  Precision: {results.precision:.1%}")
        print_flush(f"  Recall: {results.recall:.1%}")
        print_flush(f"  False Positives: {results.confusion_matrix['FP']}")
        print_flush(f"  False Negatives: {results.confusion_matrix['FN']}")
        
        return metrics
        
    except Exception as e:
        print_flush(f"ERROR in configuration {config.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a comparison table from results."""
    
    rows = []
    baseline_precision = None
    baseline_recall = None
    
    for result in results:
        if result is None:
            continue
            
        config = result['config']
        
        # Set baseline for comparison
        if config['name'] == 'S-Base':
            baseline_precision = result['precision']
            baseline_recall = result['recall']
        
        # Calculate deltas from baseline
        precision_delta = None
        recall_delta = None
        if baseline_precision is not None:
            precision_delta = (result['precision'] - baseline_precision) * 100
            recall_delta = (result['recall'] - baseline_recall) * 100
        
        rows.append({
            'Configuration': config['name'],
            'Description': config['description'],
            'Precision': f"{result['precision']:.1%}",
            'Recall': f"{result['recall']:.1%}",
            'F1': f"{result['f1_score']:.3f}",
            'FP': result['false_positives'],
            'FN': result['false_negatives'],
            'Mass Tol': config['mass_tolerance'],
            'Threshold': config['threshold'],
            'FP Penalty': config['fp_penalty'],
            'Δ Precision': f"{precision_delta:+.1f}%" if precision_delta is not None else "",
            'Δ Recall': f"{recall_delta:+.1f}%" if recall_delta is not None else ""
        })
    
    return pd.DataFrame(rows)


def create_impact_visualization(results: List[Dict[str, Any]], output_dir: Path):
    """Create visualization showing incremental impact of changes."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    configs = [r['config']['name'] for r in results if r]
    precisions = [r['precision'] * 100 for r in results if r]
    recalls = [r['recall'] * 100 for r in results if r]
    fps = [r['false_positives'] for r in results if r]
    fns = [r['false_negatives'] for r in results if r]
    
    # 1. Precision comparison
    ax = axes[0, 0]
    bars = ax.bar(configs, precisions, color=['red' if p < 95 else 'green' for p in precisions])
    ax.axhline(y=95, color='blue', linestyle='--', label='95% Target')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Precision (%)')
    ax.set_title('Precision Across Configurations')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, precisions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 2. Precision vs Recall tradeoff
    ax = axes[0, 1]
    for i, (config, prec, rec) in enumerate(zip(configs, precisions, recalls)):
        color = 'red' if 'Base' in config else 'orange' if 'S-' in config else 'green'
        ax.scatter(rec, prec, s=100, alpha=0.7, color=color)
        ax.annotate(config, (rec, prec), fontsize=8, 
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Recall (%)')
    ax.set_ylabel('Precision (%)')
    ax.set_title('Precision-Recall Tradeoff')
    ax.axhline(y=95, color='blue', linestyle='--', alpha=0.5, label='95% Target')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. False Positives and Negatives
    ax = axes[1, 0]
    x_pos = np.arange(len(configs))
    width = 0.35
    ax.bar(x_pos - width/2, fps, width, label='False Positives', color='red', alpha=0.7)
    ax.bar(x_pos + width/2, fns, width, label='False Negatives', color='orange', alpha=0.7)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Incremental improvements
    ax = axes[1, 1]
    baseline_idx = configs.index('S-Base') if 'S-Base' in configs else 0
    baseline_prec = precisions[baseline_idx]
    improvements = [(p - baseline_prec) for p in precisions]
    
    bars = ax.bar(configs, improvements, color=['red' if i < 0 else 'green' for i in improvements])
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Precision Change from Baseline (%)')
    ax.set_title('Incremental Precision Improvements')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (0.5 if val > 0 else -1),
                f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top', 
                fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_results.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print_flush(f"Visualization saved to {output_dir / 'ablation_results.png'}")


def main():
    """Run complete ablation study."""
    
    parser = argparse.ArgumentParser(description='CompAssign Ablation Study')
    parser.add_argument('--n-samples', type=int, default=500,
                       help='Number of MCMC samples per configuration')
    parser.add_argument('--output-dir', type=str, default='output/ablation',
                       help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with fewer configurations')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_flush("="*60)
    print_flush("COMPASSIGN ABLATION STUDY")
    print_flush("="*60)
    print_flush(f"Output directory: {output_dir}")
    print_flush(f"Samples per model: {args.n_samples}")
    
    # Generate synthetic data once
    print_flush("\n1. Generating synthetic data...")
    params, obs_df, peak_df = generate_synthetic_data(
        n_species=80,
        n_compounds=60,
        n_clusters=5,
        n_classes=10,
        obs_per_compound=20
    )
    
    print_flush(f"  RT observations: {len(obs_df)}")
    print_flush(f"  Peaks (including decoys): {len(peak_df)}")
    
    # Train RT model once (shared across all configurations)
    print_flush("\n2. Training hierarchical RT model...")
    rt_model = HierarchicalRTModel(
        n_clusters=5,
        n_species=params['n_species'],
        n_classes=params['n_classes'],
        n_compounds=params['n_compounds'],
        species_cluster=params['species_cluster'],
        compound_class=params['compound_class'],
        descriptors=params['descriptors'],
        internal_std=params['internal_std']
    )
    
    rt_model.build_model(obs_df, use_non_centered=True)
    trace_rt = rt_model.sample(
        n_samples=args.n_samples,
        n_chains=2,
        n_tune=500,
        target_accept=0.95
    )
    
    # Define configurations
    print_flush("\n3. Running ablation configurations...")
    configs = define_configurations()
    
    if args.quick:
        # Quick test: only run key configurations
        key_names = ['S-Base', 'S-Threshold', 'S-Both', 'E-Weight5']
        configs = [c for c in configs if c.name in key_names]
        print_flush(f"  Quick mode: Testing {len(configs)} key configurations")
    else:
        print_flush(f"  Testing all {len(configs)} configurations")
    
    # Run each configuration
    results = []
    for i, config in enumerate(configs, 1):
        print_flush(f"\n[{i}/{len(configs)}] {config.name}")
        result = run_single_configuration(
            config, obs_df, peak_df, params, trace_rt, args.n_samples
        )
        if result:
            results.append(result)
    
    # Create comparison table
    print_flush("\n4. Creating comparison table...")
    comparison_df = create_comparison_table(results)
    
    # Save results
    comparison_df.to_csv(output_dir / 'ablation_results.csv', index=False)
    print_flush(f"Results saved to {output_dir / 'ablation_results.csv'}")
    
    # Display table
    print_flush("\n" + "="*60)
    print_flush("ABLATION STUDY RESULTS")
    print_flush("="*60)
    print(comparison_df.to_string(index=False))
    
    # Create visualization
    print_flush("\n5. Creating visualizations...")
    create_impact_visualization(results, output_dir)
    
    # Find best configuration for >95% precision
    high_precision = comparison_df[comparison_df['Precision'].str.rstrip('%').astype(float) >= 95]
    if not high_precision.empty:
        print_flush("\n" + "="*60)
        print_flush("CONFIGURATIONS ACHIEVING >95% PRECISION")
        print_flush("="*60)
        print(high_precision.to_string(index=False))
        
        # Find minimal configuration
        best = high_precision.iloc[0]  # First one achieving target
        print_flush(f"\nMinimal configuration for >95% precision: {best['Configuration']}")
        print_flush(f"  {best['Description']}")
    else:
        print_flush("\n⚠ No configuration achieved >95% precision target")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_configurations': len(results),
        'n_samples': args.n_samples,
        'best_precision': comparison_df['Precision'].str.rstrip('%').astype(float).max(),
        'configurations_above_95': len(high_precision) if not high_precision.empty else 0,
        'results': [r for r in results if r]
    }
    
    with open(output_dir / 'ablation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print_flush("\n" + "="*60)
    print_flush("ABLATION STUDY COMPLETE")
    print_flush("="*60)
    print_flush(f"Results directory: {output_dir}/")


if __name__ == "__main__":
    main()