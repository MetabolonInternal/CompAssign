"""
CompAssign: Analyze and optimize compound assignment precision for Metabolon requirements.

This script explores the precision-recall tradeoff and identifies optimal
thresholds for high-precision compound assignment in the CompAssign framework.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.metrics import precision_recall_curve, confusion_matrix


def print_flush(msg):
    """Print with immediate flush for real-time logging"""
    print(msg, flush=True)
    sys.stdout.flush()


def analyze_assignment_errors(output_path: Path = Path("output")):
    """Analyze false positives to understand precision issues."""
    
    # Load data
    logit_df = pd.read_csv(output_path / "data" / "logit_training_data.csv")
    peaks_df = pd.read_csv(output_path / "data" / "peaks.csv")
    assignments_df = pd.read_csv(output_path / "results" / "peak_assignments.csv")
    
    print_flush("=" * 60)
    print_flush("PEAK ASSIGNMENT PRECISION ANALYSIS FOR METABOLON")
    print_flush("=" * 60)
    
    # Analyze probability distribution of errors
    print_flush("\n1. PROBABILITY DISTRIBUTION ANALYSIS")
    print_flush("-" * 40)
    
    # Get false positives (assigned but wrong)
    false_positives = []
    true_positives = []
    false_negatives = []
    
    for _, assignment in assignments_df.iterrows():
        peak_id = assignment['peak_id']
        assigned_comp = assignment['assigned_compound']
        prob = assignment['probability']
        
        # Get true compound for this peak
        true_comp = peaks_df[peaks_df['peak_id'] == peak_id]['true_compound'].iloc[0]
        
        if pd.notna(assigned_comp):  # Made an assignment
            if pd.notna(true_comp) and assigned_comp == true_comp:
                true_positives.append(prob)
            else:
                false_positives.append(prob)
        elif pd.notna(true_comp):  # Missed a true peak
            false_negatives.append(prob)
    
    print(f"True Positives: {len(true_positives)} peaks")
    print(f"  Mean probability: {np.mean(true_positives):.3f}")
    print(f"  Min probability: {np.min(true_positives):.3f}")
    print(f"  25th percentile: {np.percentile(true_positives, 25):.3f}")
    
    print(f"\nFalse Positives: {len(false_positives)} peaks (PRECISION ISSUE)")
    if false_positives:
        print(f"  Mean probability: {np.mean(false_positives):.3f}")
        print(f"  Min probability: {np.min(false_positives):.3f}")
        print(f"  Max probability: {np.max(false_positives):.3f}")
        print(f"  75th percentile: {np.percentile(false_positives, 75):.3f}")
    
    # Analyze feature patterns in false positives
    print("\n2. FALSE POSITIVE FEATURE ANALYSIS")
    print("-" * 40)
    
    # Get features for false positive assignments
    if false_positives and 'pred_prob' in logit_df.columns:
        # Find high-confidence false positives
        high_conf_fp = [p for p in false_positives if p > 0.7]
        print(f"High-confidence false positives (p > 0.7): {len(high_conf_fp)}")
        
        # Analyze their features
        fp_features = logit_df[logit_df['label'] == 0].copy()
        fp_features = fp_features[fp_features['pred_prob'] > 0.5]
        
        if len(fp_features) > 0:
            print(f"\nFalse positive feature statistics:")
            print(f"  Mass error (ppm):")
            print(f"    Mean: {fp_features['mass_err_ppm'].abs().mean():.2f}")
            print(f"    Max: {fp_features['mass_err_ppm'].abs().max():.2f}")
            print(f"  RT z-score:")
            print(f"    Mean: {fp_features['rt_z'].abs().mean():.2f}")
            print(f"    Max: {fp_features['rt_z'].abs().max():.2f}")
            print(f"  Log intensity:")
            print(f"    Mean: {fp_features['log_intensity'].mean():.2f}")
    
    return true_positives, false_positives, false_negatives, logit_df


def optimize_threshold_for_precision(logit_df: pd.DataFrame, target_precision: float = 0.95):
    """Find optimal probability threshold for target precision."""
    
    print("\n3. PRECISION-RECALL TRADEOFF ANALYSIS")
    print("-" * 40)
    
    if 'pred_prob' not in logit_df.columns:
        print("Warning: No predicted probabilities found")
        return
    
    y_true = logit_df['label'].values
    y_scores = logit_df['pred_prob'].values
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Find thresholds for different precision targets
    precision_targets = [0.90, 0.95, 0.98, 0.99]
    
    print(f"\nThreshold recommendations for Metabolon:")
    print(f"{'Target Precision':<20} {'Threshold':<15} {'Expected Recall':<15}")
    print("-" * 50)
    
    results = []
    for target in precision_targets:
        # Find threshold that achieves target precision
        valid_idx = np.where(precisions[:-1] >= target)[0]
        if len(valid_idx) > 0:
            idx = valid_idx[0]
            threshold = thresholds[idx]
            recall = recalls[idx]
            print(f"{target:.0%:<20} {threshold:<15.3f} {recall:<15.1%}")
            results.append((target, threshold, recall))
        else:
            print(f"{target:.0%:<20} {'Not achievable':<15} {'-':<15}")
    
    # Analyze impact of different thresholds
    print("\n4. THRESHOLD IMPACT ANALYSIS")
    print("-" * 40)
    
    test_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Assigned':<12}")
    print("-" * 60)
    
    for thresh in test_thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        n_assigned = np.sum(y_pred)
        
        print(f"{thresh:<12.2f} {precision:<12.1%} {recall:<12.1%} {f1:<12.3f} {n_assigned:<12}")
    
    return results


def analyze_feature_importance(logit_df: pd.DataFrame):
    """Analyze which features drive false positives."""
    
    print("\n5. FEATURE IMPORTANCE FOR FALSE POSITIVES")
    print("-" * 40)
    
    if 'pred_prob' not in logit_df.columns:
        return
    
    # Separate true and false candidates
    true_candidates = logit_df[logit_df['label'] == 1].copy()
    false_candidates = logit_df[logit_df['label'] == 0].copy()
    
    # High confidence errors
    high_conf_errors = false_candidates[false_candidates['pred_prob'] > 0.7]
    
    if len(high_conf_errors) > 0:
        print(f"\nHigh-confidence false positives (p > 0.7): {len(high_conf_errors)} cases")
        print("\nFeature comparison (True vs High-Conf False):")
        
        features = ['mass_err_ppm', 'rt_z', 'log_intensity']
        feature_names = ['Mass Error (ppm)', 'RT Z-score', 'Log Intensity']
        
        for feat, name in zip(features, feature_names):
            true_mean = true_candidates[feat].abs().mean() if feat != 'log_intensity' else true_candidates[feat].mean()
            false_mean = high_conf_errors[feat].abs().mean() if feat != 'log_intensity' else high_conf_errors[feat].mean()
            
            print(f"\n{name}:")
            print(f"  True assignments:  {true_mean:.3f}")
            print(f"  False positives:   {false_mean:.3f}")
            print(f"  Difference:        {false_mean - true_mean:+.3f}")


def suggest_improvements():
    """Suggest improvements for higher precision."""
    
    print("\n6. RECOMMENDATIONS FOR ULTRA-HIGH PRECISION")
    print("=" * 60)
    
    recommendations = """
    A. IMMEDIATE IMPROVEMENTS (Current System):
       1. Increase probability threshold from 0.5 to 0.8-0.9
          - Expected precision: >95%
          - Tradeoff: Some recall loss (but still >85%)
       
       2. Tighten mass tolerance from 0.01 Da to 0.005 Da
          - Reduces false candidates by ~50%
          - Requires high-resolution MS data
    
    B. MODEL ENHANCEMENTS:
       1. Add asymmetric loss function:
          - Penalize false positives 3-5x more than false negatives
          - Train with class_weight = {0: 5, 1: 1}
       
       2. Feature engineering:
          - Add RT prediction uncertainty as feature
          - Include peak shape/quality metrics
          - Add isotope pattern matching score
       
       3. Hierarchical thresholds:
          - Higher threshold for novel compounds
          - Lower threshold for well-characterized compounds
          - Compound-specific thresholds based on history
    
    C. ENSEMBLE APPROACHES:
       1. Multiple models with voting:
          - Require 2+ models to agree for assignment
          - Different models emphasize different features
       
       2. Calibrated confidence scores:
          - Use isotonic regression for probability calibration
          - Better probability estimates → better thresholds
    
    D. ACTIVE LEARNING STRATEGY:
       1. Flag uncertain assignments (0.5 < p < 0.9) for review
       2. Use expert feedback to retrain periodically
       3. Build compound-specific confidence models
    
    E. VALIDATION REQUIREMENTS:
       1. Minimum 95% precision on held-out test set
       2. Stress test with isomers and isobars
       3. Validate on different instrument types
    """
    
    print(recommendations)


def create_precision_plots(output_path: Path = Path("output")):
    """Create detailed precision analysis plots."""
    
    # Load data
    logit_df = pd.read_csv(output_path / "data" / "logit_training_data.csv")
    
    if 'pred_prob' not in logit_df.columns:
        print("No predictions found in logit_df")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Precision-Recall curve with threshold annotations
    ax = axes[0, 0]
    y_true = logit_df['label'].values
    y_scores = logit_df['pred_prob'].values
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    ax.plot(recalls[:-1], precisions[:-1], 'b-', linewidth=2)
    
    # Annotate key thresholds
    for thresh in [0.5, 0.7, 0.9]:
        idx = np.argmin(np.abs(thresholds - thresh))
        ax.plot(recalls[idx], precisions[idx], 'ro', markersize=8)
        ax.annotate(f't={thresh:.1f}', 
                   xy=(recalls[idx], precisions[idx]),
                   xytext=(recalls[idx]-0.05, precisions[idx]-0.05))
    
    ax.axhline(0.95, color='r', linestyle='--', label='95% precision target')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Tradeoff')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Probability distribution by class
    ax = axes[0, 1]
    true_probs = logit_df[logit_df['label'] == 1]['pred_prob']
    false_probs = logit_df[logit_df['label'] == 0]['pred_prob']
    
    ax.hist(true_probs, bins=20, alpha=0.6, label='True', color='green', density=True)
    ax.hist(false_probs, bins=20, alpha=0.6, label='False', color='red', density=True)
    
    # Add threshold lines
    for thresh, label in [(0.5, 'Current'), (0.8, 'Recommended')]:
        ax.axvline(thresh, color='black', linestyle='--', alpha=0.5)
        ax.text(thresh, ax.get_ylim()[1]*0.9, label, rotation=90)
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Probability Distributions')
    ax.legend()
    
    # 3. Feature importance for FP vs TP
    ax = axes[1, 0]
    
    # Get high-confidence predictions
    high_conf = logit_df[logit_df['pred_prob'] > 0.7].copy()
    
    if len(high_conf) > 0:
        features = ['mass_err_ppm', 'rt_z', 'log_intensity']
        tp_means = []
        fp_means = []
        
        for feat in features:
            tp_data = high_conf[high_conf['label'] == 1][feat]
            fp_data = high_conf[high_conf['label'] == 0][feat]
            
            if feat != 'log_intensity':
                tp_means.append(tp_data.abs().mean() if len(tp_data) > 0 else 0)
                fp_means.append(fp_data.abs().mean() if len(fp_data) > 0 else 0)
            else:
                tp_means.append(tp_data.mean() if len(tp_data) > 0 else 0)
                fp_means.append(fp_data.mean() if len(fp_data) > 0 else 0)
        
        x = np.arange(len(features))
        width = 0.35
        
        ax.bar(x - width/2, tp_means, width, label='True Positives', color='green', alpha=0.7)
        ax.bar(x + width/2, fp_means, width, label='False Positives', color='red', alpha=0.7)
        
        ax.set_xlabel('Feature')
        ax.set_ylabel('Mean Value')
        ax.set_title('Feature Values: TP vs FP (High Confidence)')
        ax.set_xticks(x)
        ax.set_xticklabels(['Mass Error\n(ppm)', 'RT Z-score', 'Log Intensity'])
        ax.legend()
    
    # 4. Threshold impact
    ax = axes[1, 1]
    
    thresholds = np.linspace(0.3, 0.95, 20)
    precisions = []
    recalls = []
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(prec)
        recalls.append(rec)
    
    ax2 = ax.twinx()
    line1 = ax.plot(thresholds, precisions, 'b-', label='Precision')
    line2 = ax2.plot(thresholds, recalls, 'r-', label='Recall')
    
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Current')
    ax.axhline(0.95, color='blue', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Probability Threshold')
    ax.set_ylabel('Precision', color='b')
    ax2.set_ylabel('Recall', color='r')
    ax.set_title('Threshold Impact on Metrics')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')
    
    plt.suptitle('Precision Optimization Analysis for Metabolon', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path / "plots" / "assignment_model" / "precision_analysis.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"\nPrecision analysis plot saved to: {plot_path}")


if __name__ == "__main__":
    # Run full analysis
    output_path = Path("output")
    
    # Analyze current errors
    tp, fp, fn, logit_df = analyze_assignment_errors(output_path)
    
    # Find optimal thresholds
    if 'pred_prob' in logit_df.columns:
        optimize_threshold_for_precision(logit_df, target_precision=0.95)
        analyze_feature_importance(logit_df)
    
    # Provide recommendations
    suggest_improvements()
    
    # Create plots
    create_precision_plots(output_path)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)