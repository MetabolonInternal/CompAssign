"""
Visualization for peak assignment model and overall performance evaluation.

This module provides plots for assessing the logistic regression model
and comparing RT predictions with peak assignments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')


def create_assignment_plots(logit_df: pd.DataFrame,
                           assignment_trace: az.InferenceData,
                           assignment_results: Dict[str, Any],
                           output_path: Path):
    """
    Create all plots for the peak assignment model.
    
    Parameters
    ----------
    logit_df : pd.DataFrame
        Logistic training data with features and predictions
    assignment_trace : az.InferenceData
        Posterior samples from logistic model
    assignment_results : dict
        Assignment results including confusion matrix
    output_path : Path
        Directory to save plots
    """
    plots_path = output_path / "plots" / "assignment_model"
    plots_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating assignment model plots...")
    
    # 1. Logistic coefficient posterior distributions
    plot_logistic_coefficients(assignment_trace, plots_path)
    
    # 2. Feature distributions by class
    plot_feature_distributions(logit_df, plots_path)
    
    # 3. ROC and Precision-Recall curves
    plot_roc_pr_curves(logit_df, plots_path)
    
    # 4. Probability calibration plot
    plot_probability_calibration(logit_df, plots_path)
    
    # 5. Confusion matrix heatmap
    plot_confusion_matrix(assignment_results, plots_path)
    
    # 6. Feature importance and correlations
    plot_feature_importance(assignment_trace, logit_df, plots_path)
    
    print(f"Assignment plots saved to: {plots_path}")


def plot_logistic_coefficients(trace: az.InferenceData, plots_path: Path):
    """Plot posterior distributions of logistic regression coefficients."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Coefficient names and labels
    coef_names = ['theta0', 'theta_mass', 'theta_rt', 'theta_int']
    coef_labels = ['Intercept', 'Mass Error (ppm)', 'RT Z-score', 'Log Intensity']
    
    for idx, (name, label) in enumerate(zip(coef_names, coef_labels)):
        ax = axes[idx // 2, idx % 2]
        
        if name in trace.posterior:
            # Get posterior samples
            samples = trace.posterior[name].values.flatten()
            
            # Plot distribution
            ax.hist(samples, bins=30, alpha=0.7, density=True, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Zero')
            ax.axvline(np.mean(samples), color='green', linewidth=2, 
                      label=f'Mean: {np.mean(samples):.3f}')
            
            # Add 95% HDI
            hdi = az.hdi(trace, var_names=[name], hdi_prob=0.95)
            hdi_vals = hdi[name].values
            ax.axvspan(hdi_vals[0], hdi_vals[1], alpha=0.2, color='blue',
                      label=f'95% HDI: [{hdi_vals[0]:.3f}, {hdi_vals[1]:.3f}]')
            
            ax.set_xlabel(label)
            ax.set_ylabel('Density')
            ax.set_title(f'Posterior: {label}')
            ax.legend()
    
    plt.suptitle('Logistic Model Coefficient Posteriors', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "logistic_coefficients.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_feature_distributions(logit_df: pd.DataFrame, plots_path: Path):
    """Plot feature distributions separated by true/false assignments."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    features = ['mass_err_ppm', 'rt_z', 'log_intensity']
    labels = ['Mass Error (ppm)', 'RT Z-score', 'Log Intensity']
    
    for idx, (feature, label) in enumerate(zip(features, labels)):
        ax = axes[idx]
        
        # Separate by label
        true_vals = logit_df[logit_df['label'] == 1][feature]
        false_vals = logit_df[logit_df['label'] == 0][feature]
        
        # Create violin plots
        data_to_plot = [true_vals, false_vals]
        positions = [1, 2]
        
        parts = ax.violinplot(data_to_plot, positions=positions, widths=0.7,
                              showmeans=True, showmedians=True)
        
        # Color the violin plots
        colors = ['green', 'red']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(['True\nAssignments', 'False\nCandidates'])
        ax.set_ylabel(label)
        ax.set_title(f'{label} Distribution by Class')
        ax.grid(True, alpha=0.3)
        
        # Add counts
        ax.text(1, ax.get_ylim()[1] * 0.9, f'n={len(true_vals)}', ha='center')
        ax.text(2, ax.get_ylim()[1] * 0.9, f'n={len(false_vals)}', ha='center')
    
    plt.suptitle('Feature Distributions: True vs False Assignments', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "feature_distributions.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_roc_pr_curves(logit_df: pd.DataFrame, plots_path: Path):
    """Plot ROC and Precision-Recall curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    y_true = logit_df['label'].values
    y_scores = logit_df['pred_prob'].values
    
    # ROC Curve
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='blue', linewidth=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    ax = axes[1]
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    ax.plot(recall, precision, color='green', linewidth=2,
            label=f'PR curve (AUC = {pr_auc:.3f})')
    
    # Add baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(baseline, color='red', linestyle='--', 
              label=f'Baseline (prevalence = {baseline:.3f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Classification Performance Curves', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "roc_pr_curves.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_probability_calibration(logit_df: pd.DataFrame, plots_path: Path):
    """Plot probability calibration and distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calibration plot
    ax = axes[0]
    n_bins = 10
    
    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    true_freq = []
    pred_freq = []
    counts = []
    
    for i in range(n_bins):
        mask = (logit_df['pred_prob'] >= bin_edges[i]) & (logit_df['pred_prob'] < bin_edges[i+1])
        if mask.sum() > 0:
            true_freq.append(logit_df.loc[mask, 'label'].mean())
            pred_freq.append(logit_df.loc[mask, 'pred_prob'].mean())
            counts.append(mask.sum())
        else:
            true_freq.append(np.nan)
            pred_freq.append(np.nan)
            counts.append(0)
    
    # Remove NaN values
    valid = ~np.isnan(true_freq)
    true_freq = np.array(true_freq)[valid]
    pred_freq = np.array(pred_freq)[valid]
    counts = np.array(counts)[valid]
    
    # Plot calibration
    ax.scatter(pred_freq, true_freq, s=counts*5, alpha=0.6)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    
    for i, (x, y, c) in enumerate(zip(pred_freq, true_freq, counts)):
        if c > 0:
            ax.annotate(f'n={c}', (x, y), fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_title('Probability Calibration Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Probability distribution
    ax = axes[1]
    
    # Separate by true label
    prob_true = logit_df[logit_df['label'] == 1]['pred_prob']
    prob_false = logit_df[logit_df['label'] == 0]['pred_prob']
    
    ax.hist(prob_true, bins=20, alpha=0.6, label='True assignments', 
            color='green', density=True)
    ax.hist(prob_false, bins=20, alpha=0.6, label='False candidates', 
            color='red', density=True)
    
    ax.axvline(0.5, color='black', linestyle='--', label='Threshold = 0.5')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Probability Distributions by True Class')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Probability Calibration Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "probability_calibration.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(assignment_results: Dict[str, Any], plots_path: Path):
    """Plot confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract confusion matrix values
    cm = assignment_results.get('confusion_matrix', assignment_results.get('assignment_confusion', {}))
    
    # Handle missing keys
    if 'TP' not in cm:
        print("Warning: Confusion matrix not available, skipping plot")
        plt.close()
        return
    
    # Create matrix
    conf_matrix = np.array([
        [cm['TP'], cm['FN']],
        [cm['FP'], cm['TN']]
    ])
    
    # Labels
    labels = ['Assigned', 'Not Assigned']
    
    # Create heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['True Peak', 'Decoy/Wrong'],
                yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title('Peak Assignment Confusion Matrix', fontsize=14)
    
    # Add metrics as text
    precision = assignment_results.get('assignment_precision', 
                                      cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) > 0 else 0)
    recall = assignment_results.get('assignment_recall',
                                   cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0)
    f1 = assignment_results.get('assignment_f1',
                               2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0)
    
    metrics_text = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1 Score: {f1:.3f}'
    ax.text(1.5, -0.15, metrics_text, transform=ax.transAxes,
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(plots_path / "confusion_matrix.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_feature_importance(trace: az.InferenceData, logit_df: pd.DataFrame, plots_path: Path):
    """Plot feature importance based on coefficient magnitudes and effects."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract coefficients
    coef_names = ['theta_mass', 'theta_rt', 'theta_int']
    coef_labels = ['Mass Error', 'RT Z-score', 'Log Intensity']
    
    # 1. Coefficient magnitudes with uncertainty
    ax = axes[0]
    
    means = []
    stds = []
    for name in coef_names:
        if name in trace.posterior:
            samples = trace.posterior[name].values.flatten()
            means.append(np.mean(samples))
            stds.append(np.std(samples))
    
    y_pos = np.arange(len(coef_labels))
    ax.barh(y_pos, np.abs(means), xerr=stds, alpha=0.7, 
           color=['red' if m < 0 else 'green' for m in means])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_labels)
    ax.set_xlabel('|Coefficient Value|')
    ax.set_title('Feature Importance (Coefficient Magnitude)')
    ax.grid(True, alpha=0.3)
    
    # Add values
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(np.abs(m) + s, i, f'{m:.3f}±{s:.3f}', va='center', fontsize=9)
    
    # 2. Feature correlations
    ax = axes[1]
    
    # Compute correlation matrix
    features_df = logit_df[['mass_err_ppm', 'rt_z', 'log_intensity', 'label']]
    corr_matrix = features_df.corr()
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1,
                xticklabels=['Mass Error', 'RT Z-score', 'Log Intensity', 'Label'],
                yticklabels=['Mass Error', 'RT Z-score', 'Log Intensity', 'Label'],
                ax=ax)
    ax.set_title('Feature Correlations')
    
    plt.suptitle('Feature Importance Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "feature_importance.png", dpi=100, bbox_inches='tight')
    plt.close()


def create_performance_comparison_plots(obs_df: pd.DataFrame,
                                       peak_df: pd.DataFrame,
                                       ppc_results: Dict[str, Any],
                                       assignment_results: Dict[str, Any],
                                       output_path: Path):
    """
    Create plots comparing RT model predictions with peak assignments.
    
    Parameters
    ----------
    obs_df : pd.DataFrame
        RT observations
    peak_df : pd.DataFrame
        Peak data
    ppc_results : dict
        RT model posterior predictive check results
    assignment_results : dict
        Peak assignment results
    output_path : Path
        Directory to save plots
    """
    plots_path = output_path / "plots" / "assignment_model"
    plots_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating performance comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. RT prediction accuracy by species
    ax = axes[0, 0]
    species_rmse = []
    for s in obs_df['species'].unique():
        mask = obs_df['species'] == s
        if mask.sum() > 0:
            rmse = np.sqrt(np.mean(ppc_results['residuals'][mask]**2))
            species_rmse.append((s, rmse))
    
    species_rmse = sorted(species_rmse, key=lambda x: x[1])
    species_ids, rmses = zip(*species_rmse)
    
    ax.bar(range(len(species_ids)), rmses, alpha=0.7)
    ax.set_xlabel('Species (sorted by RMSE)')
    ax.set_ylabel('RMSE')
    ax.set_title('RT Prediction Error by Species')
    ax.axhline(ppc_results['rmse'], color='red', linestyle='--', 
              label=f'Overall RMSE: {ppc_results["rmse"]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. RT prediction accuracy by compound
    ax = axes[0, 1]
    compound_rmse = []
    for c in obs_df['compound'].unique():
        mask = obs_df['compound'] == c
        if mask.sum() > 0:
            rmse = np.sqrt(np.mean(ppc_results['residuals'][mask]**2))
            compound_rmse.append((c, rmse))
    
    compound_rmse = sorted(compound_rmse, key=lambda x: x[1])
    compound_ids, rmses = zip(*compound_rmse)
    
    ax.bar(range(len(compound_ids)), rmses, alpha=0.7, color='orange')
    ax.set_xlabel('Compound (sorted by RMSE)')
    ax.set_ylabel('RMSE')
    ax.set_title('RT Prediction Error by Compound')
    ax.axhline(ppc_results['rmse'], color='red', linestyle='--',
              label=f'Overall RMSE: {ppc_results["rmse"]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Assignment success rate by peak intensity
    ax = axes[1, 0]
    
    # Load assignments if available
    assignments_file = output_path / "results" / "peak_assignments.csv"
    if assignments_file.exists():
        assignments_df = pd.read_csv(assignments_file)
        
        # Merge with peak data
        merged = peak_df.merge(assignments_df, on='peak_id', how='left')
        
        # Bin by intensity
        merged['log_intensity'] = np.log10(merged['intensity'])
        bins = np.percentile(merged['log_intensity'], [0, 25, 50, 75, 100])
        merged['intensity_bin'] = pd.cut(merged['log_intensity'], bins, 
                                        labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Calculate success rate per bin
        success_rates = []
        for bin_label in ['Low', 'Medium', 'High', 'Very High']:
            bin_data = merged[merged['intensity_bin'] == bin_label]
            if len(bin_data) > 0:
                # Check if assignment is correct
                correct = 0
                total = 0
                for _, row in bin_data.iterrows():
                    if not pd.isna(row['true_compound']):  # True peak
                        total += 1
                        if row['assigned_compound'] == row['true_compound']:
                            correct += 1
                
                if total > 0:
                    success_rates.append((bin_label, correct/total, total))
                else:
                    success_rates.append((bin_label, 0, 0))
        
        labels, rates, counts = zip(*success_rates)
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, rates, alpha=0.7, color='green')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Intensity Bin')
        ax.set_ylabel('Assignment Success Rate')
        ax.set_title('Assignment Accuracy by Peak Intensity')
        ax.set_ylim([0, 1.1])
        
        # Add counts on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'n={count}', ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3)
    
    # 4. Overall performance summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary text
    summary_text = f"""
    RT Model Performance:
    • RMSE: {ppc_results['rmse']:.3f}
    • MAE: {ppc_results['mae']:.3f}
    • 95% Coverage: {ppc_results['coverage_95']*100:.1f}%
    
    Peak Assignment Performance:
    • Precision: {assignment_results.get('assignment_precision', 0):.3f}
    • Recall: {assignment_results.get('assignment_recall', 0):.3f}
    • F1 Score: {assignment_results.get('assignment_f1', 0):.3f}
    
    Data Statistics:
    • Total RT observations: {len(obs_df)}
    • Total peaks: {len(peak_df)}
    • True peaks: {peak_df['true_compound'].notna().sum()}
    • Decoy peaks: {peak_df['true_compound'].isna().sum()}
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle('Model Performance Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path / "performance_comparison.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Performance plots saved to: {plots_path}")