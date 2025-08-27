#!/usr/bin/env python3
"""
Generate comprehensive benchmark report comparing all model configurations.

This script analyzes the verification results and produces:
- Performance comparison tables
- Advanced visualizations
- Executive summary
- HTML dashboard
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def print_flush(msg):
    """Print with immediate flush for real-time logging"""
    print(msg, flush=True)
    sys.stdout.flush()


def load_metrics(output_dir: Path):
    """Load metrics from a training run."""
    metrics = {}
    
    # Load configuration
    config_file = output_dir / "results" / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            metrics['config'] = json.load(f)
    
    # Load assignment metrics
    metrics_file = output_dir / "results" / "assignment_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics['performance'] = json.load(f)
    
    # Load threshold analysis if available
    threshold_file = output_dir / "results" / "threshold_analysis.csv"
    if threshold_file.exists():
        metrics['thresholds'] = pd.read_csv(threshold_file)
    
    # Load peak assignments
    assignments_file = output_dir / "results" / "peak_assignments.csv"
    if assignments_file.exists():
        metrics['assignments'] = pd.read_csv(assignments_file)
    
    return metrics


def create_performance_comparison_table(configs):
    """Create a comparison table of key metrics."""
    rows = []
    
    for name, data in configs.items():
        if 'performance' not in data:
            continue
            
        perf = data['performance']
        config = data.get('config', {})
        
        row = {
            'Configuration': name,
            'Model Type': config.get('model', 'N/A'),
            'Precision': f"{perf.get('precision', 0):.1%}",
            'Recall': f"{perf.get('recall', 0):.1%}",
            'F1 Score': f"{perf.get('f1_score', 0):.3f}",
            'False Positives': perf.get('confusion_matrix', {}).get('FP', 0),
            'False Negatives': perf.get('confusion_matrix', {}).get('FN', 0),
            'Mass Tolerance (Da)': config.get('mass_tolerance', 'N/A'),
            'FP Penalty': config.get('fp_penalty', 'N/A'),
            'Threshold': config.get('probability_threshold', config.get('high_precision_threshold', 'N/A'))
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def create_advanced_visualizations(configs, output_path):
    """Create comprehensive visualization suite."""
    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print_flush("Creating advanced visualizations...")
    
    # Check if we have any performance data
    has_performance = any('performance' in c for c in configs.values())
    has_thresholds = any('thresholds' in c for c in configs.values())
    
    if not has_performance:
        print_flush("   ⚠️ No performance data available - skipping visualizations")
        print_flush("   (Training may not have completed successfully)")
        return
    
    # 1. Performance Radar Chart
    create_radar_chart(configs, plots_dir)
    
    # 2. Threshold Sensitivity Heatmap
    if has_thresholds:
        create_threshold_heatmap(configs, plots_dir)
    
    # 3. ROC Curves Comparison
    if has_thresholds:
        create_roc_comparison(configs, plots_dir)
    
    # 4. Performance vs Cost Tradeoff
    if has_thresholds:
        create_cost_benefit_analysis(configs, plots_dir)
    
    # 5. Confusion Matrix Grid
    create_confusion_matrix_grid(configs, plots_dir)
    
    print_flush(f"Visualizations saved to {plots_dir}")


def create_radar_chart(configs, plots_dir):
    """Create radar chart comparing multiple metrics."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # Metrics to compare
    metrics = ['Precision', 'Recall', 'F1 Score', '1-FP Rate', 'Coverage']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(configs)))
    
    for idx, (name, data) in enumerate(configs.items()):
        if 'performance' not in data:
            continue
            
        perf = data['performance']
        cm = perf.get('confusion_matrix', {})
        
        # Calculate metrics (normalized to 0-1)
        values = [
            perf.get('precision', 0),
            perf.get('recall', 0),
            perf.get('f1_score', 0),
            1 - (cm.get('FP', 0) / max(cm.get('FP', 0) + cm.get('TN', 1), 1)),
            cm.get('TP', 0) / max(cm.get('TP', 0) + cm.get('FN', 1), 1)
        ]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Model Performance Comparison\n(Multi-Metric Radar Chart)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(plots_dir / "performance_radar.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_threshold_heatmap(configs, plots_dir):
    """Create heatmap showing performance across thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for metric_idx, metric in enumerate(['precision', 'recall']):
        matrix = []
        row_labels = []
        
        for name, data in configs.items():
            if 'thresholds' not in data:
                continue
                
            row_labels.append(name)
            row = data['thresholds'][metric].values
            matrix.append(row)
        
        if matrix:
            matrix = np.array(matrix)
            col_labels = data['thresholds']['threshold'].values
            
            sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                       xticklabels=col_labels, yticklabels=row_labels,
                       vmin=0, vmax=1, ax=axes[metric_idx], cbar_kws={'label': metric.capitalize()})
            axes[metric_idx].set_xlabel('Probability Threshold')
            axes[metric_idx].set_ylabel('Configuration')
            axes[metric_idx].set_title(f'{metric.capitalize()} Across Thresholds')
    
    plt.suptitle('Threshold Sensitivity Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / "threshold_sensitivity.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_roc_comparison(configs, plots_dir):
    """Create ROC curves comparison."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    
    for idx, (name, data) in enumerate(configs.items()):
        if 'thresholds' not in data:
            continue
            
        # Calculate TPR and FPR from threshold data
        tpr = data['thresholds']['recall'].values  # TPR = Recall
        # Estimate FPR from confusion matrix data
        fpr = 1 - data['thresholds']['precision'].values  # Approximation
        
        # Sort by FPR for proper curve
        sort_idx = np.argsort(fpr)
        fpr_sorted = fpr[sort_idx]
        tpr_sorted = tpr[sort_idx]
        
        # Calculate AUC
        auc = np.trapz(tpr_sorted, fpr_sorted)
        
        ax.plot(fpr_sorted, tpr_sorted, 'o-', linewidth=2, 
                label=f'{name} (AUC={auc:.3f})', color=colors[idx])
    
    # Add diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('ROC Curves Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_cost_benefit_analysis(configs, plots_dir):
    """Create cost-benefit analysis visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define costs (relative)
    fp_cost = 5.0  # False positive is 5x more costly
    fn_cost = 1.0  # False negative baseline cost
    
    for name, data in configs.items():
        if 'thresholds' not in data:
            continue
            
        thresholds = data['thresholds']
        
        # Calculate total cost for each threshold
        # Check if false_negatives column exists, calculate it if not
        if 'false_negatives' not in thresholds.columns:
            # Assuming we have a fixed number of true positives + false negatives
            # We can estimate false_negatives from recall
            # recall = TP / (TP + FN), so FN = TP * (1/recall - 1)
            # For simplicity, we'll use a fixed total of 1612 actual compounds
            total_actual = 1612
            thresholds['false_negatives'] = total_actual - (thresholds['recall'] * total_actual).astype(int)
        
        total_cost = (thresholds['false_positives'] * fp_cost + 
                     thresholds['false_negatives'] * fn_cost)
        
        # Plot Cost vs Threshold
        ax1.plot(thresholds['threshold'], total_cost, 'o-', linewidth=2, label=name)
        
        # Plot Precision vs Total Cost
        ax2.scatter(total_cost, thresholds['precision'], s=100, alpha=0.7, label=name)
    
    ax1.set_xlabel('Probability Threshold')
    ax1.set_ylabel('Total Cost (FP×5 + FN×1)')
    ax1.set_title('Cost vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Total Cost')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision vs Cost Tradeoff')
    ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% Target')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Cost-Benefit Analysis (FP Cost = 5×FN Cost)', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / "cost_benefit.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_confusion_matrix_grid(configs, plots_dir):
    """Create grid of confusion matrices."""
    n_configs = len([c for c in configs.values() if 'performance' in c])
    if n_configs == 0:
        print_flush("   No confusion matrices to plot (no performance data available)")
        return
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_configs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    idx = 0
    for name, data in configs.items():
        if 'performance' not in data:
            continue
            
        cm = data['performance'].get('confusion_matrix', {})
        matrix = np.array([[cm.get('TN', 0), cm.get('FP', 0)],
                          [cm.get('FN', 0), cm.get('TP', 0)]])
        
        ax = axes[idx] if n_configs > 1 else axes[0]
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Predicted Negative', 'Predicted Positive'],
                   yticklabels=['Actual Negative', 'Actual Positive'])
        
        precision = cm.get('TP', 0) / max(cm.get('TP', 0) + cm.get('FP', 0), 1)
        recall = cm.get('TP', 0) / max(cm.get('TP', 0) + cm.get('FN', 0), 1)
        
        ax.set_title(f'{name}\nPrecision: {precision:.1%}, Recall: {recall:.1%}')
        idx += 1
    
    # Hide unused subplots
    for i in range(idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Confusion Matrices Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrices_grid.png", dpi=300, bbox_inches='tight')
    plt.close()


def generate_executive_summary(configs, output_path):
    """Generate executive summary of findings."""
    summary = []
    summary.append("# CompAssign Enhanced Model Verification Report")
    summary.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    summary.append("## Executive Summary\n")
    
    # Find best performing configuration
    best_precision = 0
    best_config = None
    target_achieved = False
    
    for name, data in configs.items():
        if 'performance' in data:
            precision = data['performance'].get('precision', 0)
            if precision > best_precision:
                best_precision = precision
                best_config = name
            if precision >= 0.95:
                target_achieved = True
    
    if target_achieved:
        summary.append("✅ **SUCCESS**: Enhanced model achieves >95% precision target for production use.\n")
    else:
        summary.append("⚠️ **NEEDS TUNING**: Current configurations do not meet 95% precision target.\n")
    
    summary.append(f"**Best Configuration**: {best_config} with {best_precision:.1%} precision\n")
    
    # Key findings
    summary.append("## Key Findings\n")
    
    for name, data in configs.items():
        if 'performance' not in data:
            continue
            
        perf = data['performance']
        config = data.get('config', {})
        
        summary.append(f"\n### {name}")
        summary.append(f"- **Model Type**: {config.get('model', 'N/A')}")
        summary.append(f"- **Precision**: {perf.get('precision', 0):.1%}")
        summary.append(f"- **Recall**: {perf.get('recall', 0):.1%}")
        summary.append(f"- **False Positives**: {perf.get('confusion_matrix', {}).get('FP', 0)}")
        summary.append(f"- **Configuration**: mass_tolerance={config.get('mass_tolerance', 'N/A')}, "
                      f"fp_penalty={config.get('fp_penalty', 'N/A')}, "
                      f"threshold={config.get('probability_threshold', config.get('high_precision_threshold', 'N/A'))}")
    
    # Recommendations
    summary.append("\n## Recommendations\n")
    
    if target_achieved:
        summary.append("1. **Deploy Enhanced Model** with production settings for ultra-high precision")
        summary.append("2. **Monitor Performance** regularly to ensure precision remains >95%")
        summary.append("3. **Implement Active Learning** to continuously improve with human feedback")
    else:
        summary.append("1. **Increase FP Penalty** to 10.0 or higher")
        summary.append("2. **Tighten Mass Tolerance** to 0.003 Da")
        summary.append("3. **Raise Threshold** to 0.95 for ultra-high precision")
    
    summary.append("\n## Performance Guarantees\n")
    
    if best_precision >= 0.95:
        summary.append(f"✅ **Precision Guarantee**: {best_precision:.1%} (exceeds 95% requirement)")
        
        # Calculate false positive reduction
        standard_fp = 0
        enhanced_fp = 0
        for name, data in configs.items():
            if 'standard' in name.lower() and 'performance' in data:
                standard_fp = data['performance'].get('confusion_matrix', {}).get('FP', 0)
            if best_config in name and 'performance' in data:
                enhanced_fp = data['performance'].get('confusion_matrix', {}).get('FP', 0)
        
        if standard_fp > 0:
            fp_reduction = (standard_fp - enhanced_fp) / standard_fp * 100
            summary.append(f"✅ **False Positive Reduction**: {fp_reduction:.0f}% compared to baseline")
    else:
        summary.append("❌ **Precision**: Below 95% target - further tuning required")
    
    # Save summary
    summary_text = "\n".join(summary)
    
    with open(output_path / "executive_summary.md", 'w') as f:
        f.write(summary_text)
    
    return summary_text


def generate_html_dashboard(configs, output_path):
    """Generate interactive HTML dashboard."""
    html = []
    html.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CompAssign Model Verification Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .metric-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .success { color: #27ae60; font-weight: bold; }
        .warning { color: #f39c12; font-weight: bold; }
        .error { color: #e74c3c; font-weight: bold; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th { background-color: #3498db; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .plot-container {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .plot-container img { width: 100%; height: auto; }
        .summary-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .metric-value { font-size: 2em; font-weight: bold; }
    </style>
</head>
<body>
    <h1>🔬 CompAssign Model Verification Dashboard</h1>
    """)
    
    # Summary box
    best_precision = 0
    best_config = None
    for name, data in configs.items():
        if 'performance' in data:
            precision = data['performance'].get('precision', 0)
            if precision > best_precision:
                best_precision = precision
                best_config = name
    
    status_class = "success" if best_precision >= 0.95 else "warning"
    status_text = "✅ TARGET ACHIEVED" if best_precision >= 0.95 else "⚠️ NEEDS TUNING"
    
    html.append(f"""
    <div class="summary-box">
        <h2>Verification Status: <span class="{status_class}">{status_text}</span></h2>
        <p>Best Configuration: <strong>{best_config}</strong></p>
        <p>Highest Precision: <span class="metric-value">{best_precision:.1%}</span></p>
        <p>Target: >95% Precision for Production Use</p>
    </div>
    """)
    
    # Performance comparison table
    html.append('<div class="metric-card">')
    html.append('<h2>Performance Comparison</h2>')
    
    perf_df = create_performance_comparison_table(configs)
    html.append(perf_df.to_html(classes='performance-table', index=False))
    html.append('</div>')
    
    # Plots section
    html.append('<h2>Visualizations</h2>')
    html.append('<div class="plot-grid">')
    
    plot_files = [
        ('performance_radar.png', 'Multi-Metric Performance Radar'),
        ('threshold_sensitivity.png', 'Threshold Sensitivity Analysis'),
        ('roc_curves.png', 'ROC Curves Comparison'),
        ('cost_benefit.png', 'Cost-Benefit Analysis'),
        ('confusion_matrices_grid.png', 'Confusion Matrices'),
        ('precision_recall_comparison.png', 'Precision-Recall Curves')
    ]
    
    for plot_file, title in plot_files:
        if (output_path / 'plots' / plot_file).exists():
            html.append(f"""
            <div class="plot-container">
                <h3>{title}</h3>
                <img src="plots/{plot_file}" alt="{title}">
            </div>
            """)
    
    html.append('</div>')
    
    # Recommendations
    html.append('<div class="metric-card">')
    html.append('<h2>Recommendations</h2>')
    
    if best_precision >= 0.95:
        html.append("""
        <ul>
            <li class="success">✅ Deploy Enhanced Model with current production settings</li>
            <li>Monitor performance metrics daily for first week</li>
            <li>Implement automated alerts if precision drops below 94%</li>
            <li>Plan quarterly model retraining with accumulated feedback</li>
        </ul>
        """)
    else:
        html.append("""
        <ul>
            <li class="warning">⚠️ Further tuning required to achieve 95% precision target</li>
            <li>Increase FP penalty parameter to 10.0 or higher</li>
            <li>Reduce mass tolerance to 0.003 Da</li>
            <li>Consider ensemble approach for critical applications</li>
        </ul>
        """)
    
    html.append('</div>')
    
    html.append(f"""
    <footer>
        <p style="text-align: center; color: #7f8c8d; margin-top: 50px;">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | CompAssign v1.0
        </p>
    </footer>
</body>
</html>
    """)
    
    # Save HTML
    with open(output_path / "dashboard.html", 'w') as f:
        f.write("\n".join(html))
    
    print_flush(f"HTML dashboard saved to {output_path / 'dashboard.html'}")


def main():
    """Main execution function."""
    print_flush("="*60)
    print_flush("COMPASSIGN BENCHMARK REPORT GENERATOR")
    print_flush("="*60)
    
    # Define configurations to analyze
    verification_dir = Path("output/verification")
    
    configs = {
        "Standard Baseline": verification_dir / "standard",
        "Enhanced Production": verification_dir / "enhanced_production",
        "Enhanced Ultra-High": verification_dir / "enhanced_ultra"
    }
    
    # Load all metrics
    print_flush("\n1. Loading metrics from all configurations...")
    loaded_configs = {}
    
    for name, path in configs.items():
        if path.exists():
            print_flush(f"   Loading {name}...")
            loaded_configs[name] = load_metrics(path)
        else:
            print_flush(f"   ⚠️ {name} not found at {path}")
    
    if not loaded_configs:
        print_flush("\n❌ No configuration results found. Please run training first.")
        sys.exit(1)
    
    # Check if any configs have performance data
    has_any_performance = any('performance' in config for config in loaded_configs.values())
    
    if not has_any_performance:
        print_flush("\n⚠️ WARNING: Training appears incomplete - no performance metrics found")
        print_flush("   The training may have been interrupted or failed to complete.")
        print_flush("   Please check the training logs for errors.")
        print_flush("\n   Suggestion: Run training on a more powerful machine (e.g., server)")
        print_flush("   with the sequential script: ./scripts/run_verification.sh")
        sys.exit(1)
    
    # Generate report output directory
    report_dir = verification_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    
    # Create performance comparison table
    print_flush("\n2. Creating performance comparison table...")
    perf_table = create_performance_comparison_table(loaded_configs)
    if perf_table.empty:
        print_flush("   ⚠️ No performance data to display")
    else:
        print_flush("\n" + perf_table.to_string())
        perf_table.to_csv(report_dir / "performance_comparison.csv", index=False)
    
    # Create visualizations
    print_flush("\n3. Generating advanced visualizations...")
    create_advanced_visualizations(loaded_configs, report_dir)
    
    # Generate executive summary
    print_flush("\n4. Writing executive summary...")
    summary = generate_executive_summary(loaded_configs, report_dir)
    print_flush("\n" + summary)
    
    # Generate HTML dashboard
    print_flush("\n5. Creating HTML dashboard...")
    generate_html_dashboard(loaded_configs, report_dir)
    
    print_flush("\n" + "="*60)
    print_flush("REPORT GENERATION COMPLETE")
    print_flush("="*60)
    print_flush(f"\nResults saved to: {report_dir}")
    print_flush(f"View dashboard: {report_dir / 'dashboard.html'}")
    
    # Print final verdict
    best_precision = max(
        data.get('performance', {}).get('precision', 0) 
        for data in loaded_configs.values() 
        if 'performance' in data
    )
    
    if best_precision >= 0.95:
        print_flush("\n🎉 SUCCESS: Enhanced model ready for production deployment!")
    else:
        print_flush(f"\n⚠️ Current best precision: {best_precision:.1%} - Further tuning needed")


if __name__ == "__main__":
    main()