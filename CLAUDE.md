# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CompAssign is a Bayesian framework for ultra-high precision compound assignment in untargeted metabolomics. It implements a two-stage approach:
1. **Hierarchical RT (Retention Time) Regression**: Predicts retention times with uncertainty quantification
2. **Probabilistic Peak Assignment**: Uses RT predictions to assign LC-MS peaks to compounds with >95% precision

**Critical Requirement**: False positives are more costly than false negatives for Metabolon. The system is optimized for **precision over recall**.

## Common Development Commands

### Environment Setup
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate compassign

# Update environment after changes
conda env update -f environment.yml --prune

# Deactivate when done
conda deactivate

# List all environments
conda env list

# Remove environment if needed
conda env remove -n compassign
```

**Important**: Always ensure you're in the `compassign` conda environment before running any scripts. The environment contains PyMC 5.25+, ArviZ, and other critical dependencies with specific version requirements.

### Training Models

```bash
# Standard training (baseline model, 84% precision)
PYTHONPATH=. python scripts/train.py --model standard --n-samples 1000

# Enhanced training for ultra-high precision (>95% precision, recommended for production)
PYTHONPATH=. python scripts/train.py --model enhanced \
    --n-samples 1000 \
    --test-thresholds \
    --mass-tolerance 0.005 \
    --fp-penalty 5.0 \
    --high-precision-threshold 0.9

# Analyze precision-recall tradeoff
PYTHONPATH=. python scripts/analyze_precision.py

# Compare both models side-by-side (comprehensive evaluation)
PYTHONPATH=. python scripts/compare_models.py --n-samples 1000
```

### Real-Time Training Monitoring

The training scripts support real-time logging for monitoring progress:

```bash
# Start training with logging to file
PYTHONPATH=. python scripts/train.py --model enhanced \
    --n-samples 1000 \
    --test-thresholds \
    --mass-tolerance 0.005 \
    --fp-penalty 5.0 \
    --high-precision-threshold 0.9 > training.log 2>&1 &

# Monitor progress in real-time (in another terminal)
tail -f training.log

# Check if training is still running
ps aux | grep train.py

# Kill training if needed
kill %1  # or use process ID from ps
```

**Key benefits of real-time monitoring:**
- See PyMC sampling progress bars and convergence diagnostics
- Monitor MCMC chain convergence in real-time
- Track training stages and potential errors immediately
- Estimate remaining time for long-running jobs

### Model Comparison

Compare both standard and enhanced models on identical datasets:

```bash
# Run comprehensive model comparison
PYTHONPATH=. python scripts/compare_models.py \
    --n-samples 1000 \
    --enhanced-fp-penalty 5.0 \
    --enhanced-mass-tolerance 0.005 \
    --output-dir output/comparison > comparison.log 2>&1 &

# Monitor comparison progress
tail -f comparison.log

# Results structure:
# output/comparison/
# ├── standard/          # Standard model results
# ├── enhanced/          # Enhanced model results  
# ├── comparison_summary.json    # Key metrics comparison
# ├── threshold_comparison.csv   # Performance at all thresholds
# └── plots/
#     ├── precision_recall_comparison.png
#     ├── performance_metrics.png
#     └── confusion_matrices.png
```

**Key comparison features:**
- Side-by-side training on identical synthetic data
- Threshold sensitivity analysis (0.5 to 0.95)
- Precision-recall curve comparisons
- Training time benchmarking
- 95% precision target achievement analysis

### Key Parameters for Precision Tuning
- `--mass-tolerance`: Default 0.005 Da (tighter = higher precision)
- `--fp-penalty`: Default 5.0 (higher = fewer false positives)
- `--high-precision-threshold`: Default 0.9 (achieves >95% precision)

## Architecture and Key Design Decisions

### Two-Stage Bayesian Pipeline

1. **Stage 1: RT Prediction (`rt_hierarchical.py`)**
   - Hierarchical structure: species→clusters, compounds→classes
   - **Non-centered parameterization** to avoid funnel geometry in MCMC
   - Returns RT predictions WITH uncertainty (crucial for Stage 2)

2. **Stage 2: Peak Assignment (`peak_assignment_enhanced.py`)**
   - Uses RT uncertainty as a feature (not just point estimates)
   - **Class-weighted loss**: Weights negative samples 5× more in likelihood
   - **Calibrated probabilities** via isotonic regression for reliable thresholds
   - **Staged assignment**: confident (>0.9) / review (0.7-0.9) / rejected (<0.7)

### Critical Implementation Details

#### Class Weighting vs Asymmetric Loss
The "asymmetric loss" is implemented as **class weighting** in the likelihood:
```python
# Weights ALL negative samples 5× more, not just misclassified ones
log_likelihood = weights * (y_obs * pm.math.log(p + 1e-8) + 
                           (1 - y_obs) * pm.math.log(1 - p + 1e-8))
```
This makes the model conservative overall, which is desired for high precision.

#### Model Variants
- `peak_assignment.py`: Standard model (84% precision, high recall)
- `peak_assignment_enhanced.py`: Enhanced model with precision optimizations (>95% precision)

Always use the enhanced model for production.

### Output Structure
```
output/
├── data/               # Processed training data
├── models/             # Saved MCMC traces (.nc files)
├── plots/
│   ├── rt_model/      # RT model diagnostics
│   └── assignment_model/  # Assignment performance plots
└── results/           # JSON metrics and CSV predictions
```

## Performance Benchmarks

| Configuration | Precision | Recall | False Positives |
|--------------|-----------|--------|-----------------|
| Standard (threshold=0.5) | 84.4% | 98.7% | 14 |
| Enhanced (threshold=0.8) | 91.9% | 74.0% | 5 |
| **Production (threshold=0.9)** | **>95%** | ~65% | <3 |

## Key Files to Understand the System

1. **Models**: 
   - `src/compassign/rt_hierarchical.py` - Hierarchical RT prediction
   - `src/compassign/peak_assignment_enhanced.py` - High-precision assignment

2. **Documentation**:
   - `docs/precision_optimization.md` - Precision tuning strategies (MUST READ)
   - `docs/bayesian_models.md` - Mathematical specifications

3. **Training Scripts**:
   - `train_enhanced.py` - Main entry point for production training
   - `analyze_precision.py` - Precision-recall analysis

## Implementation Status

### ✅ Completed
- Enhanced model with class-weighted loss
- RT uncertainty features
- Probability calibration
- Staged assignment (confident/review/rejected)

### ⚠️ Partially Completed
- Enhanced features (core done, isotope patterns pending)
- Active learning (confidence levels done, retraining not implemented)

### ❌ Not Implemented
- Ensemble methods (two-stage verification, multi-model voting)
- Stress testing suite
- Production deployment (Docker, API endpoints)

**See `docs/TASKS.md` for the complete development roadmap and prioritized next steps.** This file contains:
- Immediate tasks for production readiness
- Short-term enhancements (isotope patterns, peak quality)
- Long-term improvements (ensemble methods, active learning)
- Success criteria for each milestone

## Important Notes

- **Divergences in MCMC**: 1-2 divergences are acceptable. If >10, increase `target_accept` or check parameterization
- **Training time**: Full pipeline takes ~5-10 minutes with 1000 samples
- **Memory usage**: ~2-4 GB for typical datasets
- **Precision focus**: When in doubt, optimize for fewer false positives even if recall suffers