# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CompAssign is a Bayesian framework for ultra-high precision compound assignment in untargeted metabolomics. It implements a two-stage approach:
1. **Hierarchical RT (Retention Time) Regression**: Predicts retention times with uncertainty quantification
2. **Probabilistic Peak Assignment**: Uses RT predictions to assign LC-MS peaks to compounds with >99% precision

**Critical Requirement**: False positives are more costly than false negatives for Metabolon. The system is optimized for **precision over recall**.

**Key Discovery (2025-08-27)**: Ablation study proved that simple parameter tuning (mass_tolerance=0.005, threshold=0.9) achieves 99.5% precision, outperforming complex model architectures. The codebase has been simplified accordingly.

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
# Standard training (uses recommended parameters)
./scripts/run_training.sh
# Expected: High precision with these parameters

# Quick test (100 samples for development)
./scripts/run_training.sh --quick

# Explore precision-recall tradeoff
./scripts/run_training.sh --test

# Custom configuration (for research)
./scripts/run_training.sh 1000 0.8  # 1000 samples, threshold=0.8

# Direct Python usage (advanced)
PYTHONPATH=. python scripts/train.py --n-samples 1000
```

### Real-Time Training Monitoring

The training scripts support real-time logging for monitoring progress:

```bash
# Start training with logging to file
PYTHONPATH=. python scripts/train.py \
    --n-samples 1000 \
    --mass-tolerance 0.005 \
    --probability-threshold 0.9 > training.log 2>&1 &

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

### Key Performance Metrics

**Performance with default parameters:**
- High precision achievable with recommended settings
- Training time: ~5 minutes with 1000 samples
- Performance varies with dataset characteristics

The ablation study showed these parameters work well in testing.

### Key Parameters (Updated based on challenging data testing)
- `--mass-tolerance`: 0.005 Da (filters 97.7% of false candidates with realistic data)
- `--rt-window-k`: 1.5 (optimal balance, accepts 87% of normal distribution)
- `--probability-threshold`: 0.9 (recommended for conservative assignment)

**Performance with Challenging Data (30% isomers, 20% near-isobars):**
- k=0.5: 81.5% precision, 32.9% recall (too restrictive)
- k=1.0: 77.9% precision, 56.0% recall (conservative)
- k=1.5: 76.9% precision, 78.7% recall (balanced) ✓
- k=2.0: 74.5% precision, 87.4% recall (permissive)

**Important Discovery**: 99% precision is NOT achievable with mass/RT alone when realistic data includes isomers and near-isobars. Maximum achievable precision is ~86% even with ultra-restrictive k=0.1 (which loses 92.5% of true positives).

See `docs/k_parameter_findings.md` and `docs/synthetic_data_generation.md` for details.

## Architecture and Key Design Decisions

### Two-Stage Bayesian Pipeline

1. **Stage 1: RT Prediction (`rt_hierarchical.py`)**
   - Hierarchical structure: species→clusters, compounds→classes
   - **Non-centered parameterization** to avoid funnel geometry in MCMC
   - Returns RT predictions WITH uncertainty (crucial for Stage 2)

2. **Stage 2: Peak Assignment (`peak_assignment.py`)**
   - Uses RT predictions from Stage 1
   - **Simple logistic regression** with mass and RT difference features
   - **Probability threshold** controls precision-recall tradeoff
   - Achieves >99% precision through parameter optimization alone

### Critical Implementation Details

#### Why Simple Works Better
The ablation study revealed that:
- **Mass tolerance filtering** removes most false candidates upfront
- **Conservative thresholds** handle remaining uncertainty
- Complex features (RT uncertainty, asymmetric loss) add no value
- Simpler models are more robust and easier to calibrate

#### Single Model Approach
- `peak_assignment.py`: Simplified model with configurable parameters
- Default parameters achieve >99% precision (mass_tol=0.005, threshold=0.9)
- Adjust parameters based on precision-recall requirements

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

**Production Configuration (Default):**
- Mass tolerance: 0.005 Da
- Probability threshold: 0.9
- These parameters showed strong performance in ablation studies
- Actual results will vary with your specific dataset

## Key Files to Understand the System

1. **Models**: 
   - `src/compassign/rt_hierarchical.py` - Hierarchical RT prediction
   - `src/compassign/peak_assignment.py` - Peak assignment with tuned parameters

2. **Documentation**:
   - `docs/precision_optimization.md` - Parameter tuning strategies with ablation study results
   - `docs/bayesian_models.md` - Mathematical specifications

3. **Training Scripts**:
   - `scripts/train.py` - Main training entry point
   - `scripts/ablation_study.py` - Evidence for simplification

## Implementation Status

### ✅ Completed
- Simplified to single model with parameter optimization
- Achieved 99.5% precision through parameter tuning alone
- Removed unnecessary complexity based on ablation study
- Created comprehensive migration guide

### 🎯 Current Focus
- Parameter optimization for different use cases
- Documentation and deployment simplification

### 🔮 Future Enhancements
- Uncertainty-informed RT windows for candidate generation
- Isotope pattern features
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