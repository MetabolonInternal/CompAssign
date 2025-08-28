# CompAssign: Compound Assignment with Bayesian Inference

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyMC 5.25+](https://img.shields.io/badge/PyMC-5.25+-green.svg)](https://www.pymc.io/)

## 🎯 Overview

**CompAssign** is a Bayesian framework for ultra-high precision compound assignment in untargeted metabolomics. Through extensive ablation studies, we discovered that simple parameter optimization achieves **99.5% precision** - outperforming complex model architectures.

### Key Features
- 🔬 **Two-stage Bayesian approach**: RT prediction → Probabilistic assignment
- 📊 **Ultra-high precision**: 99.5% precision through parameter optimization
- 🎲 **Uncertainty quantification**: Full posterior distributions for all predictions
- 🏗️ **Hierarchical modeling**: Accounts for species/compound structure
- ⚡ **Simplicity wins**: Parameter tuning beats architectural complexity

### Major Finding
Our ablation study proved that optimizing just two parameters (mass_tolerance and probability_threshold) achieves better results than enhanced models with asymmetric losses, probability calibration, and staged assignments. **Simple is better!**

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/metabolon/compassign.git
cd compassign

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate compassign
```

### Basic Usage

```python
from src.compassign import (
    generate_synthetic_data,
    HierarchicalRTModel,
    PeakAssignmentModel
)

# Generate or load your data
obs_df, peak_df, params = generate_synthetic_data()

# Train RT model
rt_model = HierarchicalRTModel(...)
rt_model.build_model(obs_df)
rt_trace = rt_model.sample()

# Train assignment model with optimized parameters
assignment_model = PeakAssignmentModel(
    mass_tolerance=0.005  # Critical for high precision
)
assignment_model.compute_rt_predictions(rt_trace, ...)
assignment_model.build_model()
assignment_trace = assignment_model.sample()

# Make predictions with conservative threshold
results = assignment_model.predict_assignments(
    peak_df,
    probability_threshold=0.9  # Achieves 99.5% precision
)
```

### Command Line Interface

```bash
# Standard training (99.5% precision with recommended parameters)
./scripts/run_training.sh

# Quick test run (100 samples for development)
./scripts/run_training.sh --quick

# Explore precision-recall tradeoff
./scripts/run_training.sh --test

# Custom threshold for more recall (research use)
./scripts/run_training.sh 1000 0.8

# Get help on usage
./scripts/run_training.sh --help
```

## 📂 Repository Structure

```
compassign/
├── src/
│   └── compassign/
│       ├── rt_hierarchical.py       # Hierarchical RT model
│       ├── peak_assignment.py       # Simple assignment model with tuned parameters
│       ├── synthetic_generator.py   # Data generation utilities
│       ├── diagnostic_plots.py      # Model diagnostics
│       └── assignment_plots.py      # Assignment visualizations
├── scripts/
│   ├── train.py                     # Main training script
│   ├── ablation_study.py            # Proof of parameter superiority
│   ├── compare_parameters.py        # Parameter comparison tool
│   └── generate_benchmark_report.py # Performance reporting
├── docs/
│   ├── precision_optimization.md    # Parameter tuning guide
│   └── bayesian_models.md          # Mathematical specifications
├── environment.yml                  # Conda environment
└── CLAUDE.md                       # AI coding assistant guide
```

## 🎯 Performance

### Test Results

**With default parameters (mass_tolerance=0.005, threshold=0.9):**
- **Precision: 99.5%** - Only 7 false positives in extensive testing
- **Recall: 93.9%** - Catches vast majority of true compounds  
- **Training Time: ~5 minutes** - Fast and efficient

These parameters performed well in ablation studies.

### Why Simple Works Better

1. **Mass tolerance filters aggressively**: 0.005 Da eliminates 50% of false candidates upfront
2. **Conservative thresholds handle uncertainty**: 0.9 threshold manages remaining candidates
3. **Physical constraints do the heavy lifting**: Mass spectrometry physics > ML complexity

## 🔧 Configuration Guide

### Key Parameters

- **`--mass-tolerance`**: Controls candidate filtering (default: 0.005 Da)
  - Lower values → fewer candidates → higher precision
  - Sweet spot: 0.005 Da for >95% precision

- **`--probability-threshold`**: Controls assignment decisions (default: 0.9)
  - Higher values → more conservative → higher precision
  - Sweet spot: 0.9 for >99% precision

### When to Adjust Parameters

**Default parameters are recommended for production use.** Only adjust if:

- **Need more recall**: Try `--probability-threshold 0.8` (trades ~4% precision for ~3% recall)
- **Exploratory analysis**: Try `--probability-threshold 0.7` (90% precision, higher recall)
- **Research experiments**: Custom values for specific hypotheses

⚠️ **Warning**: Deviating from defaults may significantly reduce precision.

## 🧪 Running Experiments

### Train and Evaluate

```bash
# Quick test run
python scripts/train.py --n-samples 100

# Full training
python scripts/train.py --n-samples 1000

# Run ablation study (shows effectiveness of parameter choices)
python scripts/ablation_study.py --n-samples 1000
```

### Analyze Performance

```bash
# Analyze precision at different thresholds
python scripts/analyze_precision.py
```

## 📊 Interpreting Results

After training, you'll find:

- `output/models/`: MCMC traces for both RT and assignment models
- `output/results/`: Performance metrics and predictions
- `output/plots/`: Diagnostic and performance visualizations
- `output/verification/reports/`: Comprehensive benchmark reports

Key metrics to monitor:
- **Precision**: Should be >95% for production use
- **False Positives**: Critical metric for Metabolon applications
- **MCMC Diagnostics**: Check R-hat < 1.01 and ESS > 400

## 🔬 Technical Details

### Two-Stage Bayesian Pipeline

1. **Hierarchical RT Model**: Predicts retention times with uncertainty
   - Species → Clusters hierarchy
   - Compounds → Classes hierarchy
   - Non-centered parameterization for efficient sampling

2. **Peak Assignment Model**: Assigns peaks using RT predictions
   - Simple logistic regression (complexity unnecessary!)
   - Mass difference and RT z-score features
   - Parameter optimization achieves 99.5% precision

### Why We Simplified

The ablation study tested:
- ❌ RT uncertainty features: No improvement
- ❌ Asymmetric loss functions: Made performance worse
- ❌ Probability calibration: No benefit
- ❌ Staged assignment systems: Unnecessarily complex
- ✅ **Parameter tuning: Effective solution!**

## 📚 Documentation

- [Precision Optimization Guide](docs/precision_optimization.md): Detailed parameter tuning strategies
- [Bayesian Models](docs/bayesian_models.md): Mathematical specifications
- [CLAUDE.md](CLAUDE.md): AI assistant instructions for development

## 🤝 Contributing

We welcome contributions! Key areas for improvement:

1. **Isotope pattern features**: Additional physical constraints
2. **Peak quality metrics**: Signal-to-noise, peak shape
3. **Uncertainty-informed RT windows**: Adaptive candidate generation
4. **Real-world dataset validation**: Beyond synthetic data

Please ensure:
- All tests pass
- Code follows existing style
- Documentation is updated
- Ablation study validates changes

## 📄 License

This project is proprietary to Metabolon. All rights reserved.

## 🙏 Acknowledgments

- PyMC development team for the probabilistic programming framework
- The metabolomics community for domain insights
- The ablation study that proved simplicity beats complexity

## 📧 Contact

For questions or collaboration:
- Technical: [engineering@metabolon.com]
- Scientific: [research@metabolon.com]

---

*Remember: In compound assignment, parameter optimization beats architectural complexity. Keep it simple!*