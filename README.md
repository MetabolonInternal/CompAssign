# CompAssign: Compound Assignment with Bayesian Inference

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyMC 5.25+](https://img.shields.io/badge/PyMC-5.25+-green.svg)](https://www.pymc.io/)

## 🎯 Overview

**CompAssign** is a Bayesian framework for ultra-high precision compound assignment in untargeted metabolomics. It combines hierarchical retention time (RT) modeling with probabilistic spectral matching to achieve >95% assignment precision critical for metabolomics applications.

### Key Features
- 🔬 **Two-stage Bayesian approach**: RT prediction → Probabilistic assignment
- 📊 **Ultra-high precision**: >95% precision with optimized thresholds
- 🎲 **Uncertainty quantification**: Full posterior distributions for all predictions
- 🏗️ **Hierarchical modeling**: Accounts for species/compound structure
- ⚖️ **Class-weighted loss**: Minimizes false positives for high-stakes applications

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
    EnhancedPeakAssignmentModel
)

# Generate or load your data
obs_df, peak_df, params = generate_synthetic_data()

# Train RT model
rt_model = HierarchicalRTModel(...)
rt_model.build_model(obs_df)
rt_trace = rt_model.sample()

# Train enhanced assignment model for high precision
assignment_model = EnhancedPeakAssignmentModel(
    mass_tolerance=0.005,  # Tight tolerance
    fp_penalty=5.0          # Penalize false positives
)
assignment_model.compute_rt_predictions(rt_trace, ...)
assignment_model.build_model()
assignment_trace = assignment_model.sample()

# Make predictions with high precision threshold
results = assignment_model.predict_assignments_staged(
    peak_df,
    high_precision_threshold=0.9  # >95% precision
)
```

### Command Line Interface

```bash
# Standard training (baseline model)
python scripts/train.py --model standard --n-samples 1000

# Enhanced training for ultra-high precision (production)
python scripts/train.py --model enhanced \
    --n-samples 1000 \
    --test-thresholds \
    --mass-tolerance 0.005 \
    --fp-penalty 5.0 \
    --high-precision-threshold 0.9

# Analyze precision-recall tradeoff
python scripts/analyze_precision.py
```

## 📁 Project Structure

```
compassign/
├── src/
│   └── compassign/         # Main CompAssign module
│       ├── __init__.py
│       ├── synthetic_generator.py      # Data generation
│       ├── rt_hierarchical.py          # RT prediction model
│       ├── peak_assignment.py          # Standard assignment
│       ├── peak_assignment_enhanced.py # High-precision assignment
│       ├── diagnostic_plots.py         # Model diagnostics
│       └── assignment_plots.py         # Assignment visualizations
├── docs/
│   ├── README.md                    # Detailed documentation
│   ├── bayesian_models.md          # Mathematical specifications
│   ├── precision_optimization.md   # Precision tuning guide
│   ├── results_guide.md           # Results interpretation
│   └── TASKS.md                   # Development roadmap
├── output/                        # Results directory
│   ├── data/                     # Processed data
│   ├── models/                   # Saved model traces
│   ├── plots/                    # Diagnostic plots
│   └── results/                  # Performance metrics
├── train.py                      # Main training script
├── train_enhanced.py            # Enhanced precision training
└── analyze_precision.py         # Precision analysis tools
```

## 🎓 Mathematical Framework

### Stage 1: Hierarchical RT Regression
```
RT ~ μ₀ + species_effect + compound_effect + β·descriptors + γ·internal_std + ε
```
- Hierarchical structure: species→clusters, compounds→classes
- Non-centered parameterization for efficient sampling

### Stage 2: Probabilistic Peak Assignment
```
P(match) = σ(θ₀ + θ_mass·|Δm/z| + θ_rt·|z_RT| + θ_int·log(I) + θ_unc·σ_RT)
```
- Class-weighted loss: 5× penalty for false positives
- Calibrated probabilities via isotonic regression
- Staged assignment: confident/review/rejected

## 📊 Performance

| Model | Precision | Recall | False Positives |
|-------|-----------|--------|-----------------|
| Baseline (threshold=0.5) | 84.4% | 98.7% | 14 |
| Enhanced (threshold=0.8) | 91.9% | 74.0% | 5 |
| **Enhanced (threshold=0.9)** | **>95%** | **~65%** | **<3** |

## 📚 Documentation

- [Mathematical Models](docs/bayesian_models.md) - Detailed model specifications
- [Precision Optimization](docs/precision_optimization.md) - Achieving >95% precision
- [Results Guide](docs/results_guide.md) - Interpreting outputs
- [Development Tasks](docs/TASKS.md) - Roadmap and TODOs

## 🔬 Use Cases

PRISM is designed for:
- **Clinical metabolomics** where false positives are costly
- **Biomarker discovery** requiring high-confidence assignments
- **Untargeted metabolomics** with complex biological matrices
- **Quality control** in metabolomics core facilities

## 🤝 Contributing

We welcome contributions! Key areas for improvement:
1. Isotope pattern matching
2. Peak quality metrics
3. Multi-model ensemble methods
4. Deep learning alternatives

See [TASKS.md](docs/TASKS.md) for the development roadmap.

## 📄 License

This project is proprietary to Metabolon Internal.

## 📧 Contact

For questions or support:
- Metabolon RT Team
- Internal Slack: #prism-support

---

**CompAssign**: *Bringing Bayesian precision to metabolomics compound assignment*