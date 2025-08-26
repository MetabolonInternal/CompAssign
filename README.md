# CompAssign: Compound Assignment with Bayesian Inference

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyMC 5.25+](https://img.shields.io/badge/PyMC-5.25+-green.svg)](https://www.pymc.io/)

## 🎯 Overview

**CompAssign** is a Bayesian framework for ultra-high precision compound assignment in untargeted metabolomics. It combines hierarchical retention time (RT) modeling with probabilistic spectral matching to achieve highly confident peak-to-compound assignment. The Bayesian framework allows us to report prediction confidence in a principled manner, and we can also update this in real time given user's feedback.

### Key Features
- 🔬 Two-stage Bayesian approach: RT prediction → Probabilistic assignment
- 🎲 Uncertainty quantification**: Full posterior distributions for all predictions
- 🏗️ Hierarchical modeling: Share power between rare species and compounds to improve prediction.

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

## 📚 Documentation

- [Mathematical Models](docs/bayesian_models.md) - Detailed model specifications
- [Precision Optimization](docs/precision_optimization.md) - Achieving >95% precision
- [Results Guide](docs/results_guide.md) - Interpreting outputs
- [Development Tasks](docs/TASKS.md) - Roadmap and TODOs
