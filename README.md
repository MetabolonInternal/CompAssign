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

# Run training script on synthetic data. Will also print eval
 ./scripts/run_training.s
```

## 📚 Documentation

- [Bayesian Models](docs/bayesian_models.md): Mathematical specifications
- [CLAUDE.md](CLAUDE.md): AI assistant instructions for development

## 🤝 Contributing

Feel free to contribute! Key areas for improvement:

1. **Isotope pattern features**: Additional physical constraints
2. **Peak quality metrics**: Signal-to-noise, peak shape
3. **Uncertainty-informed RT windows**: Adaptive candidate generation
4. **Real-world dataset validation**: Beyond synthetic data

Please ensure:
- All tests pass
- Code follows existing style
- Documentation is updated

## 📄 License

This project is proprietary to Metabolon. All rights reserved.