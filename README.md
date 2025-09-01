# CompAssign: Compound Assignment with Bayesian Inference

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyMC 5.25+](https://img.shields.io/badge/PyMC-5.25+-green.svg)](https://www.pymc.io/)

## 🎯 Overview

**CompAssign** is a Bayesian framework for compound assignment in untargeted metabolomics. It combines hierarchical retention time (RT) modeling with probabilistic peak matching to achieve confident peak-to-compound assignment. The Bayesian framework provides uncertainty quantification for all predictions.

### Key Features
- 🔬 **Two-stage Bayesian approach**: RT prediction → Probabilistic assignment
- 🎲 **Uncertainty quantification**: Full posterior distributions for all predictions
- 🏗️ **Hierarchical modeling**: Captures biological and chemical structure in the data

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MetabolonInternal/CompAssign.git
cd CompAssign

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate compassign

# Run training script on synthetic data with evaluation
./scripts/run_training.sh
```

## 📚 Documentation

- [Bayesian Models](docs/bayesian_models.md): Mathematical specifications
- [CLAUDE.md](CLAUDE.md): AI assistant instructions for development

## 🤝 Contributing

This project is under active development. Key areas for contribution:

1. **Real-world dataset validation**: Testing beyond synthetic data
2. **Performance optimization**: Faster MCMC sampling
3. **Additional features**: Isotope patterns, peak quality metrics

Please ensure:
- Code follows existing style
- Documentation is updated
- Changes maintain the principle of precision over recall

## 📄 License

This project is proprietary to Metabolon. All rights reserved.