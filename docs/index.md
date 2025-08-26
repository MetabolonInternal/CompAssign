# RT Regression and Peak Assignment Documentation

## Project Overview

This project implements a Bayesian approach to metabolomics data analysis, specifically addressing the challenge of assigning LC-MS peaks to known compounds using hierarchical retention time (RT) modeling.

## Documentation Contents

### 1. [Main Documentation](README.md)
Comprehensive overview including:
- System architecture
- Mathematical models
- Implementation details
- Evaluation methodology
- Usage instructions

### 2. [Bayesian Model Specifications](bayesian_models.md)
Technical details on:
- Full Bayesian model specifications
- Non-centered parameterization
- MCMC sampling strategies
- Information flow between models
- Computational considerations

### 3. [Results Interpretation Guide](results_guide.md)
Practical guide for:
- Understanding diagnostic plots
- Interpreting performance metrics
- Troubleshooting common issues
- Production recommendations

## Quick Links

### Getting Started
1. Install conda environment: See main README
2. Run basic training: `python train.py --save-data`
3. Check results: `output/results/results.json`
4. View plots: `output/plots/`

### Key Concepts

#### Hierarchical RT Model
- Predicts retention times across species/compounds
- Accounts for nested structure (species in clusters, compounds in classes)
- Uses molecular descriptors and internal standards
- Provides uncertainty quantification

#### Peak Assignment Model
- Uses RT predictions to disambiguate peaks
- Combines mass accuracy, RT match, and intensity
- Outputs probability of assignment
- Enables threshold-based decision making

### Performance Metrics

#### RT Model
- **RMSE**: Root mean square error (< 0.5 is good)
- **Coverage**: Proportion in 95% intervals (should ≈ 0.95)
- **R-hat**: Convergence diagnostic (< 1.01)

#### Assignment Model
- **Precision**: Correctness of assignments (> 0.8 is good)
- **Recall**: Proportion of peaks found (> 0.9 is good)
- **F1 Score**: Harmonic mean of precision/recall

## Algorithm Summary

### Stage 1: RT Regression
```
1. Build hierarchical Bayesian model
2. Sample posterior using NUTS
3. Generate RT predictions with uncertainty
```

### Stage 2: Peak Assignment
```
1. Generate candidate peak-compound pairs (mass filter)
2. Compute features:
   - Mass error (ppm)
   - RT z-score (using Stage 1 predictions)
   - Log intensity
3. Train logistic regression
4. Assign peaks based on probability threshold
```

### Stage 3: Evaluation
```
1. Compare predicted vs true parameters
2. Calculate classification metrics
3. Generate diagnostic plots
4. Assess calibration
```

## Key Innovations

1. **Hierarchical Modeling**: Captures biological and chemical relationships
2. **Uncertainty Propagation**: RT uncertainty informs assignment confidence
3. **Non-centered Parameterization**: Ensures efficient MCMC sampling
4. **Comprehensive Diagnostics**: 16+ plots for model assessment

## File Structure

```
rt_regression/
├── docs/                     # This documentation
│   ├── index.md             # This file
│   ├── README.md            # Main documentation
│   ├── bayesian_models.md   # Technical specifications
│   └── results_guide.md     # Interpretation guide
├── src/                     # Source code
│   ├── data/               # Data generation
│   ├── models/             # Bayesian models
│   └── visualization/      # Plotting functions
├── output/                  # Results (generated)
│   ├── data/              # Processed data
│   ├── plots/             # Diagnostic plots
│   └── results/           # Model outputs
└── train.py                # Main entry point
```

## Citation

If you use this code in your research, please cite:
```
Bayesian Hierarchical Modeling for LC-MS Peak Assignment
[Your Name], 2024
https://github.com/[your-repo]
```

## Contact

For questions or issues:
- GitHub Issues: [Link to repo]
- Email: [Your email]

## License

[Your chosen license]

---

*Generated with Claude Code - A Bayesian approach to metabolomics data analysis*