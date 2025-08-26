# CompAssign: Compound Assignment with Bayesian Inference

## Table of Contents
1. [Overview](#overview)
2. [Mathematical Models](#mathematical-models)
3. [Implementation Details](#implementation-details)
4. [Evaluation Methodology](#evaluation-methodology)
5. [Results Interpretation](#results-interpretation)
6. [Output Structure](#output-structure)
7. [Probability Calibration](probability_calibration.md) - Why and how we calibrate model confidence

## Overview

CompAssign implements a two-stage Bayesian approach for metabolomics data analysis:

1. **Hierarchical RT (Retention Time) Regression Model**: Predicts retention times for metabolites across different species/samples, accounting for hierarchical structure in the data
2. **Bayesian Logistic Peak Assignment Model**: Uses RT predictions to assign LC-MS peaks to known compounds with quantified uncertainty

The approach addresses key challenges in metabolomics:
- Missing data (not all compounds present in all samples)
- Hierarchical structure (species grouped in clusters, compounds grouped in classes)
- Peak ambiguity (multiple compounds with similar masses)
- Measurement uncertainty

## Mathematical Models

### 1. Hierarchical RT Regression Model

The model predicts retention time (RT) for compound $m$ in species $s$:

$$y_{sm} = \mu_0 + u_s + v_m + \boldsymbol{\beta}^T \mathbf{x}_m + \gamma \cdot \text{IS}_s + \epsilon_{sm}$$

Where:
- $y_{sm}$: Observed RT for compound $m$ in species $s$
- $\mu_0$: Global intercept (baseline RT)
- $u_s$: Species-specific random effect
- $v_m$: Compound-specific random effect
- $\boldsymbol{\beta}$: Coefficients for molecular descriptors
- $\mathbf{x}_m$: Molecular descriptors for compound $m$ (e.g., hydrophobicity, polarity)
- $\gamma$: Coefficient for internal standard correction
- $\text{IS}_s$: Internal standard RT for species $s$
- $\epsilon_{sm} \sim \mathcal{N}(0, \sigma_y^2)$: Observation noise

#### Hierarchical Structure

**Species hierarchy:**
- Species $s$ belongs to cluster $k(s)$
- $u_s = \alpha_{k(s)} + \delta_s$
- $\alpha_k \sim \mathcal{N}(0, \sigma_{\text{cluster}}^2)$: Cluster effect
- $\delta_s \sim \mathcal{N}(0, \sigma_{\text{species}}^2)$: Species deviation

**Compound hierarchy:**
- Compound $m$ belongs to class $c(m)$
- $v_m = \kappa_{c(m)} + \zeta_m$
- $\kappa_c \sim \mathcal{N}(0, \sigma_{\text{class}}^2)$: Class effect
- $\zeta_m \sim \mathcal{N}(0, \sigma_{\text{compound}}^2)$: Compound deviation

#### Non-centered Parameterization

To improve MCMC sampling efficiency and avoid divergences, we use non-centered parameterization:

```
cluster_eff = cluster_raw * σ_cluster
species_eff = cluster_eff[k(s)] + species_raw * σ_species
class_eff = class_raw * σ_class
compound_eff = class_eff[c(m)] + compound_raw * σ_compound
```

Where `*_raw ~ N(0, 1)` are standard normal variables.

#### Prior Distributions

- $\mu_0 \sim \mathcal{N}(5, 2)$: Centered at typical RT value
- $\boldsymbol{\beta} \sim \mathcal{N}(0, 2)$: Regularized regression coefficients
- $\gamma \sim \mathcal{N}(1, 0.5)$: Internal standard coefficient (expected ~1)
- $\sigma_* \sim \text{Exponential}(\lambda)$: Variance parameters with different rates

### 2. Bayesian Logistic Peak Assignment Model

For each peak-compound candidate pair $(p, m)$, we model the probability of a true match:

$$\text{logit}(P(y_{pm} = 1)) = \theta_0 + \theta_1 \cdot \text{MassErr}_{pm} + \theta_2 \cdot \text{RT-Z}_{pm} + \theta_3 \cdot \log(\text{Intensity}_p)$$

Where:
- $y_{pm} \in \{0, 1\}$: Binary indicator (1 = true match)
- $\theta_0$: Intercept (baseline log-odds)
- $\theta_1$: Mass error coefficient (ppm)
- $\theta_2$: RT z-score coefficient
- $\theta_3$: Log intensity coefficient

#### Feature Engineering

1. **Mass Error (ppm)**:
   $$\text{MassErr}_{pm} = \frac{m/z_{\text{observed}} - m/z_{\text{theoretical}}}{m/z_{\text{theoretical}}} \times 10^6$$

2. **RT Z-score**:
   $$\text{RT-Z}_{pm} = \frac{\text{RT}_{\text{observed}} - \hat{\text{RT}}_{sm}}{\sigma_{\text{RT},sm}}$$
   
   Where $\hat{\text{RT}}_{sm}$ and $\sigma_{\text{RT},sm}$ come from the RT model posterior

3. **Log Intensity**:
   $$\log_{10}(\text{Intensity}_p)$$

#### Prior Distributions

- $\theta_i \sim \mathcal{N}(0, 2)$ for all coefficients

## Implementation Details

### Technology Stack

- **Language**: Python 3.11
- **Probabilistic Programming**: PyMC 5.25.1
- **MCMC Sampling**: NUTS (No-U-Turn Sampler)
- **Visualization**: ArviZ, Matplotlib, Seaborn
- **Data Processing**: NumPy, Pandas

### Project Structure

```
rt_regression/
├── src/
│   ├── data/
│   │   └── synthetic_generator.py    # Synthetic data generation
│   ├── models/
│   │   ├── rt_hierarchical.py       # RT regression model
│   │   └── peak_assignment.py       # Logistic assignment model
│   └── visualization/
│       ├── diagnostic_plots.py      # MCMC diagnostics
│       └── assignment_plots.py      # Assignment performance plots
├── output/
│   ├── data/                        # Generated/processed data
│   ├── plots/                       # All diagnostic plots
│   └── results/                     # Model outputs and metrics
├── docs/                            # Documentation
└── train.py                         # Main training pipeline
```

### Key Implementation Features

1. **Non-centered Parameterization**: Prevents divergences in hierarchical models
2. **Adaptive NUTS Sampling**: 
   - Target acceptance: 0.95
   - Max tree depth: 12
   - Initialization: adapt_diag
3. **Posterior Predictive Checks**: Validates model calibration
4. **Comprehensive Diagnostics**: R-hat, ESS, energy plots, trace plots

### Workflow Pipeline

1. **Data Generation/Loading**
   - Generate synthetic data with known parameters
   - Create hierarchical structure (clusters, classes)
   - Add measurement noise and decoy peaks

2. **RT Model Training**
   - Build hierarchical Bayesian model
   - Sample posterior using NUTS
   - Compute posterior predictive distributions
   - Generate diagnostic plots

3. **Peak Assignment**
   - Generate candidate peak-compound pairs (mass tolerance filter)
   - Extract features using RT predictions
   - Train logistic regression model
   - Assign peaks to compounds

4. **Evaluation**
   - Compare estimated vs true parameters
   - Calculate prediction metrics
   - Generate comprehensive visualizations

## Evaluation Methodology

### RT Model Evaluation

1. **Parameter Recovery**
   - Compare posterior means to true parameter values
   - Assess bias and variance of estimates

2. **Predictive Performance**
   - **RMSE** (Root Mean Square Error): $\sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$
   - **MAE** (Mean Absolute Error): $\frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$
   - **Coverage**: Proportion of observations within 95% prediction intervals

3. **Convergence Diagnostics**
   - **R-hat**: Potential scale reduction factor (should be < 1.01)
   - **ESS**: Effective sample size (should be > 100)
   - **Divergences**: Number of divergent transitions (should be 0)

### Peak Assignment Evaluation

1. **Classification Metrics**
   - **Precision**: $\frac{TP}{TP + FP}$ (accuracy of positive predictions)
   - **Recall**: $\frac{TP}{TP + FN}$ (proportion of true peaks found)
   - **F1 Score**: $2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

2. **Probability Calibration**
   - Compare predicted probabilities to observed frequencies
   - Assess discrimination using ROC and PR curves
   - **AUC-ROC**: Area under ROC curve (discrimination ability)
   - **AUC-PR**: Area under Precision-Recall curve

3. **Confusion Matrix Analysis**
   - True Positives (TP): Correctly assigned peaks
   - False Positives (FP): Incorrect assignments
   - False Negatives (FN): Missed true peaks
   - True Negatives (TN): Correctly rejected decoys

## Results Interpretation

### Typical Results (Synthetic Data)

#### RT Model Performance
- **RMSE**: ~0.39 (below true noise σ = 0.5)
- **MAE**: ~0.31
- **95% Coverage**: ~98.7% (well-calibrated)
- **Parameter Recovery**: Most within 10% of true values

#### Peak Assignment Performance
- **Precision**: ~84.4%
- **Recall**: ~98.7%
- **F1 Score**: ~91.0%

#### Key Findings
1. **RT z-score** is the most informative feature (coefficient ~-1.5)
2. **Log intensity** positively correlates with true assignments
3. **Mass error** has smaller but significant effect

### Interpreting Diagnostic Plots

#### RT Model Plots

1. **trace_plots.png**: Check for convergence (chains should mix well)
2. **energy_plot.png**: Energy transitions should overlap with marginal energy
3. **posterior_vs_true.png**: Assess parameter recovery accuracy
4. **ppc_plots.png**: Observed vs predicted should follow diagonal
5. **residual_diagnostics.png**: Residuals should be normally distributed

#### Assignment Model Plots

1. **logistic_coefficients.png**: Posterior distributions of feature weights
2. **feature_distributions.png**: Feature separation between true/false
3. **roc_pr_curves.png**: Classification performance (AUC > 0.9 is excellent)
4. **probability_calibration.png**: Predicted probabilities should match frequencies
5. **confusion_matrix.png**: Visual summary of classification errors

## Output Structure

```
output/
├── data/
│   ├── observations.csv           # RT training data
│   ├── peaks.csv                  # Peak data with true labels
│   ├── true_parameters.json       # True parameter values
│   └── logit_training_data.csv    # Peak assignment training data
├── plots/
│   ├── rt_model/                  # RT model diagnostic plots
│   │   ├── trace_plots.png       # MCMC traces
│   │   ├── energy_plot.png       # Energy diagnostics
│   │   ├── posterior_vs_true.png # Parameter recovery
│   │   ├── pairs_plot.png        # Parameter correlations
│   │   ├── forest_plot.png       # Variance components
│   │   ├── ppc_plots.png         # Posterior predictive checks
│   │   ├── residual_diagnostics.png # Residual analysis
│   │   └── convergence_diagnostics.png # R-hat and ESS
│   └── assignment_model/          # Assignment model plots
│       ├── logistic_coefficients.png # Coefficient posteriors
│       ├── feature_distributions.png # Feature separation
│       ├── roc_pr_curves.png     # ROC and PR curves
│       ├── probability_calibration.png # Calibration plots
│       ├── confusion_matrix.png  # Confusion matrix
│       ├── feature_importance.png # Feature analysis
│       └── performance_comparison.png # Overall performance
└── results/
    ├── config.json                # Run configuration
    ├── parameter_summary.csv      # RT model parameters
    ├── predictions.csv            # RT predictions
    ├── assignment_parameters.csv  # Logistic model parameters
    ├── peak_assignments.csv      # Final assignments
    └── results.json              # Summary metrics
```

### Key Output Files

1. **results.json**: Overall performance metrics
2. **predictions.csv**: RT predictions with uncertainty
3. **peak_assignments.csv**: Peak-to-compound assignments with probabilities
4. **parameter_summary.csv**: Full posterior summaries with convergence diagnostics

## Usage

### Basic Training
```bash
python train.py --n-species 25 --n-compounds 20 --save-data
```

### Advanced Options
```bash
python train.py \
    --n-species 30 \
    --n-compounds 25 \
    --n-samples 1000 \
    --n-tune 1500 \
    --n-chains 4 \
    --target-accept 0.95 \
    --mass-tolerance 0.01 \
    --probability-threshold 0.5 \
    --save-data
```

### Parameters
- `--n-species`: Number of species/samples
- `--n-compounds`: Number of unique compounds
- `--n-samples`: MCMC samples per chain
- `--n-tune`: MCMC tuning steps
- `--n-chains`: Number of MCMC chains
- `--target-accept`: NUTS target acceptance rate
- `--mass-tolerance`: Mass tolerance (Da) for candidates
- `--probability-threshold`: Minimum probability for assignment
- `--save-data`: Save generated data to CSV
- `--skip-model`: Skip RT model training
- `--skip-assignment`: Skip peak assignment model

## References

1. Carpenter, B., et al. (2017). "Stan: A probabilistic programming language." Journal of Statistical Software.
2. Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). "Probabilistic programming in Python using PyMC3." PeerJ Computer Science.
3. Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo." JMLR.
4. Betancourt, M. (2017). "A conceptual introduction to Hamiltonian Monte Carlo." arXiv preprint.