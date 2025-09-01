# Bayesian Models: Mathematical Specifications

## Abstract

This document provides the formal mathematical specifications for the two-stage Bayesian framework implemented in CompAssign for metabolomics compound assignment. The framework consists of a hierarchical Bayesian model for retention time (RT) prediction followed by a calibrated logistic regression model with 9 optimized features for peak-to-compound assignment.

## 1. Hierarchical Retention Time Model

The retention time model employs a hierarchical Bayesian structure that captures the nested relationships inherent in metabolomics data, where species are grouped into biological clusters and compounds are grouped into chemical classes.

### 1.1 Model Specification

For a retention time observation $y_{ij}$ of compound $j$ in species $i$, the model is specified as:

```math
y_{ij} = \mu_0 + \alpha_{s(i)} + \beta_{c(j)} + \boldsymbol{\theta}^\top \mathbf{x}_j + \gamma \cdot \mathrm{IS}_{i} + \epsilon_{ij}
```

where:

* $\mu_0$ represents the global intercept
* $\alpha_{s(i)}$ denotes the species-specific effect for species $i$
* $\beta_{c(j)}$ denotes the compound-specific effect for compound $j$
* $\boldsymbol{\theta}$ is a vector of regression coefficients for molecular descriptors
* $\mathbf{x}_j$ is the molecular descriptor vector for compound $j$
* $\gamma$ is the coefficient for internal standard correction
* $\text{IS}_i$ is the internal standard retention time for species $i$
* $\epsilon_{ij} \sim \mathcal{N}(0, \sigma_y^2)$ represents observation noise

### 1.2 Hierarchical Structure

The hierarchical structure is implemented through nested random effects. Species are grouped into $K$ clusters, and compounds are grouped into $C$ classes:

```math
\alpha_{s(i)} = \mu_{k(s)} + \delta_{s(i)}
```

```math
\beta_{c(j)} = \nu_{\ell(c)} + \zeta_{c(j)}
```

where $k(s)$ maps species $s$ to its cluster, and $\ell(c)$ maps compound $c$ to its class.

### 1.3 Non-Centered Parameterization

To improve MCMC sampling efficiency and avoid the funnel geometry common in hierarchical models, we employ a non-centered parameterization. The random effects are reparameterized as:

```math
\mu_k = \tilde{\mu}_k \cdot \sigma_{\mathrm{cluster}}, \quad \tilde{\mu}_k \sim \mathcal{N}(0, 1)
```

```math
\delta_s = \mu_{k(s)} + \tilde{\delta}_s \cdot \sigma_{\mathrm{species}}, \quad \tilde{\delta}_s \sim \mathcal{N}(0, 1)
```

```math
\nu_c = \tilde{\nu}_c \cdot \sigma_{\mathrm{class}}, \quad \tilde{\nu}_c \sim \mathcal{N}(0, 1)
```

```math
\zeta_j = \nu_{\ell(j)} + \tilde{\zeta}_j \cdot \sigma_{\mathrm{compound}}, \quad \tilde{\zeta}_j \sim \mathcal{N}(0, 1)
```

### 1.4 Prior Specifications

The model employs weakly informative priors that allow the data to dominate while providing reasonable regularization:

* Variance components: $\sigma_{\cdot} \sim \text{Exponential}(\lambda)$ with rates chosen to reflect expected variability

  * $\sigma_{\text{cluster}} \sim \text{Exponential}(1.0)$
  * $\sigma_{\text{species}} \sim \text{Exponential}(2.0)$
  * $\sigma_{\text{class}} \sim \text{Exponential}(1.0)$
  * $\sigma_{\text{compound}} \sim \text{Exponential}(3.0)$
  * $\sigma_y \sim \text{Exponential}(2.0)$

* Fixed effects:

  * $\mu_0 \sim \mathcal{N}(5.0, 2.0)$ (centered at typical RT values)
  * $\boldsymbol{\theta} \sim \mathcal{N}(0, 2.0)$ (regression coefficients)
  * $\gamma \sim \mathcal{N}(1.0, 0.5)$ (internal standard coefficient with tighter prior)

## 2. Peak Assignment Model

The peak assignment model uses the RT predictions from the hierarchical model to assign observed peaks to candidate compounds through calibrated logistic regression with an optimized 9-feature set.

### 2.1 Feature Construction

For each peak-compound candidate pair $(p, c)$, nine essential features are computed, organized into three categories:

#### Core Features (Primary Signals)
1. **Mass error** (in ppm): $\Delta m_{pc} = \frac{m_p - m_c}{m_c} \times 10^6$
2. **RT z-score**: $z_{pc} = \frac{t_p - \hat{t}_c}{\hat{\sigma}_c}$
3. **Log intensity**: $\ell_p = \log(I_p)$

#### Confidence Scores (Calibrated Probabilities)
4. **Mass confidence**: $\psi_m = \exp(-|\Delta m_{pc}| / 10)$
5. **RT confidence**: $\psi_t = \exp(-|z_{pc}| / 3)$
6. **Combined confidence**: $\psi_{mt} = \psi_m \times \psi_t$

#### Context Features (Prior Information)
7. **Log compound mass**: $\kappa_c = \log(m_c)$
8. **Log RT uncertainty**: $\upsilon_c = \log(\hat{\sigma}_c + 10^{-6})$
9. **Log relative intensity**: $\rho_p = \log(I_p / \tilde{I})$

where:
* $m_p$ and $t_p$ are the observed mass and RT of peak $p$
* $m_c$ is the theoretical mass of compound $c$
* $(\hat{t}_c, \hat{\sigma}_c)$ are the predicted RT mean and standard deviation for compound $c$ from the hierarchical model
* $\tilde{I}$ is the median intensity across all peaks
* $I_p$ is the intensity of peak $p$

### 2.2 Calibrated Logistic Regression Model

The probability that peak $p$ corresponds to compound $c$ is modeled as:

```math
\text{logit}\,P(y_{pc} = 1 \mid \mathbf{f}_{pc}) = \phi_0 + \sum_{k=1}^{9} \phi_k f_k
```

where $\mathbf{f} = [\Delta m_{pc}, z_{pc}, \ell_p, \psi_m, \psi_t, \psi_{mt}, \kappa_c, \upsilon_c, \rho_p]^T$ is the 9-dimensional feature vector.

### 2.3 Feature Standardization and Class Balancing

To improve model stability and calibration:

1. **Feature Standardization**: All features are standardized to zero mean and unit variance before training
2. **Class Balancing**: Positive examples (true matches) are upweighted during training to account for class imbalance
3. **Temperature Scaling**: Post-training calibration is applied to ensure well-calibrated probabilities

### 2.4 Prior Specifications

The logistic regression employs weakly informative normal priors adapted for standardized features:

```math
\phi_0 \sim \mathcal{N}(0, 1.0)
```

```math
\phi_k \sim \mathcal{N}(0, 1.0), \quad k \in \{1, \ldots, 9\}
```

These tighter priors (compared to unstandardized features) provide appropriate regularization for the standardized feature space.

## 3. Inference and Implementation

### 3.1 Posterior Sampling

Both models employ the No-U-Turn Sampler (NUTS), an adaptive variant of Hamiltonian Monte Carlo, for posterior inference. The sampler automatically tunes its parameters during a warm-up phase to achieve efficient exploration of the posterior distribution.

### 3.2 Uncertainty Propagation

A key feature of the framework is the propagation of uncertainty from the RT model to the assignment model. The posterior distribution of RT predictions provides not only point estimates $\hat{t}_c$ but also uncertainty estimates $\hat{\sigma}_c$, which are incorporated into both the RT z-score feature and the log RT uncertainty feature. This allows the assignment model to appropriately down-weight candidates with highly uncertain RT predictions.

### 3.3 Assignment Strategy

After obtaining posterior probabilities for all peak-compound pairs, assignments are made using a multi-stage filtering approach:

1. **Mass filter**: Retain candidates where $|\Delta m_{pc}| < \tau_m$ (typically 0.005 Da)
2. **RT filter**: Retain candidates where $|z_{pc}| < k$ (typically $k = 1.5$)
3. **Probability threshold**: Accept assignments where $P(y_{pc} = 1) > \tau_p$ (typically 0.7 after calibration)

The greedy matching algorithm ensures each peak is assigned to at most one compound, selecting the highest probability match above the threshold.

## 4. Model Optimization and Performance

### 4.1 Feature Selection

Through systematic ablation studies, the feature set was optimized from an initial 16 features to 9 essential features, eliminating redundant features while maintaining performance:

**Removed features** (found to be redundant):
- Squared terms ($\Delta m_{pc}^2$, $z_{pc}^2$) - redundant with confidence scores
- Absolute values ($|\Delta m_{pc}|$, $|z_{pc}|$) - redundant with squared terms
- Normalized features - redundant with standardization
- Interaction terms - minimal contribution

This 44% reduction in features maintains the same F1 score (98.7%) while improving model interpretability and training efficiency.

### 4.2 Performance Characteristics

The optimized model achieves:
- **F1 Score**: 98.7%
- **Precision**: >98% (minimizing false positives as per design goal)
- **Recall**: >99% (near-complete recovery of true compounds)

These metrics are achieved with temperature-calibrated probabilities, ensuring that the probability threshold has meaningful interpretation.

## 5. Computational Considerations

The non-centered parameterization in the hierarchical model significantly improves sampling efficiency by eliminating the correlation between hierarchical parameters that causes the "funnel" geometry in centered parameterizations. This results in better effective sample sizes and reduced autocorrelation in the MCMC chains.

The two-stage approach allows for modular development and testing, with the RT model providing interpretable predictions that can be validated independently before being used in the assignment model. This separation also enables the use of different training datasets for each stage if desired.

## References

The mathematical framework draws from established principles in Bayesian hierarchical modeling and logistic regression, adapted specifically for the challenges of metabolomics data analysis where both biological variability (across species) and chemical variability (across compounds) must be simultaneously modeled. The feature engineering incorporates domain knowledge about mass spectrometry and chromatography to create informative representations for compound assignment.