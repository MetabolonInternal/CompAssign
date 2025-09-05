# Bayesian Models: Mathematical Specifications

## Abstract

This document provides the formal mathematical specifications for the two-stage Bayesian framework implemented in CompAssign for metabolomics compound assignment. The framework consists of a hierarchical Bayesian model for retention time (RT) prediction (with standardized covariates, non-centered sampling, and sum-to-zero constraints) followed by a calibrated logistic regression model with 9 features for peak-to-compound assignment.

## 1. Hierarchical Retention Time Model

The RT model employs a hierarchical Bayesian structure that captures the nested relationships in metabolomics data, where species are grouped into biological clusters and compounds are grouped into chemical classes. Molecular descriptors and internal standard (IS) measurements are standardized before entering the linear predictor.

### 1.1 Model Specification

For a retention time observation \$y\_{ij}\$ of compound \$j\$ in species \$i\$, the model is:

```math
y_{ij} = \mu_0 + \alpha_{s(i)} + \beta_{c(j)} + \boldsymbol{\theta}^\top \mathbf{x}_j^{\star} + \gamma \cdot \mathrm{IS}_{i}^{\star} + \epsilon_{ij}
```

where:

* \$\mu\_0\$ is the global intercept,
* \$\alpha\_{s(i)}\$ is the species effect for species \$i\$,
* \$\beta\_{c(j)}\$ is the compound effect for compound \$j\$,
* \$\mathbf{x}\_j^{\star}\$ is the **standardized** molecular descriptor vector for compound \$j\$ (feature-wise mean subtracted, divided by std, computed from the training set),
* \$\mathrm{IS}\_{i}^{\star}\$ is the **standardized** internal-standard measurement for species \$i\$,
* \$\boldsymbol{\theta}\$ and \$\gamma\$ are regression coefficients,
* \$\epsilon\_{ij} \sim \mathcal{N}(0, \sigma\_y^2)\$ represents observation noise.

### 1.2 Hierarchical Structure

Species are grouped into \$K\$ clusters, and compounds are grouped into \$C\$ classes. We use nested random effects:

```math
\alpha_{s(i)} \sim \mathcal{N}(\mu_{k(s)}, \sigma_{\mathrm{species}}^2), \qquad \mu_k \sim \mathcal{N}(0, \sigma_{\mathrm{cluster}}^2)
```

```math
\beta_{c(j)} \sim \mathcal{N}(\nu_{\ell(c)}, \sigma_{\mathrm{compound}}^2), \qquad \nu_c \sim \mathcal{N}(0, \sigma_{\mathrm{class}}^2)
```

To avoid confounding with \$\mu\_0\$, the realized cluster, class, species, and compound effects are constrained to sum to zero within each level (sum-to-zero constraints).

### 1.3 Non-Centered Parameterization

To improve sampling efficiency and avoid funnel pathologies, we employ a non-centered parameterization for all hierarchical effects:

```math
\mu_k = \tilde{\mu}_k \cdot \sigma_{\mathrm{cluster}}, \quad \tilde{\mu}_k \sim \mathcal{N}(0, 1)
```

```math
\alpha_s = \mu_{k(s)} + \tilde{\alpha}_s \cdot \sigma_{\mathrm{species}}, \quad \tilde{\alpha}_s \sim \mathcal{N}(0, 1)
```

```math
\nu_c = \tilde{\nu}_c \cdot \sigma_{\mathrm{class}}, \quad \tilde{\nu}_c \sim \mathcal{N}(0, 1)
```

```math
\zeta_j = \nu_{\ell(j)} + \tilde{\zeta}_j \cdot \sigma_{\mathrm{compound}}, \quad \tilde{\zeta}_j \sim \mathcal{N}(0, 1)
```

(With sum-to-zero centering applied to each realized group of effects as noted above.)

### 1.4 Prior Specifications

Weakly informative priors are used, adapted to standardized covariates:

* Variance components:

  * \$\sigma\_{\text{cluster}} \sim \text{Exponential}(1.0)\$
  * \$\sigma\_{\text{species}} \sim \text{Exponential}(2.0)\$
  * \$\sigma\_{\text{class}} \sim \text{Exponential}(1.0)\$
  * \$\sigma\_{\text{compound}} \sim \text{Exponential}(3.0)\$
  * \$\sigma\_y \sim \text{Exponential}(2.0)\$

* Fixed effects (standardized covariates):

  * \$\mu\_0 \sim \mathcal{N}(\bar{t}, 5.0)\$, where \$\bar{t}\$ is the empirical mean RT in the training data
  * \$\boldsymbol{\theta} \sim \mathcal{N}(0, 1.0)\$
  * \$\gamma \sim \mathcal{N}(0, 1.0)\$

## 2. Peak Assignment Model

The assignment model consumes the RT predictions from the hierarchical model and produces calibrated, interpretable probabilities for peak-to-compound matches using a 9-feature logistic regression.

### 2.1 Feature Construction

For each peak–compound pair \$(p, c)\$, nine features are computed in three groups:

#### Core (Primary Signals)

1. **Mass error (ppm)**: \$\Delta m\_{pc} = \dfrac{m\_p - m\_c}{m\_c} \times 10^6\$
2. **RT \$z\$-score**: \$z\_{pc} = \dfrac{t\_p - \hat{t}\_c}{\hat{\sigma}\_c}\$
3. **Log intensity**: \$\ell\_p = \log(1 + I\_p)\$

#### Confidence Scores (Soft penalties)

4. **Mass confidence**: \$\psi\_m = \exp!\big(-|\Delta m\_{pc}| / 10\big)\$
5. **RT confidence**: \$\psi\_t = \exp!\big(-|z\_{pc}| / 3\big)\$
6. **Combined confidence**: \$\psi\_{mt} = \psi\_m \times \psi\_t\$

#### Context (Priors / Uncertainty)

7. **Log compound mass**: \$\kappa\_c = \log(m\_c)\$
8. **Log RT uncertainty**: \$\upsilon\_c = \log(\hat{\sigma}\_c + 10^{-6})\$
9. **Log relative intensity**: \$\rho\_p = \log!\big(1 + I\_p / \tilde{I}\_{s(p)}\big)\$

where:

* \$m\_p\$ and \$t\_p\$ are the observed mass and RT of peak \$p\$,
* \$m\_c\$ is the theoretical mass for compound \$c\$,
* \$(\hat{t}\_c, \hat{\sigma}\_c)\$ are the predictive mean and **predictive** standard deviation for compound \$c\$ (see §3.2),
* \$\tilde{I}\_{s(p)}\$ is the median peak intensity within the same species as peak \$p\$.

### 2.2 Calibrated Logistic Regression

Let \$\mathbf{f}*{pc} = \[\Delta m*{pc}, z\_{pc}, \ell\_p, \psi\_m, \psi\_t, \psi\_{mt}, \kappa\_c, \upsilon\_c, \rho\_p]^\top\$ denote the 9-dimensional feature vector **after standardization**. The assignment probability is

```math
\text{logit}\,P(y_{pc}=1 \mid \mathbf{f}_{pc}) = \phi_0 + \sum_{k=1}^{9} \phi_k f_k
```

Probabilities are calibrated post-hoc. By default, **temperature scaling** is applied to the logits \$\eta\$:

```math
\tilde{\eta} = \eta / T, \qquad p_{\mathrm{cal}} = \sigma(\tilde{\eta})
```

An **isotonic regression** option is also available.

### 2.3 Standardization, Class Balancing, Calibration

1. **Standardization**: All 9 features are standardized to zero mean and unit variance. The scaler is fit on the training subset and applied to all rows.
2. **Class Balancing**: During likelihood construction, the minority class is over-sampled to balance positives and negatives, preventing a bias toward the majority class.
3. **Calibration**: Temperature \$T\$ is fitted by minimizing negative log-likelihood on a held-out calibration split when provided (otherwise on the available data). Isotonic calibration can be selected as an alternative.

### 2.4 Priors for Logistic Weights

With standardized features, we use tight normal priors:

```math
\phi_0 \sim \mathcal{N}(0, 1.0)
```

```math
\phi_k \sim \mathcal{N}(0, 1.0), \quad k \in \{1, \ldots, 9\}
```

## 3. Inference and Implementation

### 3.1 Posterior Sampling

Both the hierarchical RT model and the logistic regression are fit with the No-U-Turn Sampler (NUTS). The RT model uses non-centered parameterizations and sum-to-zero constraints for stable, efficient sampling; the logistic model is trained on standardized, class-balanced data and then calibrated.

### 3.2 Uncertainty Propagation

Uncertainty from the RT model is propagated into the assignment model. For species \$s\$ and compound \$c\$, let \$t\_{sc}^{(r)}\$ denote the draw-wise linear predictor of RT (including hierarchical effects and standardized covariates) and \$\sigma\_y^{(r)}\$ the observation noise on draw \$r\$. We use:

```math
\hat{t}_c = \mathbb{E}[t_{sc}] \quad\text{and}\quad \hat{\sigma}_c^2 = \mathrm{Var}[t_{sc}] + \mathbb{E}[\sigma_y^2]
```

This yields **predictive** uncertainty, not merely posterior uncertainty of parameters.

### 3.3 Assignment Strategy

After computing calibrated probabilities for all candidate pairs:

1. **Mass filter**: Keep candidates with \$|\Delta m\_{pc}| < \tau\_m\$ (default \$\tau\_m = 0.005\$ Da).
2. **RT window**: Keep candidates with \$|z\_{pc}| < k\$ (default \$k = 1.5\$).
3. **Probability threshold**: Accept if \$P(y\_{pc}=1) \ge \tau\_p\$ with a **meaningful default** \$\tau\_p = 0.5\$ (due to probability calibration).

A greedy matching ensures each peak is assigned to at most one compound, choosing the highest probability above threshold.

## 4. Softmax Assignment Model with Null Class

The softmax variant enforces exclusivity constraints directly and includes explicit null assignments for unmatched peaks.

### 4.1 Model Specification

For each peak $i$ with candidate set $C_i$ (compounds passing mass filter), we model assignment as a categorical distribution over $K_i + 1$ classes (candidates plus null):

```math
p(y_i = k | \mathbf{X}_i, \boldsymbol{\theta}) = \frac{\exp(\eta_{ik} / T)}{\sum_{j=0}^{K_i} \exp(\eta_{ij} / T)}
```

where the logits are:

```math
\eta_{i0} = \theta_{\text{null}} + \log \pi_{\text{null}}
```

```math
\eta_{ik} = \theta_0 + \boldsymbol{\theta}^\top \mathbf{x}_{ik} + \log \pi_{s(i),c_k} \quad \text{for } k > 0
```

Here:
- $k = 0$ represents the null class (no compound assigned)
- $\mathbf{x}_{ik}$ are the 9 features for peak $i$ and candidate $c_k$
- $\pi_{sc}$ is the prior probability that compound $c$ occurs in species $s$
- $T$ is a temperature parameter for calibration

### 4.2 Compound Presence Priors

We maintain Beta-Bernoulli priors for compound presence:

```math
\pi_{sc} \sim \text{Beta}(\alpha_{sc}, \beta_{sc})
```

Initially $\alpha_{sc} = \beta_{sc} = 1$ (uniform). These are updated online:
- Positive annotation: $\alpha_{sc} \leftarrow \alpha_{sc} + 1$
- Negative annotation: $\beta_{sc} \leftarrow \beta_{sc} + 1$

### 4.3 Key Advantages

1. **Exclusivity**: Softmax ensures exactly one assignment per peak
2. **Null handling**: Explicit modeling of unmatched peaks
3. **Online learning**: Presence priors update with human feedback
4. **Calibration**: Temperature scaling provides in-model calibration

### 4.4 Implementation Details

- **Ragged to padded**: Variable candidate counts handled via masking
- **Numerical stability**: Invalid slots masked with $\eta = -10^9$
- **Temperature prior**: $\log T \sim \mathcal{N}(0, 0.5)$ ensures $T > 0$

## 5. Active Learning Integration

The softmax model supports precision-focused active learning via expected false positive reduction:

```math
A_{\text{FP}}(i) = \mathbb{I}[q_i^{\max} \geq \tau] \cdot (1 - q_i^{\max})
```

where $q_i^{\max} = \max_k p(y_i = k)$ is the top probability for peak $i$.
