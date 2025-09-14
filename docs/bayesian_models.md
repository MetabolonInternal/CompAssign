# Bayesian Models: Mathematical Specifications

## Abstract

This document provides the formal mathematical specifications for the two-stage Bayesian framework implemented in CompAssign for metabolomics compound assignment. The framework consists of a hierarchical Bayesian model for retention time (RT) prediction (with standardized covariates, non-centered sampling, and sum-to-zero constraints) followed by a hierarchical softmax peak assignment model with presence priors and an explicit null class.

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

The assignment model consumes the RT predictions from the hierarchical model and produces calibrated, interpretable probabilities for peak-to-compound matches using a hierarchical softmax with an explicit null class and compound presence priors. The model is exchangeable with respect to the ordering of non-null candidates.

### 2.1 Candidate Filtering and Feature Construction

For peak \(i\) and compound \(c\) in species \(s(i)\), we first define the candidate set \(C_i\) by mass and RT filters:

```math
|m_i - m_c| \leq \tau_m, \qquad \left| \frac{t_i - \hat{t}_{s(i)c}}{\hat{\sigma}_{s(i)c}} \right| \leq k
```

For each candidate \(c_k \in C_i\), we compute a minimal 4-feature vector (then standardize feature-wise using training statistics):

```math
\mathbf{x}_{ik} = \left[\, \Delta m^{\mathrm{ppm}}_{ik},\ z^{\mathrm{RT}}_{ik},\ \log(1 + I_i),\ \log(\hat{\sigma}_{s(i)c_k} + 10^{-6}) \,\right]^\top
```

where \(\Delta m^{\mathrm{ppm}}\) is the mass error in ppm, \(z^{\mathrm{RT}}\) is the RT z-score using the RT model’s predictive mean and variance, and \(I_i\) is the peak intensity. An extended 9-feature variant is available for ablations, but the default model uses the 4-feature set above.

### 2.2 Likelihood with Hierarchical Uncertainty

We model assignment for peak \(i\) over \(K_i + 1\) classes (\(K_i\) candidates plus an explicit null class \(k=0\)):

```math
\eta_{i0} = \theta_{\text{null}} + \log \pi_{\text{null}}
```

```math
\eta_{ik} = \theta_0 + \boldsymbol{\theta}^\top \mathbf{x}_{ik} + \log \pi_{s(i), c_k} \quad (k=1,\ldots,K_i)
```

To capture uncertainty at the logit level, we introduce a shared noise scale \(\sigma_{\text{logit}}\):

```math
\tilde{\eta}_{ik} \mid \eta_{ik}, \sigma_{\text{logit}} \sim \mathcal{N}(\eta_{ik},\ \sigma_{\text{logit}}^2)
```

Class probabilities are the softmax of the noisy logits:

```math
\mathbf{p}_i = \operatorname{softmax}\big(\, [\tilde{\eta}_{i0},\tilde{\eta}_{i1},\ldots,\tilde{\eta}_{iK_i}] \,\big), \qquad y_i \sim \operatorname{Categorical}(\mathbf{p}_i)
```

Masking handles ragged candidate sets; invalid slots are assigned a large negative logit before softmax for numerical stability.

### 2.3 Priors and Presence Priors

With standardized features, we place tight Normal priors:

```math
\theta_0 \sim \mathcal{N}(0,1), \qquad \boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \qquad \theta_{\text{null}} \sim \mathcal{N}(-1,1)
```

```math
\sigma_{\text{logit}} \sim \operatorname{HalfNormal}(0.5)
```

Compound presence priors enter as log-offsets:

```math
\pi_{sc} \sim \operatorname{Beta}(\alpha_{sc}, \beta_{sc}), \quad \log \pi_{s(i), c_k} \text{ added to } \eta_{ik}
```

The null class has a separate global prior \(\pi_{\text{null}} \sim \operatorname{Beta}(\alpha_{\text{null}}, \beta_{\text{null}})\). Presence priors are updated online from annotations: positive labels increment \(\alpha_{sc}\), negative labels increment \(\beta_{sc}\).

### 2.4 Exchangeability and Decision Rule

All non-null candidate slots share the same parameterization; only the null slot is distinguished. Therefore, the model is exchangeable with respect to the ordering of candidates \(k>0\). At prediction time, we compute \(\mathbf{p}_i\) over the test candidate set and apply a probability threshold \(\tau_p\): assign the argmax compound if its probability \(\geq \tau_p\) and \(k>0\); otherwise assign null. Many-to-one assignments (multiple peaks per compound) are allowed by design; there is no global one-to-one matching across peaks.

## 3. Inference and Implementation

### 3.1 Posterior Sampling

Both the hierarchical RT model and the softmax assignment model are fit with the No-U-Turn Sampler (NUTS). The RT model uses non-centered parameterizations and sum-to-zero constraints for stable, efficient sampling. The assignment model uses standardized features and hierarchical logit noise (\(\sigma_{\text{logit}}\)).

### 3.2 Uncertainty Propagation

Uncertainty from the RT model is propagated into the assignment model. For species \$s\$ and compound \$c\$, let \$t\_{sc}^{(r)}\$ denote the draw-wise linear predictor of RT (including hierarchical effects and standardized covariates) and \$\sigma\_y^{(r)}\$ the observation noise on draw \$r\$. We use:

```math
\hat{t}_c = \mathbb{E}[t_{sc}] \quad\text{and}\quad \hat{\sigma}_c^2 = \mathrm{Var}[t_{sc}] + \mathbb{E}[\sigma_y^2]
```

This yields **predictive** uncertainty, not merely posterior uncertainty of parameters.

### 3.3 Assignment Strategy

After filtering candidates by mass and RT window, we compute softmax probabilities with presence priors and hierarchical logit noise. The decision rule is:

1. **Probability threshold**: Accept the top compound if \(\max_k p(y_i=k) \ge \tau_p\) and \(k>0\); otherwise assign null.
2. **Many-to-one allowed**: No global matching is enforced across peaks; multiple peaks may be assigned to the same compound.
## 4. Implementation Notes

- **Ragged to padded**: Variable candidate counts handled via masking.
- **Numerical stability**: Invalid slots masked with a large negative logit (e.g., \(-10^9\)).
- **Exchangeability**: Non-null candidate slots are treated identically; ordering does not affect probabilities.
- **Training vs. testing candidates**: To avoid leakage, decoy compounds can be excluded from training candidate sets and included at test time with presence priors.

## 5. Active Learning Integration

The softmax model supports precision-focused active learning via expected false positive reduction:

```math
A_{\text{FP}}(i) = \mathbb{I}[q_i^{\max} \geq \tau] \cdot (1 - q_i^{\max})
```

where $q_i^{\max} = \max_k p(y_i = k)$ is the top probability for peak $i$.
