# Bayesian Models: Mathematical Specifications

## Abstract

This report describes the two-stage Bayesian framework used in **CompAssign** for assigning LC–MS peaks to compounds. Stage 1 predicts retention time (RT) with a hierarchical model that respects the data’s nested structure and uses standardized covariates, non-centered parameterization, and sum-to-zero constraints. Critically, compound-level sharing is achieved via descriptor-informed effects from a fixed 20‑dimensional SMILES embedding rather than discrete “compound classes.” Stage 2 takes those RT predictions and assigns peaks to compounds with a hierarchical softmax. That model includes an explicit **null** option (“no suitable compound”) and **presence priors** that encode prior likelihoods of observing particular compounds in particular species. All equations below are formal specifications; the surrounding text clarifies notation and purpose.

## 1. Hierarchical Retention Time Model

This model predicts RTs while borrowing statistical strength across related groups. Species are organized into biological clusters, and compounds share via a descriptor-informed prior based on a 20‑dimensional SMILES embedding. Run-level covariates (e.g., internal-standard channels) are standardized before modeling so their coefficients live on a comparable scale.

### 1.1 Model specification

The observed RT $y_{ijr}$ (compound $j$ measured in species $i$ on run $r$) is modeled as a baseline $\mu_0$ plus additive adjustments: a species effect $\alpha_{s(i)}$, a compound effect $\beta_{c(j)}$ (informed by SMILES descriptors), and a contribution from the standardized run covariates $\mathbf{x}^\star_r$ with compound-specific weights $\mathbf{w}_{c(j)}$. The residual $\epsilon_{ijr}$ captures remaining noise with variance $\sigma_y^2$.

```math
y_{ijr}=\mu_0+\alpha_{s(i)}+\beta_{c(j)}+\mathbf{w}_{c(j)}^{\top}\mathbf{x}^{\star}_r+\epsilon_{ijr},
\qquad \epsilon_{ijr}\sim\mathcal{N}(0,\sigma_y^2).
```

#### 1.1a Notation and intuition (dot product term)

- $\mu_0$ (intercept): grand mean RT across all species, compounds, and runs (with centered effects).
- $\alpha_{s(i)}$ (species effect): adjustment for species $s(i)$ after clustering/shrinkage.
- $\beta_{c(j)}$ (compound effect): compound baseline RT; when descriptors are available it is informed by the compound’s descriptor vector $\mathbf{z}_{c(j)}$ via $\mathbf{z}_{c(j)}^{\top}\boldsymbol{\theta}_\beta$ (see Section 1.2).
- $\mathbf{x}^{\star}_r\in\mathbb{R}^P$ (standardized run covariates): per‑run features (e.g., internal‑standard channels) z‑scored over the training runs so each coordinate has mean 0 and variance 1.
- $\mathbf{w}_{c(j)}\in\mathbb{R}^P$ (compound‑specific covariate weights): how sensitive compound $c(j)$ is to each run covariate.

The product $\mathbf{w}_{c(j)}^{\top}\mathbf{x}^{\star}_r$ is a dot product (inner product). Written with summation it is

```math
\langle \mathbf{x}^{\star}_r,\, \mathbf{w}_{c(j)} \rangle\;=\;\sum_{p=1}^{P} x^{\star}_{r,p}\,w_{c(j),p}.
```

Intuition: $\mathbf{x}^{\star}_r$ captures run‑to‑run shifts, and $\mathbf{w}_{c(j)}$ tells how much compound $c(j)$’s RT moves with those shifts. Using standardized $\mathbf{x}^{\star}_r$ keeps the scale of $\mathbf{w}_{c(j)}$ interpretable and improves numerical stability.

### 1.2 Hierarchical structure

Species effects vary around their cluster means, and compound effects are tied to molecular descriptors rather than to discrete classes. Sum-to-zero constraints (stated after the equations) re-center species effects so they don’t absorb the global intercept.

```math
\alpha_{s(i)}\sim\mathcal{N}(\mu_{k(s)},\sigma^2_{\mathrm{species}}),\qquad \mu_k\sim\mathcal{N}(0,\sigma^2_{\mathrm{cluster}}),
```

Descriptor-informed compound effect (no class level):

```math
\beta_{c}\mid \boldsymbol{\theta}_\beta,\,\sigma_{\mathrm{compound}},\,\mathbf{z}_c\ \sim\ \mathcal{N}\bigl(\mathbf{z}_c^{\top}\boldsymbol{\theta}_\beta,\ \sigma^2_{\mathrm{compound}}\bigr),
```

where $\mathbf{z}_c\in\mathbb{R}^{d}$ is the standardized 20‑dimensional SMILES embedding for compound $c$, and $\boldsymbol{\theta}_\beta\in\mathbb{R}^{d}$ maps descriptors to baseline RT.

### 1.3 Non-centered parameterization

These expressions say “draw standardized latent variables $\tilde{\cdot}\sim\mathcal{N}(0,1)$, then scale them by the relevant variance components.” This reparameterization usually samples better in hierarchical models. The realized species effects are centered to sum to zero across species. For compounds, we use a descriptor-informed mean plus a non-centered residual.

```math
\mu_k=\tilde{\mu}_k\,\sigma_{\mathrm{cluster}},\ \ \tilde{\mu}_k\sim\mathcal{N}(0,1), \qquad
\alpha_s=\mu_{k(s)}+\tilde{\alpha}_s\,\sigma_{\mathrm{species}},\ \ \tilde{\alpha}_s\sim\mathcal{N}(0,1),
```

Descriptor-informed compound effect (non-centered):

```math
\tilde{\beta}_c\sim\mathcal{N}(0,1),\qquad \beta_c=\mathbf{z}_c^{\top}\boldsymbol{\theta}_\beta+\tilde{\beta}_c\,\sigma_{\mathrm{compound}}.
```

### 1.4 Prior specifications

The prior choices reflect weak information on a standardized scale. Exponential or Half‑Student‑t priors on standard deviations favor smaller between‑group variation but allow larger values if supported by data; Normal priors on fixed effects keep coefficients near zero unless the data push them away.

Variance components (example choices):

```math
\sigma_{\mathrm{cluster}}\sim\mathrm{Exponential}(1.0),\qquad
\sigma_{\mathrm{species}}\sim\mathrm{Exponential}(2.0),\qquad
\sigma_{\mathrm{compound}}\sim\mathrm{Half\text{-}Student\,t}(\nu{=}3,\,\text{scale}{=}0.5),\qquad
\sigma_y\sim\mathrm{Exponential}(2.0).
```

Fixed effects and descriptor weights:

```math
\mu_0\sim\mathcal{N}(\bar{t},5.0),\qquad \boldsymbol{\theta}_\beta\sim\mathcal{N}(\mathbf{0},\tau_\beta^2\,\mathbf{I}_d),\ \ \tau_\beta\in[0.3,0.7].
```

Run‑covariate weights per compound $\mathbf{w}_c$ can either be kept as independent, weakly regularized parameters,

```math
\mathbf{w}_c\sim\mathcal{N}(\mathbf{0},\sigma_\gamma^2\,\mathbf{I}_P),\qquad \sigma_\gamma\sim\mathrm{Half\,Normal}(0.5),
```

or (optionally) given a descriptor‑informed prior $\mathbf{w}_c\mid \Theta_\gamma,\sigma_\gamma\sim\mathcal{N}(\Theta_\gamma^{\top}\mathbf{z}_c,\sigma_\gamma^2\,\mathbf{I}_P)$ with $\Theta_\gamma\sim\mathcal{N}(\mathbf{0},\tau_\gamma^2\,\mathbf{I})$.

<!-- Run‑level random effects omitted intentionally in the current model to favor
generalisation to unseen runs. If future data demand it, prefer modelling effects
at the batch/worksheet level rather than per‑run idiosyncrasies. -->

### 1.5 Molecular descriptors (20‑dim SMILES embedding)

We use per‑compound molecular descriptors $\mathbf{z}_c\in\mathbb{R}^{d}$ with a fixed dimensionality $d=20$ to enable sharing across chemically similar compounds and to provide calibrated priors for rarely observed or unseen compounds.

Sources and construction (computed offline; no runtime dependency):

- Start from a pretrained SMILES encoder (e.g., ChemBERTa or a comparable transformer) to obtain a high‑dimensional embedding (typically 256–768).
- Fit PCA on the compound library embeddings and reduce to $d=20$ dimensions, followed by whitening (unit variance per component). Persist the PCA mean and components to apply consistently to future compounds.
- As a lightweight alternative, generate ECFP6 (radius 3) 1024‑bit fingerprints and apply TruncatedSVD to 20 dimensions; results are similar for small/medium libraries.

Standardization and identifiability:

- After reduction, z‑score each component across compounds to mean 0 and variance 1, and do not include an intercept term in $\mathbf{z}_c$. With centered descriptors, $\mu_0$ remains interpretable as the grand mean RT.
- Missing descriptors can be imputed to the zero vector in standardized space; uncertainty then relies more on $\sigma_{\mathrm{compound}}$.

Dimensionality rationale:

- For PyMC/NUTS and typical library sizes, $d=20$ balances expressiveness and sampling stability. Larger $d$ increase cost and may require stronger regularization without improving generalization.
- Choose $d$ empirically if needed (e.g., 12, 16, 20, 32) and prefer the smallest $d$ that matches held‑out performance and maintains good sampler diagnostics.

Why this helps rare compounds:

- For a rarely observed compound $c$, the posterior for $\beta_c$ shrinks toward the descriptor‑predicted mean $\mathbf{z}_c^{\top}\boldsymbol{\theta}_\beta$ rather than a broad class average. Uncertainty propagates both the mapping uncertainty in $\boldsymbol{\theta}_\beta$ and the residual scale $\sigma_{\mathrm{compound}}$, yielding calibrated predictions.

Indexing recap (symbols used above):

- $i$ (observation row), $s(i)$ (species of row $i$), $c(i)$ (compound of row $i$), $r(i)$ (run of row $i$)
- $y_i$ (observed RT), $\mathbf{x}_{r}$ (standardized run covariates; length $P$), $\mathbf{z}_c$ (SMILES embedding; length $d=20$)
- $\mu_0$ (intercept), $\alpha_s$ (species effect), $\beta_c$ (compound effect), $\mathbf{w}_c$ (compound‑specific covariate weights)
- $\boldsymbol{\theta}_\beta$ (descriptor→RT weights), $\Theta_\gamma$ (descriptor→covariate‑weight mapping; optional)
- $\sigma_{\mathrm{cluster}},\sigma_{\mathrm{species}},\sigma_{\mathrm{compound}},\sigma_\gamma,\sigma_y$ (scales)

## 2. Peak Assignment Model

This stage produces a probability for each feasible peak–compound match. It brings together simple, standardized features, prior knowledge about which compounds tend to be present, and an explicit **null** option for “no match”.

### 2.1 Candidate filtering and features

The first line defines the candidate set: a mass-window filter (tolerance $\tau_m$) and an RT-window filter using the RT model’s prediction $\hat{t}_{s(i)c}$ and its uncertainty $\hat{\sigma}_{s(i)c}$ expressed as a $z$-score cut $k$.

```math
\lvert m_i-m_c\rvert\le \tau_m,\qquad
\left\lvert\frac{t_i-\hat{t}_{s(i)c}}{\hat{\sigma}_{s(i)c}}\right\rvert\le k.
```

The second line builds a four-component feature vector for each candidate: mass error in ppm, RT $z$-score, log-intensity, and log-uncertainty from the RT model (the $10^{-6}$ avoids $\log 0$). These features are standardized with training statistics before modeling.

```math
\mathbf{x}_{ik}=\bigl[\ \Delta m^{\mathrm{ppm}}_{ik},\ z^{\mathrm{RT}}_{ik},\ \log(1+I_i),\ \log(\hat{\sigma}_{s(i)c_k}+10^{-6})\ \bigr]^\top,
```

### 2.2 Likelihood with hierarchical uncertainty

Here $\eta_{ik}$ are the logits (pre-softmax scores). For the null class ($k=0$), the score combines a bias term and the prior odds of predicting null; for a candidate compound, it adds a linear function of features and a presence prior specific to species $s(i)$ and compound $c_k$.

```math
\eta_{i0}=\theta_{\mathrm{null}}+\log \pi_{\mathrm{null}},\qquad
\eta_{ik}=\theta_0+\boldsymbol{\theta}^\top\mathbf{x}_{ik}+\log \pi_{s(i),c_k}\ \ (k=1,\ldots,K_i).
```

To reflect residual uncertainty at the logit level, we jitter each $\eta_{ik}$ by Gaussian noise with shared scale $\sigma_{\mathrm{logit}}$.

```math
\tilde{\eta}_{ik}\mid \eta_{ik},\sigma_{\mathrm{logit}}\sim\mathcal{N}(\eta_{ik},\sigma_{\mathrm{logit}}^2),
```

Softmax then converts noisy logits into class probabilities $\mathbf{p}_i$, and $y_i$ denotes the sampled class for peak $i$. Padding/masking is used so peaks with different numbers of candidates can be handled uniformly.

```math
\mathbf{p}_i=\mathrm{softmax}\!\left(\,[\tilde{\eta}_{i0},\tilde{\eta}_{i1},\ldots,\tilde{\eta}_{iK_i}]\,\right),\qquad
y_i\sim\mathrm{Categorical}(\mathbf{p}_i).
```

### 2.3 Priors and presence priors

The first line gives tight Normal priors for the softmax intercept and weights (appropriate because features are standardized); the second line sets a Half-Normal prior on the logit noise scale.

```math
\theta_0\sim\mathcal{N}(0,1),\qquad \boldsymbol{\theta}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),\qquad \theta_{\mathrm{null}}\sim\mathcal{N}(-1,1),
```

```math
\sigma_{\mathrm{logit}}\sim\mathrm{HalfNormal}(0.5).
```

Presence priors $\pi_{s,c}$ encode how likely compound $c$ is to appear in species $s$; their logs act as offsets to the logits. The null class has its own global prior. The final sentences explain how these Beta priors are updated from labels in batch training and active learning settings.

```math
\pi_{s,c}\sim\mathrm{Beta}(\alpha_{s,c},\beta_{s,c}),\qquad
\eta_{ik}\leftarrow \eta_{ik}+\log \pi_{s(i),c_k}.
```

The null class has a separate global prior
$\pi_{\mathrm{null}}\sim\mathrm{Beta}(\alpha_{\mathrm{null}},\beta_{\mathrm{null}})$.
During batch supervised training, each observed positive pair $(s,c)$ increments $\alpha_{s,c}$ and each labeled null increments $\alpha_{\mathrm{null}}$; batch negatives are not added to avoid bias from incomplete candidate sets. Under active learning, positives increment $\alpha_{s,c}$ and null labels increment $\alpha_{\mathrm{null}}$, and we apply a light negative signal to other candidates for the annotated peak by adding a small increment (default weight $0.25$) to $\beta_{s,c'}$. This policy is configurable.

### 2.4 Exchangeability and decision rule

All non-null candidates share the same parameterization—their ordering is irrelevant—while the null class has its own bias. At prediction time, we compute $\mathbf{p}_i$ and accept the top compound if its probability exceeds $\tau_p$; otherwise we return null. Many-to-one assignments are allowed: multiple peaks may map to the same compound.

## 3. Inference and Implementation

### 3.1 Posterior sampling

Both stages are fitted with NUTS (No-U-Turn Sampler). The RT model uses non-centered parameterizations and sum-to-zero constraints to improve mixing. The assignment model uses standardized features and a hierarchical logit noise scale $\sigma_{\mathrm{logit}}$.

### 3.2 Uncertainty propagation

The next two quantities are the predictive mean and variance used to score RT consistency in Stage 2. $\hat{t}_{sc}$ averages the draw-wise linear predictors for species $s$, compound $c$. $\hat{\sigma}^2_{sc}$ adds the across-draw variability of the predictor to the expected observation noise, yielding a full predictive variance (not just parameter uncertainty).

```math
\hat{t}_{sc}=\mathbb{E}[t_{sc}], \qquad
\hat{\sigma}^2_{sc}=\mathrm{Var}[t_{sc}]+\mathbb{E}[\sigma_y^2],
```

### 3.3 Assignment strategy

After the mass/RT filters, the softmax model computes probabilities using features and presence priors. The rule is simple: choose the highest-probability compound if $\max_k p(y_i=k)\ge \tau_p$; otherwise return null. There is no one-to-one matching constraint across peaks.

### 3.4 Diagnostics and checks

We run posterior predictive checks (PPC) to summarize accuracy (RMSE, MAE) and coverage (95%) by species and overall. Standard MCMC diagnostics are also enforced: no Hamiltonian divergences, $\hat{R}\le 1.01$, and $\mathrm{ESS}_{\mathrm{bulk}}\ge 200$. Any failures are logged and written to JSON artifacts.

## 4. Calibration Metrics

Accuracy alone is not enough; probability estimates should be **honest**. We therefore report a top-1 expected calibration error (ECE), which compares the model’s maximum predicted probability with the empirical frequency of being correct, a one-vs-rest ECE averaged across classes (including the null), and the one-vs-rest Brier score (mean squared error of predicted probabilities against binary outcomes). Lower is better for all three.

## 4. Implementation Notes

Variable candidate counts are handled by masking and padding; masked slots receive a large negative logit (e.g., $-10^9$) for stability. Exchangeability is enforced by treating all non-null candidates identically, so the order of candidates does not matter. To avoid leakage, decoy compounds can be excluded during training and reintroduced at test time with presence priors applied.

## 5. Active Learning Integration

For targeted curation, we prioritize examples where confirming the label is most likely to reduce false positives. The next expression scores each peak $i$: it activates only when the top probability exceeds a screening threshold $\tau$, and then prefers uncertain high-probability cases $(1-q_i^{\max})$.

```math
A_{\mathrm{FP}}(i)=\mathbb{I}\!\left[q_i^{\max}\ge \tau\right]\cdot\left(1-q_i^{\max}\right),
```

where $q_i^{\max}=\max_k p(y_i=k)$.

## 6. Evaluation Metrics (Many-to-Many)

Each peak can be assigned zero, one, or multiple compounds, and the **null** outcome remains a legitimate choice (“no suitable compound”). For peak $i$, $T_i$ is the true set and $P_i(\tau,k)$ the predicted set formed by keeping candidates with probability at least $\tau$ and optionally truncating to the top $k$. The pair $(\tau,k)$ is the **operating point**, i.e., the exact thresholding and capping used to report results.

To measure identification quality across the dataset, we count true positives (TP), false positives (FP), and false negatives (FN) over all peak–compound pairs and then compute precision, recall, and F1:

```math
\mathrm{Precision}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\qquad
\mathrm{Recall}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\qquad
\mathrm{F1}=\frac{2\,\mathrm{Precision}\,\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}.
```

For a per-peak perspective (each peak contributes equally), we compute $\mathrm{Precision}_i$, $\mathrm{Recall}_i$, and $\mathrm{F1}_i$ from $P_i$ versus $T_i$ for each peak and then average them. We also report the mean Jaccard index $\lvert P_i\cap T_i\rvert/\lvert P_i\cup T_i\rvert$ to summarize set overlap.

Because returning **null** is a deliberate decision, we report the false-assignment rate among true-null peaks, $\mathrm{FAR}_{\mathrm{null}}$, and its complement $1-\mathrm{FAR}_{\mathrm{null}}$. We also report the overall assignment rate (AR): the fraction of peaks that receive any non-null assignment at the chosen operating point.

Finally, we assess probability calibration. Among predictions made with probability $0.8$, roughly $80\%$ should be correct if the model is well calibrated. We summarize this with an expected calibration error and a Brier score (mean squared error of probabilities versus binary outcomes). We also track absolute cardinality error, $\bigl|\lvert P_i\rvert-\lvert T_i\rvert\bigr|$, averaged across peaks to detect whether the method systematically over- or under-predicts how many compounds a peak should have. Unless stated otherwise, $\tau=0.7$ and $k=2$ are used as the default operating point.
