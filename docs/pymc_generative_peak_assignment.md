# Toward a Fully Bayesian Peak Assignment in PyMC

This document explains:

- What the current softmax model (`src/compassign/peak_assignment_softmax.py`) is doing and why it runs fast.
- Why that design under‑utilizes PyMC’s strengths despite being “Bayesian enough” for small discriminative models.
- A detailed proposal for a proper generative, hierarchical PyMC model that better captures uncertainty, structure, and active‑learning signal—along with trade‑offs and a phased rollout plan.

The intended audience is reviewers and collaborators evaluating whether and how to evolve the current pipeline toward a more expressive Bayesian model.

---

## 1) Executive Summary

- The existing softmax model is a compact Bayesian discriminative classifier over per‑candidate features. It samples a handful of global parameters (feature weights, biases, temperature) and plugs in presence priors and RT predictions. It’s vectorized and trains only on labeled rows; this is why it finishes in seconds.
- While effective and fast for multi‑candidate assignment, it does not fully leverage PyMC’s strengths: uncertainty propagation from measurement models (mass/RT), hierarchical pooling across species/compounds, and generative modeling of the null/background.
- We propose migrating toward a generative, hierarchical mixture in which each peak is explained by one of K candidates (including null). The model marginalizes assignment indicators (no discrete sampling), uses a logistic‑normal presence prior with partial pooling, and models mass/RT/intensity/shape via robust likelihoods. The result is a principled, end‑to‑end uncertainty model that should improve calibration and active‑learning utility at the cost of more runtime and engineering effort.

---

## 2) Current Model (Softmax with Null)

File: `src/compassign/peak_assignment_softmax.py`

### 2.1 What it does

- Candidate gating: For each peak i and species s, filter candidates by mass tolerance (|m_i − M_c| ≤ δ_m) and RT z window (|z_i| ≤ k). This yields variable numbers of valid candidates (5–8 on MEDIUM difficulty), plus a null option.
- Features: For each valid (peak, candidate) slot k, build a short feature vector X[i,k] (mass_err_ppm, rt_z, log_intensity, confidence transforms, plus allowed shape cues like peak width/asymmetry).
- Presence priors: A Beta–Bernoulli presence prior is maintained online (empirical‑Bayes). Its log means are added to the logits as fixed offsets per species–compound.
- Discriminative softmax: A single set of coefficients θ maps features to logits per slot; a separate bias handles the null. A temperature parameter T calibrates the softmax.
- Likelihood: `pm.Categorical` for labeled rows only; unlabeled rows are skipped (label = −1). All computations are batched across a padded (N, K_max, F) tensor with a mask.
- Prediction: The model returns posterior mean probabilities for each peak’s [null, candidates…] via the deterministic node `p`. Assignments are argmax above a threshold.

### 2.2 Why it’s fast

- Tiny parameter space: ~10–15 parameters total (θ, θ0, θ_null, log_T). No per‑peak or per‑compound random effects inside the PyMC model.
- Vectorized likelihood: All peaks/candidates evaluated in one tensor; no discrete sampling.
- Train on labeled rows only: The Categorical likelihood ignores unlabeled peaks, so the effective data size is small.

### 2.3 Strengths

- Enforces multiclass exclusivity with a proper null.
- Calibrated probabilities (temperature) and good speed for iterative AL.
- Clean integration of online presence priors (as fixed logits) and candidate gating.

### 2.4 Limitations

- Primarily discriminative: Relies on engineered features; does not model measurement processes (mass, RT, intensity) generatively.
- Empirical‑Bayes presence: Uses posterior means as fixed offsets; uncertainty in presence is not propagated through `p`.
- Limited pooling: No random effects to share signal across species and compounds.
- Null/background unmodeled: Null probabilities are learned discriminatively rather than via a background data model.

---

## 3) A Proper PyMC Model: Generative Hierarchical Mixture

We propose a fully continuous, marginalized assignment model. It models peak observables directly given candidate identity, integrates hierarchical presence priors, and marginalizes the (otherwise discrete) assignment variable using `logsumexp` in a `pm.Potential`.

### 3.1 Notation

- Peaks: i = 1..N, with species s[i].
- Candidate library: c = 1..C with theoretical mass M_c and (optionally) descriptors d_c for RT.
- Candidate set for peak i: K_i (indices into {1..C}), plus null at k = 0.
- Observables per peak i: x_i = {m_i, rt_i, I_i, shape_i}.

### 3.2 Presence prior with partial pooling (logistic–normal)

We model the prior mixture weights for each peak’s candidate set via species and compound effects:

- Base logits (before normalization):
  
  l_{i,k} = α + a_{s[i]} + b_{c[i,k]}         for candidate k ≥ 1
  
  l_{i,0} = α_null                            for the null component

- Priors:
  
  a_s ~ Normal(0, τ_s),     b_c ~ Normal(0, τ_c),
  
  α, α_null ~ Normal(0, 1), τ_s, τ_c ~ HalfNormal(1)

The per‑peak mixture weights are then π_{i,k} = softmax_k(l_{i,k}). This induces **partial pooling**: species and compounds share information without a full S×C parameter explosion.

### 3.3 Measurement models (robust, generative)

- Mass:
  
  m_i | z_i = c ~ StudentT(ν_m, M_c + Δ_adduct[k], σ_m)
  
  where Δ_adduct[k] is a known offset per adduct/isotope slot (optional). Use weakly informative priors on σ_m, ν_m.

- Retention time (two options):
  
  Option A (empirical μ):
  
  rt_i | z_i = c ~ StudentT(ν_rt, μ_c + δ_{s[i]}, σ_rt)
  
  where μ_c, σ_rt are set from a separate RT predictor (or lightly learned) and δ_s captures species drift.
  
  Option B (hierarchical RT regression):
  
  μ_c = β0 + β^T f(d_c) + u_c,  u_c ~ Normal(0, τ_μ),  δ_s ~ Normal(0, σ_δ)
  
  rt_i | z_i = c ~ StudentT(ν_rt, μ_c + δ_{s[i]}, σ_rt)

- Intensity/shape (optional layers):
  
  log I_i | z_i=c ~ Normal(μ_Ic, σ_I),  μ_Ic ~ Normal(μ_I0, τ_I)
  
  shape_i | z_i=c ~ appropriate light‑tailed likelihoods (e.g., Gamma/LogNormal for widths)

- Null/background component: define a broad likelihood for (m, rt, I, shape), e.g., mixtures or heavy‑tailed distributions that match background patterns.

All components are continuous; no discrete sampling of z_i.

### 3.4 Marginalized likelihood (no discrete z)

For each peak i, define the component log‑likelihoods (masked by valid candidates):

- loglik[i,k] = log p_mass(m_i | k) + log p_rt(rt_i | k) + [optional intensity/shape terms]

Combine with the (logistic–normal) prior logits:

- wlog[i,k] = l_{i,k} (from Section 3.2), with k=0 for null.

Total log‑likelihood over peaks:

- L = Σ_i logsumexp_k( wlog[i,k] + loglik[i,k] ), over valid k including null

In PyMC, implement `L` via a `pm.Potential(L)`. This is computationally efficient and avoids discrete latent variables.

### 3.5 Posterior responsibilities and assignments

At each draw, compute responsibilities:

- r[i,k] = softmax_k( wlog[i,k] + loglik[i,k] )

Expose r (and its posterior mean) as a `pm.Deterministic`. For decisions, pick argmax or apply a threshold on r[i,k] for k≥1 (non‑null) and use k=0 for null otherwise. These are principled posterior probabilities per candidate.

### 3.6 Why this better uses PyMC

- Hierarchical pooling (a_s, b_c) shares signal across species and compounds.
- End‑to‑end uncertainty: noise scales, drifts, RT means, presence weights are sampled; responsibilities reflect parameter uncertainty.
- Robustness: Student‑T likelihoods tolerate outliers; explicit null background avoids forcing spurious matches.
- Interpretability: The model speaks the language of the data (mass/RT/intensity/shape) rather than only feature engineering.

---

## 4) Computational Design

- Candidate gating stays: Maintain the current mass/RT pre‑filters to bound K_i and keep the mixture small per peak.
- Vectorization: Build `(N, K_max)` tensors for wlog and loglik with a boolean mask; rely on `pytensor` broadcasting and `pm.math.logsumexp`.
- Priors and identifiability:
  - Centering: use non‑centered parameterizations for hierarchical terms (a_s, b_c) if divergences occur.
  - Null dominance: consider regularizing α_null to prevent degenerate solutions.
  - Avoid label switching: candidates are externally identified (compound IDs), so mixture label switching is not an issue; if the null is internally a mixture, apply ordering or shrinkage constraints.
- Performance expectations: The model is larger than the softmax but still continuous; NUTS should handle it. Expect minutes (not seconds). Start with small `ν` fixed (e.g., ν_m=5, ν_rt=5) and broaden later if needed.

---

## 5) Active Learning and Uncertainty

- Uncertainty measures derive from posterior draws of r[i,k]: entropy, margin, variation ratio, and mutual information (MI) are straightforward.
- Compared to discriminative softmax, responsibilities here incorporate presence prior and measurement uncertainty coherently, which should improve the rank‑ordering of informative peaks.

---

## 6) Pros and Cons

### Advantages

- Principled uncertainty propagation from measurement models and presence priors.
- Hierarchical pooling → better data efficiency, less overfitting, stronger calibration.
- Explicit null/background modeling → fewer spurious matches; clearer abstention behavior.
- Naturally supports AL via posterior responsibilities and MI.

### Trade‑offs / Risks

- Runtime: slower than the compact softmax (minutes vs. seconds).
- Complexity: more parameters and moving parts; requires careful priors and diagnostics (divergences, R‑hat, ESS).
- Engineering: additional data preparation (adduct maps, null background spec), more involved code.

---

## 7) Phased Rollout Plan

To de‑risk, implement in phases and validate at each step.

- Phase 1 (minimal generative):
  - Keep current candidate gating.
  - Presence logits: α, a_s, b_c (logistic–normal, partial pooling).
  - Likelihood: mass + RT as Student‑T with fixed ν, learn σ_m, σ_rt, species drift δ_s.
  - RT means μ_c from existing predictor (empirical), or simple learned per‑compound effects with shrinkage.
  - Null: broad Student‑T × Student‑T baseline.
  - Outcome: responsibilities r[i,k], assignments, calibration & AL metrics. Compare to softmax on MEDIUM.

- Phase 2 (RT hierarchy):
  - Add hierarchical regression for μ_c from descriptors d_c with shrinkage; learn σ_δ.
  - Optionally consume posterior draws from an external RT model to propagate uncertainty directly.

- Phase 3 (intensity/shape & adducts):
  - Add log‑intensity and shape likelihoods with modest priors; refine null background.
  - Introduce adduct/isotope offsets Δ_adduct per slot (optional mixture inside candidate slots if needed).

- Phase 4 (refinement & efficiency):
  - Non‑centered parameterizations; weakly informative hyperpriors; tuning target_accept.
  - Profiling and batching if memory becomes tight.

### Acceptance criteria per phase

- Calibration: ECE/MCE comparable or better than softmax; responsibilities reflect uncertainty patterns.
- AL uplift: Consistent +10–15% improvement after 100–150 annotations on MEDIUM difficulty.
- Stability: No pathological divergences at recommended draws/chains; R‑hat < 1.01 typically.

---

## 8) Implementation Sketch (PyMC)

Pseudo‑PyMC structure (omitting boilerplate and data prep):

```python
with pm.Model() as model:
    # Presence logits (logistic–normal)
    alpha = pm.Normal('alpha', 0., 1.)
    alpha_null = pm.Normal('alpha_null', 0., 1.)
    tau_s = pm.HalfNormal('tau_s', 1.)
    tau_c = pm.HalfNormal('tau_c', 1.)
    a_s = pm.Normal('a_s', 0., tau_s, shape=n_species)
    b_c = pm.Normal('b_c', 0., tau_c, shape=n_compounds)

    # RT components (Phase 1: empirical μ_c, learn δ_s and σ_rt)
    delta_s = pm.Normal('delta_s', 0., 0.5, shape=n_species)
    sigma_rt = pm.HalfNormal('sigma_rt', 0.5)
    nu_rt = 5  # or pm.Exponential('nu_rt', 1/10) if inferring

    # Mass components
    sigma_m = pm.HalfNormal('sigma_m', 0.01)
    nu_m = 5

    # Build component log-likelihoods for each (i,k)
    # Shapes: wlog/loglik -> (N, K_max)
    wlog = pm.math.ones((N, K_max)) * (-1e9)
    # Null logits
    wlog = pm.math.set_subtensor(wlog[:, 0], alpha_null)
    # Candidate logits with pooling
    # idx_sc maps (i,k) -> (species_idx, compound_idx)
    wlog_candidates = alpha + a_s[idx_sc_species] + b_c[idx_sc_comp]
    wlog = pm.math.set_subtensor(wlog[mask_candidates], wlog_candidates)

    # Mass log-likelihood
    # mass_pred = M_c[compound_idx] + adduct_offset[k] (optional)
    mass_ll = pm.logp(pm.StudentT.dist(nu_m, mu=mass_pred, sigma=sigma_m), mass_obs)

    # RT log-likelihood
    rt_mu = rt_mu_c[compound_idx] + delta_s[species_idx]
    rt_ll = pm.logp(pm.StudentT.dist(nu_rt, mu=rt_mu, sigma=sigma_rt), rt_obs)

    # Combine (plus optional intensity/shape terms)
    loglik = mass_ll + rt_ll

    # Mask invalid slots, sum with wlog, and marginalize via logsumexp
    total = pm.math.logsumexp(wlog + loglik_masked, axis=1)
    pm.Potential('mixture_ll', pm.math.sum(total))

    # Deterministic responsibilities for downstream use
    responsibilities = pm.Deterministic('r', pm.math.softmax(wlog + loglik_masked, axis=1))
```

The central ideas: continuous hierarchy, vectorized logsumexp marginalization, and responsibilities as first‑class outputs.

---

## 9) Comparison Against Current Softmax

| Dimension            | Softmax (current)                              | Generative Mixture (proposed)                   |
|----------------------|-----------------------------------------------|-------------------------------------------------|
| Nature               | Discriminative over features                  | Generative over observables                     |
| Presence prior       | Empirical‑Bayes offsets                       | Logistic–normal with pooling (sampled)          |
| Null                 | Bias + features                               | Explicit background likelihood                  |
| RT/mass modeling     | Engineered features                           | Robust Student‑T per component                  |
| Uncertainty          | Parameter posterior on θ only                 | Propagated across presence & measurement noise  |
| AL signals           | Entropy/margin on softmax θ                   | Responsibilities + MI from generative model     |
| Speed                | Seconds                                       | Minutes (vectorized, no discrete z)             |

We expect better calibration and AL ranking from the generative approach, especially in ambiguous multi‑candidate regimes, at the cost of runtime and complexity.

---

## 10) Risks and Mitigations

- Divergences / identifiability: Use non‑centered parameterizations, HalfNormal priors, and moderate target_accept (0.9–0.95). Start simple (Phase 1) and layer complexity gradually.
- Background model misspecification: Begin with simple heavy‑tailed null; revise after inspecting residuals.
- Runtime: Keep K_max reasonable with candidate gating; start with 2 chains × 1000–1500 draws and scale.

---

## 11) Conclusion

The current softmax is a pragmatic, fast, and surprisingly effective multiclass baseline. However, if we want PyMC to materially improve decision quality—especially under ambiguity and for active learning—we should invest in a marginalized, hierarchical generative mixture with partial pooling. The proposed phased plan balances rigor with practicality, enabling incremental validation and clear acceptance criteria at each step.

---

## Appendix A: Variable Glossary

- i: peak index (1..N)
- k: candidate slot (0..K_i), with 0 = null
- c: compound ID (1..C)
- s: species ID (1..S)
- M_c: theoretical mass for compound c
- d_c: descriptors for compound c (for RT regression)
- x_i: observables for peak i (mass m_i, RT rt_i, intensity I_i, shape)
- a_s, b_c: species/compound random effects in presence logits
- α, α_null: global biases for candidate and null
- δ_s: species RT drift; σ_m, σ_rt: measurement scales; ν_m, ν_rt: degrees of freedom
- r[i,k]: responsibility (posterior P(z_i=k | data, params))

