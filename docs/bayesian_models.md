# CompAssign Bayesian Model Specifications

## 1. Hierarchical RT Regression Model

### Full Bayesian Specification

The complete generative model for retention times:

#### Hyperpriors (Variance Components)
```
σ_cluster ~ Exponential(1.0)
σ_species ~ Exponential(2.0)
σ_class ~ Exponential(1.0)
σ_compound ~ Exponential(3.0)
σ_y ~ Exponential(2.0)
```

#### Random Effects (Non-centered)
```
# Cluster level
cluster_raw[k] ~ Normal(0, 1) for k = 1, ..., K
cluster_eff[k] = cluster_raw[k] * σ_cluster

# Species level (hierarchical)
species_raw[s] ~ Normal(0, 1) for s = 1, ..., S
species_eff[s] = cluster_eff[k(s)] + species_raw[s] * σ_species

# Class level
class_raw[c] ~ Normal(0, 1) for c = 1, ..., C
class_eff[c] = class_raw[c] * σ_class

# Compound level (hierarchical)
compound_raw[m] ~ Normal(0, 1) for m = 1, ..., M
compound_eff[m] = class_eff[c(m)] + compound_raw[m] * σ_compound
```

#### Fixed Effects
```
μ₀ ~ Normal(5.0, 2.0)        # Global intercept
β_j ~ Normal(0, 2.0)         # Descriptor coefficients, j = 1, ..., J
γ ~ Normal(1.0, 0.5)         # Internal standard coefficient
```

#### Likelihood
```
y_sm ~ Normal(μ_sm, σ_y)

where:
μ_sm = μ₀ + species_eff[s] + compound_eff[m] + Σⱼ β_j * x_mj + γ * IS_s
```

### Why Non-centered Parameterization?

The centered parameterization would be:
```
species_eff[s] ~ Normal(cluster_eff[k(s)], σ_species)
```

This creates a "funnel" geometry when σ_species is small:
- The posterior becomes highly concentrated near cluster_eff[k(s)]
- NUTS has difficulty exploring this narrow region
- Results in divergent transitions

Non-centered parameterization:
```
species_raw[s] ~ Normal(0, 1)
species_eff[s] = cluster_eff[k(s)] + species_raw[s] * σ_species
```

Benefits:
- Separates location and scale
- Creates better geometry for NUTS
- Reduces divergences dramatically

### Posterior Inference

We use NUTS (No-U-Turn Sampler) for posterior inference:

1. **Hamiltonian Monte Carlo**: Uses gradient information for efficient exploration
2. **Automatic tuning**: Adapts step size and mass matrix during warmup
3. **No-U-Turn criterion**: Automatically determines trajectory length

Key sampling parameters:
- `target_accept = 0.95`: High to reduce divergences
- `max_treedepth = 12`: Allow longer trajectories
- `adapt_diag`: Diagonal mass matrix adaptation

### Model Checking

#### Convergence Diagnostics
- **R-hat** (potential scale reduction): 
  ```
  R̂ = √[(variance between chains + variance within chains) / variance within chains]
  ```
  Should be < 1.01 for all parameters

- **ESS** (Effective Sample Size):
  ```
  ESS = N / (1 + 2Σ_k ρ_k)
  ```
  Where ρ_k is autocorrelation at lag k

#### Posterior Predictive Checks
Generate new data from posterior:
```
ỹ_sm ~ Normal(μ̃_sm, σ̃_y)
```
Where parameters are drawn from posterior

Check calibration:
- Coverage: P(y_true ∈ [ỹ_2.5%, ỹ_97.5%]) ≈ 0.95
- RMSE: √[E[(y - ỹ)²]]

## 2. Bayesian Logistic Regression for Peak Assignment

### Model Specification

#### Prior
```
θ ~ MVN(0, σ²I)
```
Where θ = [θ₀, θ₁, θ₂, θ₃]ᵀ

In PyMC:
```python
theta0 ~ Normal(0, 2)      # Intercept
theta_mass ~ Normal(0, 2)  # Mass error effect
theta_rt ~ Normal(0, 2)    # RT z-score effect  
theta_int ~ Normal(0, 2)   # Intensity effect
```

#### Likelihood
```
y_i ~ Bernoulli(p_i)
logit(p_i) = θ₀ + θ₁x₁ᵢ + θ₂x₂ᵢ + θ₃x₃ᵢ
```

Where:
- x₁ᵢ: Mass error (ppm)
- x₂ᵢ: RT z-score
- x₃ᵢ: log₁₀(intensity)

### Feature Engineering Details

#### RT Z-score Calculation
Uses posterior predictive distribution from RT model:

```python
# For each (species s, compound m) pair:
μ_pred = E[μ_sm | data]  # Posterior mean
σ_pred = √[Var(μ_sm | data) + σ_y²]  # Total uncertainty

z_score = (RT_observed - μ_pred) / σ_pred
```

This incorporates:
1. Parameter uncertainty from RT model
2. Observation noise
3. Hierarchical structure

#### Mass Error Normalization
```
mass_error_ppm = (m/z_observed - m/z_theoretical) / m/z_theoretical * 10⁶
```

Typical ranges:
- High-resolution MS: ±5 ppm
- Low-resolution MS: ±50 ppm

### Posterior Predictive Distribution

For new peak p with features x_p:
```
P(assignment | x_p, data) = ∫ σ(θᵀx_p) p(θ | data) dθ
```

In practice, use Monte Carlo:
```
P̂(assignment | x_p) = 1/S Σₛ σ(θ⁽ˢ⁾ᵀx_p)
```
Where θ⁽ˢ⁾ are posterior samples

### Decision Rule

Assign peak p to compound m* where:
```
m* = argmax_m P(assignment | x_pm) if max_m P(assignment | x_pm) > τ
     = NULL                         otherwise
```

Where τ is probability threshold (default 0.5)

## 3. Information Flow Between Models

### Sequential Bayesian Learning

1. **Stage 1**: RT Model
   - Input: (species, compound, RT) observations
   - Output: Posterior p(θ_RT | y_RT)
   
2. **Stage 2**: Use RT Predictions
   - Extract: E[RT_sm | y_RT], Var(RT_sm | y_RT)
   - Compute: z-scores for candidate assignments
   
3. **Stage 3**: Logistic Model
   - Input: Features including RT z-scores
   - Output: P(assignment | features, y_RT)

### Uncertainty Propagation

RT model uncertainty → RT z-score → Assignment probability

1. **Parametric uncertainty**: From posterior variance of RT parameters
2. **Aleatory uncertainty**: From observation noise σ_y
3. **Combined**: z-score denominator includes both

This ensures:
- Uncertain RT predictions → larger z-score denominator → weaker evidence
- Confident RT predictions → smaller z-score denominator → stronger evidence

## 4. Model Extensions

### Potential Improvements

1. **Spike-and-slab priors** for feature selection:
   ```
   β_j ~ π δ₀ + (1-π) Normal(0, σ²)
   ```

2. **Hierarchical logistic model** for compounds:
   ```
   θ_compound[m] ~ Normal(θ_class[c(m)], σ_θ)
   ```

3. **Non-linear RT relationships**:
   ```
   f(x) = Σᵢ wᵢ φᵢ(x)  # Basis expansion
   ```

4. **Multi-level assignment** with mixture model:
   ```
   P(peak) = π₁P(true) + π₂P(isomer) + π₃P(noise)
   ```

### Active Learning Strategy

Use posterior uncertainty to guide experiments:

1. **Uncertainty sampling**: 
   ```
   x* = argmax_x H[P(y|x)]  # Maximum entropy
   ```

2. **Expected information gain**:
   ```
   x* = argmax_x I(θ; y|x)  # Mutual information
   ```

3. **Variance reduction**:
   ```
   x* = argmin_x E[Var(θ | y, x)]
   ```

## 5. Computational Considerations

### Scaling to Large Datasets

1. **Minibatch ADVI** for approximate inference:
   ```python
   approx = pm.fit(n=10000, method='advi', 
                   callbacks=[pm.callbacks.CheckParametersConvergence()])
   ```

2. **GPU acceleration** with JAX/NumPyro:
   ```python
   pm.sample(nuts_sampler='numpyro', 
            nuts_sampler_kwargs={'chain_method': 'vectorized'})
   ```

3. **Sparse matrices** for large presence/absence data

4. **Variational inference** for real-time predictions

### Memory Management

For large hierarchies:
- Use sparse representations for presence matrix
- Implement batch processing for predictions
- Cache RT predictions for repeated queries

## 6. Model Validation

### Cross-validation Strategies

1. **Leave-one-species-out**: Test generalization to new species
2. **Leave-one-compound-out**: Test for new compounds
3. **Time-based split**: For temporal validation
4. **Stratified k-fold**: Maintain class balance

### Calibration Metrics

1. **Expected Calibration Error (ECE)**:
   ```
   ECE = Σᵦ (nᵦ/N) |acc(Bᵦ) - conf(Bᵦ)|
   ```

2. **Brier Score**:
   ```
   BS = 1/N Σᵢ (pᵢ - yᵢ)²
   ```

3. **Log Loss**:
   ```
   LL = -1/N Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
   ```