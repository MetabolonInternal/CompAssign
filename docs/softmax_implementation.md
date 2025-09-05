# Softmax Peak Assignment Model - Technical Documentation

## Overview

This document describes the implementation of a peak-wise softmax assignment model with explicit null class handling and compound presence priors. This model replaces the pairwise logistic regression approach with a more principled probabilistic framework.

## Mathematical Formulation

### Model Structure

For each peak $i$ with candidate compound set $C_i$, we model assignment as a categorical distribution over $K_i + 1$ classes (candidates plus null):

$$p(y_i = k | X_i, \theta) = \frac{\exp(\eta_{ik} / T)}{\sum_{j=0}^{K_i} \exp(\eta_{ij} / T)}$$

Where the logits are:
- Null class (k=0): $\eta_{i0} = \theta_{null} + \log \pi_{null}$
- Compound classes (k>0): $\eta_{ik} = \theta_0 + \theta^T x_{ik} + \log \pi_{s(i),c_k}$

### Key Components

1. **Temperature Scaling**: $T$ provides in-model calibration
2. **Presence Priors**: $\pi_{sc} \sim Beta(\alpha_{sc}, \beta_{sc})$ for compound prevalence
3. **Feature Vector**: 9-dimensional features identical to logistic model
4. **Exclusivity**: Softmax ensures exactly one assignment per peak

## Implementation Architecture

### Core Modules

```
src/compassign/
├── peak_assignment_softmax.py   # Main softmax model
├── presence_prior.py            # Beta-Bernoulli priors
├── oracles.py                   # Human annotator simulators
├── eval_loop.py                 # Evaluation harness
└── active_learning.py           # Acquisition functions
```

### Class Hierarchy

```python
PeakAssignmentSoftmaxModel
├── compute_rt_predictions()     # Reuses existing RT prediction logic
├── generate_training_data()     # Creates padded tensors for peaks
├── build_model()                # PyMC model with softmax
├── sample()                     # MCMC sampling
├── predict_probs()              # Posterior predictive probabilities
└── assign()                     # Make assignments with thresholding

PresencePrior
├── init()                       # Initialize Beta priors
├── log_prior_odds()            # Get log probabilities
├── update_positive()           # Update after positive annotation
├── update_negative()           # Update after negative annotation
└── update_null()               # Update after null annotation
```

## Performance Guidance

Performance depends on dataset characteristics (e.g., candidate density, RT prediction quality). Use the validation suite to measure precision/recall/F1 and calibration (ECE/MCE) on your data.

For the multi‑candidate MEDIUM difficulty generator used in this repo’s validation:
- Start with `mass_tolerance=0.15` and `rt_window_k=2.0` to allow 5–8 candidates per peak.
- Adjust the assignment threshold to target your baseline (e.g., 0.30–0.40 for ~60–70% precision when demonstrating AL uplift).

## Active Learning Integration

### Acquisition Functions

1. **Expected FP Reduction**: $A_{FP}(i) = \mathbb{I}[q_i^{max} \geq \tau] \cdot (1 - q_i^{max})$
2. **Entropy**: $H(p_i) = -\sum_k p_{ik} \log p_{ik}$
3. **Mutual Information**: $I(y_i; \theta) = H[p(y_i)] - \mathbb{E}_\theta[H[p(y_i|\theta)]]$

### Update Cycle

```python
# 1. Select peaks for annotation
batch = select_batch(probs, acquisition='hybrid')

# 2. Get oracle labels
labels = oracle.label_peaks(batch)

# 3. Update presence priors
for peak, label in zip(batch, labels):
    if label == null:
        presence.update_null()
    else:
        presence.update_positive(species, compound)

# 4. Refit model
model.build_model()
model.sample()
```

## API Reference

### PeakAssignmentSoftmaxModel

```python
class PeakAssignmentSoftmaxModel:
    def __init__(self,
                 mass_tolerance: float = 0.005,
                 rt_window_k: float = 3.0,
                 use_temperature: bool = True,
                 standardize_features: bool = True,
                 random_seed: int = 42)
    
    def compute_rt_predictions(self,
                              trace_rt: az.InferenceData,
                              n_species: int,
                              n_compounds: int,
                              descriptors: np.ndarray,
                              internal_std: np.ndarray,
                              rt_model=None) -> Dict
    
    def generate_training_data(self,
                              peak_df: pd.DataFrame,
                              compound_mass: np.ndarray,
                              n_compounds: int,
                              species_cluster: np.ndarray,
                              init_presence: Optional[PresencePrior] = None) -> Dict
    
    def build_model(self) -> pm.Model
    
    def sample(self,
              draws: int = 1000,
              tune: int = 1000,
              chains: int = 2,
              target_accept: float = 0.95,
              seed: int = 42) -> az.InferenceData
    
    def predict_probs(self) -> Dict[int, np.ndarray]
    
    def assign(self, 
              prob_threshold: float = 0.5) -> SoftmaxAssignmentResults
```

### PresencePrior

```python
class PresencePrior:
    @classmethod
    def init(cls,
            n_species: int,
            n_compounds: int,
            smoothing: float = 1.0,
            null_prior: Tuple[float, float] = (1.0, 1.0),
            empirical_counts: Optional[Dict] = None) -> PresencePrior
    
    def log_prior_odds(self, species_idx: int) -> np.ndarray
    
    def log_prior_null(self) -> float
    
    def update_positive(self, species_idx: int, compound_idx: int)
    
    def update_negative(self, species_idx: int, compound_idx: int)
    
    def update_null(self)
```

## Migration Guide

### From Logistic to Softmax

```python
# Old (Logistic)
model = PeakAssignmentModel(
    mass_tolerance=0.005,
    rt_window_k=1.5,
    calibration_method='temperature'
)
results = model.predict_assignments(
    peak_df=peak_df,
    probability_threshold=0.5
)

# New (Softmax)
model = PeakAssignmentSoftmaxModel(
    mass_tolerance=0.005,
    rt_window_k=2.5,  # Slightly wider for better recall
    use_temperature=True
)
results = model.assign(
    prob_threshold=0.70  # Higher threshold for equivalent precision
)
```

### Key Differences

1. **Threshold**: Softmax requires τ ≈ 0.70 for high precision (vs 0.50 for logistic)
2. **RT Window**: Slightly wider (2.5σ vs 1.5σ) compensates for null class
3. **Results**: Includes per-peak probability distributions and null assignments
4. **Training**: ~2x slower but better calibrated

## Troubleshooting

### Common Issues

1. **Low Precision at Default Threshold**
   - Solution: Increase threshold to 0.70-0.75
   - Root cause: More conservative null assignments

2. **No Candidates Found**
   - Check mass tolerance and RT window settings
   - Verify RT predictions are properly computed

3. **PyMC Errors**
   - Ensure PyMC v5.25+ is installed
   - Use `pm.Data` instead of deprecated `pm.ConstantData`

4. **Slow Training**
   - Reduce chains to 2 for development
   - Use `draws=500, tune=500` for quick tests
   - Enable parallel sampling with `cores=4`

## Future Enhancements

1. **Hierarchical Temperature**: Species-specific temperature parameters
2. **Online Updates**: Laplace approximation for faster retraining
3. **Compound Class Priors**: Share information across similar compounds
4. **GPU Acceleration**: PyMC JAX backend for large-scale problems
