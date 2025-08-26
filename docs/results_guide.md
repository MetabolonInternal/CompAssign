# CompAssign Results Interpretation Guide

## Quick Start: What to Check First

### 1. Check Overall Performance (`output/results/results.json`)

```json
{
  "model_results": {
    "rmse": 0.389,           // Should be < 0.5 for good RT predictions
    "coverage_95": 0.987,    // Should be close to 0.95
    "assignment_precision": 0.844,  // Higher is better (fewer false positives)
    "assignment_recall": 0.987,     // Higher is better (fewer missed peaks)
    "assignment_f1": 0.910          // Overall assignment quality
  }
}
```

**Interpretation:**
- **RMSE < 0.5**: Excellent RT prediction accuracy
- **Coverage ≈ 95%**: Model uncertainty is well-calibrated
- **Precision > 80%**: Most assignments are correct
- **Recall > 95%**: Very few true peaks are missed
- **F1 > 0.9**: Excellent overall performance

### 2. Check Convergence (`output/plots/convergence_diagnostics.png`)

Look for:
- All R-hat values < 1.01 (green bars)
- ESS values > 100 (higher is better)

**Red flags:**
- R-hat > 1.01: Chains haven't converged
- ESS < 100: Need more samples

## Understanding Each Plot

### RT Model Diagnostic Plots

#### 1. `trace_plots.png` - MCMC Convergence
![Trace Plot Structure]
- **Left panels**: Parameter values over iterations
  - Should look like "fuzzy caterpillars"
  - Chains should overlap and mix well
- **Right panels**: Posterior distributions
  - Should be smooth and unimodal

**Problems to watch for:**
- Chains stuck at different values → increase tuning
- Trending patterns → chains haven't converged
- Multimodal distributions → model identification issues

#### 2. `energy_plot.png` - Sampling Quality
- **Blue**: Energy transitions (should be wide)
- **Black**: Marginal energy distribution
- Should overlap significantly

**Problems:**
- Separation indicates inefficient sampling
- May need to reparameterize model

#### 3. `posterior_vs_true.png` - Parameter Recovery
Shows estimated vs true parameters (synthetic data only)
- **Green line**: Posterior mean
- **Red dashed**: True value
- **Blue histogram**: Posterior distribution

**Good signs:**
- Green and red lines are close
- Narrow distributions (high certainty)
- True value within 95% HDI

#### 4. `ppc_plots.png` - Posterior Predictive Checks

**Panel 1: Observed vs Predicted**
- Points should follow diagonal line
- Scatter indicates prediction error

**Panel 2: Prediction Intervals**
- Black dots: observed values
- Blue band: 95% prediction interval
- Red line: predicted mean
- ~95% of points should be in blue band

**Panel 3: Residual Distribution**
- Should be centered at 0
- Approximately normal

**Panel 4: Q-Q Plot**
- Points should follow diagonal
- Deviations indicate non-normal residuals

#### 5. `residual_diagnostics.png` - Detailed Residual Analysis

**Panel 1: Residuals vs Fitted**
- Should show no pattern
- Random scatter around 0

**Panel 2: Scale-Location**
- Check for homoscedasticity
- Spread should be constant

**Panel 3: Residuals by Order**
- Check for temporal patterns
- Should be random

**Panel 4: ACF Plot**
- Most bars within red bands
- Indicates independence

### Assignment Model Plots

#### 6. `logistic_coefficients.png` - Feature Importance

Shows posterior distributions of logistic regression coefficients:
- **Intercept**: Baseline log-odds
- **Mass Error**: Effect of mass deviation (expect negative)
- **RT Z-score**: Effect of RT mismatch (expect negative)
- **Log Intensity**: Effect of peak intensity (expect positive)

**Interpretation:**
- Coefficients not overlapping 0 are significant
- Larger |coefficient| = more important feature
- Sign indicates direction of effect

#### 7. `feature_distributions.png` - Feature Separation

Violin plots showing feature distributions:
- **Green**: True assignments
- **Red**: False candidates

**Good separation:**
- RT Z-score: True assignments near 0, false far from 0
- Mass error: True assignments near 0
- Log intensity: True assignments higher

#### 8. `roc_pr_curves.png` - Classification Performance

**ROC Curve (left)**:
- AUC > 0.9: Excellent
- AUC > 0.8: Good
- AUC > 0.7: Acceptable
- Diagonal line: Random performance

**PR Curve (right)**:
- Important for imbalanced data
- Higher curve = better performance
- Red line: baseline (random)

#### 9. `probability_calibration.png` - Probability Reliability

**Calibration Plot (left)**:
- Points should follow diagonal
- Above diagonal: underconfident
- Below diagonal: overconfident

**Probability Distribution (right)**:
- Green: true assignments (should be high)
- Red: false candidates (should be low)
- Good separation at threshold (0.5)

#### 10. `confusion_matrix.png` - Assignment Errors

Heatmap showing:
```
              True Peak  Decoy/Wrong
Assigned        TP         FP
Not Assigned    FN         TN
```

**Ideal pattern:**
- High TP (top-left): correct assignments
- Low FP (top-right): few false assignments
- Low FN (bottom-left): few missed peaks
- High TN (bottom-right): correctly rejected

### Performance Comparison Plot

#### 11. `performance_comparison.png` - Overall Analysis

**Panel 1: RT Error by Species**
- Identifies problematic species
- Bar height = RMSE
- Red line = overall RMSE

**Panel 2: RT Error by Compound**
- Identifies difficult compounds
- May indicate descriptor issues

**Panel 3: Assignment Success by Intensity**
- Shows intensity bias
- Higher intensity should = higher success

**Panel 4: Summary Statistics**
- Quick reference for all metrics

## Common Issues and Solutions

### Issue 1: High RMSE (> 0.5)
**Possible causes:**
- Insufficient data
- Missing important descriptors
- Wrong hierarchical structure

**Solutions:**
- Add more training data
- Include more molecular descriptors
- Check species/compound groupings

### Issue 2: Poor Coverage (far from 95%)
**If < 95%:** Overconfident predictions
- Increase variance priors
- Check for model misspecification

**If > 95%:** Underconfident predictions
- Tighten variance priors
- May have too much regularization

### Issue 3: Low Assignment Precision
**Causes:**
- Many false positives
- Mass tolerance too wide
- RT predictions not discriminative

**Solutions:**
- Tighten mass tolerance
- Improve RT model
- Adjust probability threshold

### Issue 4: Low Assignment Recall
**Causes:**
- Missing true peaks
- Threshold too high
- Features not informative

**Solutions:**
- Lower probability threshold
- Check feature engineering
- Ensure RT model covers all compounds

### Issue 5: Divergences in Sampling
**Shown in log:** "X divergences after tuning"

**Solutions:**
1. Increase `target_accept` (up to 0.99)
2. Increase `max_treedepth` (up to 15)
3. Use more tuning steps
4. Check model parameterization

## Output Files Reference

### Data Files (`output/data/`)
- `observations.csv`: RT training data
- `peaks.csv`: All peaks with true labels
- `logit_training_data.csv`: Features for assignment model
- `true_parameters.json`: Ground truth (synthetic only)

### Results Files (`output/results/`)
- `config.json`: Run configuration
- `parameter_summary.csv`: All parameter estimates with uncertainty
- `predictions.csv`: RT predictions for all observations
- `assignment_parameters.csv`: Logistic model coefficients
- `peak_assignments.csv`: Final peak→compound mapping
- `results.json`: Summary metrics

### Reading Parameter Summary
```csv
parameter, mean, sd, hdi_3%, hdi_97%, ess_bulk, r_hat
mu0, 5.12, 0.15, 4.82, 5.41, 1523, 1.00
```

- **mean**: Point estimate
- **sd**: Uncertainty
- **hdi_3%, hdi_97%**: 94% credible interval
- **ess_bulk**: Effective samples (> 100 good)
- **r_hat**: Convergence (< 1.01 good)

## Recommendations for Production Use

1. **Always check convergence first**
   - No divergences
   - R-hat < 1.01
   - ESS > 400

2. **Validate on held-out data**
   - Don't trust training metrics alone
   - Use cross-validation

3. **Monitor prediction uncertainty**
   - Wide intervals indicate unreliable predictions
   - May need more data in those regions

4. **Adjust thresholds based on use case**
   - High precision needed: increase threshold
   - High recall needed: decrease threshold

5. **Regular model updates**
   - Retrain as new data becomes available
   - Monitor for distribution shift