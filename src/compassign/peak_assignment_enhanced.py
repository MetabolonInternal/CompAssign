"""
Enhanced Bayesian logistic regression model for ultra-high precision peak assignment.

This module implements precision-focused improvements:
- Asymmetric loss function (penalizes false positives more)
- RT uncertainty as additional feature
- Probability calibration for better thresholds
- Tighter mass tolerance defaults
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict


@dataclass
class EnhancedAssignmentResults:
    """Container for enhanced peak assignment results."""
    assignments: Dict[int, Optional[int]]  # peak_id -> compound_id
    probabilities: Dict[int, float]  # peak_id -> max probability
    calibrated_probabilities: Dict[int, float]  # peak_id -> calibrated probability
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Dict[str, int]  # TP, FP, TN, FN counts
    confidence_levels: Dict[str, List[int]]  # confident, review, rejected peaks


class EnhancedPeakAssignmentModel:
    """
    Enhanced Bayesian logistic regression for ultra-high precision peak assignment.
    
    Optimized for Metabolon's requirements where false positives are more costly
    than false negatives.
    """
    
    def __init__(self, 
                 mass_tolerance: float = 0.005,  # Tighter default for higher precision
                 fp_penalty: float = 5.0):  # Weight for false positive penalty
        """
        Initialize the enhanced peak assignment model.
        
        Parameters
        ----------
        mass_tolerance : float
            Mass tolerance in Da for candidate generation (default 0.005 for high precision)
        fp_penalty : float
            Penalty weight for false positives (default 5.0)
        """
        self.mass_tolerance = mass_tolerance
        self.fp_penalty = fp_penalty
        self.model = None
        self.trace = None
        self.logit_df = None
        self.rt_predictions = {}
        self.calibrator = None
        
    def compute_rt_predictions(self, 
                              trace_rt: az.InferenceData,
                              n_species: int,
                              n_compounds: int,
                              descriptors: np.ndarray,
                              internal_std: np.ndarray) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """
        Compute RT predictions and uncertainties from the RT model posterior.
        
        Enhanced version includes parameter uncertainty quantification.
        """
        print("Computing RT predictions with uncertainty quantification...")
        
        # Extract posterior samples for full uncertainty
        mu0_samples = trace_rt.posterior['mu0'].values.flatten()
        species_samples = trace_rt.posterior['species_eff'].values  # shape: (chains*draws, n_species)
        compound_samples = trace_rt.posterior['compound_eff'].values  # shape: (chains*draws, n_compounds)
        beta_samples = trace_rt.posterior['beta'].values  # shape: (chains*draws, n_descriptors)
        gamma_samples = trace_rt.posterior['gamma'].values.flatten()
        sigma_y_samples = trace_rt.posterior['sigma_y'].values.flatten()
        
        # Reshape for easier indexing
        n_samples = len(mu0_samples)
        species_samples = species_samples.reshape(n_samples, -1)
        compound_samples = compound_samples.reshape(n_samples, -1)
        beta_samples = beta_samples.reshape(n_samples, -1)
        
        rt_predictions = {}
        
        for s in range(n_species):
            for m in range(n_compounds):
                # Sample RT predictions from posterior
                rt_samples = []
                for i in range(0, n_samples, 10):  # Subsample for efficiency
                    rt_pred = (mu0_samples[i] 
                              + species_samples[i, s] 
                              + compound_samples[i, m] 
                              + np.dot(descriptors[m], beta_samples[i]) 
                              + gamma_samples[i] * internal_std[s])
                    rt_samples.append(rt_pred)
                
                rt_samples = np.array(rt_samples)
                
                # Mean and total uncertainty (parameter + observation)
                mean_rt = np.mean(rt_samples)
                param_std = np.std(rt_samples)  # Parameter uncertainty
                obs_std = np.mean(sigma_y_samples)  # Observation noise
                total_std = np.sqrt(param_std**2 + obs_std**2)
                
                rt_predictions[(s, m)] = (mean_rt, total_std)
        
        self.rt_predictions = rt_predictions
        return rt_predictions
    
    def generate_training_data(self,
                              peak_df: pd.DataFrame,
                              compound_mass: np.ndarray,
                              n_compounds: int) -> pd.DataFrame:
        """
        Generate enhanced training data with additional features.
        """
        if not self.rt_predictions:
            raise ValueError("RT predictions not computed. Run compute_rt_predictions first.")
        
        print(f"Generating candidates with tight mass tolerance ±{self.mass_tolerance} Da...")
        
        logit_data = []
        
        for _, peak in peak_df.iterrows():
            peak_id = int(peak['peak_id'])
            s = int(peak['species'])
            true_comp = peak['true_compound']
            if not pd.isna(true_comp):
                true_comp = int(true_comp)
            else:
                true_comp = None
            
            mz = peak['mass']
            rt = peak['rt']
            intensity = peak['intensity']
            
            # Tighter mass tolerance for higher precision
            for m in range(n_compounds):
                if abs(mz - compound_mass[m]) <= self.mass_tolerance:
                    # Standard features
                    mass_error_ppm = (mz - compound_mass[m]) / compound_mass[m] * 1e6
                    
                    # Enhanced RT features
                    rt_pred_mean, rt_pred_std = self.rt_predictions[(s, m)]
                    rt_z = (rt - rt_pred_mean) / rt_pred_std if rt_pred_std > 0 else 0
                    
                    # NEW: RT uncertainty as feature
                    rt_uncertainty = rt_pred_std / rt_pred_mean if rt_pred_mean > 0 else 1.0
                    
                    # Log intensity
                    log_intensity = np.log10(intensity) if intensity > 0 else 0
                    
                    # NEW: Absolute RT difference for extreme outliers
                    rt_abs_diff = abs(rt - rt_pred_mean)
                    
                    # Label
                    label = 1 if (true_comp is not None and true_comp == m) else 0
                    
                    # NEW: Weight for asymmetric loss
                    weight = 1.0 if label == 1 else self.fp_penalty
                    
                    logit_data.append((peak_id, s, m, mass_error_ppm, rt_z, 
                                     log_intensity, rt_uncertainty, rt_abs_diff,
                                     label, weight))
        
        self.logit_df = pd.DataFrame(logit_data, 
                                     columns=['peak_id', 'species', 'compound', 
                                             'mass_err_ppm', 'rt_z', 'log_intensity',
                                             'rt_uncertainty', 'rt_abs_diff',
                                             'label', 'weight'])
        
        print(f"Enhanced training pairs: {len(self.logit_df)}")
        print(f"  Positive (true) assignments: {self.logit_df['label'].sum()}")
        print(f"  Negative (false) assignments: {(1 - self.logit_df['label']).sum()}")
        print(f"  False positive penalty weight: {self.fp_penalty}x")
        
        return self.logit_df
    
    def build_model(self) -> pm.Model:
        """
        Build enhanced PyMC model with asymmetric loss function.
        """
        if self.logit_df is None:
            raise ValueError("Training data not generated. Run generate_training_data first.")
        
        with pm.Model() as model:
            # Priors with informative constraints based on domain knowledge
            theta0 = pm.Normal('theta0', 0.0, 2.0)  # intercept
            
            # Mass error should strongly predict (negative for absolute error)
            theta_mass = pm.Normal('theta_mass', -2.0, 1.0)  
            
            # RT z-score should strongly predict (negative)
            theta_rt = pm.Normal('theta_rt', -3.0, 1.0)
            
            # Intensity somewhat predictive (positive)
            theta_int = pm.Normal('theta_int', 0.5, 1.0)
            
            # NEW: RT uncertainty should be negative (higher uncertainty = lower probability)
            theta_rt_unc = pm.Normal('theta_rt_unc', -1.0, 1.0)
            
            # NEW: Absolute RT difference (strong negative)
            theta_rt_abs = pm.Normal('theta_rt_abs', -2.0, 1.0)
            
            # Linear predictor with enhanced features
            logit_p = (theta0 
                      + theta_mass * np.abs(self.logit_df['mass_err_ppm'].values)
                      + theta_rt * np.abs(self.logit_df['rt_z'].values)
                      + theta_int * self.logit_df['log_intensity'].values
                      + theta_rt_unc * self.logit_df['rt_uncertainty'].values
                      + theta_rt_abs * self.logit_df['rt_abs_diff'].values)
            
            # Convert to probability
            p = pm.Deterministic('p', pm.math.sigmoid(logit_p))
            
            # Weighted Bernoulli likelihood for asymmetric loss
            y_obs = self.logit_df['label'].values
            weights = self.logit_df['weight'].values
            
            # Log-likelihood with weights
            log_likelihood = weights * (y_obs * pm.math.log(p + 1e-8) + 
                                       (1 - y_obs) * pm.math.log(1 - p + 1e-8))
            
            # Add as potential (weighted likelihood)
            pm.Potential('weighted_loglik', log_likelihood)
        
        self.model = model
        return model
    
    def sample(self,
              n_samples: int = 2000,  # More samples for better calibration
              n_tune: int = 2000,
              n_chains: int = 4,  # More chains for robustness
              target_accept: float = 0.95,
              random_seed: int = 42) -> az.InferenceData:
        """
        Sample from the posterior with enhanced settings.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        print(f"\nSampling enhanced model with {n_chains} chains, {n_samples} samples each...")
        print(f"Asymmetric loss: False positives weighted {self.fp_penalty}x")
        
        with self.model:
            self.trace = pm.sample(
                n_samples,
                tune=n_tune,
                chains=n_chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=True,
                cores=2  # Use parallel sampling
            )
        
        # Enhanced convergence checks
        rhat_vals = az.rhat(self.trace).to_dataframe()
        max_rhat = rhat_vals.max().max()
        
        if max_rhat > 1.01:
            print(f"\nWarning: Maximum R-hat = {max_rhat:.3f} (should be < 1.01)")
        else:
            print(f"\nConvergence excellent: Maximum R-hat = {max_rhat:.3f}")
        
        # Check effective sample size
        ess = az.ess(self.trace).to_dataframe()
        min_ess = ess.min().min()
        print(f"Minimum effective sample size: {min_ess:.0f}")
        
        return self.trace
    
    def calibrate_probabilities(self) -> IsotonicRegression:
        """
        Calibrate probabilities using isotonic regression for better thresholds.
        """
        if self.trace is None:
            raise ValueError("No trace available. Run sample first.")
        
        print("\nCalibrating probabilities for optimal thresholds...")
        
        # Get raw predictions
        theta0_mean = self.trace.posterior['theta0'].mean().item()
        theta_mass_mean = self.trace.posterior['theta_mass'].mean().item()
        theta_rt_mean = self.trace.posterior['theta_rt'].mean().item()
        theta_int_mean = self.trace.posterior['theta_int'].mean().item()
        theta_rt_unc_mean = self.trace.posterior['theta_rt_unc'].mean().item()
        theta_rt_abs_mean = self.trace.posterior['theta_rt_abs'].mean().item()
        
        # Compute raw probabilities
        logit = (theta0_mean 
                + theta_mass_mean * np.abs(self.logit_df['mass_err_ppm'])
                + theta_rt_mean * np.abs(self.logit_df['rt_z'])
                + theta_int_mean * self.logit_df['log_intensity']
                + theta_rt_unc_mean * self.logit_df['rt_uncertainty']
                + theta_rt_abs_mean * self.logit_df['rt_abs_diff'])
        
        raw_probs = 1 / (1 + np.exp(-logit))
        
        # Fit isotonic regression for calibration
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(raw_probs, self.logit_df['label'])
        
        # Store both raw and calibrated probabilities
        self.logit_df['pred_prob_raw'] = raw_probs
        self.logit_df['pred_prob'] = self.calibrator.transform(raw_probs)
        
        # Compare calibration
        raw_mean_pos = raw_probs[self.logit_df['label'] == 1].mean()
        raw_mean_neg = raw_probs[self.logit_df['label'] == 0].mean()
        cal_mean_pos = self.logit_df['pred_prob'][self.logit_df['label'] == 1].mean()
        cal_mean_neg = self.logit_df['pred_prob'][self.logit_df['label'] == 0].mean()
        
        print(f"Probability calibration results:")
        print(f"  True positives - Raw: {raw_mean_pos:.3f}, Calibrated: {cal_mean_pos:.3f}")
        print(f"  True negatives - Raw: {raw_mean_neg:.3f}, Calibrated: {cal_mean_neg:.3f}")
        print(f"  Separation improved: {(cal_mean_pos - cal_mean_neg) - (raw_mean_pos - raw_mean_neg):.3f}")
        
        return self.calibrator
    
    def predict_assignments_staged(self, 
                                  peak_df: pd.DataFrame,
                                  high_precision_threshold: float = 0.9,
                                  review_threshold: float = 0.7) -> EnhancedAssignmentResults:
        """
        Staged prediction with confidence levels for active learning.
        
        Parameters
        ----------
        peak_df : pd.DataFrame
            Peak data
        high_precision_threshold : float
            Threshold for high-confidence assignments (default 0.9)
        review_threshold : float
            Threshold for review queue (default 0.7)
            
        Returns
        -------
        EnhancedAssignmentResults
            Enhanced results with confidence levels
        """
        if self.trace is None:
            raise ValueError("No trace available. Run sample first.")
        
        # Calibrate probabilities if not done
        if 'pred_prob' not in self.logit_df.columns:
            self.calibrate_probabilities()
        
        print(f"\nStaged assignment with thresholds:")
        print(f"  High confidence: ≥ {high_precision_threshold}")
        print(f"  Review queue: {review_threshold} - {high_precision_threshold}")
        print(f"  Rejected: < {review_threshold}")
        
        # Extract coefficients
        theta0_mean = self.trace.posterior['theta0'].mean().item()
        theta_mass_mean = self.trace.posterior['theta_mass'].mean().item()
        theta_rt_mean = self.trace.posterior['theta_rt'].mean().item()
        theta_int_mean = self.trace.posterior['theta_int'].mean().item()
        theta_rt_unc_mean = self.trace.posterior['theta_rt_unc'].mean().item()
        theta_rt_abs_mean = self.trace.posterior['theta_rt_abs'].mean().item()
        
        print(f"\nEnhanced model coefficients:")
        print(f"  Intercept: {theta0_mean:.3f}")
        print(f"  |Mass error|: {theta_mass_mean:.3f}")
        print(f"  |RT z-score|: {theta_rt_mean:.3f}")
        print(f"  Log intensity: {theta_int_mean:.3f}")
        print(f"  RT uncertainty: {theta_rt_unc_mean:.3f}")
        print(f"  RT abs diff: {theta_rt_abs_mean:.3f}")
        
        # Assign peaks with confidence levels
        assignments = {}
        probabilities = {}
        calibrated_probs = {}
        confidence_levels = {'confident': [], 'review': [], 'rejected': []}
        
        for peak_id, group in self.logit_df.groupby('peak_id'):
            # Find candidate with max calibrated probability
            idx_max = group['pred_prob'].idxmax()
            comp_pred = int(group.loc[idx_max, 'compound'])
            prob_cal = group.loc[idx_max, 'pred_prob']
            prob_raw = group.loc[idx_max, 'pred_prob_raw']
            
            calibrated_probs[peak_id] = prob_cal
            probabilities[peak_id] = prob_raw
            
            if prob_cal >= high_precision_threshold:
                assignments[peak_id] = comp_pred
                confidence_levels['confident'].append(peak_id)
            elif prob_cal >= review_threshold:
                assignments[peak_id] = None  # Needs review
                confidence_levels['review'].append(peak_id)
            else:
                assignments[peak_id] = None
                confidence_levels['rejected'].append(peak_id)
        
        # Compute metrics at high precision threshold
        TP = FP = FN = TN = 0
        
        for _, peak in peak_df.iterrows():
            peak_id = int(peak['peak_id'])
            true_comp = peak['true_compound']
            if not pd.isna(true_comp):
                true_comp = int(true_comp)
            else:
                true_comp = None
            
            # Only count confident assignments
            if peak_id in confidence_levels['confident']:
                assigned_comp = assignments.get(peak_id)
                if true_comp is not None and assigned_comp == true_comp:
                    TP += 1
                else:
                    FP += 1
            else:
                if true_comp is not None:
                    FN += 1
                else:
                    TN += 1
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\nEnhanced assignment results (threshold={high_precision_threshold}):")
        print(f"  Confident assignments: {len(confidence_levels['confident'])}")
        print(f"  Review queue: {len(confidence_levels['review'])}")
        print(f"  Rejected: {len(confidence_levels['rejected'])}")
        print(f"\nPerformance metrics:")
        print(f"  True Positives: {TP}")
        print(f"  False Positives: {FP} (minimized)")
        print(f"  False Negatives: {FN}")
        print(f"  True Negatives: {TN}")
        print(f"  PRECISION: {precision:.3f} (target: >0.95)")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1_score:.3f}")
        
        return EnhancedAssignmentResults(
            assignments=assignments,
            probabilities=probabilities,
            calibrated_probabilities=calibrated_probs,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            confusion_matrix={'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN},
            confidence_levels=confidence_levels
        )
    
    def get_parameter_summary(self) -> pd.DataFrame:
        """Get enhanced summary statistics for model parameters."""
        if self.trace is None:
            raise ValueError("No trace available. Run sample first.")
        
        summary = az.summary(self.trace)
        
        # Add interpretation
        print("\nParameter interpretation:")
        print("  Negative coefficients → feature reduces assignment probability")
        print("  |Mass error| and |RT z-score| should be strongly negative")
        print("  RT uncertainty should be negative (high uncertainty → low confidence)")
        
        return summary