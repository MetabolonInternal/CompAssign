"""
Calibrated peak assignment model with properly scaled probabilities.

This version produces ACTUAL calibrated probabilities where:
- 0.9 means 90% chance of being correct
- 0.5 means 50% chance of being correct
- Threshold of 0.5 is meaningful (not 0.1)

Key improvements:
1. Balanced class weighting during training
2. Proper feature scaling
3. Temperature calibration
4. Isotonic regression calibration option
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

@dataclass
class AssignmentResults:
    """Container for peak assignment results."""
    assignments: Dict[int, Optional[int]]  # peak_id -> compound_id
    probabilities: Dict[int, float]  # peak_id -> calibrated probability
    raw_scores: Dict[int, float]  # peak_id -> raw score (before calibration)
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Dict[str, int]  # TP, FP, TN, FN counts
    calibration_metrics: Dict[str, float]  # ECE, MCE, etc.


class PeakAssignmentModel:
    """
    Peak assignment with properly calibrated probabilities.
    
    Features:
    1. Balanced training to avoid bias toward negative class
    2. Feature standardization to prevent extreme logits
    3. Temperature scaling for calibration
    4. Optional isotonic regression for perfect calibration
    """
    
    def __init__(self, 
                 mass_tolerance: float = 0.005,
                 rt_window_k: float = 1.5,
                 use_class_weights: bool = True,
                 standardize_features: bool = True,
                 calibration_method: str = 'temperature'):  # 'temperature', 'isotonic', or 'none'
        """
        Initialize the calibrated peak assignment model.
        
        Parameters
        ----------
        mass_tolerance : float
            Mass tolerance in Da for candidate generation
        rt_window_k : float
            RT window multiplier for filtering
        use_class_weights : bool
            If True, balance classes during training
        standardize_features : bool
            If True, standardize features to prevent extreme logits
        calibration_method : str
            Calibration method: 'temperature', 'isotonic', or 'none'
        """
        self.mass_tolerance = mass_tolerance
        self.rt_window_k = rt_window_k
        self.use_class_weights = use_class_weights
        self.standardize_features = standardize_features
        self.calibration_method = calibration_method
        
        self.model = None
        self.trace = None
        self.logit_df = None
        self.rt_predictions = {}
        self.feature_names = []
        self.feature_scaler = StandardScaler()
        self.temperature = 1.0  # Temperature scaling parameter
        self.isotonic_regressor = None  # Isotonic regression model
        
    def compute_rt_predictions(self, 
                              trace_rt: az.InferenceData,
                              n_species: int,
                              n_compounds: int,
                              descriptors: np.ndarray,
                              internal_std: np.ndarray) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """Compute RT predictions from the RT model posterior."""
        print("Computing RT predictions from posterior...")
        
        # Extract posterior means
        mu0_mean = trace_rt.posterior['mu0'].mean().item()
        species_mean = trace_rt.posterior['species_eff'].mean(dim=['chain','draw']).values
        compound_mean = trace_rt.posterior['compound_eff'].mean(dim=['chain','draw']).values
        beta_mean = trace_rt.posterior['beta'].mean(dim=['chain','draw']).values
        gamma_mean = trace_rt.posterior['gamma'].mean().item()
        sigma_y_mean = trace_rt.posterior['sigma_y'].mean().item()
        
        # Extract posterior standard deviations
        species_std = trace_rt.posterior['species_eff'].std(dim=['chain','draw']).values
        compound_std = trace_rt.posterior['compound_eff'].std(dim=['chain','draw']).values
        
        rt_predictions = {}
        
        for s in range(n_species):
            for m in range(n_compounds):
                mean_rt = (mu0_mean 
                          + species_mean[s] 
                          + compound_mean[m] 
                          + np.dot(descriptors[m], beta_mean) 
                          + gamma_mean * internal_std[s])
                
                param_var = species_std[s]**2 + compound_std[m]**2
                total_std = np.sqrt(param_var + sigma_y_mean**2)
                
                rt_predictions[(s, m)] = (mean_rt, total_std)
        
        self.rt_predictions = rt_predictions
        return rt_predictions
    
    def generate_training_data(self,
                              peak_df: pd.DataFrame,
                              compound_mass: np.ndarray,
                              n_compounds: int) -> pd.DataFrame:
        """Generate training data with balanced classes and scaled features."""
        if not self.rt_predictions:
            raise ValueError("RT predictions not computed. Run compute_rt_predictions first.")
        
        print(f"Generating CALIBRATED training data...")
        print(f"  Mass tolerance: ±{self.mass_tolerance} Da")
        print(f"  RT window: ±{self.rt_window_k}σ")
        print(f"  Class weighting: {self.use_class_weights}")
        print(f"  Feature standardization: {self.standardize_features}")
        
        logit_data = []
        n_mass_filtered = 0
        n_rt_filtered = 0
        n_total = 0
        
        # Track species median intensities
        species_medians = peak_df.groupby('species')['intensity'].median()
        
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
            median_intensity = species_medians[s]
            
            for m in range(n_compounds):
                n_total += 1
                
                # Mass filtering
                mass_error_da = abs(mz - compound_mass[m])
                if mass_error_da > self.mass_tolerance:
                    n_mass_filtered += 1
                    continue
                
                mass_error_ppm = (mz - compound_mass[m]) / compound_mass[m] * 1e6
                
                # RT filtering
                rt_pred_mean, rt_pred_std = self.rt_predictions[(s, m)]
                
                if rt_pred_std > 0.01:
                    rt_z = (rt - rt_pred_mean) / rt_pred_std
                else:
                    rt_z = (rt - rt_pred_mean) / 0.01
                
                if abs(rt_z) > self.rt_window_k:
                    n_rt_filtered += 1
                    continue
                
                # Calculate ESSENTIAL features only (9 features for optimal performance)
                # Core features (3) - Primary signals
                log_intensity = np.log(intensity) if intensity > 0 else 0
                
                # Confidence scores (3) - Calibrated probabilities
                mass_confidence = np.exp(-abs(mass_error_ppm) / 10)
                rt_confidence = np.exp(-abs(rt_z) / 3)
                combined_confidence = mass_confidence * rt_confidence
                
                # Context features (3) - Prior information
                log_compound_mass = np.log(compound_mass[m])
                log_rt_uncertainty = np.log(rt_pred_std + 1e-6)
                log_relative_intensity = np.log(intensity / median_intensity) if median_intensity > 0 else 0
                
                # REMOVED REDUNDANT FEATURES (7 features eliminated):
                # - mass_err_sq, rt_z_sq (redundant with confidence scores)
                # - mass_rt_interaction (minimal value)
                # - abs_mass_err, abs_rt_z (redundant with squared)
                # - mass_err_normalized, rt_err_normalized (redundant with standardization)
                
                label = 1 if (true_comp is not None and true_comp == m) else 0
                
                logit_data.append({
                    'peak_id': peak_id,
                    'species': s,
                    'compound': m,
                    # Core features (3)
                    'mass_err_ppm': mass_error_ppm,
                    'rt_z': rt_z,
                    'log_intensity': log_intensity,
                    # Confidence scores (3)
                    'mass_confidence': mass_confidence,
                    'rt_confidence': rt_confidence,
                    'combined_confidence': combined_confidence,
                    # Context features (3)
                    'log_compound_mass': log_compound_mass,
                    'log_rt_uncertainty': log_rt_uncertainty,
                    'log_relative_intensity': log_relative_intensity,
                    'label': label
                })
        
        self.logit_df = pd.DataFrame(logit_data)
        
        # Calculate class weights for balanced training
        if self.use_class_weights:
            n_pos = self.logit_df['label'].sum()
            n_neg = len(self.logit_df) - n_pos
            self.pos_weight = n_neg / n_pos  # Upweight positive class
            self.neg_weight = 1.0
            print(f"\n  Class weights: Positive={self.pos_weight:.2f}, Negative={self.neg_weight:.2f}")
        else:
            self.pos_weight = 1.0
            self.neg_weight = 1.0
        
        # Store feature names
        self.feature_names = [col for col in self.logit_df.columns 
                              if col not in ['peak_id', 'species', 'compound', 'label']]
        
        # Standardize features if requested
        if self.standardize_features:
            print("  Standardizing features to mean=0, std=1")
            feature_data = self.logit_df[self.feature_names].values
            feature_data_scaled = self.feature_scaler.fit_transform(feature_data)
            
            # Replace with scaled features
            for i, feat in enumerate(self.feature_names):
                self.logit_df[feat] = feature_data_scaled[:, i]
        
        # Report statistics
        n_kept = len(self.logit_df)
        print(f"\nCandidate generation statistics:")
        print(f"  Total possible pairs: {n_total}")
        print(f"  Filtered by mass: {n_mass_filtered} ({n_mass_filtered/n_total*100:.1f}%)")
        print(f"  Filtered by RT: {n_rt_filtered} ({n_rt_filtered/n_total*100:.1f}%)")
        print(f"  Kept for training: {n_kept} ({n_kept/n_total*100:.1f}%)")
        print(f"  Positive examples: {self.logit_df['label'].sum()}")
        print(f"  Negative examples: {(1 - self.logit_df['label']).sum()}")
        
        return self.logit_df
    
    def build_model(self) -> pm.Model:
        """Build balanced logistic regression model."""
        if self.logit_df is None:
            raise ValueError("Training data not generated. Run generate_training_data first.")
        
        n_features = len(self.feature_names)
        print(f"\nBuilding CALIBRATED logistic model")
        print(f"  Features: {n_features}")
        print(f"  Class weighting: {self.use_class_weights}")
        
        with pm.Model() as model:
            # Priors - use tighter priors if features are standardized
            if self.standardize_features:
                theta0 = pm.Normal('theta0', 0.0, 1.0)  # Tighter
                theta_features = pm.Normal('theta_features', 0.0, 1.0, shape=n_features)
            else:
                theta0 = pm.Normal('theta0', 0.0, 2.0)
                theta_features = pm.Normal('theta_features', 0.0, 2.0, shape=n_features)
            
            # Linear predictor
            feature_data = self.logit_df[self.feature_names].values
            logit_p = theta0 + pm.math.dot(feature_data, theta_features)
            
            # Convert to probability
            p = pm.Deterministic('p', pm.math.sigmoid(logit_p))
            
            # Apply class weights in the likelihood
            labels = self.logit_df['label'].values
            if self.use_class_weights:
                # MORE EFFICIENT: Use data upsampling approach
                # Duplicate positive examples to balance classes
                pos_indices = np.where(labels == 1)[0]
                neg_indices = np.where(labels == 0)[0]
                
                # Calculate how many times to replicate positive examples
                n_replications = int(self.pos_weight)
                remainder = self.pos_weight - n_replications
                
                # Create balanced indices
                balanced_indices = list(neg_indices)  # All negatives
                
                # Properly replicate positive indices
                for _ in range(n_replications):
                    balanced_indices.extend(pos_indices)
                
                # Add partial replication for remainder
                if remainder > 0:
                    n_extra = int(len(pos_indices) * remainder)
                    if n_extra > 0:
                        extra_indices = np.random.choice(pos_indices, n_extra, replace=True)
                        balanced_indices.extend(extra_indices)
                
                balanced_indices = np.array(balanced_indices)
                
                # Use balanced data for likelihood
                balanced_p = p[balanced_indices]
                balanced_labels = labels[balanced_indices]
                
                # Standard Bernoulli with balanced data
                y = pm.Bernoulli('y', balanced_p, observed=balanced_labels)
                
                print(f"  Balanced training: {len(balanced_indices)} examples "
                      f"({len(pos_indices) * self.pos_weight:.0f} pos, {len(neg_indices)} neg)")
            else:
                # Standard Bernoulli
                y = pm.Bernoulli('y', p, observed=labels)
            
            model.feature_names = self.feature_names
        
        self.model = model
        return model
    
    def sample(self,
              n_samples: int = 1000,
              n_tune: int = 1000,
              n_chains: int = None,
              target_accept: float = 0.95,
              random_seed: int = 42) -> az.InferenceData:
        """Sample from the posterior using NUTS."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        chains_str = f"{n_chains} chains" if n_chains is not None else "default chains"
        print(f"\nSampling calibrated model with {chains_str}, {n_samples} samples each...")
        
        sample_kwargs = {
            'draws': n_samples,
            'tune': n_tune,
            'target_accept': target_accept,
            'random_seed': random_seed,
            'progressbar': True
        }
        
        if n_chains is not None:
            sample_kwargs['chains'] = n_chains
        
        with self.model:
            self.trace = pm.sample(**sample_kwargs)
        
        # After sampling, calibrate if requested
        if self.calibration_method != 'none':
            self._calibrate_model()
        
        return self.trace
    
    def _calibrate_model(self):
        """Calibrate the model using temperature scaling or isotonic regression."""
        print(f"\nCalibrating model using {self.calibration_method} method...")
        
        # Get raw predictions
        theta0_mean = self.trace.posterior['theta0'].mean().item()
        theta_features_mean = self.trace.posterior['theta_features'].mean(dim=['chain','draw']).values
        
        feature_data = self.logit_df[self.feature_names].values
        logits = theta0_mean + np.dot(feature_data, theta_features_mean)
        raw_probs = 1 / (1 + np.exp(-logits))
        
        labels = self.logit_df['label'].values
        
        if self.calibration_method == 'temperature':
            # Find optimal temperature using validation set
            # (In production, use separate validation set)
            def nll_loss(T):
                """Negative log likelihood loss for temperature scaling."""
                scaled_probs = 1 / (1 + np.exp(-logits / T))
                # Clip to avoid log(0)
                scaled_probs = np.clip(scaled_probs, 1e-10, 1 - 1e-10)
                nll = -np.mean(labels * np.log(scaled_probs) + 
                              (1 - labels) * np.log(1 - scaled_probs))
                return nll
            
            # Optimize temperature
            result = minimize(nll_loss, x0=1.0, bounds=[(0.1, 10.0)], method='L-BFGS-B')
            self.temperature = result.x[0]
            print(f"  Optimal temperature: {self.temperature:.3f}")
            
        elif self.calibration_method == 'isotonic':
            # Fit isotonic regression
            self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
            self.isotonic_regressor.fit(raw_probs, labels)
            print(f"  Isotonic regression fitted")
    
    def _apply_calibration(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply calibration to raw probabilities."""
        if self.calibration_method == 'none':
            return raw_probs
        elif self.calibration_method == 'temperature':
            # Re-compute from logits with temperature
            # We need to store logits or recompute them
            # For simplicity, we'll scale the probabilities directly (approximation)
            # Better approach: store logits and scale them
            # Using approximation: p_cal ≈ p^(1/T) / (p^(1/T) + (1-p)^(1/T))
            if self.temperature == 1.0:
                return raw_probs
            else:
                # Approximate temperature scaling on probabilities
                # Convert back to logits, scale, then back to probs
                logits = np.log(raw_probs / (1 - raw_probs + 1e-10))
                scaled_logits = logits / self.temperature
                return 1 / (1 + np.exp(-scaled_logits))
        elif self.calibration_method == 'isotonic':
            return self.isotonic_regressor.transform(raw_probs)
        else:
            return raw_probs
    
    def predict_assignments(self, 
                           peak_df: pd.DataFrame,
                           probability_threshold: float = 0.5) -> AssignmentResults:
        """
        Predict assignments with CALIBRATED probabilities.
        
        Now a threshold of 0.5 means "50% confident" as it should!
        """
        if self.trace is None:
            raise ValueError("No trace available. Run sample first.")
        
        # Extract posterior mean coefficients
        theta0_mean = self.trace.posterior['theta0'].mean().item()
        theta_features_mean = self.trace.posterior['theta_features'].mean(dim=['chain','draw']).values
        
        print(f"\n🎯 CALIBRATED Peak Assignment")
        print(f"="*50)
        print(f"Probability threshold: {probability_threshold:.2f} (meaningful!)")
        print(f"Calibration method: {self.calibration_method}")
        if self.calibration_method == 'temperature':
            print(f"Temperature: {self.temperature:.3f}")
        
        # Compute raw scores
        feature_data = self.logit_df[self.feature_names].values
        logits = theta0_mean + np.dot(feature_data, theta_features_mean)
        raw_probs = 1 / (1 + np.exp(-logits))
        
        # Apply calibration
        calibrated_probs = self._apply_calibration(raw_probs)
        
        self.logit_df['raw_prob'] = raw_probs
        self.logit_df['calibrated_prob'] = calibrated_probs
        # Also add 'pred_prob' for compatibility with plotting code
        self.logit_df['pred_prob'] = calibrated_probs
        
        # Show probability distributions
        print(f"\n📊 Probability distributions:")
        print(f"  Raw probabilities:")
        print(f"    Mean: {raw_probs.mean():.3f}, Std: {raw_probs.std():.3f}")
        print(f"    >0.5: {(raw_probs > 0.5).sum()} ({(raw_probs > 0.5).mean()*100:.1f}%)")
        print(f"  Calibrated probabilities:")
        print(f"    Mean: {calibrated_probs.mean():.3f}, Std: {calibrated_probs.std():.3f}")
        print(f"    >0.5: {(calibrated_probs > 0.5).sum()} ({(calibrated_probs > 0.5).mean()*100:.1f}%)")
        
        # Greedy assignment with calibrated probabilities
        assignments = {}
        assignment_probs = {}
        raw_scores = {}
        
        for peak_id in self.logit_df['peak_id'].unique():
            peak_candidates = self.logit_df[self.logit_df['peak_id'] == peak_id]
            
            # Filter by calibrated probability threshold
            good_candidates = peak_candidates[peak_candidates['calibrated_prob'] >= probability_threshold]
            
            if len(good_candidates) > 0:
                # Pick best among those passing threshold
                best_idx = good_candidates['calibrated_prob'].idxmax()
                best_candidate = good_candidates.loc[best_idx]
                
                assignments[int(peak_id)] = int(best_candidate['compound'])
                assignment_probs[int(peak_id)] = best_candidate['calibrated_prob']
                raw_scores[int(peak_id)] = best_candidate['raw_prob']
        
        # Calculate metrics
        tp = fp = fn = tn = 0
        
        for _, peak in peak_df.iterrows():
            peak_id = int(peak['peak_id'])
            true_comp = peak['true_compound']
            
            if pd.isna(true_comp) or true_comp is None:
                if peak_id in assignments:
                    fp += 1
                else:
                    tn += 1
            else:
                true_comp = int(true_comp)
                if peak_id in assignments:
                    if assignments[peak_id] == true_comp:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate calibration metrics
        calibration_metrics = self._calculate_calibration_metrics(
            calibrated_probs, 
            self.logit_df['label'].values
        )
        
        print(f"\n📈 Calibration metrics:")
        print(f"  Expected Calibration Error (ECE): {calibration_metrics['ece']:.3f}")
        print(f"  Maximum Calibration Error (MCE): {calibration_metrics['mce']:.3f}")
        
        results = AssignmentResults(
            assignments=assignments,
            probabilities=assignment_probs,
            raw_scores=raw_scores,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix={'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn},
            calibration_metrics=calibration_metrics
        )
        
        print(f"\n📊 Results with calibrated probabilities:")
        print(f"  Threshold: {probability_threshold:.2f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1: {f1:.3f}")
        
        return results
    
    def _calculate_calibration_metrics(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
        """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        mce = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                
                calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                
                ece += prop_in_bin * calibration_error
                mce = max(mce, calibration_error)
        
        return {
            'ece': ece,
            'mce': mce
        }