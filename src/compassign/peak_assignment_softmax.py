"""
Peak assignment model using softmax with null class and presence priors.

This module implements a Bayesian softmax model for peak-to-compound assignment
that enforces exclusivity constraints (one compound per peak), includes a null
option for unmatched peaks, and incorporates compound presence priors that can
be updated online from human feedback.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

from .presence_prior import PresencePrior


@dataclass
class SoftmaxAssignmentResults:
    """Container for softmax peak assignment results."""
    assignments: Dict[int, Optional[int]]  # peak_id -> compound_id or None
    top_prob: Dict[int, float]  # peak_id -> max probability
    per_peak_probs: Dict[int, np.ndarray]  # peak_id -> [p_null, p_c1, ...]
    precision: float
    recall: float
    f1: float
    confusion_matrix: Dict[str, int]  # TP, FP, TN, FN
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error


class PeakAssignmentSoftmaxModel:
    """
    Peak assignment using softmax over candidates with null option.
    
    Key improvements over pairwise logistic:
    - Enforces exclusivity: exactly one compound (or null) per peak
    - Includes null class for unmatched peaks
    - Incorporates compound presence priors
    - Temperature-based calibration
    - Supports online updates from human feedback
    """
    
    def __init__(self, 
                 mass_tolerance: float = 0.005,
                 rt_window_k: float = 3.0,
                 use_temperature: bool = True,
                 standardize_features: bool = True,
                 random_seed: int = 42):
        """
        Initialize the softmax peak assignment model.
        
        Parameters
        ----------
        mass_tolerance : float
            Mass tolerance in Da for candidate generation
        rt_window_k : float
            RT window multiplier (in standard deviations)
        use_temperature : bool
            If True, include temperature parameter for calibration
        standardize_features : bool
            If True, standardize features before model fitting
        random_seed : int
            Random seed for reproducibility
        """
        self.mass_tolerance = mass_tolerance
        self.rt_window_k = rt_window_k
        self.use_temperature = use_temperature
        self.standardize_features = standardize_features
        self.random_seed = random_seed
        
        # Initialize random state
        self.rng = np.random.default_rng(random_seed)
        
        # Model components
        self.presence: Optional[PresencePrior] = None
        self.model: Optional[pm.Model] = None
        self.trace: Optional[az.InferenceData] = None
        
        # Data storage
        self.K_max: int = 0  # Maximum number of candidates (including null)
        self.feature_names: List[str] = []
        self.train_pack: Dict[str, Any] = {}
        self.feature_scaler = StandardScaler()
        self.rt_predictions = {}
    
    def compute_rt_predictions(self, 
                              trace_rt: az.InferenceData,
                              n_species: int,
                              n_compounds: int,
                              descriptors: np.ndarray,
                              internal_std: np.ndarray,
                              rt_model=None) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """
        Compute RT predictions with proper predictive variance.
        
        This reuses the exact logic from PeakAssignmentModel to ensure
        consistency in RT predictions between models.
        """
        print("Computing RT predictions from posterior...")
        
        post = trace_rt.posterior
        
        # Flatten chains and draws
        def flat(x):
            v = x.values
            return v.reshape(-1, *v.shape[2:])
        
        mu0 = flat(post['mu0'])
        beta = flat(post['beta'])
        gamma = flat(post['gamma'])
        sp = flat(post['species_eff'])
        cp = flat(post['compound_eff'])
        sigy = flat(post['sigma_y'])
        
        # Standardize descriptors if RT model provided
        if rt_model is not None and hasattr(rt_model, '_desc_mean'):
            desc_std = (descriptors - rt_model._desc_mean) / rt_model._desc_std
            is_std = (internal_std - rt_model._is_mean) / rt_model._is_std
        else:
            desc_std = descriptors
            is_std = internal_std
        
        rt_predictions = {}
        
        for s in range(n_species):
            sp_s = sp[:, s]
            for m in range(n_compounds):
                # Predictive mean and variance
                loc = (mu0 + sp_s + cp[:, m] + 
                       (beta @ desc_std[m]) + gamma * is_std[s])
                
                # Proper predictive variance: includes parameter uncertainty + noise
                var = np.var(loc, ddof=1) + np.mean(sigy**2)
                
                rt_predictions[(s, m)] = (float(np.mean(loc)), float(np.sqrt(var)))
        
        self.rt_predictions = rt_predictions
        print(f"  Computed predictions for {len(rt_predictions)} species-compound pairs")
        return rt_predictions
    
    def generate_training_data(self,
                              peak_df: pd.DataFrame,
                              compound_mass: np.ndarray,
                              n_compounds: int,
                              species_cluster: np.ndarray,
                              init_presence: Optional[PresencePrior] = None,
                              initial_labeled_fraction: float = 0.0,
                              initial_labeled_n: Optional[int] = None,
                              random_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate training data for softmax model.
        
        Creates padded tensors with candidates + null for each peak.
        
        Parameters
        ----------
        peak_df : pd.DataFrame
            DataFrame with peak information
        compound_mass : np.ndarray
            Array of compound masses
        n_compounds : int
            Number of compounds
        species_cluster : np.ndarray
            Cluster assignment for each species
        init_presence : Optional[PresencePrior]
            Initial presence prior (if None, creates uniform)
            
        Returns
        -------
        Dict[str, Any]
            Training data pack with padded tensors and mappings
        """
        if not self.rt_predictions:
            raise ValueError("RT predictions not computed. Run compute_rt_predictions first.")
        
        print(f"\nGenerating SOFTMAX training data...")
        print(f"  Mass tolerance: ±{self.mass_tolerance} Da")
        print(f"  RT window: ±{self.rt_window_k}σ")
        print(f"  Including NULL class for each peak")
        
        # Initialize presence prior if not provided
        n_species = len(np.unique(peak_df['species']))
        if init_presence is None:
            self.presence = PresencePrior.init(n_species, n_compounds, smoothing=1.0)
        else:
            self.presence = init_presence
        
        # Track species median intensities
        species_medians = peak_df.groupby('species')['intensity'].median()
        
        # First pass: collect candidates for each peak to find K_max
        peak_candidates = []
        peak_labels = []
        peak_species = []
        peak_ids = []
        
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
            
            # Extract additional features if available (allowed, no ground-truth leakage)
            peak_quality = peak.get('peak_quality', None)
            sn_ratio = peak.get('sn_ratio', None)
            peak_width = peak.get('peak_width', None)
            peak_width_rt = peak.get('peak_width_rt', None)
            peak_asymmetry = peak.get('peak_asymmetry', None)
            
            # Find candidates by mass
            candidates = []
            features = []
            
            for m in range(n_compounds):
                mass_error_da = abs(mz - compound_mass[m])
                if mass_error_da > self.mass_tolerance:
                    continue
                
                mass_error_ppm = (mz - compound_mass[m]) / compound_mass[m] * 1e6
                
                # RT filtering
                rt_pred_mean, rt_pred_std = self.rt_predictions[(s, m)]
                
                if rt_pred_std > 0.01:
                    rt_z = (rt - rt_pred_mean) / rt_pred_std
                else:
                    rt_z = (rt - rt_pred_mean) / 0.01
                
                if abs(rt_z) > self.rt_window_k:
                    continue
                
                # Core features (always present)
                log_intensity = np.log1p(intensity)
                
                # Confidence scores
                mass_confidence = np.exp(-abs(mass_error_ppm) / 10)
                rt_confidence = np.exp(-abs(rt_z) / 3)
                combined_confidence = mass_confidence * rt_confidence
                
                # Context features
                log_compound_mass = np.log(compound_mass[m])
                log_rt_uncertainty = np.log(rt_pred_std + 1e-6)
                log_relative_intensity = np.log1p(intensity / median_intensity) if median_intensity > 0 else 0
                
                # Build feature list
                feat = [
                    mass_error_ppm,
                    rt_z,
                    log_intensity,
                    mass_confidence,
                    rt_confidence,
                    combined_confidence,
                    log_compound_mass,
                    log_rt_uncertainty,
                    log_relative_intensity
                ]
                
                # Add discriminative features if available
                if peak_quality is not None:
                    feat.append(peak_quality)
                if sn_ratio is not None:
                    feat.append(np.log1p(sn_ratio))  # Log transform for scale
                if peak_width is not None:
                    feat.append(peak_width)
                # Add permitted peak shape features if present
                if peak_width_rt is not None:
                    feat.append(peak_width_rt)
                if peak_asymmetry is not None:
                    feat.append(peak_asymmetry)
                
                candidates.append(m)
                features.append(feat)
            
            # Store candidate info
            peak_candidates.append((candidates, features))
            peak_species.append(s)
            peak_ids.append(peak_id)
            
            # Determine label (0 for null, 1+ for compound position)
            if true_comp is None:
                peak_labels.append(0)  # Null
            elif true_comp in candidates:
                peak_labels.append(candidates.index(true_comp) + 1)  # 1-indexed position
            else:
                # True compound not in candidates (filtered out)
                peak_labels.append(-1)  # Will be masked in training
        
        # Find maximum number of candidates (plus 1 for null)
        if not peak_candidates:
            raise ValueError("No peaks to process. Check data generation.")
        
        max_candidates = max(len(cands[0]) for cands in peak_candidates) if peak_candidates else 0
        if max_candidates == 0:
            print("WARNING: No valid candidates found for any peak. Check mass tolerance and RT window settings.")
        
        self.K_max = max_candidates + 1  # +1 for null class
        
        # Determine number of features from first non-empty candidate set
        n_features = 9  # Default to base features
        for cands, feats in peak_candidates:
            if feats:
                n_features = len(feats[0])
                break
        
        # Store feature names dynamically
        self.feature_names = [
            'mass_err_ppm', 'rt_z', 'log_intensity',
            'mass_confidence', 'rt_confidence', 'combined_confidence',
            'log_compound_mass', 'log_rt_uncertainty', 'log_relative_intensity'
        ]
        
        # Add names for additional features if present
        if n_features > 9:
            # Check what additional features are available
            first_peak = peak_df.iloc[0] if len(peak_df) > 0 else None
            if first_peak is not None:
                if 'peak_quality' in first_peak and not pd.isna(first_peak.get('peak_quality')):
                    self.feature_names.append('peak_quality')
                if 'sn_ratio' in first_peak and not pd.isna(first_peak.get('sn_ratio')):
                    self.feature_names.append('log_sn_ratio')
                if 'peak_width' in first_peak and not pd.isna(first_peak.get('peak_width')):
                    self.feature_names.append('peak_width')
                if 'peak_width_rt' in first_peak and not pd.isna(first_peak.get('peak_width_rt')):
                    self.feature_names.append('peak_width_rt')
                if 'peak_asymmetry' in first_peak and not pd.isna(first_peak.get('peak_asymmetry')):
                    self.feature_names.append('peak_asymmetry')
        
        print(f"\nCandidate statistics:")
        print(f"  Number of peaks: {len(peak_candidates)}")
        print(f"  Max candidates per peak: {self.K_max - 1} (+ null)")
        print(f"  Features per candidate: {n_features} ({len(self.feature_names)} named)")
        print(f"  Peaks with true compound: {sum(1 for l in peak_labels if l > 0)}")
        print(f"  Null peaks: {sum(1 for l in peak_labels if l == 0)}")
        print(f"  Filtered peaks: {sum(1 for l in peak_labels if l == -1)}")
        
        # Build padded tensors with dynamic feature size
        N = len(peak_candidates)
        X = np.zeros((N, self.K_max, n_features), dtype=np.float32)
        mask = np.zeros((N, self.K_max), dtype=bool)
        true_labels = np.array(peak_labels, dtype=np.int32)
        labels = np.full_like(true_labels, -1, dtype=np.int32)
        
        # Mappings
        peak_to_row = {pid: i for i, pid in enumerate(peak_ids)}
        row_to_candidates = []
        
        # Fill padded tensors
        for i, ((candidates, features), s) in enumerate(zip(peak_candidates, peak_species)):
            # First slot is always null (no features needed, will use theta_null)
            mask[i, 0] = True
            row_to_candidates.append([None] + candidates)  # None for null
            
            # Fill candidate features
            for k, feat in enumerate(features, start=1):
                X[i, k, :] = feat
                mask[i, k] = True
        
        # Standardize features if requested (only on valid slots)
        if self.standardize_features:
            print("  Standardizing features...")
            # Extract valid features for fitting scaler
            valid_features = []
            for i in range(N):
                for k in range(1, self.K_max):  # Skip null slot
                    if mask[i, k]:
                        valid_features.append(X[i, k, :])
            
            if len(valid_features) > 0:
                valid_features = np.array(valid_features)
                self.feature_scaler.fit(valid_features)
                
                # Apply standardization
                for i in range(N):
                    for k in range(1, self.K_max):
                        if mask[i, k]:
                            X[i, k, :] = self.feature_scaler.transform(X[i, k, :].reshape(1, -1))[0]
        
        # Optionally reveal a small seed set of labels to enable initial supervised training
        if initial_labeled_n is None and initial_labeled_fraction > 0:
            valid_idx = np.where(true_labels >= 0)[0]
            n_seed = max(1, int(len(valid_idx) * initial_labeled_fraction))
            rng = np.random.default_rng(self.random_seed if random_seed is None else random_seed)
            if len(valid_idx) > 0:
                seed_idx = rng.choice(valid_idx, size=min(n_seed, len(valid_idx)), replace=False)
                labels[seed_idx] = true_labels[seed_idx]
        elif initial_labeled_n is not None and initial_labeled_n > 0:
            valid_idx = np.where(true_labels >= 0)[0]
            rng = np.random.default_rng(self.random_seed if random_seed is None else random_seed)
            if len(valid_idx) > 0:
                n_seed = min(initial_labeled_n, len(valid_idx))
                seed_idx = rng.choice(valid_idx, size=n_seed, replace=False)
                labels[seed_idx] = true_labels[seed_idx]

        # Store training pack
        self.train_pack = {
            'X': X,
            'mask': mask,
            'labels': labels,
            'true_labels': true_labels,
            'peak_to_row': peak_to_row,
            'row_to_candidates': row_to_candidates,
            'peak_species': np.array(peak_species),
            'peak_ids': np.array(peak_ids),
            'n_compounds': n_compounds,
            'n_species': n_species
        }
        
        return self.train_pack
    
    def build_model(self) -> pm.Model:
        """
        Build softmax model with null class and presence priors.
        
        Returns
        -------
        pm.Model
            PyMC model for softmax assignment
        """
        if not self.train_pack:
            raise ValueError("Training data not generated. Run generate_training_data first.")
        
        print(f"\nBuilding SOFTMAX model")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Max candidates: {self.K_max}")
        print(f"  Temperature scaling: {self.use_temperature}")
        
        X = self.train_pack['X']
        mask = self.train_pack['mask']
        labels = self.train_pack['labels']
        peak_species = self.train_pack['peak_species']
        row_to_candidates = self.train_pack['row_to_candidates']
        
        N = X.shape[0]
        n_features = len(self.feature_names)
        
        with pm.Model() as model:
            # Priors
            if self.standardize_features:
                # Tighter priors for standardized features
                theta0 = pm.Normal('theta0', 0.0, 1.0)
                theta_features = pm.Normal('theta_features', 0.0, 1.0, shape=n_features)
                theta_null = pm.Normal('theta_null', 0.0, 1.0)
            else:
                theta0 = pm.Normal('theta0', 0.0, 2.0)
                theta_features = pm.Normal('theta_features', 0.0, 2.0, shape=n_features)
                theta_null = pm.Normal('theta_null', 0.0, 2.0)
            
            # Temperature parameter (on log scale to ensure positive)
            if self.use_temperature:
                log_T = pm.Normal('log_T', 0.0, 0.5)  # exp(0) = 1.0 by default
                T = pm.Deterministic('T', pm.math.exp(log_T))
            else:
                T = 1.0
            
            # Prepare data as PyMC constants (v5 uses Data)
            X_tensor = pm.Data('X', X)
            mask_tensor = pm.Data('mask', mask)
            
            # Pre-compute presence priors for all species-compound pairs
            log_pi_null = self.presence.log_prior_null()
            
            # Build presence prior matrix
            presence_matrix = np.zeros((N, self.K_max), dtype='float32')
            for i in range(N):
                s = peak_species[i]
                candidates = row_to_candidates[i]
                log_pi_compounds = self.presence.log_prior_odds(s)
                
                presence_matrix[i, 0] = log_pi_null  # Null prior
                for k in range(1, self.K_max):
                    if mask[i, k]:
                        c = candidates[k]
                        presence_matrix[i, k] = log_pi_compounds[c]
            
            presence_tensor = pm.Data('presence_priors', presence_matrix)
            
            # Compute features contribution for all candidates
            # Shape: (N, K_max)
            feature_contrib = pm.math.sum(X_tensor * theta_features[None, None, :], axis=2)
            
            # Build logits using PyMC operations
            # Initialize with very negative values for masked slots
            eta = pm.math.ones((N, self.K_max)) * (-1e9)
            
            # Null logits (column 0)
            null_logits = theta_null + presence_tensor[:, 0]
            
            # Candidate logits (columns 1+)
            candidate_logits = theta0 + feature_contrib[:, 1:] + presence_tensor[:, 1:]
            
            # Combine null and candidate logits
            eta = pm.math.concatenate([null_logits[:, None], candidate_logits], axis=1)
            
            # Apply mask: set invalid slots to very negative
            eta = pm.math.where(mask_tensor, eta, -1e9)
            
            # Name the tensor for debugging
            eta_tensor = pm.Deterministic('eta', eta)
            
            # Apply temperature and compute softmax
            scaled_eta = eta_tensor / T
            
            # Softmax probabilities
            p = pm.Deterministic('p', pm.math.softmax(scaled_eta, axis=1))
            
            # Filter to only peaks with known labels for training
            train_mask = labels >= 0
            train_idx = np.where(train_mask)[0]
            
            if len(train_idx) > 0:
                # Categorical likelihood
                y = pm.Categorical('y', 
                                  p=p[train_idx, :], 
                                  observed=labels[train_idx])
            
            model.feature_names = self.feature_names
        
        self.model = model
        return model
    
    def sample(self,
              draws: int = 1000,
              tune: int = 1000,
              chains: int = 2,
              target_accept: float = 0.95,
              seed: int = 42) -> az.InferenceData:
        """
        Sample from the posterior using NUTS.
        
        Parameters
        ----------
        draws : int
            Number of samples to draw
        tune : int
            Number of tuning steps
        chains : int
            Number of chains
        target_accept : float
            Target acceptance rate
        seed : int
            Random seed
            
        Returns
        -------
        az.InferenceData
            Posterior samples
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        print(f"\nSampling SOFTMAX model with {chains} chains, {draws} samples each...")
        
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=seed,
                progressbar=True
            )
        
        return self.trace
    
    def predict_probs(self) -> Dict[int, np.ndarray]:
        """
        Compute posterior predictive probabilities for each peak.
        
        Returns
        -------
        Dict[int, np.ndarray]
            peak_id -> probability vector [p_null, p_c1, ...]
        """
        if self.trace is None:
            raise ValueError("No trace available. Run sample first.")
        
        print("\nComputing posterior predictive probabilities...")
        
        # Get posterior samples
        post = self.trace.posterior
        
        # Average probabilities across chains and draws
        p_samples = post['p'].values  # Shape: (chains, draws, N, K_max)
        p_mean = p_samples.mean(axis=(0, 1))  # Shape: (N, K_max)
        
        # Map to peak IDs
        peak_probs = {}
        for i, peak_id in enumerate(self.train_pack['peak_ids']):
            # Extract valid probabilities (where mask is True)
            mask_i = self.train_pack['mask'][i]
            valid_probs = p_mean[i, mask_i]
            
            # Renormalize (should already sum to 1, but ensure numerical stability)
            valid_probs = valid_probs / valid_probs.sum()
            
            peak_probs[int(peak_id)] = valid_probs
        
        return peak_probs
    
    def predict_prob_samples(self) -> Dict[int, np.ndarray]:
        """
        Return posterior samples of per-peak probabilities.
        
        Returns
        -------
        Dict[int, np.ndarray]
            peak_id -> array of shape (n_samples, K_i) with valid classes only
        """
        if self.trace is None:
            raise ValueError("No trace available. Run sample first.")
        
        # Get posterior samples
        p = self.trace.posterior['p'].values  # (chains, draws, N, K_max)
        samples = p.reshape(-1, p.shape[2], p.shape[3])  # (S, N, K_max)
        
        out = {}
        for i, peak_id in enumerate(self.train_pack['peak_ids']):
            mask_i = self.train_pack['mask'][i]
            pi = samples[:, i, mask_i]  # (S, K_i)
            # Renormalize to ensure proper probabilities
            pi = pi / np.clip(pi.sum(axis=1, keepdims=True), 1e-12, None)
            out[int(peak_id)] = pi
        
        return out
    
    def assign(self, prob_threshold: float = 0.5) -> SoftmaxAssignmentResults:
        """
        Assign compounds to peaks based on softmax probabilities.
        
        Parameters
        ----------
        prob_threshold : float
            Minimum probability for assignment (default: 0.5)
            
        Returns
        -------
        SoftmaxAssignmentResults
            Assignment results with metrics
        """
        if self.trace is None:
            raise ValueError("No trace available. Run sample first.")
        
        print(f"\nSoftmax Peak Assignment")
        print(f"="*50)
        print(f"Probability threshold: {prob_threshold:.2f}")
        
        # Get probabilities
        peak_probs = self.predict_probs()
        
        # Make assignments
        assignments = {}
        top_probs = {}
        
        for peak_id, probs in peak_probs.items():
            # Find argmax
            best_idx = np.argmax(probs)
            best_prob = probs[best_idx]
            
            top_probs[peak_id] = best_prob
            
            # Check threshold and null
            if best_prob >= prob_threshold and best_idx > 0:
                # Get compound from mapping
                row = self.train_pack['peak_to_row'][peak_id]
                candidates = self.train_pack['row_to_candidates'][row]
                assignments[peak_id] = candidates[best_idx]
            else:
                # Either null or below threshold
                assignments[peak_id] = None
        
        # Calculate metrics
        true_labels = self.train_pack.get('true_labels', self.train_pack['labels'])
        peak_ids = self.train_pack['peak_ids']
        row_to_candidates = self.train_pack['row_to_candidates']
        
        tp = fp = fn = tn = 0
        all_probs = []
        all_correct = []
        
        for i, (peak_id, label) in enumerate(zip(peak_ids, true_labels)):
            if label < 0:
                continue  # Skip filtered peaks
            
            pred = assignments.get(peak_id)
            candidates = row_to_candidates[i]
            
            # Get true compound
            if label == 0:
                true_comp = None
            else:
                true_comp = candidates[label]
            
            # Classification metrics
            if true_comp is None:
                if pred is None:
                    tn += 1
                else:
                    fp += 1
            else:
                if pred == true_comp:
                    tp += 1
                elif pred is None:
                    fn += 1
                else:
                    fp += 1
                    fn += 1  # Wrong assignment counts as both FP and FN
            
            # Calibration data
            all_probs.append(top_probs[peak_id])
            all_correct.append(1 if pred == true_comp else 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate calibration metrics
        ece, mce = self._calculate_calibration_metrics(
            np.array(all_probs), 
            np.array(all_correct)
        )
        
        print(f"\nResults:")
        print(f"  Assignments made: {sum(1 for v in assignments.values() if v is not None)}")
        print(f"  Null assignments: {sum(1 for v in assignments.values() if v is None)}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1: {f1:.3f}")
        print(f"  ECE: {ece:.3f}")
        print(f"  MCE: {mce:.3f}")
        
        return SoftmaxAssignmentResults(
            assignments=assignments,
            top_prob=top_probs,
            per_peak_probs=peak_probs,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix={'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn},
            ece=ece,
            mce=mce
        )
    
    def _calculate_calibration_metrics(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
        """Calculate Expected and Maximum Calibration Error."""
        if len(probs) == 0 or len(labels) == 0:
            return 0.0, 0.0
        
        # Ensure arrays are same length and valid
        probs = np.clip(probs, 1e-12, 1 - 1e-12)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0
        mce = 0
        
        for j in range(n_bins):
            bin_lower = bin_boundaries[j]
            bin_upper = bin_boundaries[j + 1]
            
            if j == 0:
                in_bin = (probs >= bin_lower) & (probs <= bin_upper)
            else:
                in_bin = (probs > bin_lower) & (probs <= bin_upper)
            
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0 and np.sum(in_bin) > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                
                calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                
                ece += prop_in_bin * calibration_error
                mce = max(mce, calibration_error)
        
        return ece, mce
