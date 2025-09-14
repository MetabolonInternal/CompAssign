"""
Simplified Bayesian peak assignment model for publication.

This module implements a cleaner, more theoretically grounded Bayesian softmax model
with minimal features, hierarchical uncertainty (instead of temperature), and
presence priors for active learning.

Key simplifications:
1. Minimal feature set (4-5 features only)
2. Hierarchical Normal uncertainty instead of temperature scaling
3. Clean mathematical formulation for publication
4. Maintains presence priors for active learning capability
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

from .presence_prior import PresencePrior


@dataclass
class AssignmentResults:
    """Container for peak assignment results with many-to-one support."""
    assignments: Dict[int, Optional[int]]  # peak_id -> compound_id or None
    top_prob: Dict[int, float]  # peak_id -> max probability
    per_peak_probs: Dict[int, np.ndarray]  # peak_id -> [p_null, p_c1, ...]
    
    # Peak-level metrics (many-to-one aware)
    precision: float  # Peak-level precision
    recall: float  # Peak-level recall
    f1: float  # Peak-level F1
    confusion_matrix: Dict[str, int]  # TP, FP, TN, FN
    
    # Compound-level metrics (PRIMARY)
    compound_precision: float  # What fraction of predicted compounds are correct?
    compound_recall: float  # What fraction of true compounds were found?
    compound_f1: float  # Harmonic mean of compound P/R
    compound_metrics: Dict[str, Any]  # Additional compound-level stats
    
    # Coverage metrics
    coverage_per_compound: Dict[int, float]  # compound_id -> fraction of peaks found
    mean_coverage: float  # Average coverage across all compounds
    
    # Calibration
    ece: float  # Expected Calibration Error
    
    # Groupings for analysis
    peaks_by_compound: Dict[int, List[int]]  # compound_id -> list of peak_ids assigned to it
    true_peaks_by_compound: Dict[int, List[int]]  # compound_id -> list of true peak_ids


class PeakAssignment:
    """
    Hierarchical Bayesian model for peak assignment.
    
    This model uses:
    - Minimal discriminative features (4 only)
    - Hierarchical uncertainty modeling with learnable noise
    - Beta-Bernoulli presence priors for active learning
    - Explicit null class for unmatched peaks
    
    Mathematical formulation:
    η_mean = θ₀ + X @ θ + log_π  # Linear predictor with presence priors
    η ~ Normal(η_mean, σ)         # Hierarchical uncertainty
    p = softmax(η)                # Category probabilities
    y ~ Categorical(p)            # Assignment
    """
    
    # Define minimal feature set
    MINIMAL_FEATURES = [
        'mass_err_ppm',       # Fundamental: mass accuracy
        'rt_z',               # Fundamental: RT accuracy
        'log_intensity',      # Peak quality
        'log_rt_uncertainty'  # Prediction confidence (important for AL)
    ]
    
    def __init__(self,
                 mass_tolerance: float = 0.005,
                 rt_window_k: float = 1.5,
                 random_seed: int = 42):
        """
        Initialize the hierarchical Bayesian model.
        
        Parameters
        ----------
        mass_tolerance : float
            Mass tolerance in Da for candidate generation
        rt_window_k : float
            RT window multiplier (in standard deviations)
        random_seed : int
            Random seed for reproducibility
        """
        self.mass_tolerance = mass_tolerance
        self.rt_window_k = rt_window_k
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
        (Identical to softmax model for compatibility)
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
                
                # Proper predictive variance
                var = np.var(loc, ddof=1) + np.mean(sigy**2)
                
                rt_predictions[(s, m)] = (float(np.mean(loc)), float(np.sqrt(var)))
        
        self.rt_predictions = rt_predictions
        print(f"  Computed predictions for {len(rt_predictions)} species-compound pairs")
        return rt_predictions
    
    def generate_training_data(self,
                              peak_df: pd.DataFrame,
                              compound_mass: np.ndarray,
                              n_compounds: int,
                              compound_info: Optional[pd.DataFrame] = None,
                              init_presence: Optional[PresencePrior] = None,
                              initial_labeled_fraction: float = 0.0,
                              random_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate training data with minimal features.
        """
        if not self.rt_predictions:
            raise ValueError("RT predictions not computed. Run compute_rt_predictions first.")
        
        print(f"\nGenerating training data...")
        
        # Count decoys if present (will be printed later)
        
        # Initialize presence prior if not provided
        n_species = len(np.unique(peak_df['species']))
        if init_presence is None:
            self.presence = PresencePrior.init(n_species, n_compounds, smoothing=1.0)
        else:
            self.presence = init_presence
        
        # Track species median intensities
        species_medians = peak_df.groupby('species')['intensity'].median()
        
        # Collect candidates for each peak
        # We'll generate TWO sets: training (no decoys) and test (all compounds)
        peak_candidates_train = []  # Training candidates (no decoys)
        peak_candidates_test = []   # Test candidates (all compounds)
        peak_labels = []
        true_compounds_list = []
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
            
            # Find candidates by mass - TRAINING SET (no decoys)
            candidates_train = []
            features_train = []
            
            for m in range(n_compounds):
                # Skip decoys during training to prevent data leakage
                if compound_info is not None and 'is_decoy' in compound_info.columns:
                    if compound_info.iloc[m]['is_decoy']:
                        continue
                
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
                
                # Build feature list - fixed minimal feature set (4 features)
                feat = [
                    mass_error_ppm,                    # Core
                    rt_z,                              # Core
                    np.log1p(intensity),               # Quality
                    np.log(rt_pred_std + 1e-6)        # Uncertainty
                ]
                
                candidates_train.append(m)
                features_train.append(feat)
            
            # Find candidates - TEST SET (all compounds including decoys)
            candidates_test = []
            features_test = []
            
            for m in range(n_compounds):
                # Include ALL compounds for test evaluation
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
                
                # Build same features for test candidates (4 features)
                feat = [
                    mass_error_ppm,
                    rt_z,
                    np.log1p(intensity),
                    np.log(rt_pred_std + 1e-6)
                ]
                
                candidates_test.append(m)
                features_test.append(feat)
            
            # Store both candidate sets
            peak_candidates_train.append((candidates_train, features_train))
            peak_candidates_test.append((candidates_test, features_test))
            peak_species.append(s)
            peak_ids.append(peak_id)
            
            # Track true compound id for evaluation (independent of candidate presence)
            true_compounds_list.append(true_comp)

            # Determine label (0 for null, 1+ for compound position)
            if true_comp is None:
                peak_labels.append(0)  # Null
            elif true_comp in candidates_train:
                peak_labels.append(candidates_train.index(true_comp) + 1)  # 1-indexed position
            else:
                # True compound not in candidates (filtered out)
                peak_labels.append(-1)  # Will be masked in training
        
        # Find maximum number of candidates for both sets
        max_candidates_train = max(len(cands[0]) for cands in peak_candidates_train) if peak_candidates_train else 0
        max_candidates_test = max(len(cands[0]) for cands in peak_candidates_test) if peak_candidates_test else 0
        
        # Use the MAXIMUM of both for the model architecture
        # This ensures the model can handle both training and test candidates
        self.K_max = max(max_candidates_train, max_candidates_test) + 1  # +1 for null class
        self.K_max_train = max_candidates_train + 1  # For training-specific operations
        self.K_max_test = max_candidates_test + 1  # For test-specific operations
        
        # Determine number of features (fixed minimal set)
        n_features = 4
        
        # Store feature names (fixed minimal set)
        self.feature_names = self.MINIMAL_FEATURES
        
        # Log simplified statistics
        if compound_info is not None and 'is_decoy' in compound_info.columns:
            n_decoys = compound_info['is_decoy'].sum()
            print(f"  Excluding {n_decoys} decoy compounds from training")
        print(f"  Model slots: {self.K_max} ({self.K_max_train} used in training, {self.K_max_test} in test)")
        
        # Build padded tensors for BOTH training and test
        # Both use self.K_max (the maximum) for consistent model architecture
        N = len(peak_candidates_train)
        # Training tensors
        X_train = np.zeros((N, self.K_max, n_features), dtype=np.float32)
        mask_train = np.zeros((N, self.K_max), dtype=bool)
        
        # Test tensors (same dimensions as training for model compatibility)
        X_test = np.zeros((N, self.K_max, n_features), dtype=np.float32)
        mask_test = np.zeros((N, self.K_max), dtype=bool)
        
        true_labels = np.array(peak_labels, dtype=np.int32)
        labels = np.full_like(true_labels, -1, dtype=np.int32)
        
        # Mappings for both sets
        peak_to_row = {pid: i for i, pid in enumerate(peak_ids)}
        row_to_candidates_train = []
        row_to_candidates_test = []
        
        # Fill padded tensors for both training and test
        for i, (train_data, test_data, s) in enumerate(zip(peak_candidates_train, peak_candidates_test, peak_species)):
            candidates_train, features_train = train_data
            candidates_test, features_test = test_data
            
            # Store mappings
            row_to_candidates_train.append([None] + candidates_train)  # None for null
            row_to_candidates_test.append([None] + candidates_test)  # None for null
            
            # Fill training tensors
            mask_train[i, 0] = True  # Null slot
            for k, feat in enumerate(features_train, start=1):
                X_train[i, k, :] = feat
                mask_train[i, k] = True
            
            # Fill test tensors
            mask_test[i, 0] = True  # Null slot
            for k, feat in enumerate(features_test, start=1):
                X_test[i, k, :] = feat
                mask_test[i, k] = True
        
        # Standardize features using TRAINING data statistics
        valid_features = []
        for i in range(N):
            for k in range(1, self.K_max):  # Skip null slot
                if mask_train[i, k]:
                    valid_features.append(X_train[i, k, :])
        
        if len(valid_features) > 0:
            valid_features = np.array(valid_features)
            self.feature_scaler.fit(valid_features)
            
            # Apply standardization to BOTH train and test tensors
            # Training tensors
            for i in range(N):
                for k in range(1, self.K_max):
                    if mask_train[i, k]:
                        X_train[i, k, :] = self.feature_scaler.transform(X_train[i, k, :].reshape(1, -1))[0]
            
            # Test tensors (using same scaler fitted on training data)
            for i in range(N):
                for k in range(1, self.K_max):
                    if mask_test[i, k]:
                        X_test[i, k, :] = self.feature_scaler.transform(X_test[i, k, :].reshape(1, -1))[0]
        
        # Optionally reveal a small seed set of labels
        if initial_labeled_fraction > 0:
            valid_idx = np.where(true_labels >= 0)[0]
            n_seed = max(1, int(len(valid_idx) * initial_labeled_fraction))
            rng = np.random.default_rng(self.random_seed if random_seed is None else random_seed)
            if len(valid_idx) > 0:
                seed_idx = rng.choice(valid_idx, size=min(n_seed, len(valid_idx)), replace=False)
                labels[seed_idx] = true_labels[seed_idx]
        
        # Store training pack with BOTH train and test tensors
        self.train_pack = {
            'X': X_train,  # Training features (no decoys)
            'mask': mask_train,  # Training mask
            'X_test': X_test,  # Test features (includes decoys)
            'mask_test': mask_test,  # Test mask
            'labels': labels,
            'true_labels': true_labels,
            'true_compounds': np.array(true_compounds_list, dtype=object),
            'peak_to_row': peak_to_row,
            'row_to_candidates': row_to_candidates_train,  # Training candidates (no decoys)
            'row_to_candidates_test': row_to_candidates_test,  # Test candidates (all)
            'peak_species': np.array(peak_species),
            'peak_ids': np.array(peak_ids),
            'n_compounds': n_compounds,
            'n_species': n_species
        }
        
        return self.train_pack
    
    def build_model(self) -> pm.Model:
        """
        Build hierarchical Bayesian model with learnable noise.
        """
        if not self.train_pack:
            raise ValueError("Training data not generated. Run generate_training_data first.")
        
        # Building model
        
        X = self.train_pack['X']
        mask = self.train_pack['mask']
        labels = self.train_pack['labels']
        peak_species = self.train_pack['peak_species']
        row_to_candidates = self.train_pack['row_to_candidates']
        
        N = X.shape[0]
        n_features = len(self.feature_names)
        
        with pm.Model() as model:
            # Priors - tighter for standardized features
            theta0 = pm.Normal('theta0', 0.0, 1.0)
            theta_features = pm.Normal('theta_features', 0.0, 1.0, shape=n_features)
            theta_null = pm.Normal('theta_null', -1.0, 1.0)  # Slight bias against null
            
            # Hierarchical uncertainty modeling
            sigma_logit = pm.HalfNormal('sigma_logit', 0.5)
            
            # Prepare data as PyMC constants
            X_tensor = pm.Data('X', X)
            mask_tensor = pm.Data('mask', mask)
            
            # Pre-compute presence priors
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
            
            # Compute features contribution
            feature_contrib = pm.math.sum(X_tensor * theta_features[None, None, :], axis=2)
            
            # Build mean logits
            null_logits = theta_null + presence_tensor[:, 0]
            candidate_logits = theta0 + feature_contrib[:, 1:] + presence_tensor[:, 1:]
            eta_mean = pm.math.concatenate([null_logits[:, None], candidate_logits], axis=1)
            
            # Apply mask
            eta_mean = pm.math.where(mask_tensor, eta_mean, -1e9)
            
            # Apply hierarchical uncertainty
            eta_noise = pm.Normal('eta_noise', 0, 1, shape=(N, self.K_max))
            eta = pm.Deterministic('eta', eta_mean + sigma_logit * eta_noise)
            
            # Ensure masked values stay very negative
            eta_final = pm.math.where(mask_tensor, eta, -1e9)
            
            # Softmax probabilities
            p = pm.Deterministic('p', pm.math.softmax(eta_final, axis=1))
            
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
        """Sample from the posterior using NUTS."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        print(f"\nSampling model with {chains} chains, {draws} samples each...")
        
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
    
    def predict_probs(self, use_test_candidates: bool = True) -> Dict[int, np.ndarray]:
        """Compute posterior predictive probabilities for each peak.
        
        Parameters
        ----------
        use_test_candidates : bool
            If True, compute probabilities for test candidates (includes decoys).
            If False, use training candidates (no decoys).
        """
        if self.trace is None:
            raise ValueError("No trace available. Run sample first.")
        
        # For test evaluation, we need to compute probabilities on test tensors
        if use_test_candidates and 'X_test' in self.train_pack:
            # Compute probabilities for test candidates
            X = self.train_pack['X_test']
            mask = self.train_pack['mask_test']
            K_max = self.K_max  # Using unified dimension
            
            # Get model parameters from trace
            post = self.trace.posterior
            theta_features_samples = post['theta_features'].values  # Shape: (chains, draws, n_features)
            theta0_samples = post['theta0'].values  # Shape: (chains, draws)
            theta_null_samples = post['theta_null'].values  # Shape: (chains, draws)
            sigma_logit_samples = post['sigma_logit'].values  # Shape: (chains, draws)
            
            # Compute logits for test features
            N = X.shape[0]
            chains, draws = theta_features_samples.shape[:2]
            
            # Compute presence priors for test candidates
            peak_species = self.train_pack['peak_species']
            log_pi_null = self.presence.log_prior_null()
            candidates_map = self.train_pack['row_to_candidates_test']
            
            # Build presence prior matrix for test candidates
            presence_matrix_test = np.zeros((N, self.K_max), dtype=float)
            for i in range(N):
                s = peak_species[i]
                candidates = candidates_map[i]
                log_pi_compounds = self.presence.log_prior_odds(s)
                
                presence_matrix_test[i, 0] = log_pi_null  # Null prior
                for k in range(1, self.K_max):
                    if mask[i, k] and k < len(candidates):
                        c = candidates[k]
                        if c is not None:
                            presence_matrix_test[i, k] = log_pi_compounds[c]
            
            # Streaming mean across chains and draws (vectorized over peaks)
            p_sum = np.zeros((N, self.K_max), dtype=float)
            count = 0
            rng = np.random.default_rng(self.random_seed)

            for chain in range(chains):
                for draw in range(draws):
                    theta_features = theta_features_samples[chain, draw]  # (n_features,)
                    theta0 = float(theta0_samples[chain, draw])
                    theta_null = float(theta_null_samples[chain, draw])
                    sigma_logit = float(sigma_logit_samples[chain, draw])

                    # Feature contribution (N, K_max)
                    feature_contrib = np.tensordot(X, theta_features, axes=([2], [0]))

                    # Logits with presence priors (vectorized)
                    logits = theta0 + feature_contrib + presence_matrix_test
                    logits[:, 0] = theta_null + presence_matrix_test[:, 0]

                    # Mask invalid slots
                    logits = np.where(mask, logits, -np.inf)

                    # Add hierarchical logit noise
                    if sigma_logit > 0:
                        noise = rng.standard_normal(size=logits.shape) * sigma_logit
                        logits = logits + noise

                    # Softmax across valid slots
                    m = np.max(logits, axis=1, keepdims=True)
                    exp_logits = np.exp(logits - m)
                    exp_logits = np.where(mask, exp_logits, 0.0)
                    denom = np.sum(exp_logits, axis=1, keepdims=True)
                    p = np.divide(exp_logits, denom, out=np.zeros_like(exp_logits), where=denom > 0)

                    p_sum += p
                    count += 1

            p_mean = p_sum / max(count, 1)
        else:
            # Use training probabilities (original behavior)
            post = self.trace.posterior
            p_samples = post['p'].values  # Shape: (chains, draws, N, K_max)
            p_mean = p_samples.mean(axis=(0, 1))  # Shape: (N, K_max)
            mask = self.train_pack['mask']
            candidates_map = self.train_pack['row_to_candidates']
        
        # Map to peak IDs
        peak_probs = {}
        for i, peak_id in enumerate(self.train_pack['peak_ids']):
            # Extract valid probabilities (where mask is True)
            mask_i = mask[i]
            valid_probs = p_mean[i, mask_i]
            
            # Renormalize (should already sum to 1, but ensure numerical stability)
            valid_probs = valid_probs / valid_probs.sum()
            
            peak_probs[int(peak_id)] = valid_probs
        
        return peak_probs
    
    def assign(self, prob_threshold: float = 0.5, eval_peak_ids: Optional[set] = None, 
               compound_info: Optional[pd.DataFrame] = None) -> AssignmentResults:
        """
        Assign compounds to peaks with many-to-one support.
        Now properly evaluates multiple peaks from the same compound as all correct.
        Also properly accounts for decoy compounds as false positives.
        
        Parameters
        ----------
        prob_threshold : float
            Probability threshold for assignment
        eval_peak_ids : Optional[set]
            Peak IDs to evaluate (if None, evaluate all)
        compound_info : Optional[pd.DataFrame]
            Compound information including 'is_decoy' flag
        """
        if self.trace is None:
            raise ValueError("No trace available. Run sample first.")
        
        # Silent - no printing in library code
        
        # Get probabilities using TEST candidates (includes decoys)
        peak_probs = self.predict_probs(use_test_candidates=True)
        
        # Use test candidate mappings if available
        if 'row_to_candidates_test' in self.train_pack:
            candidates_map = self.train_pack['row_to_candidates_test']
        else:
            candidates_map = self.train_pack['row_to_candidates']
        
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
                candidates = candidates_map[row]
                assignments[peak_id] = candidates[best_idx]
            else:
                # Either null or below threshold
                assignments[peak_id] = None
        
        # Get ground truth
        true_labels = self.train_pack.get('true_labels', self.train_pack['labels'])
        peak_ids = self.train_pack['peak_ids']
        # Use true compound ids independent of candidate presence
        true_compounds = self.train_pack.get('true_compounds')
        # Keep row_to_candidates for mapping indices to compound IDs where needed
        row_to_candidates = self.train_pack['row_to_candidates']
        train_labels = self.train_pack.get('labels', np.full_like(true_labels, -1))
        
        # Build mappings for many-to-one evaluation
        true_peaks_by_compound = defaultdict(list)
        pred_peaks_by_compound = defaultdict(list)
        
        # FIRST PASS: Build compound mappings from ALL peaks (for compound-level metrics)
        for i, peak_id in enumerate(peak_ids):
            pred = assignments.get(peak_id)
            true_comp = true_compounds[i] if true_compounds is not None else None
            if pd.notna(true_comp) if isinstance(true_comp, (np.floating, float)) else (true_comp is not None):
                true_comp = int(true_comp)
                true_peaks_by_compound[true_comp].append(peak_id)
            
            # Track predicted compounds
            if pred is not None:
                pred_peaks_by_compound[pred].append(peak_id)
        
        # SECOND PASS: Calculate peak-level metrics (test peaks only)
        tp = fp = fn = tn = 0
        all_probs = []
        all_correct = []
        
        eval_set = set(map(int, eval_peak_ids)) if eval_peak_ids is not None else set(map(int, peak_ids))
        
        for i, (peak_id, label) in enumerate(zip(peak_ids, true_labels)):
            if int(peak_id) not in eval_set:
                continue
            # Skip TRAINING peaks if they were labeled (evaluate test set only)
            if train_labels[i] >= 0:
                continue
            
            pred = assignments.get(peak_id)
            # Use true compound id directly (may be outside candidate set)
            tc = true_compounds[i] if true_compounds is not None else None
            true_comp = None if (tc is None or (isinstance(tc, float) and np.isnan(tc))) else int(tc)
            
            # CORRECTED Peak-level metrics for many-to-one
            if true_comp is None:
                # Noise peak
                if pred is None:
                    tn += 1  # Correctly identified as noise
                else:
                    fp += 1  # Incorrectly assigned to a compound
            else:
                # Real peak
                if pred == true_comp:
                    tp += 1  # Correctly assigned (many-to-one is OK!)
                elif pred is None:
                    fn += 1  # Missed assignment
                else:
                    fp += 1  # Wrong compound
                    fn += 1  # Also counts as missing the true compound
            
            # Calibration data
            all_probs.append(top_probs[peak_id])
            all_correct.append(1 if pred == true_comp else 0)
        
        # Peak-level metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # COMPOUND-LEVEL METRICS (PRIMARY)
        true_compounds_set = set(true_peaks_by_compound.keys())
        pred_compounds = set(pred_peaks_by_compound.keys())
        
        # Separate real vs decoy compounds if compound_info provided
        decoy_compounds = set()
        if compound_info is not None and 'is_decoy' in compound_info.columns:
            # Get decoy compound IDs from the compound_id column
            decoy_compounds = set(compound_info[compound_info['is_decoy']]['compound_id'].values)
            
            # Filter predicted compounds into real vs decoy
            pred_real = pred_compounds - decoy_compounds
            pred_decoys = pred_compounds & decoy_compounds
            
            # True positives: predicted real compounds that are in true_compounds
            correct_compounds = pred_real & true_compounds_set
            # False positives: predicted decoys + predicted real not in true_compounds
            false_positive_compounds = pred_decoys | (pred_real - true_compounds_set)
            # Missed compounds: true compounds not predicted
            missed_compounds = true_compounds_set - pred_compounds
            
            # Calculate metrics with decoy awareness
            total_predicted = len(pred_compounds)
            compound_precision = len(correct_compounds) / total_predicted if total_predicted > 0 else 0
            compound_recall = len(correct_compounds) / len(true_compounds_set) if true_compounds_set else 0
        else:
            # Original calculation (no decoy info)
            correct_compounds = pred_compounds & true_compounds_set
            false_positive_compounds = pred_compounds - true_compounds_set
            missed_compounds = true_compounds_set - pred_compounds
            
            compound_precision = len(correct_compounds) / len(pred_compounds) if pred_compounds else 0
            compound_recall = len(correct_compounds) / len(true_compounds_set) if true_compounds_set else 0
            pred_decoys = set()
        
        compound_f1 = 2 * compound_precision * compound_recall / (compound_precision + compound_recall) \
                      if (compound_precision + compound_recall) > 0 else 0
        
        # COVERAGE METRICS
        coverage_per_compound = {}
        for compound in true_compounds_set:
            true_peaks = set(true_peaks_by_compound[compound])
            pred_peaks = set(pred_peaks_by_compound.get(compound, []))
            coverage = len(true_peaks & pred_peaks) / len(true_peaks) if true_peaks else 0
            coverage_per_compound[compound] = coverage
        
        mean_coverage = np.mean(list(coverage_per_compound.values())) if coverage_per_compound else 0
        
        # Calculate calibration
        ece = self._calculate_ece(np.array(all_probs), np.array(all_correct))
        
        # Don't print here - let train.py handle all output
        
        return AssignmentResults(
            assignments=assignments,
            top_prob=top_probs,
            per_peak_probs=peak_probs,
            # Peak-level
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix={'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn},
            # Compound-level (PRIMARY)
            compound_precision=compound_precision,
            compound_recall=compound_recall,
            compound_f1=compound_f1,
            compound_metrics={
                'identified': len(correct_compounds),
                'total': len(true_compounds_set),
                'false_positives': len(false_positive_compounds),
                'missed': len(missed_compounds),
                'decoys_assigned': len(pred_decoys) if compound_info is not None else 0
            },
            # Coverage
            coverage_per_compound=coverage_per_compound,
            mean_coverage=mean_coverage,
            # Calibration
            ece=ece,
            # Groupings
            peaks_by_compound=dict(pred_peaks_by_compound),
            true_peaks_by_compound=dict(true_peaks_by_compound)
        )
    
    def _calculate_ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
        """Calculate Expected Calibration Error."""
        if len(probs) == 0 or len(labels) == 0:
            return 0.0
        
        probs = np.clip(probs, 1e-12, 1 - 1e-12)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0
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
        
        return ece
