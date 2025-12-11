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

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.preprocessing import StandardScaler

from .presence_prior import PresencePrior
from ..utils import RunMetadataError, compute_all_chemical_features, ensure_run_metadata
from ..utils import (
    ADDUCT_FEATURE_MAP as GLOBAL_ADDUCT_FEATURE_MAP,
    MATCHING_TRANSFORMS,
)


logger = logging.getLogger(__name__)


@dataclass
class AssignmentResults:
    """Container for peak assignment results (many-to-many by default)."""

    # Predictions
    assignments: Dict[int, List[int]]  # peak_id -> list of predicted compound_ids (may be empty)
    top_prob: Dict[int, float]  # peak_id -> max probability
    per_peak_probs: Dict[int, np.ndarray]  # peak_id -> [p_null, p_c1, ...]

    # Pair-based (micro) metrics over (peak, compound) pairs
    precision: float
    recall: float
    f1: float
    confusion_matrix: Dict[str, int]  # TP, FP, FN at pair level

    # Peak-level set metrics (macro)
    f1_macro: float
    jaccard_macro: float
    far_null: float
    tnr_null: float
    assignment_rate: float

    # Compound-level identification (decoy-aware)
    compound_precision: float
    compound_recall: float
    compound_f1: float
    compound_metrics: Dict[str, Any]

    # Coverage metrics
    coverage_per_compound: Dict[int, float]
    mean_coverage: float

    # Calibration
    ece: float  # Pairwise ECE (micro)
    ece_ovr: float  # Macro OVR-ECE across classes
    brier_ovr: float  # OVR Brier score across classes
    cardinality_mae: float  # Mean abs error of set sizes

    # Groupings for analysis
    peaks_by_compound: Dict[int, List[int]]
    true_peaks_by_compound: Dict[int, List[int]]


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

    # Shared transforms sourced from `ion_transforms` to keep parity with data generation
    ADDUCT_TRANSFORMS = MATCHING_TRANSFORMS
    ADDUCT_FEATURE_MAP = GLOBAL_ADDUCT_FEATURE_MAP

    # Define enhanced feature set with chemical relationships and adduct types
    MINIMAL_FEATURES = [
        "mass_err_ppm",  # Fundamental: mass accuracy
        "rt_z",  # Fundamental: RT accuracy
        "log_intensity",  # Peak quality
        "log_rt_uncertainty",  # Prediction confidence (important for AL)
        "has_isotope",  # Binary: has M+1 isotope peak
        "isotope_score",  # Quality of isotope pattern
        "n_adducts",  # Count of related adduct peaks
        "rt_cluster_size",  # Number of peaks at same RT
        "n_correlated",  # Peaks with correlated intensities
    ] + [feature for _, feature in ADDUCT_FEATURE_MAP]

    def __init__(
        self,
        mass_tolerance_ppm: float = 10.0,
        rt_window_k: float = 1.5,
        rt_std_floor: float = 0.05,
        random_seed: int = 42,
    ):
        """
        Initialize the hierarchical Bayesian model.

        Parameters
        ----------
        mass_tolerance_ppm : float
            Mass tolerance (ppm) for candidate generation
        rt_window_k : float
            RT window multiplier (in standard deviations)
        random_seed : int
            Random seed for reproducibility
        """
        self.mass_tolerance_ppm = mass_tolerance_ppm
        self.rt_window_k = rt_window_k
        self.rt_std_floor = rt_std_floor
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
        self.feature_diagnostics = pd.DataFrame()
        self.feature_scaler = StandardScaler()
        self.rt_predictions = {}
        self.run_covariate_columns: List[str] = []

    def compute_rt_predictions(
        self,
        trace_rt: az.InferenceData,
        n_species: int,
        n_compounds: int,
        *,
        peak_df: Optional[pd.DataFrame] = None,
        run_metadata: Optional[pd.DataFrame] = None,
        run_covariate_columns: Optional[Sequence[str]] = None,
        run_features: Optional[np.ndarray] = None,
        run_species: Optional[np.ndarray] = None,
        rt_model=None,
    ) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """
        Compute RT predictions with proper predictive variance.
        (Identical to softmax model)
        """
        logger.info("Computing RT predictions from RT posterior")

        try:
            run_meta = ensure_run_metadata(
                run_features,
                run_species=run_species,
                peak_df=run_metadata if run_metadata is not None else peak_df,
                covariate_columns=run_covariate_columns,
            )
        except RunMetadataError as err:
            raise ValueError(str(err)) from err

        run_features_arr = run_meta.features
        run_species_arr = run_meta.species
        self.run_covariate_columns = run_meta.covariate_columns

        if rt_model is None:
            raise ValueError("rt_model instance required to compute run-standardised predictions")

        species_idx = np.repeat(np.arange(n_species, dtype=int), n_compounds)
        compound_idx = np.tile(np.arange(n_compounds, dtype=int), n_species)

        representative_run = np.zeros(n_species, dtype=int)
        for s in range(n_species):
            mask = np.where(run_species_arr == s)[0]
            if mask.size == 0:
                raise ValueError(f"No runs available for species {s}")
            representative_run[s] = mask[0]
        run_idx = representative_run[species_idx]

        rt_mean, rt_std = rt_model.predict_new(
            species_idx=species_idx,
            compound_idx=compound_idx,
            run_idx=run_idx,
            run_features=run_features_arr,
        )

        rt_predictions = {}
        for idx, (s, m) in enumerate(zip(species_idx, compound_idx)):
            rt_predictions[(int(s), int(m))] = (float(rt_mean[idx]), float(rt_std[idx]))

        self.rt_predictions = rt_predictions
        logger.info("Computed RT predictions for %d species-compound pairs", len(rt_predictions))
        return rt_predictions

    def generate_training_data(
        self,
        peak_df: pd.DataFrame,
        compound_mass: np.ndarray,
        n_compounds: int,
        compound_info: Optional[pd.DataFrame] = None,
        init_presence: Optional[PresencePrior] = None,
        initial_labeled_fraction: float = 0.0,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate training data with minimal features.
        """
        if not self.rt_predictions:
            raise ValueError("RT predictions not computed. Run compute_rt_predictions first.")

        required_cols = {"peak_id", "species", "mass", "rt", "intensity", "true_compound"}
        missing = required_cols.difference(peak_df.columns)
        if missing:
            raise ValueError(
                "peak_df must contain columns 'peak_id', 'species', 'mass', 'rt', 'intensity', "
                f"'true_compound'; missing {sorted(missing)}"
            )

        if len(compound_mass) < n_compounds:
            raise ValueError("compound_mass length must be >= n_compounds")

        if not np.all(np.isfinite(peak_df[["mass", "rt", "intensity"]].to_numpy(dtype=float))):
            raise ValueError("Peak dataframe contains non-finite mass/rt/intensity values")

        logger.info("Generating assignment training data for %d peaks", len(peak_df))

        # Initialize presence prior if not provided
        n_species = len(np.unique(peak_df["species"]))
        if init_presence is None:
            self.presence = PresencePrior.init(n_species, n_compounds, smoothing=1.0)
        else:
            self.presence = init_presence

        # Collect candidates for each peak
        # We'll generate TWO sets: training (no decoys) and test (all compounds)
        peak_candidates_train = []  # Training candidates (no decoys)
        peak_candidates_test = []  # Test candidates (all compounds)
        peak_labels = []
        true_compounds_list = []
        peak_species = []
        peak_ids = []
        feature_records: List[Dict[str, Any]] = []

        null_label_counts = defaultdict(int)
        non_null_label_counts = defaultdict(int)

        for _, peak in peak_df.iterrows():
            peak_id = int(peak["peak_id"])
            s = int(peak["species"])
            true_comp = peak["true_compound"]
            if not pd.isna(true_comp):
                true_comp = int(true_comp)
            else:
                true_comp = None

            mz = peak["mass"]
            rt = peak["rt"]
            intensity = peak["intensity"]

            # Compute chemical features for this peak (once per peak)
            chemical_features = compute_all_chemical_features(mz, rt, intensity, peak_df, s)

            # Find candidates by mass - TRAINING SET (no decoys)
            candidates_train = []
            features_train = []

            for m in range(n_compounds):
                # Skip decoys during training to prevent data leakage
                if compound_info is not None and "is_decoy" in compound_info.columns:
                    if compound_info.iloc[m]["is_decoy"]:
                        continue

                # Check if peak could be ANY adduct of this compound
                matched_adduct = None
                min_mass_error = float("inf")
                best_theoretical_mass = None

                for adduct_name, mass_shift in self.ADDUCT_TRANSFORMS.items():
                    theoretical_mass = compound_mass[m] + mass_shift
                    if theoretical_mass < 0:  # Skip invalid masses
                        continue
                    mass_error_da = abs(mz - theoretical_mass)

                    mass_tolerance_da = self.mass_tolerance_ppm * max(theoretical_mass, 1e-6) * 1e-6

                    if mass_error_da <= mass_tolerance_da and mass_error_da < min_mass_error:
                        matched_adduct = adduct_name
                        min_mass_error = mass_error_da
                        best_theoretical_mass = theoretical_mass

                if matched_adduct is None:
                    continue  # No adduct form matches

                # Calculate mass error using the matched adduct mass
                mass_error_ppm = (mz - best_theoretical_mass) / best_theoretical_mass * 1e6

                # RT filtering
                rt_pred_mean, rt_pred_std = self.rt_predictions[(s, m)]

                rt_scale = max(rt_pred_std, self.rt_std_floor)
                rt_z = (rt - rt_pred_mean) / rt_scale

                if abs(rt_z) > self.rt_window_k:
                    continue

                # Build enhanced feature list with chemical relationships and adduct type indicators
                feat = [
                    mass_error_ppm,  # Core
                    rt_z,  # Core
                    np.log1p(intensity),  # Quality
                    np.log(rt_pred_std + 1e-6),  # Uncertainty
                    chemical_features["has_isotope"],  # Binary: has isotope
                    chemical_features["isotope_score"],  # Isotope quality
                    chemical_features["n_adducts"],  # Adduct count
                    chemical_features["rt_cluster_size"],  # RT clustering
                    chemical_features["n_correlated"],  # Correlated peaks
                ]
                feat.extend(float(matched_adduct == key) for key, _ in self.ADDUCT_FEATURE_MAP)

                candidates_train.append(m)
                features_train.append(feat)

                feature_records.append(
                    {
                        "peak_id": peak_id,
                        "species": s,
                        "candidate_id": m,
                        "is_true": float(true_comp == m),
                        **{name: value for name, value in zip(self.MINIMAL_FEATURES, feat)},
                    }
                )

            # Find candidates - TEST SET (all compounds including decoys)
            candidates_test = []
            features_test = []

            for m in range(n_compounds):
                # Include ALL compounds for test evaluation
                # Check if peak could be ANY adduct of this compound
                matched_adduct = None
                min_mass_error = float("inf")
                best_theoretical_mass = None

                for adduct_name, mass_shift in self.ADDUCT_TRANSFORMS.items():
                    theoretical_mass = compound_mass[m] + mass_shift
                    if theoretical_mass < 0:  # Skip invalid masses
                        continue
                    mass_error_da = abs(mz - theoretical_mass)

                    mass_tolerance_da = self.mass_tolerance_ppm * max(theoretical_mass, 1e-6) * 1e-6

                    if mass_error_da <= mass_tolerance_da and mass_error_da < min_mass_error:
                        matched_adduct = adduct_name
                        min_mass_error = mass_error_da
                        best_theoretical_mass = theoretical_mass

                if matched_adduct is None:
                    continue  # No adduct form matches

                # Calculate mass error using the matched adduct mass
                mass_error_ppm = (mz - best_theoretical_mass) / best_theoretical_mass * 1e6

                # RT filtering
                rt_pred_mean, rt_pred_std = self.rt_predictions[(s, m)]

                rt_scale = max(rt_pred_std, self.rt_std_floor)
                rt_z = (rt - rt_pred_mean) / rt_scale

                if abs(rt_z) > self.rt_window_k:
                    continue

                # Build same enhanced features for test candidates (match training order)
                feat = [
                    mass_error_ppm,
                    rt_z,
                    np.log1p(intensity),
                    np.log(rt_pred_std + 1e-6),
                    chemical_features["has_isotope"],
                    chemical_features["isotope_score"],
                    chemical_features["n_adducts"],
                    chemical_features["rt_cluster_size"],
                    chemical_features["n_correlated"],
                ]
                feat.extend(float(matched_adduct == key) for key, _ in self.ADDUCT_FEATURE_MAP)

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
        max_candidates_train = (
            max(len(cands[0]) for cands in peak_candidates_train) if peak_candidates_train else 0
        )
        max_candidates_test = (
            max(len(cands[0]) for cands in peak_candidates_test) if peak_candidates_test else 0
        )

        # Use the MAXIMUM of both for the model architecture
        # This ensures the model can handle both training and test candidates
        self.K_max = max(max_candidates_train, max_candidates_test) + 1  # +1 for null class
        self.K_max_train = max_candidates_train + 1  # For training-specific operations
        self.K_max_test = max_candidates_test + 1  # For test-specific operations

        # Determine number of features (enhanced set with chemical features)
        n_features = len(self.MINIMAL_FEATURES)  # Currently 16 features

        # Store feature names (fixed minimal set)
        self.feature_names = self.MINIMAL_FEATURES

        # Log simplified statistics
        if compound_info is not None and "is_decoy" in compound_info.columns:
            n_decoys = int(compound_info["is_decoy"].sum())
            if n_decoys:
                logger.info("Excluding %d decoy compounds from training", n_decoys)
        logger.info(
            "Model candidate slots: %d (train %d, test %d)",
            self.K_max,
            self.K_max_train,
            self.K_max_test,
        )

        # Build padded tensors for BOTH training and test
        # Both use self.K_max (the maximum) for consistent model architecture
        N = len(peak_candidates_train)
        # Training tensors
        X_train = np.zeros((N, self.K_max, n_features), dtype=np.float32)
        mask_train = np.zeros((N, self.K_max), dtype=bool)

        # Test tensors use the same dimensions as training
        X_test = np.zeros((N, self.K_max, n_features), dtype=np.float32)
        mask_test = np.zeros((N, self.K_max), dtype=bool)

        true_labels = np.array(peak_labels, dtype=np.int32)
        labels = np.full_like(true_labels, -1, dtype=np.int32)

        # Mappings for both sets
        peak_to_row = {pid: i for i, pid in enumerate(peak_ids)}
        row_to_candidates_train = []
        row_to_candidates_test = []

        # Fill padded tensors for both training and test
        for i, (train_data, test_data, s) in enumerate(
            zip(peak_candidates_train, peak_candidates_test, peak_species)
        ):
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
                        X_train[i, k, :] = self.feature_scaler.transform(
                            X_train[i, k, :].reshape(1, -1)
                        )[0]

            # Test tensors (using same scaler fitted on training data)
            for i in range(N):
                for k in range(1, self.K_max):
                    if mask_test[i, k]:
                        X_test[i, k, :] = self.feature_scaler.transform(
                            X_test[i, k, :].reshape(1, -1)
                        )[0]

        # Optionally reveal a small seed set of labels
        if initial_labeled_fraction > 0:
            valid_idx = np.where(true_labels >= 0)[0]
            n_seed = max(1, int(len(valid_idx) * initial_labeled_fraction))
            rng = np.random.default_rng(self.random_seed if random_seed is None else random_seed)
            if len(valid_idx) > 0:
                seed_idx = rng.choice(valid_idx, size=min(n_seed, len(valid_idx)), replace=False)
                labels[seed_idx] = true_labels[seed_idx]

        # Batch-positive presence prior updates (one-per-(species, compound))
        # - Increment alpha for each labeled positive compound once per species
        # - Increment null prior for labeled nulls
        updated_pairs = set()
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue  # unlabeled in batch
            s = peak_species[i]
            if lab == 0:
                null_label_counts[int(s)] += 1
            else:
                # labeled positive; map to compound id from training candidates
                if lab < len(row_to_candidates_train[i]):
                    c = row_to_candidates_train[i][lab]
                    if c is not None:
                        key = (int(s), int(c))
                        if key not in updated_pairs:
                            self.presence.update_positive(int(s), int(c), weight=1.0)
                            updated_pairs.add(key)
                non_null_label_counts[int(s)] += 1

        # Post-process presence priors with softened counts to prevent null domination
        for species_idx, count in null_label_counts.items():
            weight = float(np.log1p(count))
            if weight > 0:
                self.presence.update_null(weight=weight)

        for species_idx, count in non_null_label_counts.items():
            weight = float(np.log1p(count))
            if weight > 0:
                self.presence.update_not_null(weight=weight)

        # Cache raw (pre-standardized) feature diagnostics for downstream analysis
        if feature_records:
            self.feature_diagnostics = pd.DataFrame(feature_records)
        else:
            self.feature_diagnostics = pd.DataFrame(
                columns=["peak_id", "species", "candidate_id", "is_true", *self.MINIMAL_FEATURES]
            )

        # Store training pack with BOTH train and test tensors
        self.train_pack = {
            "X": X_train,  # Training features (no decoys)
            "mask": mask_train,  # Training mask
            "X_test": X_test,  # Test features (includes decoys)
            "mask_test": mask_test,  # Test mask
            "labels": labels,
            "true_labels": true_labels,
            "true_compounds": np.array(true_compounds_list, dtype=object),
            "peak_to_row": peak_to_row,
            "row_to_candidates": row_to_candidates_train,  # Training candidates (no decoys)
            "row_to_candidates_test": row_to_candidates_test,  # Test candidates (all)
            "peak_species": np.array(peak_species),
            "peak_ids": np.array(peak_ids),
            "n_compounds": n_compounds,
            "n_species": n_species,
        }

        return self.train_pack

    def build_model(self) -> pm.Model:
        """
        Build hierarchical Bayesian model with learnable noise.
        """
        if not self.train_pack:
            raise ValueError("Training data not generated. Run generate_training_data first.")

        # Building model

        X = self.train_pack["X"]
        mask = self.train_pack["mask"]
        labels = self.train_pack["labels"]
        peak_species = self.train_pack["peak_species"]
        row_to_candidates = self.train_pack["row_to_candidates"]

        N = X.shape[0]
        n_features = len(self.feature_names)

        with pm.Model() as model:
            # Priors - tighter for standardized features
            theta0 = pm.Normal("theta0", 0.0, 1.0)
            theta_features = pm.Normal("theta_features", 0.0, 1.0, shape=n_features)
            theta_null = pm.Normal("theta_null", -1.0, 1.0)  # Slight bias against null

            # Hierarchical uncertainty modeling
            sigma_logit = pm.HalfNormal("sigma_logit", 0.5)

            # Prepare data as PyMC constants
            X_tensor = pm.Data("X", X)
            mask_tensor = pm.Data("mask", mask)

            # Pre-compute presence priors
            log_pi_null = self.presence.log_prior_null()

            # Build presence prior matrix
            presence_matrix = np.zeros((N, self.K_max), dtype="float32")
            for i in range(N):
                s = peak_species[i]
                candidates = row_to_candidates[i]
                log_pi_compounds = self.presence.log_prior_prob(s)

                presence_matrix[i, 0] = log_pi_null  # Null prior
                for k in range(1, self.K_max):
                    if mask[i, k]:
                        c = candidates[k]
                        presence_matrix[i, k] = log_pi_compounds[c]

            presence_tensor = pm.Data("presence_priors", presence_matrix)

            # Compute features contribution
            feature_contrib = pm.math.sum(X_tensor * theta_features[None, None, :], axis=2)

            # Build mean logits
            null_logits = theta_null + presence_tensor[:, 0]
            candidate_logits = theta0 + feature_contrib[:, 1:] + presence_tensor[:, 1:]
            eta_mean = pm.math.concatenate([null_logits[:, None], candidate_logits], axis=1)

            # Apply mask
            eta_mean = pm.math.where(mask_tensor, eta_mean, -1e9)

            # Apply hierarchical uncertainty
            eta_noise = pm.Normal("eta_noise", 0, 1, shape=(N, self.K_max))
            eta = pm.Deterministic("eta", eta_mean + sigma_logit * eta_noise)

            # Ensure masked values stay very negative
            eta_final = pm.math.where(mask_tensor, eta, -1e9)

            # Softmax probabilities
            p = pm.Deterministic("p", pm.math.softmax(eta_final, axis=1))

            # Filter to only peaks with known labels for training
            train_mask = labels >= 0
            train_idx = np.where(train_mask)[0]

            if len(train_idx) > 0:
                # Categorical likelihood
                pm.Categorical("y", p=p[train_idx, :], observed=labels[train_idx])

            model.feature_names = self.feature_names

        self.model = model
        return model

    def sample(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        target_accept: float = 0.95,
        seed: int = 42,
    ) -> az.InferenceData:
        """Sample from the posterior using NUTS."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")

        logger.info(
            "Sampling peak assignment model with %d chains, %d draws, target_accept %.2f",
            chains,
            draws,
            target_accept,
        )

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=seed,
                progressbar=True,
            )

        return self.trace

    def _select_probability_inputs(
        self, use_test_candidates: bool
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Optional[int]]]]:
        if use_test_candidates and "X_test" in self.train_pack:
            X = self.train_pack["X_test"]
            mask = self.train_pack["mask_test"]
            candidates_map = self.train_pack["row_to_candidates_test"]
        else:
            X = self.train_pack["X"]
            mask = self.train_pack["mask"]
            candidates_map = self.train_pack["row_to_candidates"]
        return X, mask, candidates_map

    def _build_presence_matrix(
        self,
        peak_species: np.ndarray,
        candidates_map: List[List[Optional[int]]],
        mask: np.ndarray,
    ) -> np.ndarray:
        N = len(peak_species)
        presence_matrix = np.zeros((N, self.K_max), dtype=float)
        log_pi_null = self.presence.log_prior_null()
        for i in range(N):
            s = int(peak_species[i])
            log_pi_compounds = self.presence.log_prior_prob(s)
            presence_matrix[i, 0] = log_pi_null
            candidates = candidates_map[i]
            for k in range(1, self.K_max):
                if not mask[i, k]:
                    continue
                if k < len(candidates):
                    c = candidates[k]
                    if c is not None:
                        presence_matrix[i, k] = log_pi_compounds[c]
        return presence_matrix

    def _posterior_prob_samples(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        presence_matrix: np.ndarray,
        use_noise: bool,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        post = self.trace.posterior
        theta_features_samples = post["theta_features"].values
        theta0_samples = post["theta0"].values
        theta_null_samples = post["theta_null"].values
        sigma_logit_samples = post["sigma_logit"].values

        chains, draws = theta_features_samples.shape[:2]
        N = X.shape[0]
        prob_samples = np.zeros((chains * draws, N, self.K_max), dtype=float)

        if rng is None:
            rng = np.random.default_rng(self.random_seed)

        idx = 0
        for chain in range(chains):
            for draw in range(draws):
                theta_features = theta_features_samples[chain, draw]
                theta0 = float(np.asarray(theta0_samples[chain, draw]).item())
                theta_null = float(np.asarray(theta_null_samples[chain, draw]).item())
                sigma_logit = float(np.asarray(sigma_logit_samples[chain, draw]).item())

                feature_contrib = np.tensordot(X, theta_features, axes=([2], [0]))
                logits = theta0 + feature_contrib + presence_matrix
                logits[:, 0] = theta_null + presence_matrix[:, 0]
                logits = np.where(mask, logits, -np.inf)

                if use_noise and sigma_logit > 0:
                    noise = rng.standard_normal(size=logits.shape) * sigma_logit
                    logits = logits + noise

                m = np.max(logits, axis=1, keepdims=True)
                exp_logits = np.exp(logits - m)
                exp_logits = np.where(mask, exp_logits, 0.0)
                denom = np.sum(exp_logits, axis=1, keepdims=True)
                probs = np.divide(exp_logits, denom, out=np.zeros_like(exp_logits), where=denom > 0)

                prob_samples[idx] = probs
                idx += 1

        return prob_samples

    def predict_prob_samples(
        self,
        use_test_candidates: bool = True,
        max_samples: Optional[int] = None,
    ) -> Dict[int, np.ndarray]:
        """Return posterior probability samples per peak for MI-style acquisition."""
        if self.trace is None:
            raise ValueError("No trace available. Run sample first.")
        if self.presence is None:
            raise ValueError("Presence priors not initialized. Run generate_training_data first.")

        X, mask, candidates_map = self._select_probability_inputs(use_test_candidates)
        peak_species = self.train_pack["peak_species"]

        presence_matrix = self._build_presence_matrix(peak_species, candidates_map, mask)
        prob_samples = self._posterior_prob_samples(
            X,
            mask,
            presence_matrix,
            use_noise=True,
            rng=np.random.default_rng(self.random_seed),
        )

        if max_samples is not None and max_samples > 0 and max_samples < prob_samples.shape[0]:
            prob_samples = prob_samples[:max_samples]

        peak_probs: Dict[int, np.ndarray] = {}
        peak_ids = self.train_pack["peak_ids"]
        for i, peak_id in enumerate(peak_ids):
            mask_i = mask[i]
            samples = prob_samples[:, i, mask_i]
            denom = samples.sum(axis=1, keepdims=True)
            samples = np.divide(samples, denom, out=np.zeros_like(samples), where=denom > 0)
            peak_probs[int(peak_id)] = samples

        return peak_probs

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
        if self.presence is None:
            raise ValueError("Presence priors not initialized. Run generate_training_data first.")

        X, mask, candidates_map = self._select_probability_inputs(use_test_candidates)
        peak_species = self.train_pack["peak_species"]

        presence_matrix = self._build_presence_matrix(peak_species, candidates_map, mask)
        prob_samples = self._posterior_prob_samples(
            X,
            mask,
            presence_matrix,
            use_noise=True,
            rng=np.random.default_rng(self.random_seed),
        )
        p_mean = prob_samples.mean(axis=0)

        # Map to peak IDs
        peak_probs = {}
        for i, peak_id in enumerate(self.train_pack["peak_ids"]):
            # Extract valid probabilities (where mask is True)
            mask_i = mask[i]
            valid_probs = p_mean[i, mask_i]

            # Renormalize (should already sum to 1, but ensure numerical stability)
            valid_probs = valid_probs / valid_probs.sum()

            peak_probs[int(peak_id)] = valid_probs

        return peak_probs

    def assign(
        self,
        prob_threshold: float = 0.5,
        eval_peak_ids: Optional[set] = None,
        compound_info: Optional[pd.DataFrame] = None,
        max_predictions_per_peak: Optional[int] = 2,
    ) -> AssignmentResults:
        """
        Assign compounds to peaks with many-to-many support (default).
        Builds predicted sets per peak using a probability threshold and optional top-k cap,
        then computes pair-based micro metrics, peak-level macro metrics, compound-level
        identification, coverage, and calibration.

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
        if "row_to_candidates_test" in self.train_pack:
            candidates_map = self.train_pack["row_to_candidates_test"]
        else:
            candidates_map = self.train_pack["row_to_candidates"]

        # Make assignments (many-to-many)
        assignments: Dict[int, List[int]] = {}
        top_probs = {}

        for peak_id, probs in peak_probs.items():
            # Find argmax
            best_idx = np.argmax(probs)
            best_prob = probs[best_idx]

            top_probs[peak_id] = best_prob

            row = self.train_pack["peak_to_row"][peak_id]
            candidates = candidates_map[row]
            # Gather non-null candidates meeting threshold
            mask_row = (self.train_pack.get("mask_test", self.train_pack["mask"]))[row]
            valid_idx = np.where(mask_row)[0]
            cand_entries = []
            for j, k in enumerate(valid_idx):
                if k == 0:
                    continue
                c = candidates[k]
                if c is None:
                    continue
                p = float(probs[j])
                if p >= prob_threshold:
                    cand_entries.append((int(c), p))
            cand_entries.sort(key=lambda x: x[1], reverse=True)
            if max_predictions_per_peak is not None and max_predictions_per_peak > 0:
                cand_entries = cand_entries[:max_predictions_per_peak]
            assignments[peak_id] = [cid for cid, _ in cand_entries]

        # Get ground truth
        true_labels = self.train_pack.get("true_labels", self.train_pack["labels"])
        peak_ids = self.train_pack["peak_ids"]
        # Use true compound ids independent of candidate presence
        true_compounds = self.train_pack.get("true_compounds")
        train_labels = self.train_pack.get("labels", np.full_like(true_labels, -1))

        # Build mappings for compound-level evaluation
        true_peaks_by_compound = defaultdict(list)
        pred_peaks_by_compound = defaultdict(list)

        # Define evaluation set (IDs) for consistency across metrics
        eval_set = (
            set(map(int, eval_peak_ids)) if eval_peak_ids is not None else set(map(int, peak_ids))
        )

        # FIRST PASS: Build compound mappings from TEST-ONLY peaks in eval set
        # This mirrors peak-level filtering and avoids optimistic compound metrics.
        for i, peak_id in enumerate(peak_ids):
            # Restrict to held-out/test peaks and any explicit eval subset
            if int(peak_id) not in eval_set:
                continue
            if train_labels[i] >= 0:
                continue  # skip training-labeled peaks

            preds = assignments.get(peak_id, [])
            true_comp = true_compounds[i] if true_compounds is not None else None
            if (
                pd.notna(true_comp)
                if isinstance(true_comp, (np.floating, float))
                else (true_comp is not None)
            ):
                true_comp = int(true_comp)
                true_peaks_by_compound[true_comp].append(peak_id)

            # Track predicted compounds
            for pred in preds:
                pred_peaks_by_compound[int(pred)].append(peak_id)

        # SECOND PASS: Build pairwise arrays and peak-level sets (test peaks only)
        tp = fp = fn = 0
        pair_probs: List[float] = []
        pair_labels: List[int] = []
        ovr_by_class_probs: Dict[int, List[float]] = {}
        ovr_by_class_labels: Dict[int, List[int]] = {}
        peak_f1s: List[float] = []
        peak_jaccards: List[float] = []
        null_flags: List[int] = []
        assigned_flags: List[int] = []

        # Reuse the same eval_set defined above for peak-level metrics

        for i, (peak_id, label) in enumerate(zip(peak_ids, true_labels)):
            if int(peak_id) not in eval_set:
                continue
            # Skip TRAINING peaks if they were labeled (evaluate test set only)
            if train_labels[i] >= 0:
                continue

            preds = set(assignments.get(peak_id, []))
            tc = true_compounds[i] if true_compounds is not None else None
            true_set = set()
            if isinstance(tc, (list, tuple, np.ndarray)):
                true_set = {int(x) for x in tc if not (isinstance(x, float) and np.isnan(x))}
            else:
                if not (tc is None or (isinstance(tc, float) and np.isnan(tc))):
                    true_set = {int(tc)}

            # Peak-level set metrics
            inter = len(preds & true_set)
            prec_i = inter / len(preds) if len(preds) > 0 else (1.0 if len(true_set) == 0 else 0.0)
            rec_i = (
                inter / len(true_set) if len(true_set) > 0 else (1.0 if len(preds) == 0 else 0.0)
            )
            f1_i = (
                (2 * prec_i * rec_i / (prec_i + rec_i))
                if (prec_i + rec_i) > 0
                else (1.0 if len(preds) == 0 and len(true_set) == 0 else 0.0)
            )
            union = len(preds | true_set)
            jacc_i = (inter / union) if union > 0 else 1.0
            peak_f1s.append(f1_i)
            peak_jaccards.append(jacc_i)
            null_flags.append(1 if len(true_set) == 0 else 0)
            assigned_flags.append(1 if len(preds) > 0 else 0)

            # Pair universe for this peak: all candidate classes plus ensure true classes included
            mask_i = (self.train_pack.get("mask_test", self.train_pack["mask"]))[i]
            candidates_eval = self.train_pack.get(
                "row_to_candidates_test", self.train_pack["row_to_candidates"]
            )[i]
            valid_idx = np.where(mask_i)[0]
            # Map probs for valid non-null slots
            probs_vec = peak_probs[peak_id]
            prob_map = {}
            for j, k in enumerate(valid_idx):
                if k == 0:
                    continue
                c = candidates_eval[k]
                if c is None:
                    continue
                prob_map[int(c)] = float(probs_vec[j])
            U_i = set(prob_map.keys()) | true_set
            for c in U_i:
                y = 1 if c in true_set else 0
                yhat = 1 if c in preds else 0
                p = prob_map.get(c, 0.0)
                if yhat == 1 and y == 1:
                    tp += 1
                elif yhat == 1 and y == 0:
                    fp += 1
                elif yhat == 0 and y == 1:
                    fn += 1
                pair_probs.append(p)
                pair_labels.append(y)
                ovr_by_class_probs.setdefault(int(c), []).append(p)
                ovr_by_class_labels.setdefault(int(c), []).append(y)

        # Pair-based micro metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Peak-level macro metrics and null detection
        f1_macro = float(np.mean(peak_f1s)) if peak_f1s else 0.0
        jaccard_macro = float(np.mean(peak_jaccards)) if peak_jaccards else 0.0
        true_null_mask = np.array(null_flags, dtype=bool)
        assigned_arr = np.array(assigned_flags, dtype=bool)
        if true_null_mask.any():
            far_null = float(np.mean(assigned_arr[true_null_mask]))
            tnr_null = 1.0 - far_null
        else:
            far_null = 0.0
            tnr_null = 1.0
        assignment_rate = float(np.mean(assigned_arr)) if assigned_arr.size > 0 else 0.0

        # COMPOUND-LEVEL METRICS (PRIMARY)
        true_compounds_set = set(true_peaks_by_compound.keys())
        pred_compounds = set(pred_peaks_by_compound.keys())

        # Separate real vs decoy compounds if compound_info provided
        decoy_compounds = set()
        if compound_info is not None and "is_decoy" in compound_info.columns:
            # Get decoy compound IDs from the compound_id column
            decoy_compounds = set(compound_info[compound_info["is_decoy"]]["compound_id"].values)

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
            compound_precision = (
                len(correct_compounds) / total_predicted if total_predicted > 0 else 0
            )
            compound_recall = (
                len(correct_compounds) / len(true_compounds_set) if true_compounds_set else 0
            )
        else:
            # Original calculation (no decoy info)
            correct_compounds = pred_compounds & true_compounds_set
            false_positive_compounds = pred_compounds - true_compounds_set
            missed_compounds = true_compounds_set - pred_compounds

            compound_precision = (
                len(correct_compounds) / len(pred_compounds) if pred_compounds else 0
            )
            compound_recall = (
                len(correct_compounds) / len(true_compounds_set) if true_compounds_set else 0
            )
            pred_decoys = set()

        compound_f1 = (
            2 * compound_precision * compound_recall / (compound_precision + compound_recall)
            if (compound_precision + compound_recall) > 0
            else 0
        )

        # COVERAGE METRICS
        coverage_per_compound = {}
        for compound in true_compounds_set:
            true_peaks = set(true_peaks_by_compound[compound])
            pred_peaks = set(pred_peaks_by_compound.get(compound, []))
            coverage = len(true_peaks & pred_peaks) / len(true_peaks) if true_peaks else 0
            coverage_per_compound[compound] = coverage

        mean_coverage = (
            np.mean(list(coverage_per_compound.values())) if coverage_per_compound else 0
        )

        # Calibration
        ece = (
            self._calculate_ece(np.array(pair_probs), np.array(pair_labels)) if pair_probs else 0.0
        )
        ece_ovr_vals = []
        for cls, probs_list in ovr_by_class_probs.items():
            labels_list = ovr_by_class_labels.get(cls, [])
            if len(probs_list) > 0 and len(labels_list) > 0:
                ece_cls = self._calculate_ece(np.array(probs_list), np.array(labels_list))
                ece_ovr_vals.append(ece_cls)
        ece_ovr = float(np.mean(ece_ovr_vals)) if ece_ovr_vals else 0.0
        brier_ovr = (
            float(np.mean([(p - y) ** 2 for p, y in zip(pair_probs, pair_labels)]))
            if pair_probs
            else 0.0
        )
        # Cardinality calibration
        # true_set size currently at most 1 in synthetic generation; keep generic formulation
        cardinality_mae = (
            float(
                np.mean(
                    [
                        abs(
                            len(assignments.get(int(pid), []))
                            - (0 if (tc is None or (isinstance(tc, float) and np.isnan(tc))) else 1)
                        )
                        for pid, tc in zip(peak_ids, true_compounds)
                    ]
                )
            )
            if len(peak_ids) > 0
            else 0.0
        )

        # Don't print here - let train.py handle all output

        return AssignmentResults(
            assignments=assignments,
            top_prob=top_probs,
            per_peak_probs=peak_probs,
            # Pair-based micro
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix={"TP": tp, "FP": fp, "FN": fn},
            # Peak-level macro
            f1_macro=f1_macro,
            jaccard_macro=jaccard_macro,
            far_null=far_null,
            tnr_null=tnr_null,
            assignment_rate=assignment_rate,
            # Compound-level (PRIMARY)
            compound_precision=compound_precision,
            compound_recall=compound_recall,
            compound_f1=compound_f1,
            compound_metrics={
                "identified": len(correct_compounds),
                "total": len(true_compounds_set),
                "false_positives": len(false_positive_compounds),
                "missed": len(missed_compounds),
                "decoys_assigned": len(pred_decoys) if compound_info is not None else 0,
            },
            # Coverage
            coverage_per_compound=coverage_per_compound,
            mean_coverage=mean_coverage,
            # Calibration
            ece=ece,
            ece_ovr=ece_ovr,
            brier_ovr=brier_ovr,
            cardinality_mae=cardinality_mae,
            # Groupings
            peaks_by_compound=dict(pred_peaks_by_compound),
            true_peaks_by_compound=dict(true_peaks_by_compound),
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
