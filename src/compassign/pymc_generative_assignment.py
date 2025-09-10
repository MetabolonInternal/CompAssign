"""
Generative, hierarchical PyMC model for peak assignment with marginalized responsibilities.

Implements a fully continuous mixture with:
- Logistic–normal presence prior with partial pooling over species and compounds
- Robust Student-T likelihoods for mass and RT per candidate
- Explicit null/background component (broad heavy-tailed baseline)
- Vectorized marginalization via logsumexp (no discrete z sampling)

This mirrors the design in docs/pymc_generative_peak_assignment.md (Phase 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


@dataclass
class GenerativeAssignmentResults:
    """Container for generative peak assignment results."""

    assignments: Dict[int, Optional[int]]  # peak_id -> compound_id or None
    top_prob: Dict[int, float]  # peak_id -> max probability
    per_peak_probs: Dict[int, np.ndarray]  # peak_id -> [p_null, p_c1, ...]
    precision: float
    recall: float
    f1: float
    confusion_matrix: Dict[str, int]  # TP, FP, TN, FN
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error


class GenerativeAssignmentModel:
    """
    Generative, hierarchical peak assignment model with marginalized responsibilities.

    API intentionally echoes the softmax model where helpful, but does not aim
    for strict parity. Provides predict_probs() and assign() for evaluation and
    active learning integration.
    """

    def __init__(
        self,
        mass_tolerance: float = 0.005,
        rt_window_k: float = 3.0,
        random_seed: int = 42,
    ):
        self.mass_tolerance = mass_tolerance
        self.rt_window_k = rt_window_k
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # Data + model state
        self.model: Optional[pm.Model] = None
        self.trace: Optional[az.InferenceData] = None
        self.train_pack: Dict[str, Any] = {}
        self.K_max: int = 0
        self.rt_predictions: Dict[Tuple[int, int], Tuple[float, float]] = {}

    # ---- RT predictions (reused logic) -------------------------------------------------
    def compute_rt_predictions(
        self,
        trace_rt: az.InferenceData,
        n_species: int,
        n_compounds: int,
        descriptors: np.ndarray,
        internal_std: np.ndarray,
        rt_model: Optional[Any] = None,
    ) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """
        Compute RT predictions with proper predictive variance using RT posterior.
        Returns dict[(species, compound)] -> (mean, std).
        """
        print("Computing RT predictions from posterior...")

        post = trace_rt.posterior

        def flat(x):
            v = x.values
            return v.reshape(-1, *v.shape[2:])

        mu0 = flat(post["mu0"])  # [S]
        beta = flat(post["beta"])  # [S, F]
        gamma = flat(post["gamma"])  # [S]
        sp = flat(post["species_eff"])  # [S, n_species]
        cp = flat(post["compound_eff"])  # [S, n_compounds]
        sigy = flat(post["sigma_y"])  # [S]

        # Standardize using the RT model's parameters if available
        if rt_model is not None and hasattr(rt_model, "_desc_mean"):
            desc_std = (descriptors - rt_model._desc_mean) / (rt_model._desc_std + 1e-8)
            is_std = (internal_std - rt_model._is_mean) / (rt_model._is_std + 1e-8)
        else:
            # Fallback: use raw values (less ideal)
            desc_std = descriptors
            is_std = internal_std

        rt_predictions: Dict[Tuple[int, int], Tuple[float, float]] = {}

        for s in range(n_species):
            sp_s = sp[:, s]
            for m in range(n_compounds):
                loc = mu0 + sp_s + cp[:, m] + (beta @ desc_std[m]) + gamma * is_std[s]
                var = np.var(loc, ddof=1) + np.mean(sigy ** 2)
                rt_predictions[(s, m)] = (float(np.mean(loc)), float(np.sqrt(max(var, 1e-12))))

        self.rt_predictions = rt_predictions
        print(f"  Computed predictions for {len(rt_predictions)} species-compound pairs")
        return rt_predictions

    # ---- Data preparation ---------------------------------------------------------------
    def generate_training_data(
        self,
        peak_df: pd.DataFrame,
        compound_mass: np.ndarray,
        n_compounds: int,
        initial_labeled_fraction: float = 0.0,
        initial_labeled_n: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Construct padded (N, K_max) tensors for candidates + null per peak.
        Stores minimal features X for AL heuristics; not used by the model.
        """
        if not self.rt_predictions:
            raise ValueError("RT predictions not computed. Run compute_rt_predictions first.")

        print("\nGenerating GENERATIVE training data...")
        print(f"  Mass tolerance: ±{self.mass_tolerance} Da")
        print(f"  RT window: ±{self.rt_window_k}σ")
        print(f"  Including NULL component per peak")

        # Summary values for null background (broad heavy tails)
        # Use observed peak stats for centering
        mass_obs_all = peak_df["mass"].values.astype(float)
        rt_obs_all = peak_df["rt"].values.astype(float)
        mass_null_mu = float(np.median(mass_obs_all)) if len(mass_obs_all) else 0.0
        rt_null_mu = float(np.median(rt_obs_all)) if len(rt_obs_all) else 0.0

        # Track species medians for a simple intensity feature
        species_medians = peak_df.groupby("species")["intensity"].median()

        peak_candidates: List[Tuple[List[int], List[List[float]]]] = []
        peak_labels: List[int] = []
        peak_species: List[int] = []
        peak_ids: List[int] = []

        for _, peak in peak_df.iterrows():
            peak_id = int(peak["peak_id"]) if "peak_id" in peak else int(_)
            s = int(peak["species"]) if "species" in peak else 0
            true_comp = peak.get("true_compound", None)
            if not pd.isna(true_comp):
                true_comp = int(true_comp)
            else:
                true_comp = None

            mz = float(peak["mass"]) if "mass" in peak else float(peak["mz"])  # alias support
            rt = float(peak["rt"]) if "rt" in peak else 0.0
            intensity = float(peak.get("intensity", 0.0))
            median_intensity = float(species_medians.get(s, 1.0))

            candidates: List[int] = []
            # Minimal features for AL embedding: mass_err_ppm, rt_z, log_intensity
            features: List[List[float]] = []

            for m in range(n_compounds):
                mass_error_da = abs(mz - compound_mass[m])
                if mass_error_da > self.mass_tolerance:
                    continue

                mass_err_ppm = (mz - compound_mass[m]) / max(compound_mass[m], 1e-9) * 1e6

                rt_pred_mean, rt_pred_std = self.rt_predictions[(s, m)]
                rt_sigma = max(rt_pred_std, 1e-2)
                rt_z = (rt - rt_pred_mean) / rt_sigma
                if abs(rt_z) > self.rt_window_k:
                    continue

                log_intensity = float(np.log1p(intensity))
                rel_intensity = float(np.log1p(intensity / median_intensity)) if median_intensity > 0 else 0.0

                feat = [float(mass_err_ppm), float(rt_z), log_intensity, rel_intensity]
                candidates.append(m)
                features.append(feat)

            peak_candidates.append((candidates, features))
            peak_species.append(s)
            peak_ids.append(peak_id)

            # Determine label (0 for null, 1+ for candidate position)
            if true_comp is None:
                peak_labels.append(0)
            elif true_comp in candidates:
                peak_labels.append(candidates.index(true_comp) + 1)
            else:
                peak_labels.append(-1)

        if not peak_candidates:
            raise ValueError("No peaks to process. Check data generation.")

        max_candidates = max(len(cands[0]) for cands in peak_candidates) if peak_candidates else 0
        self.K_max = max_candidates + 1  # +1 for null
        print("\nCandidate statistics:")
        print(f"  Number of peaks: {len(peak_candidates)}")
        print(f"  Max candidates per peak: {self.K_max - 1} (+ null)")

        # Build padded tensors
        N = len(peak_candidates)
        X = np.zeros((N, self.K_max, 4), dtype=np.float32)  # minimal features for AL
        mask = np.zeros((N, self.K_max), dtype=bool)
        species_idx = np.array(peak_species, dtype=np.int32)
        labels = np.full((N,), -1, dtype=np.int32)
        true_labels = np.array(peak_labels, dtype=np.int32)

        # Observations
        mass_obs = np.zeros((N, self.K_max), dtype=np.float32)
        rt_obs = np.zeros((N, self.K_max), dtype=np.float32)
        comp_idx = -np.ones((N, self.K_max), dtype=np.int32)

        # Mappings
        peak_to_row = {pid: i for i, pid in enumerate(peak_ids)}
        row_to_candidates: List[List[Optional[int]]] = []

        # Fill tensors
        for i, ((cands, feats), s) in enumerate(zip(peak_candidates, peak_species)):
            # Null slot always valid
            mask[i, 0] = True
            row_to_candidates.append([None] + cands)

            # Observed peak values (same across candidate slots)
            mz = float(peak_df.iloc[i]["mass"]) if "mass" in peak_df.columns else float(peak_df.iloc[i]["mz"])
            rt = float(peak_df.iloc[i]["rt"]) if "rt" in peak_df.columns else 0.0
            mass_obs[i, 0] = mz
            rt_obs[i, 0] = rt

            for k, feat in enumerate(feats, start=1):
                X[i, k, :] = np.array(feat, dtype=np.float32)
                mask[i, k] = True
                mass_obs[i, k] = mz
                rt_obs[i, k] = rt
                comp_idx[i, k] = cands[k - 1]

        # Optionally seed a small set of labels
        if initial_labeled_fraction > 0 and initial_labeled_n is None:
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

        # Store pack
        self.train_pack = {
            "X": X,
            "mask": mask,
            "labels": labels,
            "true_labels": true_labels,
            "species_idx": species_idx,
            "peak_to_row": peak_to_row,
            "row_to_candidates": row_to_candidates,
            "peak_ids": np.array(peak_ids),
            "mass_obs": mass_obs,
            "rt_obs": rt_obs,
            "compound_idx": comp_idx,
            "mass_null_mu": mass_null_mu,
            "rt_null_mu": rt_null_mu,
            "n_compounds": n_compounds,
            "compound_mass": np.array(compound_mass, dtype=float),
        }

        return self.train_pack

    # ---- Model building and sampling ---------------------------------------------------
    def build_model(self) -> pm.Model:
        if not self.train_pack:
            raise ValueError("Training data not generated. Run generate_training_data first.")

        print("\nBuilding GENERATIVE model")
        X = self.train_pack["X"]
        mask = self.train_pack["mask"]
        labels = self.train_pack["labels"]
        species_idx = self.train_pack["species_idx"]
        comp_idx = self.train_pack["compound_idx"]
        mass_obs = self.train_pack["mass_obs"]
        rt_obs = self.train_pack["rt_obs"]
        compound_mass = self.train_pack["compound_mass"]
        n_compounds = int(self.train_pack["n_compounds"])
        N = X.shape[0]
        K_max = X.shape[1]
        
        print(f"  Data shapes: N={N}, K_max={K_max}, n_compounds={n_compounds}")

        # Prepare predicted RT means for candidates aligned with comp_idx
        # Create arrays of shape (N, K_max) for rt_pred_mean and rt_pred_std
        rt_mu_pred = np.zeros((N, K_max), dtype=np.float32)
        rt_sigma_pred = np.zeros((N, K_max), dtype=np.float32)
        for i in range(N):
            s = int(species_idx[i])
            for k in range(1, K_max):  # candidates only
                c = int(comp_idx[i, k])
                if c >= 0:
                    mu, sd = self.rt_predictions[(s, c)]
                    rt_mu_pred[i, k] = mu
                    rt_sigma_pred[i, k] = max(sd, 1e-2)

        with pm.Model() as model:
            # Presence logits: logistic–normal with partial pooling
            alpha = pm.Normal("alpha", 0.0, 1.0)
            alpha_null = pm.Normal("alpha_null", 0.0, 1.0)
            tau_s = pm.HalfNormal("tau_s", 1.0)
            tau_c = pm.HalfNormal("tau_c", 1.0)
            a_s = pm.Normal("a_s", 0.0, tau_s, shape=int(species_idx.max()) + 1)
            b_c = pm.Normal("b_c", 0.0, tau_c, shape=n_compounds)

            # RT species drift and scales
            delta_s = pm.Normal("delta_s", 0.0, 0.5, shape=int(species_idx.max()) + 1)
            sigma_rt = pm.HalfNormal("sigma_rt", 0.5)
            nu_rt = 5

            # Mass scale
            sigma_m = pm.HalfNormal("sigma_m", 0.01)
            nu_m = 5

            # Null/background: broad heavy-tailed baseline
            mass_null_mu = float(self.train_pack["mass_null_mu"])
            rt_null_mu = float(self.train_pack["rt_null_mu"])
            sigma_m_null = pm.HalfNormal("sigma_m_null", 0.1)
            sigma_rt_null = pm.HalfNormal("sigma_rt_null", 1.0)
            nu_null = 3

            # Data containers
            mask_t = pm.Data("mask", mask)
            species_t = pm.Data("species_idx", species_idx)
            comp_idx_t = pm.Data("compound_idx", comp_idx)
            mass_obs_t = pm.Data("mass_obs", mass_obs)
            rt_obs_t = pm.Data("rt_obs", rt_obs)
            rt_mu_pred_t = pm.Data("rt_mu_pred", rt_mu_pred)
            rt_sigma_pred_t = pm.Data("rt_sigma_pred", rt_sigma_pred)
            compound_mass_t = pm.Data("compound_mass", compound_mass)

            # Build presence logits and likelihoods with clone() to avoid stalling
            import pytensor.tensor as pt
            
            # Build presence logits per slot
            wlog = pm.math.ones((N, K_max)) * (-1e9)
            # Null slot logits
            wlog = pt.set_subtensor(wlog[:, 0], alpha_null)
            
            # Use clone() to break graph dependencies - this is the key fix
            wlog = wlog.clone()
            
            # Candidates
            ii, kk = np.where(mask & (comp_idx >= 0))
            if len(ii) > 0:
                # For PyTensor advanced indexing
                ii_t = pm.Data("ii", ii)
                kk_t = pm.Data("kk", kk)
                sp_for_slot = species_t[ii_t]
                comp_for_slot = comp_idx_t[ii_t, kk_t]
                cand_logits = alpha + a_s[sp_for_slot] + b_c[comp_for_slot]
                wlog = pt.set_subtensor(wlog[ii_t, kk_t], cand_logits)
            
            # Build log-likelihood per slot
            loglik = pm.math.zeros((N, K_max)) + (-1e9)
            
            if len(ii) > 0:
                # Clone to break dependencies
                loglik = loglik.clone()
                # Candidate mass log-lik
                cand_mass_mu = compound_mass_t[comp_for_slot]
                cand_mass_obs = mass_obs_t[ii_t, kk_t]
                mass_ll_cand = pm.logp(pm.StudentT.dist(nu_m, mu=cand_mass_mu, sigma=sigma_m), cand_mass_obs)
                
                # Candidate RT log-lik
                rt_mu_slot = rt_mu_pred_t[ii_t, kk_t] + delta_s[sp_for_slot]
                rt_obs_slot = rt_obs_t[ii_t, kk_t]
                rt_sd_slot = pm.math.maximum(rt_sigma_pred_t[ii_t, kk_t], 1e-2)
                rt_sigma_eff = pm.math.sqrt(rt_sd_slot**2 + sigma_rt**2)
                rt_ll_cand = pm.logp(pm.StudentT.dist(nu_rt, mu=rt_mu_slot, sigma=rt_sigma_eff), rt_obs_slot)
                
                cand_ll = mass_ll_cand + rt_ll_cand
                loglik = pt.set_subtensor(loglik[ii_t, kk_t], cand_ll)
            
            # Clone again before null likelihood
            loglik = loglik.clone()
            
            # Null log-likelihood per slot (k=0)
            mass_ll_null = pm.logp(
                pm.StudentT.dist(nu_null, mu=mass_null_mu, sigma=sigma_m_null), mass_obs_t[:, 0]
            )
            rt_ll_null = pm.logp(
                pm.StudentT.dist(nu_null, mu=rt_null_mu, sigma=sigma_rt_null), rt_obs_t[:, 0]
            )
            null_ll = mass_ll_null + rt_ll_null
            loglik = pt.set_subtensor(loglik[:, 0], null_ll)
            
            # Mask invalid slots
            very_neg = -1e9
            wlog_masked = pm.math.where(mask_t, wlog, very_neg)
            loglik_masked = pm.math.where(mask_t, loglik, very_neg)
            
            # Marginalize over assignments per peak
            total = pm.math.logsumexp(wlog_masked + loglik_masked, axis=1)
            pm.Potential("mixture_ll", pm.math.sum(total))
            
            # Deterministic responsibilities
            r = pm.Deterministic("r", pm.math.softmax(wlog_masked + loglik_masked, axis=1))

            # Optional: supervised labels if provided
            train_idx = np.where(labels >= 0)[0]
            if len(train_idx) > 0:
                y_obs = labels[train_idx]
                r_train = r[train_idx]
                pm.Categorical("y", p=r_train, observed=y_obs)

        print("  Model built successfully!")
        self.model = model
        return model

    def sample(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: Optional[int] = None,
        target_accept: float = 0.9,
        max_treedepth: int = 12,
        random_seed: Optional[int] = None,
        init: str = "adapt_diag",
    ) -> az.InferenceData:
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        sample_kwargs = {
            "draws": draws,
            "tune": tune,
            "target_accept": target_accept,
            "max_treedepth": max_treedepth,
            "random_seed": self.random_seed if random_seed is None else random_seed,
            "init": init,
            "progressbar": True,
            "return_inferencedata": True,
        }
        if chains is not None:
            sample_kwargs["chains"] = chains

        with self.model:
            self.trace = pm.sample(**sample_kwargs)
        return self.trace

    # ---- Predictions and assignments ----------------------------------------------------
    def predict_probs(self) -> Dict[int, np.ndarray]:
        """Posterior mean responsibilities per peak, as dict peak_id -> probs array."""
        if self.trace is None:
            # Fallback: compute heuristic responsibilities from likelihoods under default params
            return self._predict_probs_prior_mode()

        r = self.trace.posterior["r"].values  # (chains, draws, N, K_max)
        r_mean = r.mean(axis=(0, 1))

        mask = self.train_pack["mask"]
        peak_ids = self.train_pack["peak_ids"]
        probs: Dict[int, np.ndarray] = {}
        for i, pid in enumerate(peak_ids):
            pmask = mask[i]
            pvec = r_mean[i].copy()
            # Zero out invalid slots and renormalize
            pvec[~pmask] = 0.0
            s = float(pvec.sum())
            if s <= 0:
                # default to null if no valid slots (shouldn't happen with null always valid)
                pvec[:] = 0.0
                pvec[0] = 1.0
            else:
                pvec = pvec / s
            probs[int(pid)] = pvec
        return probs

    def assign(self, prob_threshold: float = 0.75, eval_peak_ids: Optional[set] = None) -> GenerativeAssignmentResults:
        peak_probs = self.predict_probs()
        assignments: Dict[int, Optional[int]] = {}
        top_probs: Dict[int, float] = {}

        for peak_id, probs in peak_probs.items():
            k_best = int(np.argmax(probs))
            p_best = float(probs[k_best])
            top_probs[peak_id] = p_best
            if k_best > 0 and p_best >= prob_threshold:
                row = self.train_pack["peak_to_row"][int(peak_id)]
                candidates = self.train_pack["row_to_candidates"][row]
                assignments[int(peak_id)] = int(candidates[k_best]) if candidates[k_best] is not None else None
            else:
                assignments[int(peak_id)] = None

        # Metrics
        true_labels = self.train_pack.get("true_labels", self.train_pack.get("labels"))
        peak_ids = self.train_pack["peak_ids"]
        row_to_candidates = self.train_pack["row_to_candidates"]

        tp = fp = fn = tn = 0
        all_probs: List[float] = []
        all_correct: List[int] = []

        eval_set = set(map(int, eval_peak_ids)) if eval_peak_ids is not None else set(map(int, peak_ids))
        for i, pid in enumerate(peak_ids):
            if int(pid) not in eval_set:
                continue
            label = int(true_labels[i]) if true_labels is not None else -1
            if label < 0:
                continue
            pred = assignments.get(int(pid))
            candidates = row_to_candidates[i]
            if label == 0:
                true_comp = None
            else:
                true_comp = candidates[label]

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
                    fn += 1

            all_probs.append(top_probs[int(pid)])
            all_correct.append(1 if pred == true_comp else 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        ece, mce = self._calculate_calibration_metrics(np.array(all_probs), np.array(all_correct))

        print("\nResults:")
        print(f"  Assignments made: {sum(1 for v in assignments.values() if v is not None)}")
        print(f"  Null assignments: {sum(1 for v in assignments.values() if v is None)}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1: {f1:.3f}")
        print(f"  ECE: {ece:.3f}")
        print(f"  MCE: {mce:.3f}")

        return GenerativeAssignmentResults(
            assignments=assignments,
            top_prob=top_probs,
            per_peak_probs=peak_probs,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix={"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            ece=ece,
            mce=mce,
        )

    # ---- Internals ---------------------------------------------------------------------
    def _predict_probs_prior_mode(self) -> Dict[int, np.ndarray]:
        """
        Heuristic responsibilities without sampling: use simple defaults.
        Useful for fast tests and when trace is unavailable.
        """
        pack = self.train_pack
        mask = pack["mask"]
        species_idx = pack["species_idx"]
        comp_idx = pack["compound_idx"]
        mass_obs = pack["mass_obs"]
        rt_obs = pack["rt_obs"]
        compound_mass = pack["compound_mass"]
        N, K_max = mask.shape

        # Defaults match priors: zero effects, moderate scales
        alpha = 0.0
        alpha_null = 0.0
        a_s = np.zeros(int(species_idx.max()) + 1)
        b_c = np.zeros_like(compound_mass)
        sigma_m = 0.01
        sigma_rt = 0.5
        nu_m = 5
        nu_rt = 5

        # Build wlog
        wlog = np.full((N, K_max), -1e9, dtype=float)
        wlog[:, 0] = alpha_null
        for i in range(N):
            s = int(species_idx[i])
            for k in range(1, K_max):
                c = int(comp_idx[i, k])
                if c >= 0 and mask[i, k]:
                    wlog[i, k] = alpha + a_s[s] + b_c[c]

        # Build loglik
        loglik = np.full((N, K_max), -1e9, dtype=float)
        # Candidate slots
        for i in range(N):
            for k in range(1, K_max):
                c = int(comp_idx[i, k])
                if c >= 0 and mask[i, k]:
                    m_mu = compound_mass[c]
                    m_x = mass_obs[i, k]
                    # Student-T logpdf approx via Normal for speed (test only)
                    mass_ll = -0.5 * ((m_x - m_mu) / sigma_m) ** 2 - np.log(sigma_m + 1e-12)

                    # RT using predicted mu from earlier compute (when available)
                    s = int(species_idx[i])
                    mu_rt, sd_rt = self.rt_predictions.get((s, c), (rt_obs[i, k], 0.2))
                    rt_sigma_eff = np.sqrt(sd_rt**2 + sigma_rt**2)
                    rt_ll = -0.5 * ((rt_obs[i, k] - mu_rt) / rt_sigma_eff) ** 2 - np.log(rt_sigma_eff + 1e-12)

                    loglik[i, k] = mass_ll + rt_ll

        # Null: broad baseline centered at medians (use stored values)
        mass_null_mu = float(pack["mass_null_mu"]) if "mass_null_mu" in pack else float(np.median(mass_obs[:, 0]))
        rt_null_mu = float(pack["rt_null_mu"]) if "rt_null_mu" in pack else float(np.median(rt_obs[:, 0]))
        sigma_m_null = 0.1
        sigma_rt_null = 1.0
        for i in range(N):
            loglik[i, 0] = (
                -0.5 * ((mass_obs[i, 0] - mass_null_mu) / sigma_m_null) ** 2
                - np.log(sigma_m_null + 1e-12)
                - 0.5 * ((rt_obs[i, 0] - rt_null_mu) / sigma_rt_null) ** 2
                - np.log(sigma_rt_null + 1e-12)
            )

        # Combine and softmax
        very_neg = -1e9
        wlog_masked = np.where(mask, wlog, very_neg)
        loglik_masked = np.where(mask, loglik, very_neg)
        logits = wlog_masked + loglik_masked
        logits -= logits.max(axis=1, keepdims=True)
        probs_mat = np.exp(logits)
        probs_mat /= probs_mat.sum(axis=1, keepdims=True)

        peak_ids = self.train_pack["peak_ids"]
        return {int(pid): probs_mat[i] for i, pid in enumerate(peak_ids)}

    def _calculate_calibration_metrics(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
        if len(probs) == 0 or len(labels) == 0:
            return 0.0, 0.0
        probs = np.clip(probs, 1e-12, 1 - 1e-12)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
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
        return float(ece), float(mce)
