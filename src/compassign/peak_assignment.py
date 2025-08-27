"""
Bayesian logistic regression model for peak-to-compound assignment.

This module implements a logistic model that uses mass accuracy, RT predictions,
and intensity to assign LC-MS peaks to known compounds.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AssignmentResults:
    """Container for peak assignment results."""
    assignments: Dict[int, Optional[int]]  # peak_id -> compound_id
    probabilities: Dict[int, float]  # peak_id -> max probability
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Dict[str, int]  # TP, FP, TN, FN counts


class PeakAssignmentModel:
    """
    Bayesian logistic regression for peak-to-compound assignment.
    
    Uses mass error, RT z-score (from RT model predictions), and intensity
    as features to predict whether a peak-compound pair is a true match.
    """
    
    def __init__(self, mass_tolerance: float = 0.005):
        """
        Initialize the peak assignment model.
        
        Parameters
        ----------
        mass_tolerance : float
            Mass tolerance in Da for candidate generation (default: 0.005 Da = 5 ppm)
            Note: Ablation study showed 0.005 Da achieves >95% precision
        """
        self.mass_tolerance = mass_tolerance
        self.model = None
        self.trace = None
        self.logit_df = None
        self.rt_predictions = {}
        
    def compute_rt_predictions(self, 
                              trace_rt: az.InferenceData,
                              n_species: int,
                              n_compounds: int,
                              descriptors: np.ndarray,
                              internal_std: np.ndarray) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """
        Compute RT predictions and uncertainties from the RT model posterior.
        
        Parameters
        ----------
        trace_rt : az.InferenceData
            Posterior samples from RT model
        n_species : int
            Number of species
        n_compounds : int
            Number of compounds
        descriptors : np.ndarray
            Molecular descriptors matrix
        internal_std : np.ndarray
            Internal standard RTs
            
        Returns
        -------
        dict
            Dictionary mapping (species, compound) -> (mean_rt, std_rt)
        """
        print("Computing RT predictions from posterior...")
        
        # Extract posterior means
        mu0_mean = trace_rt.posterior['mu0'].mean().item()
        species_mean = trace_rt.posterior['species_eff'].mean(dim=['chain','draw']).values
        compound_mean = trace_rt.posterior['compound_eff'].mean(dim=['chain','draw']).values
        beta_mean = trace_rt.posterior['beta'].mean(dim=['chain','draw']).values
        gamma_mean = trace_rt.posterior['gamma'].mean().item()
        sigma_y_mean = trace_rt.posterior['sigma_y'].mean().item()
        
        # Extract posterior standard deviations for uncertainty
        species_std = trace_rt.posterior['species_eff'].std(dim=['chain','draw']).values
        compound_std = trace_rt.posterior['compound_eff'].std(dim=['chain','draw']).values
        
        rt_predictions = {}
        
        for s in range(n_species):
            for m in range(n_compounds):
                # Predicted mean RT
                mean_rt = (mu0_mean 
                          + species_mean[s] 
                          + compound_mean[m] 
                          + np.dot(descriptors[m], beta_mean) 
                          + gamma_mean * internal_std[s])
                
                # Predicted uncertainty (combine parameter uncertainty and observation noise)
                param_var = species_std[s]**2 + compound_std[m]**2
                total_std = np.sqrt(param_var + sigma_y_mean**2)
                
                rt_predictions[(s, m)] = (mean_rt, total_std)
        
        self.rt_predictions = rt_predictions
        return rt_predictions
    
    def generate_training_data(self,
                              peak_df: pd.DataFrame,
                              compound_mass: np.ndarray,
                              n_compounds: int) -> pd.DataFrame:
        """
        Generate training data for the logistic model.
        
        Parameters
        ----------
        peak_df : pd.DataFrame
            Peak data with columns: peak_id, species, true_compound, mass, rt, intensity
        compound_mass : np.ndarray
            True masses of compounds
        n_compounds : int
            Number of compounds
            
        Returns
        -------
        pd.DataFrame
            Training data with features and labels
        """
        if not self.rt_predictions:
            raise ValueError("RT predictions not computed. Run compute_rt_predictions first.")
        
        print(f"Generating candidate assignments with mass tolerance ±{self.mass_tolerance} Da...")
        
        logit_data = []
        
        # Iterate over peaks to generate candidate assignments
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
            
            # Check candidate compounds by mass tolerance
            for m in range(n_compounds):
                if abs(mz - compound_mass[m]) <= self.mass_tolerance:
                    # Compute features
                    mass_error_ppm = (mz - compound_mass[m]) / compound_mass[m] * 1e6
                    
                    # RT z-score using predictions
                    rt_pred_mean, rt_pred_std = self.rt_predictions[(s, m)]
                    rt_z = (rt - rt_pred_mean) / rt_pred_std if rt_pred_std > 0 else 0
                    
                    # Log intensity
                    log_intensity = np.log10(intensity) if intensity > 0 else 0
                    
                    # Label (1 if true match, 0 otherwise)
                    label = 1 if (true_comp is not None and true_comp == m) else 0
                    
                    logit_data.append((peak_id, s, m, mass_error_ppm, rt_z, log_intensity, label))
        
        self.logit_df = pd.DataFrame(logit_data, 
                                     columns=['peak_id', 'species', 'compound', 
                                             'mass_err_ppm', 'rt_z', 'log_intensity', 'label'])
        
        print(f"Logistic training pairs count: {len(self.logit_df)}")
        print(f"Positive (true) assignments: {self.logit_df['label'].sum()}")
        print(f"Negative (false) assignments: {(1 - self.logit_df['label']).sum()}")
        
        return self.logit_df
    
    def build_model(self) -> pm.Model:
        """
        Build the PyMC logistic regression model.
        
        Returns
        -------
        pm.Model
            The constructed PyMC model
        """
        if self.logit_df is None:
            raise ValueError("Training data not generated. Run generate_training_data first.")
        
        with pm.Model() as model:
            # Priors for logistic regression coefficients
            theta0 = pm.Normal('theta0', 0.0, 2.0)  # intercept
            theta_mass = pm.Normal('theta_mass', 0.0, 2.0)  # mass error coefficient
            theta_rt = pm.Normal('theta_rt', 0.0, 2.0)  # RT z-score coefficient
            theta_int = pm.Normal('theta_int', 0.0, 2.0)  # intensity coefficient
            
            # Linear predictor
            logit_p = (theta0 
                      + theta_mass * self.logit_df['mass_err_ppm'].values 
                      + theta_rt * self.logit_df['rt_z'].values 
                      + theta_int * self.logit_df['log_intensity'].values)
            
            # Convert to probability
            p = pm.Deterministic('p', pm.math.sigmoid(logit_p))
            
            # Bernoulli likelihood
            y = pm.Bernoulli('y', p, observed=self.logit_df['label'].values)
        
        self.model = model
        return model
    
    def sample(self,
              n_samples: int = 1000,
              n_tune: int = 1000,
              n_chains: int = 2,
              target_accept: float = 0.95,
              random_seed: int = 42) -> az.InferenceData:
        """
        Sample from the posterior using NUTS.
        
        Parameters
        ----------
        n_samples : int
            Number of samples per chain
        n_tune : int
            Number of tuning steps
        n_chains : int
            Number of MCMC chains
        target_accept : float
            Target acceptance probability
        random_seed : int
            Random seed
            
        Returns
        -------
        az.InferenceData
            Posterior samples
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        print(f"\nSampling logistic model with {n_chains} chains, {n_samples} samples each...")
        
        with self.model:
            self.trace = pm.sample(
                n_samples,
                tune=n_tune,
                chains=n_chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=True
            )
        
        # Check convergence
        rhat_vals = az.rhat(self.trace).to_dataframe()
        max_rhat = rhat_vals.max().max()
        if max_rhat > 1.01:
            print(f"\nWarning: Maximum R-hat = {max_rhat:.3f} (should be < 1.01)")
        
        return self.trace
    
    def predict_assignments(self, 
                           peak_df: pd.DataFrame,
                           probability_threshold: float = 0.9) -> AssignmentResults:
        """
        Predict peak assignments using the posterior mean coefficients.
        
        Parameters
        ----------
        peak_df : pd.DataFrame
            Peak data
        probability_threshold : float
            Minimum probability to make an assignment
            
        Returns
        -------
        AssignmentResults
            Assignment results with metrics
        """
        if self.trace is None:
            raise ValueError("No trace available. Run sample first.")
        
        # Extract posterior mean coefficients
        theta0_mean = self.trace.posterior['theta0'].mean().item()
        theta_mass_mean = self.trace.posterior['theta_mass'].mean().item()
        theta_rt_mean = self.trace.posterior['theta_rt'].mean().item()
        theta_int_mean = self.trace.posterior['theta_int'].mean().item()
        
        print(f"\nLogistic model coefficients:")
        print(f"  Intercept: {theta0_mean:.3f}")
        print(f"  Mass error: {theta_mass_mean:.3f}")
        print(f"  RT z-score: {theta_rt_mean:.3f}")
        print(f"  Log intensity: {theta_int_mean:.3f}")
        
        # Compute predicted probabilities
        self.logit_df['pred_prob'] = 1 / (1 + np.exp(-(
            theta0_mean 
            + theta_mass_mean * self.logit_df['mass_err_ppm'] 
            + theta_rt_mean * self.logit_df['rt_z'] 
            + theta_int_mean * self.logit_df['log_intensity']
        )))
        
        # Assign each peak to highest-probability compound
        assignments = {}
        probabilities = {}
        
        for peak_id, group in self.logit_df.groupby('peak_id'):
            # Find candidate with max probability
            idx_max = group['pred_prob'].idxmax()
            comp_pred = int(group.loc[idx_max, 'compound'])
            prob_max = group.loc[idx_max, 'pred_prob']
            
            if prob_max >= probability_threshold:
                assignments[peak_id] = comp_pred
                probabilities[peak_id] = prob_max
            else:
                assignments[peak_id] = None
                probabilities[peak_id] = prob_max
        
        # Compute metrics
        TP = FP = FN = TN = 0
        
        for _, peak in peak_df.iterrows():
            peak_id = int(peak['peak_id'])
            true_comp = peak['true_compound']
            if not pd.isna(true_comp):
                true_comp = int(true_comp)
            else:
                true_comp = None
            
            assigned_comp = assignments.get(peak_id)
            
            if assigned_comp is not None:
                if true_comp is not None and assigned_comp == true_comp:
                    TP += 1  # Correct assignment
                else:
                    FP += 1  # Wrong assignment
            else:
                if true_comp is not None:
                    FN += 1  # Missed true peak
                else:
                    TN += 1  # Correctly rejected decoy
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\nPeak assignment results:")
        print(f"  True Positives: {TP}")
        print(f"  False Positives: {FP}")
        print(f"  False Negatives: {FN}")
        print(f"  True Negatives: {TN}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1_score:.3f}")
        
        return AssignmentResults(
            assignments=assignments,
            probabilities=probabilities,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            confusion_matrix={'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}
        )
    
    def get_parameter_summary(self) -> pd.DataFrame:
        """Get summary statistics for model parameters."""
        if self.trace is None:
            raise ValueError("No trace available. Run sample first.")
        
        return az.summary(self.trace)