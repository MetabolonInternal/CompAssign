#!/usr/bin/env python3
"""
Synthetic data creation for CompAssign training with hierarchical structure.
Generates metabolomics data with isomers, near-isobars, and realistic noise.

DAT-004 alignment: This generator now creates causal RT covariates and RTs that
follow the hierarchical model used in training. Specifically, we generate:
 - Internal standard measurements per species correlated with species effects
 - RT observations: y_sc = mu0 + alpha_s + beta_c + (IS_s - \bar{IS})·w_c + eps

This replaces the previous ad hoc "predicted_rt" pathway and ensures that the
RT model can learn meaningful coefficients and predictive variance.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compassign.utils import (
    ADDUCT_DEFS,
    FRAGMENT_DEFS,
    ISOTOPE_SHIFT,
    ISOTOPE_INTENSITY_FACTOR,
)

from compassign.utils import SyntheticDataset, load_chemberta_pca20
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[3]
EMBEDDINGS_PATH = REPO_ROOT / "resources" / "metabolites" / "embeddings_chemberta_pca20.parquet"


def create_metabolomics_data(
    n_compounds: int = 60,
    n_peaks_per_compound: int = 3,
    n_noise_peaks: int = 100,
    n_species: int = 3,
    n_runs_per_species: int = 3,
    n_internal_standards: int = 10,
    isomer_fraction: float = 0.3,
    near_isobar_fraction: float = 0.2,
    mass_error_ppm: float = 5.0,
    rt_uncertainty_range: Tuple[float, float] = (0.05, 0.5),
    decoy_fraction: float = 0.5,  # Fraction of compounds that are decoys (never present)
    presence_prob: float = 0.4,  # Probability a real compound appears per species
    near_isobar_ppm_max: float | None = None,  # Align near-isobar mass deltas to the ppm threshold
    near_isobar_rt_sd: float | None = None,  # RT proximity for near-isobars (minutes)
    *,
    fixed_runs_per_species_compound: Optional[int] = None,
    desc_tau_beta: Optional[float] = None,
    desc_sigma_compound: Optional[float] = None,
    sigma_y_override: Optional[float] = None,
    anchor_budget_min: Optional[int] = None,
    anchor_budget_max: Optional[int] = None,
    rare_budget_min: Optional[int] = None,
    rare_budget_max: Optional[int] = None,
    pair_radius_quantile: Optional[float] = None,
    gamma_scale: Optional[float] = None,
    anchor_free_frac: Optional[float] = None,
    unseen_eval_per_compound: Optional[int] = None,
    # New knobs to control gamma generation amplitude
    gamma_class_sd: Optional[float] = None,
    gamma_weight_scale: Optional[float] = None,
    species_gamma_sd: Optional[float] = None,
    seed: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int], Dict[int, float], Dict[str, Any]]:
    # Alignment knobs for ambiguity realism (defaults if not provided)
    # Max ppm offset for near-isobars (expressed around base mass)
    near_isobar_ppm_max = 15.0 if near_isobar_ppm_max is None else float(near_isobar_ppm_max)
    # RT proximity for near-isobars (minutes). Keep comparable to sigma_y.
    near_isobar_rt_sd = 0.4 if near_isobar_rt_sd is None else float(near_isobar_rt_sd)
    """
    Create synthetic metabolomics data with hierarchical structure.

    Parameters
    ----------
    n_compounds : int
        Number of compounds in the library
    n_peaks_per_compound : int
        Average peaks per compound
    n_noise_peaks : int
        Number of noise peaks to add
    n_species : int
        Number of species/samples
    n_internal_standards : int
        Number of internal standards measured per species/run
    isomer_fraction : float
        Fraction of compounds that are isomers
    near_isobar_fraction : float
        Fraction of compounds that are near-isobars
    mass_error_ppm : float
        Baseline 1σ mass measurement error (ppm)
    rt_uncertainty_range : tuple
        Range of RT prediction uncertainties

    Returns
    -------
    peak_df : pd.DataFrame
        Peak data with mass, RT, intensity, species, run, true assignments, and embedded
        run-level covariate columns named ``run_covariate_0``, ``run_covariate_1``, ...
    compound_df : pd.DataFrame
        Compound library with masses and properties
    true_assignments : dict
        Mapping of peak_id to true compound_id (or None for noise)
    rt_uncertainties : dict
        Observation-noise proxy per compound (homoscedastic sigma_y)
    hierarchical_params : dict
        Hierarchical model parameters (clusters, classes, run metadata)
    """
    if seed is not None:
        np.random.seed(int(seed))

    # Create hierarchical structure
    n_clusters = min(8, n_species // 5 + 1)  # Reasonable number of clusters
    n_classes = min(10, n_compounds // 6 + 1)  # Chemical classes

    # Assign species to clusters
    species_cluster = np.random.choice(n_clusters, size=n_species)

    # Assign compounds to chemical classes
    compound_class = np.random.choice(n_classes, size=n_compounds)

    # Generate compound library with clustered masses for more overlaps
    # Create mass clusters to increase density and overlaps
    n_mass_clusters = max(3, n_compounds // 10)
    cluster_centers = np.random.uniform(200, 600, n_mass_clusters)
    base_masses = []
    for i in range(n_compounds):
        cluster = np.random.choice(n_mass_clusters)
        # Masses clustered within ±50 Da of cluster centers
        mass = cluster_centers[cluster] + np.random.uniform(-50, 50)
        base_masses.append(mass)
    base_masses = np.array(base_masses)
    # Base RT scale reference used for initialization/reporting
    base_rts = np.random.uniform(1, 15, n_compounds)

    # Create isomers (same mass, different RT)
    n_isomers = int(n_compounds * isomer_fraction)
    isomer_groups = {}

    for i in range(n_isomers):
        if i < n_isomers // 2:
            # Create new isomer group
            group_id = len(isomer_groups)
            isomer_groups[group_id] = base_masses[i]
        else:
            # Add to existing group
            if len(isomer_groups) > 0:
                group_id = np.random.choice(len(isomer_groups))
                base_masses[i] = isomer_groups[group_id]
                # MUCH closer RTs for isomers - often hard to separate
                base_rts[i] = base_rts[group_id % n_compounds] + np.random.normal(0, 0.3)

    # Create near-isobars (similar mass AND RT) using ppm-relative mass deltas
    n_near_isobars = int(n_compounds * near_isobar_fraction)
    for i in range(n_isomers, n_isomers + n_near_isobars):
        # Find a nearby compound
        ref_idx = np.random.choice(i)
        # Draw a ppm offset within realistic instrument tolerance
        rho_ppm = np.random.uniform(-near_isobar_ppm_max, near_isobar_ppm_max)
        delta_da = base_masses[ref_idx] * rho_ppm * 1e-6
        base_masses[i] = base_masses[ref_idx] + delta_da
        # Keep RT close at the scale of minutes comparable to sigma_y
        base_rts[i] = base_rts[ref_idx] + np.random.normal(0, near_isobar_rt_sd)

    # Build compound dataframe; predicted_rt is initialized and set later to a
    # species-agnostic baseline RT mean
    compound_df = pd.DataFrame(
        {
            "compound_id": range(n_compounds),
            "true_mass": base_masses,
            "predicted_rt": base_rts,  # temporary; overwritten below
            "type": [
                "isomer"
                if i < n_isomers
                else "near_isobar"
                if i < n_isomers + n_near_isobars
                else "normal"
                for i in range(n_compounds)
            ],
            "class": compound_class,
        }
    )

    # --- Hierarchical RT generative model ---

    # Hierarchical effects (with sum-to-zero centering)
    # Variance scales chosen to yield realistic minutes-scale variability
    sigma_cluster = 1.0
    sigma_species = 0.7
    sigma_class = 1.0
    # Plausible residual scale for compound baselines (minutes)
    sigma_compound = 0.6 if desc_sigma_compound is None else float(desc_sigma_compound)
    sigma_y = 0.4 if sigma_y_override is None else float(sigma_y_override)  # observation noise

    # Cluster, class base effects
    cluster_raw = np.random.normal(0.0, 1.0, size=n_clusters) * sigma_cluster
    cluster_eff = cluster_raw - cluster_raw.mean()

    class_raw = np.random.normal(0.0, 1.0, size=n_classes) * sigma_class
    class_eff = class_raw - class_raw.mean()

    # Species effects conditional on clusters
    species_raw = np.random.normal(0.0, 1.0, size=n_species) * sigma_species
    species_base = cluster_eff[species_cluster] + species_raw
    alpha_species = species_base - species_base.mean()

    # Descriptor-informed compound baseline β_c
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Missing ChemBERTa embedding artifact at {EMBEDDINGS_PATH}. "
            "Build it with: scripts/data_prep/build_chem_library.sh"
        )
    emb = load_chemberta_pca20(EMBEDDINGS_PATH)
    if emb.features.shape[0] < n_compounds:
        raise ValueError(
            f"Embedding library too small ({emb.features.shape[0]}) for n_compounds={n_compounds}"
        )
    # Cluster the library and sample with per-cluster quotas
    X_lib = emb.features
    n_lib = X_lib.shape[0]
    K = 16 if n_compounds >= 60 else max(8, min(16, n_compounds // 2))
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_lib)
    counts = np.bincount(cluster_labels, minlength=K)
    p = n_compounds / float(n_lib)
    min_per_cluster = 2
    quotas = [max(min_per_cluster, int(round(p * c))) for c in counts]
    total = int(sum(quotas))
    # Adjust quotas to match exactly n_compounds
    if total > n_compounds:
        # Reduce from clusters with largest quotas first
        while total > n_compounds:
            idx = int(np.argmax(quotas))
            if quotas[idx] > min_per_cluster:
                quotas[idx] -= 1
                total -= 1
            else:
                # Find next candidate
                for j in np.argsort(quotas)[::-1]:
                    if quotas[j] > min_per_cluster:
                        quotas[j] -= 1
                        total -= 1
                        break
                else:
                    break
    elif total < n_compounds:
        # Add to clusters with most available elements
        deficit = n_compounds - total
        order = np.argsort(counts)[::-1]
        i = 0
        while deficit > 0:
            quotas[int(order[i % K])] += 1
            deficit -= 1
            i += 1

    selected = []
    for k in range(K):
        pool = np.where(cluster_labels == k)[0]
        if quotas[k] > len(pool):
            take = len(pool)
        else:
            take = quotas[k]
        if take > 0:
            chosen = np.random.choice(pool, size=take, replace=False)
            selected.append(chosen)
    sel_idx = np.concatenate(selected)
    # If overshoot/undershoot remains, trim or pad randomly
    if len(sel_idx) > n_compounds:
        sel_idx = np.random.choice(sel_idx, size=n_compounds, replace=False)
    elif len(sel_idx) < n_compounds:
        remaining = np.setdiff1d(np.arange(n_lib), sel_idx, assume_unique=False)
        extra = np.random.choice(remaining, size=(n_compounds - len(sel_idx)), replace=False)
        sel_idx = np.concatenate([sel_idx, extra])

    Z = X_lib[sel_idx]
    chem_id_sel = emb.chem_id[sel_idx]
    # Define compound classes from chemistry (k-means clusters in Z)
    sel_clusters = cluster_labels[sel_idx]
    compound_class = sel_clusters.astype(int)
    n_classes = int(np.max(compound_class) + 1) if len(compound_class) else 0

    # Optionally mark a fraction of clusters as anchor-free (for anchor-contrast experiments)
    anchor_free_set: set[int] = set()
    if anchor_free_frac is not None and anchor_free_frac > 0:
        n_af = int(np.ceil(float(anchor_free_frac) * K))
        n_af = max(1, min(K, n_af))
        # Deterministic w.r.t. global RNG
        anchor_free_set = set(np.random.choice(np.arange(K), size=n_af, replace=False).tolist())
    # Assign compound groups per cluster with neighbor pairing
    # Goal: each rare/unseen is a nearest neighbor of some anchor within the cluster
    sel_clusters = cluster_labels[sel_idx]
    comp_group = [None] * n_compounds  # aligned to selection order
    for k in range(K):
        local_idxs = np.where(sel_clusters == k)[0]
        if local_idxs.size == 0:
            continue
        # Counts per cluster
        m = int(local_idxs.size)
        if k in anchor_free_set:
            # No anchors in these clusters; split roughly half rare/half unseen
            n_anchor = 0
            n_rare = int(round(0.5 * m))
            n_unseen = m - n_rare
        else:
            n_anchor = int(round(0.5 * m))
            n_rare = int(round(0.25 * m))
            n_unseen = m - n_anchor - n_rare
        # Choose anchors
        anchors_local = (
            np.random.choice(local_idxs, size=n_anchor, replace=False)
            if n_anchor > 0
            else np.array([], dtype=int)
        )
        for li in anchors_local:
            comp_group[int(li)] = "anchor"
        # Remaining candidates pool
        pool = [int(x) for x in local_idxs if x not in set(anchors_local.tolist())]
        assigned_rare: List[int] = []
        assigned_unseen: List[int] = []
        taken: set[int] = set()
        # Optional radius threshold per cluster based on within-cluster distances
        rad_thresh = None
        if pair_radius_quantile is not None and local_idxs.size >= 2:
            sub = Z[local_idxs]
            dmat = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
            triu = dmat[np.triu_indices_from(dmat, k=1)]
            if triu.size > 0:
                rad_thresh = float(np.quantile(triu, float(pair_radius_quantile)))
        # Pair neighbors around anchors
        for li in anchors_local:
            if len(assigned_rare) >= n_rare and len(assigned_unseen) >= n_unseen:
                break
            anchor_vec = Z[int(li)]
            # Compute distances to pool items not yet taken
            candidates = [p for p in pool if p not in taken]
            if not candidates:
                break
            cand_vecs = Z[np.array(candidates)]
            dists = np.linalg.norm(cand_vecs - anchor_vec[None, :], axis=1)
            order = np.argsort(dists)
            for oi in order:
                cj = int(candidates[int(oi)])
                if cj in taken:
                    continue
                if rad_thresh is not None and float(dists[int(oi)]) > rad_thresh:
                    continue
                if len(assigned_rare) < n_rare:
                    assigned_rare.append(cj)
                    taken.add(cj)
                elif len(assigned_unseen) < n_unseen:
                    assigned_unseen.append(cj)
                    taken.add(cj)
                if len(assigned_rare) >= n_rare and len(assigned_unseen) >= n_unseen:
                    break
        # If quotas remain, fill randomly from leftover pool
        leftover = [p for p in pool if p not in taken]
        if len(assigned_rare) < n_rare and leftover:
            need = n_rare - len(assigned_rare)
            extra = np.random.choice(leftover, size=min(need, len(leftover)), replace=False)
            assigned_rare.extend(int(x) for x in np.atleast_1d(extra))
            for x in np.atleast_1d(extra):
                if int(x) in leftover:
                    leftover.remove(int(x))
        if len(assigned_unseen) < n_unseen and leftover:
            need = n_unseen - len(assigned_unseen)
            extra = np.random.choice(leftover, size=min(need, len(leftover)), replace=False)
            assigned_unseen.extend(int(x) for x in np.atleast_1d(extra))
        # Write back group labels
        for li in assigned_rare:
            comp_group[int(li)] = "rare"
        for li in assigned_unseen:
            comp_group[int(li)] = "unseen"
    # Any None (due to rounding edge cases) default to anchor
    comp_group = [g if g is not None else "anchor" for g in comp_group]
    if sum(1 for g in comp_group if g == "rare") == 0:
        # Guarantee at least one rare compound so downstream splits have coverage
        anchor_candidates = [idx for idx, g in enumerate(comp_group) if g == "anchor"]
        unseen_candidates = [idx for idx, g in enumerate(comp_group) if g == "unseen"]
        pool = anchor_candidates or unseen_candidates
        if pool:
            chosen = int(np.random.choice(pool))
            comp_group[chosen] = "rare"
    # Standardise descriptors (PCA is whitened; this is a safeguard)
    Z_mean = Z.mean(axis=0)
    Z_std = Z.std(axis=0) + 1e-8
    Zs = (Z - Z_mean) / Z_std
    d = Zs.shape[1]
    # Plausible descriptor mapping strength: target moderate EV for β_c
    tau_beta = 0.16 if desc_tau_beta is None else float(desc_tau_beta)
    theta_beta = np.random.normal(0.0, tau_beta, size=d)
    delta_c = np.random.normal(0.0, sigma_compound, size=n_compounds)
    beta_compound = Zs @ theta_beta + delta_c
    beta_compound = beta_compound - beta_compound.mean()

    # Global intercept near center of chromatogram
    mu0 = np.random.uniform(5.0, 11.0)

    # Run-level RT shifts for each species (simulating batch effects)
    rt_shift_range = (0.1, 0.5)
    species_rt_shifts = np.random.uniform(-rt_shift_range[1], rt_shift_range[1], size=n_species)
    n_runs_per_species = max(1, int(n_runs_per_species))
    run_species = np.repeat(np.arange(n_species, dtype=int), n_runs_per_species)
    run_ids = np.arange(run_species.size, dtype=int)
    run_specific_shift = np.random.normal(0.0, 0.08, size=run_species.size)

    # Internal-standard panel with correlated measurements per species
    n_is = int(max(1, n_internal_standards))
    is_base_rt = np.random.uniform(2.0, 15.0, size=n_is)
    is_base_rt.sort()

    sigma_is_cluster = 0.25
    sigma_is_species = 0.2
    sigma_is_measure = 0.06

    is_cluster_shift = np.random.normal(0.0, sigma_is_cluster, size=(n_clusters, n_is))
    is_species_shift = np.random.normal(0.0, sigma_is_species, size=(n_species, n_is))

    # Low-rank latent structure to create realistic correlation across IS channels
    n_is_factors = min(3, n_is)
    factor_scores = np.random.normal(0.0, 1.0, size=(n_species, n_is_factors))
    factor_loadings = np.random.normal(0.0, 0.3, size=(n_is_factors, n_is))

    species_is_mean = (
        is_base_rt
        + is_cluster_shift[species_cluster]
        + is_species_shift
        + factor_scores @ factor_loadings
    )

    run_features = np.zeros((run_species.size, n_is), dtype=float)
    for rid, species in enumerate(run_species):
        base_vec = species_is_mean[species]
        shared_shift = species_rt_shifts[species]
        run_features[rid] = (
            base_vec
            + shared_shift
            + run_specific_shift[rid]
            + np.random.normal(0.0, sigma_is_measure, size=n_is)
        )

    run_features_centered = run_features - run_features.mean(axis=0, keepdims=True)

    # Derive species-level internal-standard summaries for baselines
    internal_std_measurements = np.zeros((n_species, n_is), dtype=float)
    for species in range(n_species):
        mask = run_species == species
        internal_std_measurements[species] = run_features[mask].mean(axis=0)

    internal_std_centered = internal_std_measurements - internal_std_measurements.mean(
        axis=0, keepdims=True
    )

    # Compound-specific sensitivities to internal standards (lasso-like signal)
    cls_sd = 0.35 if gamma_class_sd is None else float(gamma_class_sd)
    base_scale = 0.4 if gamma_weight_scale is None else float(gamma_weight_scale)
    class_is_base = np.random.normal(0.0, cls_sd, size=(n_classes, n_is))
    compound_is_weights = class_is_base[compound_class] + np.random.normal(
        0.0, 0.12, size=(n_compounds, n_is)
    )
    # Keep weights centered and with controlled magnitude (minutes-scale effects)
    compound_is_weights -= compound_is_weights.mean(axis=0, keepdims=True)
    compound_is_weights *= base_scale

    # Multicell heterogeneity: species-specific gamma deviations (per-IS slope)
    # These create differing RT–covariate relationships across species/cells.
    # Species-specific gamma heterogeneity (minutes per unit covariate)
    # Default off unless explicitly enabled by caller
    gamma_species_sd = 0.0 if species_gamma_sd is None else float(species_gamma_sd)
    species_gamma_shift = np.random.normal(0.0, gamma_species_sd, size=(n_species, n_is))

    # Species-agnostic baseline RT per compound
    baseline_rt = mu0 + beta_compound
    predicted_rt = baseline_rt
    predicted_rt = np.clip(predicted_rt, 0.5, 16.0)

    rt_pred_sigma = np.random.uniform(
        rt_uncertainty_range[0], rt_uncertainty_range[1], size=n_compounds
    )

    # Attach chem_id from sampled embedding rows
    try:
        compound_df["chem_id"] = chem_id_sel
    except NameError:
        pass
    # Update class column to chemistry-aligned classes
    compound_df["class"] = compound_class
    compound_df["predicted_rt"] = predicted_rt
    compound_df["rt_prediction_std"] = rt_pred_sigma

    # Generate peaks using the hierarchical RT generative story
    peaks = []
    peak_id = 0
    true_assignments = {}

    # Mark decoy compounds (NEVER appear in samples in the realistic setting)
    # We intentionally remove the previous "leaky decoy" mechanism to avoid
    # asymmetric difficulty that does not reflect production usage of decoys.
    n_decoys = int(n_compounds * decoy_fraction)
    decoy_compounds = set(np.random.choice(n_compounds, size=n_decoys, replace=False))
    real_compounds = set(range(n_compounds)) - decoy_compounds

    compound_df["is_decoy"] = [i in decoy_compounds for i in range(n_compounds)]
    # Attach compound group labels (mark decoys distinctly)
    try:
        compound_df["compound_group"] = [
            ("decoy" if compound_df.loc[i, "is_decoy"] else comp_group[i])
            for i in range(n_compounds)
        ]
        if (compound_df["compound_group"] == "rare").sum() == 0:
            candidates = compound_df.index[
                (~compound_df["is_decoy"])
                & (compound_df["compound_group"].isin(["anchor", "unseen"]))
            ].tolist()
            if candidates:
                chosen = int(np.random.choice(candidates))
                compound_df.loc[chosen, "compound_group"] = "rare"
    except Exception:
        pass
    # Keep the column; it is always False in this generator
    compound_df["is_leaky_decoy"] = [False for _ in range(n_compounds)]

    # Log the RT shifts for transparency
    print(f"  Applied run-level RT shifts (minutes): {species_rt_shifts.round(3).tolist()}")

    # Per-species mass calibration offsets in ppm to simulate drift
    species_mass_offset_ppm = np.random.normal(0.0, mass_error_ppm * 0.25, size=n_species)

    def draw_mass_error_ppm(species_idx: int, scale: float = 1.0) -> float:
        """Sample a ppm mass error for the given species."""
        local_sigma = mass_error_ppm * scale * np.random.uniform(0.7, 1.3)
        return species_mass_offset_ppm[species_idx] + np.random.normal(0.0, local_sigma)

    # Build hard budgets per compound based on group
    budgets = np.zeros(n_compounds, dtype=int)
    for cid in range(n_compounds):
        if compound_df.loc[cid, "is_decoy"]:
            budgets[cid] = 0
        else:
            g = (
                compound_df.loc[cid, "compound_group"]
                if "compound_group" in compound_df.columns
                else "anchor"
            )
            if g == "anchor":
                lo = 3 if anchor_budget_min is None else int(anchor_budget_min)
                hi = 5 if anchor_budget_max is None else int(anchor_budget_max)
                budgets[cid] = int(np.random.randint(lo, hi + 1))
            elif g == "rare":
                # Keep rare extremely scarce in baseline generator to satisfy tests
                if rare_budget_min is None and rare_budget_max is None:
                    budgets[cid] = 2
                else:
                    rlo = 2 if rare_budget_min is None else int(rare_budget_min)
                    rhi = 2 if rare_budget_max is None else int(rare_budget_max)
                    budgets[cid] = int(np.random.randint(rlo, rhi + 1))
            elif g == "unseen":
                budgets[cid] = 0
            else:
                budgets[cid] = int(np.random.randint(2, 4))

    # Allocate budgets to species uniformly (one observation per budget unit)
    alloc_by_species: Dict[int, List[int]] = {s: [] for s in range(n_species)}
    for cid in range(n_compounds):
        for _ in range(int(budgets[cid])):
            s = int(np.random.choice(n_species))
            alloc_by_species[s].append(cid)

    # Generate real peaks
    runs_by_species = {s: run_ids[run_species == s] for s in range(n_species)}
    if "compound_group" in compound_df.columns:
        comp_groups_list = compound_df["compound_group"].astype(str).tolist()
    else:
        comp_groups_list = ["anchor"] * len(compound_df)
    fixed_K = None
    if (
        "fixed_runs_per_species_compound" in locals()
        and fixed_runs_per_species_compound is not None
    ):
        try:
            fixed_K = int(fixed_runs_per_species_compound)
        except Exception:
            fixed_K = None
    rare_counts: Dict[int, int] = {}
    for species in range(n_species):
        species_runs = runs_by_species[species]
        present_compounds = (
            list(range(n_compounds)) if fixed_K is not None else alloc_by_species.get(species, [])
        )

        for compound_id in present_compounds:
            if fixed_K is not None and compound_df.loc[compound_id, "is_decoy"]:
                continue
            compound = compound_df.iloc[compound_id]
            # Generate multiple peaks per compound (isotopes, fragments)
            n_primary = (
                fixed_K if fixed_K is not None else max(1, np.random.poisson(n_peaks_per_compound))
            )
            # For rare compounds in non-rich mode, hard-cap total labeled observations to <=2 across dataset
            if (
                fixed_K is None
                and compound_df.get(
                    "compound_group", pd.Series(["anchor"]).repeat(len(compound_df)).values
                )[compound_id]
                == "rare"
            ):
                current = rare_counts.get(int(compound_id), 0)
                if current >= 2:
                    continue
                n_primary = 1
            for _ in range(n_primary):
                # Choose run per primary observation to avoid single-run leakage
                run_choice = int(np.random.choice(species_runs))
                ppm_error = draw_mass_error_ppm(species, scale=1.0)
                measured_mass = compound["true_mass"] * (1.0 + ppm_error * 1e-6)

                # RT mean per species–compound from hierarchical story
                # Now includes run-level shift for this species
                mu_sc = (
                    mu0
                    + alpha_species[species]
                    + beta_compound[compound_id]
                    + float(
                        np.dot(
                            run_features_centered[run_choice],
                            compound_is_weights[compound_id] + species_gamma_shift[species],
                        )
                    )
                    + species_rt_shifts[species]
                    + run_specific_shift[run_choice]
                )
                # Optionally down-scale the γ contribution (experiment knob)
                if gamma_scale is not None:
                    mu_sc = (
                        mu0
                        + alpha_species[species]
                        + beta_compound[compound_id]
                        + float(
                            np.dot(
                                run_features_centered[run_choice],
                                (compound_is_weights[compound_id] + species_gamma_shift[species])
                                * float(gamma_scale),
                            )
                        )
                        + species_rt_shifts[species]
                        + run_specific_shift[run_choice]
                    )

                rt_scale = sigma_y * (1.0 if fixed_K is not None else np.random.uniform(0.7, 1.4))
                measured_rt = np.random.normal(mu_sc, rt_scale)

                base_intensity = np.exp(np.random.normal(12, 1.5))
                if compound["is_decoy"]:
                    base_intensity *= np.random.uniform(0.4, 0.8)

                peaks.append(
                    {
                        "peak_id": peak_id,
                        "species": species,
                        "run": run_choice,
                        "mass": measured_mass,
                        "rt": measured_rt,
                        "intensity": base_intensity,
                        "true_compound": compound_id,
                    }
                )
                true_assignments[peak_id] = compound_id
                peak_id += 1
                if fixed_K is None and comp_groups_list[compound_id] == "rare":
                    rare_counts[int(compound_id)] = rare_counts.get(int(compound_id), 0) + 1

                # Satellite chemistry peaks kept within tolerance but tied to same compound
                satellites: List[Tuple[str, float, float]] = []

                if (
                    not (fixed_K is None and comp_groups_list[compound_id] == "rare")
                    and np.random.random() < 0.6
                ):
                    # Use primary isotope shift/intensity from shared definitions
                    iso_delta = ISOTOPE_SHIFT
                    satellites.append(("M+13C", iso_delta, ISOTOPE_INTENSITY_FACTOR))

                for adduct in ADDUCT_DEFS:
                    if np.random.random() < 0.3:
                        # Use actual adduct mass differences, not scaled down
                        satellites.append(
                            (adduct["name"], adduct["delta"], adduct["intensity_factor"])
                        )

                for fragment in FRAGMENT_DEFS:
                    if np.random.random() < 0.2:
                        # Use actual fragment mass differences
                        satellites.append(
                            (fragment["name"], fragment["delta"], fragment["intensity_factor"])
                        )

                for _, delta, intensity_factor in satellites:
                    sat_mass_nominal = compound["true_mass"] + delta
                    ppm_error_sat = draw_mass_error_ppm(species, scale=np.random.uniform(1.2, 2.0))
                    sat_mass = sat_mass_nominal * (1.0 + ppm_error_sat * 1e-6)
                    sat_rt = measured_rt + np.random.normal(0, 0.08)
                    sat_intensity = base_intensity * intensity_factor * np.random.uniform(0.5, 1.3)

                    peaks.append(
                        {
                            "peak_id": peak_id,
                            "species": species,
                            "run": run_choice,
                            "mass": sat_mass,
                            "rt": sat_rt,
                            "intensity": sat_intensity,
                            "true_compound": compound_id,
                        }
                    )
                    true_assignments[peak_id] = compound_id
                    peak_id += 1

    # Add noise peaks with realistic mimicry of real compounds
    for _ in range(n_noise_peaks):
        species = np.random.choice(n_species)
        # Choose run early so run-level shifts are consistent with recorded run
        run_choice = int(np.random.choice(runs_by_species[species]))

        if np.random.random() < 0.65 and len(compound_df) > 0:
            # Mimic real chemistry by anchoring near an existing compound
            mimic_idx = np.random.choice(len(compound_df))
            mimic_mass = compound_df.iloc[mimic_idx]["true_mass"]
            mimic_rt_mean = compound_df.iloc[mimic_idx]["predicted_rt"]

            ppm_error_noise = draw_mass_error_ppm(species, scale=np.random.uniform(1.5, 3.0))
            rt_jitter = np.random.normal(0, np.random.uniform(0.05, 1.2))

            noise_mass = mimic_mass * (1.0 + ppm_error_noise * 1e-6)
            # Add run-level shift to noise peaks anchored to compounds
            noise_rt = np.clip(
                mimic_rt_mean
                + rt_jitter
                + species_rt_shifts[species]
                + run_specific_shift[run_choice],
                0.0,
                16.0,
            )

            if np.random.random() < 0.3:
                # High-intensity chemical noise (e.g., contaminants)
                noise_intensity = np.exp(np.random.normal(11.5, 1.3))
            else:
                noise_intensity = np.exp(np.random.normal(10.0, 1.5))
        else:
            base_mass = np.random.uniform(50, 850)
            ppm_error_noise = draw_mass_error_ppm(species, scale=np.random.uniform(1.5, 3.0))
            noise_mass = base_mass * (1.0 + ppm_error_noise * 1e-6)
            # Add run-level shift to random noise peaks too
            noise_rt = np.clip(
                np.random.uniform(0, 16)
                + species_rt_shifts[species]
                + run_specific_shift[run_choice],
                0.0,
                16.0,
            )
            noise_intensity = np.exp(np.random.normal(9.5, 1.8))

        peaks.append(
            {
                "peak_id": peak_id,
                "species": species,
                "run": run_choice,
                "mass": noise_mass,
                "rt": noise_rt,
                "intensity": noise_intensity,
                "true_compound": None,
            }
        )
        true_assignments[peak_id] = None
        peak_id += 1

    # Add evaluation-only labeled rows for unseen compounds (reserved for test)
    if unseen_eval_per_compound is not None and int(unseen_eval_per_compound) > 0:
        ueval = int(unseen_eval_per_compound)
        unseen_ids = compound_df.index[
            compound_df.get("compound_group", pd.Series(["anchor"]).repeat(len(compound_df)).values)
            == "unseen"
        ].tolist()
        runs_by_species = {s: run_ids[run_species == s] for s in range(n_species)}
        for cid in unseen_ids:
            for _ in range(ueval):
                species = int(np.random.choice(n_species))
                species_runs = runs_by_species[species]
                if len(species_runs) == 0:
                    continue
                run_choice = int(np.random.choice(species_runs))
                compound = compound_df.iloc[cid]
                # Generate a single primary observation per eval unit
                ppm_error = draw_mass_error_ppm(species, scale=1.0)
                measured_mass = compound["true_mass"] * (1.0 + ppm_error * 1e-6)
                mu_sc = (
                    mu0
                    + alpha_species[species]
                    + beta_compound[cid]
                    + float(
                        np.dot(
                            run_features_centered[run_choice],
                            compound_is_weights[cid] + species_gamma_shift[species],
                        )
                    )
                    + species_rt_shifts[species]
                    + run_specific_shift[run_choice]
                )
                if gamma_scale is not None:
                    mu_sc = (
                        mu0
                        + alpha_species[species]
                        + beta_compound[cid]
                        + float(
                            np.dot(
                                run_features_centered[run_choice],
                                (compound_is_weights[cid] + species_gamma_shift[species])
                                * float(gamma_scale),
                            )
                        )
                        + species_rt_shifts[species]
                        + run_specific_shift[run_choice]
                    )
                measured_rt = float(np.random.normal(mu_sc, sigma_y))
                base_intensity = float(np.exp(np.random.normal(12, 1.5)))
                peaks.append(
                    {
                        "peak_id": peak_id,
                        "species": species,
                        "run": run_choice,
                        "mass": measured_mass,
                        "rt": measured_rt,
                        "intensity": base_intensity,
                        "true_compound": cid,
                    }
                )
                true_assignments[peak_id] = cid
                peak_id += 1

    peak_df = pd.DataFrame(peaks)
    # Post-process: for baseline (non-rich) generator, ensure rare compounds have <=2 labeled rows
    if fixed_runs_per_species_compound is None and "compound_group" in compound_df.columns:
        rare_ids = (
            compound_df.loc[compound_df["compound_group"].astype(str) == "rare", "compound_id"]
            .astype(int)
            .tolist()
        )
        if rare_ids:
            labeled_mask = peak_df["true_compound"].notna()
            keep_index = []
            rare_seen: Dict[int, int] = {}
            for idx, row in peak_df.iterrows():
                cid = row.get("true_compound")
                if pd.isna(cid) or int(cid) not in rare_ids:
                    keep_index.append(idx)
                    continue
                cnt = rare_seen.get(int(cid), 0)
                if cnt < 2:
                    keep_index.append(idx)
                    rare_seen[int(cid)] = cnt + 1
                else:
                    # drop extra rare occurrences
                    continue
            peak_df = peak_df.loc[keep_index].reset_index(drop=True)

    # Embed run-level covariates directly into the peak dataframe
    run_covariate_cols = [f"run_covariate_{i}" for i in range(run_features.shape[1])]
    run_metadata = pd.DataFrame(
        {
            "run": run_ids,
            "species": run_species,
            **{col: run_features[:, idx] for idx, col in enumerate(run_covariate_cols)},
        }
    )
    peak_df = peak_df.merge(run_metadata, on=["run", "species"], how="left")

    # Create hierarchical parameters dict
    hierarchical_params = {
        "n_clusters": n_clusters,
        "n_classes": n_classes,
        "species_cluster": species_cluster,
        "compound_class": compound_class,
        "run_species": run_species,
        "compound_internal_std_weights": compound_is_weights,
        "gamma_class_sd": cls_sd,
        "gamma_weight_scale": base_scale,
        "internal_std_base": is_base_rt,
        "run_covariate_columns": run_covariate_cols,
        "compound_features": Z if "Z" in locals() else None,
        "compound_groups": comp_group if "comp_group" in locals() else None,
        "species_gamma_shift": species_gamma_shift,
    }

    # Uncertainties dict (homoscedastic sigma_y)
    rt_uncertainties = {i: sigma_y for i in range(n_compounds)}

    return (
        peak_df,
        compound_df,
        true_assignments,
        rt_uncertainties,
        hierarchical_params,
    )


def create_synthetic_dataset(
    n_compounds: int = 60,
    n_peaks_per_compound: int = 3,
    n_noise_peaks: int = 100,
    n_species: int = 3,
    n_runs_per_species: int = 3,
    n_internal_standards: int = 10,
    isomer_fraction: float = 0.3,
    near_isobar_fraction: float = 0.2,
    mass_error_ppm: float = 5.0,
    rt_uncertainty_range: Tuple[float, float] = (0.05, 0.5),
    decoy_fraction: float = 0.5,
    presence_prob: float = 0.4,
    near_isobar_ppm_max: float | None = None,
    near_isobar_rt_sd: float | None = None,
) -> SyntheticDataset:
    """Convenience wrapper that returns a ``SyntheticDataset``.

    Mirrors ``create_metabolomics_data`` parameters and wraps its outputs into a
    small dataclass to keep downstream call sites tidy.
    """

    (
        peak_df,
        compound_df,
        true_assignments,
        rt_uncertainties,
        hierarchical_params,
    ) = create_metabolomics_data(
        n_compounds=n_compounds,
        n_peaks_per_compound=n_peaks_per_compound,
        n_noise_peaks=n_noise_peaks,
        n_species=n_species,
        n_runs_per_species=n_runs_per_species,
        n_internal_standards=n_internal_standards,
        isomer_fraction=isomer_fraction,
        near_isobar_fraction=near_isobar_fraction,
        mass_error_ppm=mass_error_ppm,
        rt_uncertainty_range=rt_uncertainty_range,
        decoy_fraction=decoy_fraction,
        presence_prob=presence_prob,
        near_isobar_ppm_max=near_isobar_ppm_max,
        near_isobar_rt_sd=near_isobar_rt_sd,
    )

    return SyntheticDataset(
        peak_df=peak_df,
        compound_df=compound_df,
        true_assignments=true_assignments,
        rt_uncertainties=rt_uncertainties,
        hierarchical_params=hierarchical_params,
    )


if __name__ == "__main__":
    # Test the function
    peaks, compounds, assignments, rt_uncert, hier_params = create_metabolomics_data()
    print(f"Generated {len(peaks)} peaks with {len(compounds)} compounds")
    print(f"True assignments: {sum(1 for v in assignments.values() if v is not None)}")
    print(f"Noise peaks: {sum(1 for v in assignments.values() if v is None)}")
    cluster_count = hier_params["n_clusters"]
    class_count = hier_params["n_classes"]
    print(f"Hierarchical params: {cluster_count} clusters, {class_count} classes")
    run_cols = hier_params.get("run_covariate_columns", [])
    print(f"Embedded run covariates: {run_cols}")
