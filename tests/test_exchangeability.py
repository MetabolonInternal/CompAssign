import numpy as np
import pandas as pd

from src.compassign.peak_assignment import PeakAssignment
from src.compassign.presence_prior import PresencePrior


def _compute_probs_from_pack(train_pack, presence: PresencePrior, theta0, theta_null, theta_features):
    """
    Deterministically compute softmax probabilities using the model's
    mathematical form, without PyMC sampling. Uses X_test/mask_test.
    Returns a list per row of (compound_id, probability) tuples.
    """
    X = train_pack["X_test"]
    mask = train_pack["mask_test"]
    peak_species = train_pack["peak_species"]
    row_to_candidates = train_pack["row_to_candidates_test"]

    N, K_max, _ = X.shape

    # Build presence prior matrix
    log_pi_null = presence.log_prior_null()
    presence_matrix = np.zeros((N, K_max), dtype=float)
    for i in range(N):
        s = int(peak_species[i])
        log_pi_compounds = presence.log_prior_prob(s)
        presence_matrix[i, 0] = log_pi_null
        candidates = row_to_candidates[i]
        for k in range(1, K_max):
            if mask[i, k] and k < len(candidates):
                c = candidates[k]
                if c is not None:
                    presence_matrix[i, k] = log_pi_compounds[c]

    # Compute logits
    feature_contrib = np.einsum("ijk,k->ij", X, theta_features)
    logits = np.full((N, K_max), -np.inf, dtype=float)
    logits[:, 0] = theta_null + presence_matrix[:, 0]
    for i in range(N):
        for k in range(1, K_max):
            if mask[i, k]:
                logits[i, k] = theta0 + feature_contrib[i, k] + presence_matrix[i, k]

    # Softmax over valid slots per row
    probs = np.zeros_like(logits)
    for i in range(N):
        valid = mask[i]
        vl = logits[i, valid]
        vl = vl - vl.max()
        exp_vl = np.exp(vl)
        probs[i, valid] = exp_vl / exp_vl.sum()

    # Map to compound IDs
    per_row_compound_probs = []
    for i in range(N):
        cand = row_to_candidates[i]
        row = []
        for k in range(1, K_max):  # exclude null
            if mask[i, k] and k < len(cand):
                c = cand[k]
                if c is not None:
                    row.append((int(c), float(probs[i, k])))
        per_row_compound_probs.append(row)
    return per_row_compound_probs


def test_candidate_order_invariance():
    # Small deterministic setup with multiple candidates per peak
    rng = np.random.default_rng(0)

    n_species = 1
    n_compounds = 6
    species = 0

    # Define compound masses clustered so that several fall within tolerance
    compound_mass = np.array([100.000, 100.003, 100.006, 120.0, 130.0, 100.009])

    # Two peaks near 100 Da to yield ~4 candidates each
    peaks = [
        {"peak_id": 0, "species": species, "mass": 100.004, "rt": 5.0, "intensity": 1e6, "true_compound": np.nan},
        {"peak_id": 1, "species": species, "mass": 100.007, "rt": 6.0, "intensity": 2e6, "true_compound": np.nan},
    ]
    peak_df = pd.DataFrame(peaks)

    # Create model with permissive windows so candidates are not filtered out
    model = PeakAssignment(mass_tolerance=0.015, rt_window_k=10.0, random_seed=0)

    # Provide RT predictions: mean near observed RT, moderate std
    model.rt_predictions = {}
    for c in range(n_compounds):
        model.rt_predictions[(species, c)] = (5.5, 0.5)

    # Initialize presence prior (uniform) explicitly for determinism
    presence = PresencePrior.init(n_species=n_species, n_compounds=n_compounds, smoothing=1.0)

    # Generate training data (no decoys here)
    model.generate_training_data(
        peak_df=peak_df,
        compound_mass=compound_mass,
        n_compounds=n_compounds,
        init_presence=presence,
        initial_labeled_fraction=0.0,
        random_seed=0,
    )

    # Fixed weights (make them non-zero to exercise exchangeability)
    theta_features = np.array([0.8, -0.6, 0.2, -0.1], dtype=float)  # for minimal features
    theta0 = 0.05
    theta_null = -0.2

    # Compute baseline probabilities
    base_rows = _compute_probs_from_pack(model.train_pack, model.presence, theta0, theta_null, theta_features)

    # Create a permuted copy of X_test/mask_test/row_to_candidates_test
    tp = model.train_pack
    Xp = tp["X_test"].copy()
    Mp = tp["mask_test"].copy()
    rows_perm = []
    K_max = Xp.shape[1]

    for i in range(Xp.shape[0]):
        # Indices of valid non-null slots
        valid_slots = [k for k in range(1, K_max) if Mp[i, k]]
        perm = rng.permutation(valid_slots)

        # Apply permutation by swapping rows in-place on copies
        X_row = Xp[i].copy()
        for src, dst in zip(valid_slots, perm):
            Xp[i, dst] = X_row[src]
        # Mask stays True on the same count of slots; we can leave Mp as is

        # Permute the candidates mapping accordingly
        cand = tp["row_to_candidates_test"][i]
        # Start with [None] placeholder
        new_cand = [cand[0]] + [None] * (K_max - 1)
        for src, dst in zip(valid_slots, perm):
            # Ensure we don't index beyond current mapping length
            if src < len(cand):
                new_cand[dst] = cand[src]
        # Fill any remaining valid slots that were not mapped (should not happen if lengths match)
        for k in valid_slots:
            if new_cand[k] is None and k < len(cand):
                new_cand[k] = cand[k]
        rows_perm.append(new_cand)

    # Build a shallow-copied train_pack with permuted test tensors/mapping
    perm_pack = dict(tp)
    perm_pack["X_test"] = Xp
    perm_pack["mask_test"] = Mp
    perm_pack["row_to_candidates_test"] = rows_perm

    # Compute probabilities after permutation
    perm_rows = _compute_probs_from_pack(perm_pack, model.presence, theta0, theta_null, theta_features)

    # Compare per-peak distributions by compound ID
    for row1, row2 in zip(base_rows, perm_rows):
        # Convert to dicts keyed by compound_id
        d1 = {c: p for c, p in row1}
        d2 = {c: p for c, p in row2}

        # The candidate sets should be identical
        assert set(d1.keys()) == set(d2.keys())

        # Probabilities must match up to small tolerance
        for cid in d1.keys():
            assert abs(d1[cid] - d2[cid]) < 1e-10
