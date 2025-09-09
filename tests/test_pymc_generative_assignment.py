import numpy as np
import pandas as pd

from src.compassign.pymc_generative_assignment import GenerativeAssignmentModel


def make_synthetic_small():
    # 2 species, 3 compounds
    n_species = 2
    n_compounds = 3
    compound_mass = np.array([100.0, 105.0, 110.0])

    # RT means per compound (same across species for simplicity)
    rt_mu_comp = np.array([5.0, 7.0, 9.0])
    rt_sd_pred = 0.05

    peaks = []
    peak_id = 0
    rng = np.random.default_rng(0)
    for s in range(n_species):
        for c in range(n_compounds):
            mz = compound_mass[c] + rng.normal(0, 0.001)
            rt = rt_mu_comp[c] + rng.normal(0, 0.05)
            intensity = float(np.exp(rng.normal(2.0, 0.2)))
            peaks.append({
                'peak_id': peak_id,
                'species': s,
                'mass': mz,
                'rt': rt,
                'intensity': intensity,
                'true_compound': c,
            })
            peak_id += 1
    # Add one noise peak per species
    for s in range(n_species):
        mz = 120.0 + rng.normal(0, 0.01)
        rt = 12.0 + rng.normal(0, 0.2)
        intensity = float(np.exp(rng.normal(2.0, 0.2)))
        peaks.append({
            'peak_id': peak_id,
            'species': s,
            'mass': mz,
            'rt': rt,
            'intensity': intensity,
            'true_compound': None,
        })
        peak_id += 1

    peak_df = pd.DataFrame(peaks)

    # Build rt_predictions dict directly
    rt_predictions = {}
    for s in range(n_species):
        for c in range(n_compounds):
            rt_predictions[(s, c)] = (float(rt_mu_comp[c]), float(rt_sd_pred))

    return peak_df, compound_mass, n_compounds, n_species, rt_predictions


def test_generative_shapes_and_masking():
    peak_df, compound_mass, n_compounds, n_species, rt_predictions = make_synthetic_small()

    model = GenerativeAssignmentModel(mass_tolerance=0.01, rt_window_k=3.0, random_seed=0)
    model.rt_predictions = rt_predictions  # set directly for test speed

    pack = model.generate_training_data(
        peak_df=peak_df,
        compound_mass=compound_mass,
        n_compounds=n_compounds,
    )

    assert 'mask' in pack and 'X' in pack and 'compound_idx' in pack
    N, K = pack['mask'].shape
    # Null slot always valid
    assert pack['mask'][:, 0].all()
    # Candidate indices valid range or -1
    comp_ok = (pack['compound_idx'][:, 1:] < n_compounds) | (pack['compound_idx'][:, 1:] == -1)
    assert comp_ok.all()


def test_generative_prior_mode_predictions_reasonable():
    peak_df, compound_mass, n_compounds, n_species, rt_predictions = make_synthetic_small()

    model = GenerativeAssignmentModel(mass_tolerance=0.02, rt_window_k=3.0, random_seed=0)
    model.rt_predictions = rt_predictions
    model.generate_training_data(peak_df, compound_mass, n_compounds)

    # No sampling: use prior-mode heuristic
    probs = model.predict_probs()
    assert isinstance(probs, dict) and len(probs) == len(peak_df)

    # Make assignments and check they are not empty
    results = model.assign(prob_threshold=0.5)
    # We expect most true peaks assigned, noise peaks likely null
    assigned = sum(1 for v in results.assignments.values() if v is not None)
    assert assigned >= (len(peak_df) - n_species) - 2  # allow some slack

