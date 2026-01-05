from __future__ import annotations

import numpy as np

from compassign.generators.create_synthetic_data import create_metabolomics_data


def test_generator_outputs_descriptor_groups_and_budgets():
    # Small, deterministic config
    n_compounds = 30
    n_species = 4
    np.random.seed(42)

    peak_df, compound_df, _ta, _rtu, params = create_metabolomics_data(
        n_compounds=n_compounds,
        n_species=n_species,
        n_runs_per_species=2,
        n_internal_standards=4,
    )

    # Descriptor features present, correct shape
    Z = params.get("compound_features")
    assert Z is not None
    assert Z.shape == (n_compounds, 20)

    # chem_id column present
    assert "chem_id" in compound_df.columns

    # compound_group present
    assert "compound_group" in compound_df.columns
    groups = compound_df["compound_group"].astype(str)

    # Count labeled observations per compound from peak_df
    labeled = peak_df[peak_df["true_compound"].notna()]
    counts = labeled.groupby("true_compound").size()

    # unseen compounds should have 0 labels
    unseen_ids = compound_df.loc[groups == "unseen", "compound_id"].tolist()
    for cid in unseen_ids:
        assert int(counts.get(cid, 0)) == 0

    # rare compounds should have at most 2 labels
    rare_ids = compound_df.loc[groups == "rare", "compound_id"].tolist()
    for cid in rare_ids:
        assert int(counts.get(cid, 0)) <= 2

    # anchor compounds should have at least 3 labels
    anchor_ids = compound_df.loc[groups == "anchor", "compound_id"].tolist()
    for cid in anchor_ids:
        assert int(counts.get(cid, 0)) >= 3
