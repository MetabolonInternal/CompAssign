from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.create_synthetic_data import create_metabolomics_data
from src.compassign.rt.hierarchical import HierarchicalRTModel


@pytest.mark.slow
def test_descriptors_improve_mae_on_rare_subset():
    np.random.seed(42)

    # Small dataset to keep MCMC light
    n_compounds = 30
    n_species = 6
    peak_df, compound_df, _ta, _rtu, params = create_metabolomics_data(
        n_compounds=n_compounds,
        n_species=n_species,
        n_runs_per_species=2,
        n_internal_standards=6,
    )

    # Build RT dataframe
    rt_df = peak_df[peak_df["true_compound"].notna()].copy()
    rt_df = rt_df.rename(columns={"true_compound": "compound"})
    rt_df = rt_df[["species", "compound", "run", "rt"]].astype(int).astype({"rt": float})

    # Run metadata
    run_meta_cols = [c for c in peak_df.columns if c.startswith("run_covariate_")]
    run_df = peak_df[["run", "species", *run_meta_cols]].drop_duplicates("run")

    # Train/test split honoring groups
    comp_groups = compound_df.set_index("compound_id")["compound_group"].to_dict()
    comp_grp_series = rt_df["compound"].map(comp_groups)
    unseen_mask = comp_grp_series == "unseen"
    rare_mask = comp_grp_series == "rare"
    anchor_mask = comp_grp_series == "anchor"

    # All unseen → test; rare → cap 2 rows in train; anchors → simple split by species
    unseen_rows = rt_df[unseen_mask]
    rare_rows = rt_df[rare_mask]
    train_rare_parts = []
    test_rare_parts = []
    for cid, sub in rare_rows.groupby("compound"):
        sub = sub.sample(frac=1.0, random_state=42)
        take = min(2, len(sub))
        train_rare_parts.append(sub.iloc[:take])
        if len(sub) > take:
            test_rare_parts.append(sub.iloc[take:])
    train_rare = pd.concat(train_rare_parts, ignore_index=True) if train_rare_parts else rare_rows.iloc[0:0]
    test_rare = pd.concat(test_rare_parts, ignore_index=True) if test_rare_parts else rare_rows.iloc[0:0]

    anchors = rt_df[anchor_mask]
    # Simple 80/20 by species label to keep deterministic; small sizes
    train_anchor = anchors.sample(frac=0.8, random_state=42)
    test_anchor = anchors.loc[~anchors.index.isin(train_anchor.index)]

    train_df = pd.concat([train_anchor, train_rare], ignore_index=True)
    test_df = pd.concat([test_anchor, test_rare, unseen_rows], ignore_index=True)

    # Prepare model args
    args = dict(
        n_clusters=params["n_clusters"],
        n_species=n_species,
        n_classes=params["n_classes"],
        n_compounds=n_compounds,
        species_cluster=params["species_cluster"],
        compound_class=params["compound_class"],
        run_metadata=run_df,
        run_covariate_columns=run_meta_cols,
    )

    # Train with descriptors
    Z = params["compound_features"]
    model_desc = HierarchicalRTModel(**args, compound_features=Z)
    model_desc.build_model(train_df)
    trace_desc = model_desc.sample(n_samples=50, n_tune=50, n_chains=1, target_accept=0.99, random_seed=123)
    mean_d, _std_d = model_desc.predict_new(
        species_idx=test_df["species"].to_numpy(),
        compound_idx=test_df["compound"].to_numpy(),
        run_idx=test_df["run"].to_numpy(),
        n_samples=50,
    )

    # Train without descriptors (ablation)
    model_nod = HierarchicalRTModel(**args, compound_features=None)
    model_nod.build_model(train_df)
    trace_nod = model_nod.sample(n_samples=50, n_tune=50, n_chains=1, target_accept=0.99, random_seed=456)
    mean_n, _std_n = model_nod.predict_new(
        species_idx=test_df["species"].to_numpy(),
        compound_idx=test_df["compound"].to_numpy(),
        run_idx=test_df["run"].to_numpy(),
        n_samples=50,
    )

    # Evaluate on rare subset only
    rare_mask_test = test_df["compound"].map(comp_groups) == "rare"
    y_true = test_df.loc[rare_mask_test, "rt"].to_numpy()
    y_d = mean_d[rare_mask_test.to_numpy()]
    y_n = mean_n[rare_mask_test.to_numpy()]

    mae_d = float(np.mean(np.abs(y_d - y_true))) if y_true.size else np.inf
    mae_n = float(np.mean(np.abs(y_n - y_true))) if y_true.size else np.inf

    # Descriptors should not be worse and typically better on rares
    assert mae_d <= mae_n + 1e-6
