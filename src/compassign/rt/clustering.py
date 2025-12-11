from __future__ import annotations

from typing import Iterable

import numpy as np


def infer_species_clusters_by_is(
    run_df,
    covariate_columns: Iterable[str],
    *,
    n_species: int,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    """
    Infer species drift clusters from run-level covariates by clustering
    per-species mean IS profiles.

    Parameters
    - run_df: pandas.DataFrame containing at least columns ["species", *covariate_columns]
    - covariate_columns: iterable of run covariate column names
    - n_species: total number of species in the dataset (used to ensure full index)
    - n_clusters: desired number of clusters (clamped to [1, n_species])
    - seed: random seed for KMeans initialisation

    Returns
    - np.ndarray of length n_species with integer cluster ids in [0, n_clusters)
    """
    # Local import to avoid adding heavy deps globally at import time
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    rdf = run_df.copy()
    # Aggregate per-species mean covariate vector
    mean_by_species = rdf.groupby("species", sort=True)[list(covariate_columns)].mean()
    # Ensure index covers all species 0..n_species-1
    if mean_by_species.index.min() != 0 or mean_by_species.index.max() != (n_species - 1):
        mean_by_species = mean_by_species.reindex(range(n_species))
        mean_by_species = mean_by_species.fillna(mean_by_species.mean())
    X = mean_by_species.to_numpy(dtype=float)
    # Standardise before KMeans
    Xs = StandardScaler().fit_transform(X)
    k = int(max(1, min(int(n_clusters), int(n_species))))
    labels = KMeans(n_clusters=k, random_state=int(seed), n_init=10).fit_predict(Xs)
    return labels.astype(int)


# report_cluster_agreement was intentionally removed per design decision
