from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from compassign.rt.prod_csv_loader import load_production_csv


def _make_synthetic_rt_csv(tmp_path: Path) -> Path:
    """Create a tiny production-style RT CSV with multiple groups per species."""
    rows = [
        # species 100 in group 0 and 1
        {
            "sampleset_id": 1,
            "worksheet_id": 10,
            "task_id": 100,
            "species": 100,
            "species_cluster": 0,
            "compound": 1000,
            "compound_class": 1,
            "rt": 1.0,
            "IS1": 0.1,
        },
        {
            "sampleset_id": 2,
            "worksheet_id": 10,
            "task_id": 101,
            "species": 100,
            "species_cluster": 1,
            "compound": 1000,
            "compound_class": 1,
            "rt": 1.1,
            "IS1": 0.2,
        },
        # species 200 only in group 0
        {
            "sampleset_id": 3,
            "worksheet_id": 11,
            "task_id": 102,
            "species": 200,
            "species_cluster": 0,
            "compound": 2000,
            "compound_class": 2,
            "rt": 2.0,
            "IS1": 0.3,
        },
    ]
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "synthetic_rt_prod.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_loader_builds_species_at_species_cluster_level(tmp_path):
    csv_path = _make_synthetic_rt_csv(tmp_path)

    loaded = load_production_csv(csv_path)

    # We expect three distinct (species, species_cluster) pairs.
    assert loaded.n_species == 3
    assert loaded.species_cluster.shape == (3,)
    # There are two distinct raw clusters (0, 1), renumbered to 0..1.
    assert sorted(set(loaded.species_cluster.tolist())) == [0, 1]

    # Species map should be keyed by (raw_species, raw_cluster).
    assert isinstance(loaded.species_map, dict)
    keys: Dict[Tuple[int, int], int] = loaded.species_map
    assert (100, 0) in keys
    assert (100, 1) in keys
    assert (200, 0) in keys
    # The two entries for species 100 must map to different indices.
    assert keys[(100, 0)] != keys[(100, 1)]

    # The remapped rt_df species indices must be in 0..n_species-1 and align with run_species.
    species_indices = loaded.rt_df["species"].to_numpy(dtype=int)
    assert species_indices.min() >= 0
    assert species_indices.max() < loaded.n_species
    assert np.all(loaded.run_species >= 0)
    assert np.all(loaded.run_species < loaded.n_species)
