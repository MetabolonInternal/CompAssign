"""Lightweight dataset containers for synthetic data.

Provides a thin dataclass wrapper around the synthetic generator outputs to keep
call signatures tidy for consumers of individual frames and dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

from .run_metadata import RunMetadata, extract_run_metadata


@dataclass(frozen=True)
class SyntheticDataset:
    """Synthetic metabolomics dataset returned by the generator.

    Attributes
    ----------
    peak_df : pd.DataFrame
        Peaks with embedded run-level covariates (``run_covariate_*``).
    compound_df : pd.DataFrame
        Compound library with masses and properties.
    true_assignments : Mapping[int, Optional[int]]
        Peak ID to true compound ID mapping (``None`` for noise peaks).
    rt_uncertainties : Mapping[int, float]
        Homoscedastic observation noise by compound.
    hierarchical_params : Dict[str, Any]
        Hierarchical structure parameters and generator metadata.
    """

    peak_df: pd.DataFrame
    compound_df: pd.DataFrame
    true_assignments: Mapping[int, Optional[int]]
    rt_uncertainties: Mapping[int, float]
    hierarchical_params: Dict[str, Any]

    def run_meta(self, covariate_columns: Optional[Sequence[str]] = None) -> RunMetadata:
        """Extract run-level metadata aligned with the dataset.

        Parameters
        ----------
        covariate_columns : Optional[Sequence[str]]
            Explicit covariate columns. If omitted, uses the generator-provided
            ``hierarchical_params['run_covariate_columns']`` when available.

        Returns
        -------
        RunMetadata
            Tidy run dataframe with aligned feature and species arrays.
        """

        cols = covariate_columns
        if cols is None:
            cols = self.hierarchical_params.get("run_covariate_columns")
        return extract_run_metadata(self.peak_df, covariate_columns=cols)
