"""
Spectral Energy Distribution (SED) preprocessing module for stellar photometry.

This module provides functions for preprocessing SED data including:
- Band merging
- Extinction corrections
- Uncertainty propagation
- Feature vector creation for machine learning
"""

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table
from typing import Tuple, Optional, Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)


def load_sed(file_path: str) -> Table:
    """
    Load an SED from a file.

    Args:
        file_path: Path to the file containing the SED

    Returns:
        Astropy Table containing the SED data
    """
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_table(file_path, sep='[ ,\t]+', engine='python')
    # Expect columns: band, mag or flux, (optional) err
    if 'band' not in df.columns:
        raise ValueError('SED file must contain a band column')
    # Standardize flux representation: if mag present convert to relative flux (placeholder zero-point)
    if 'flux' not in df.columns and 'mag' in df.columns:
        # Simple mag->flux: F = 10^(-0.4 * mag)
        df['flux'] = 10 ** (-0.4 * df['mag'].astype(float))
    if 'flux_err' not in df.columns and 'mag_err' in df.columns:
        # dF/F = ln(10)*0.4*dmag
        df['flux_err'] = df['flux'] * np.log(10) * 0.4 * df['mag_err'].astype(float)
    table = Table.from_pandas(df)
    return table


def merge_bands(sed_table: Table, band_mapping: Dict[str, str]) -> Table:
    """
    Merge different photometric bands in an SED.

    Args:
        sed_table: Input Table containing SED data
        band_mapping: Dictionary mapping original band names to merged band names

    Returns:
        Table with merged bands
    """
    if 'band' not in sed_table.colnames:
        return sed_table
    df = sed_table.to_pandas()
    df['merged_band'] = df['band'].map(lambda b: band_mapping.get(str(b), str(b)))
    # Aggregate flux by merged band (mean), propagate simple inverse variance weighting if errors
    if 'flux_err' in df.columns:
        grouped = []
        for name, g in df.groupby('merged_band'):
            if 'flux_err' in g.columns and g['flux_err'].notna().all():
                w = 1 / (g['flux_err']**2 + 1e-12)
                flux = np.sum(g['flux'] * w) / np.sum(w)
                err = np.sqrt(1 / np.sum(w))
            else:
                flux = g['flux'].mean()
                err = g['flux'].std(ddof=1) if len(g)>1 else np.nan
            grouped.append({'band': name, 'flux': flux, 'flux_err': err})
        out = pd.DataFrame(grouped)
    else:
        out = df.groupby('merged_band')['flux'].mean().reset_index().rename(columns={'merged_band':'band'})
    return Table.from_pandas(out)


def apply_extinction_correction(
    sed_table: Table,
    extinction_law: str = 'ccm89',
    ebv: Optional[float] = None,
    r_v: float = 3.1
) -> Table:
    """
    Apply extinction correction to an SED.

    Args:
        sed_table: Input Table containing SED data
        extinction_law: Extinction law to use ('ccm89', 'f99', etc.)
        ebv: Color excess E(B-V)
        r_v: Ratio of total to selective extinction

    Returns:
        Table with extinction-corrected fluxes
    """
    if ebv is None:
        return sed_table
    df = sed_table.to_pandas()
    # Placeholder: apply uniform dimming A_lambda = R_V * E(B-V)
    a_lambda = r_v * ebv
    df['flux'] = df['flux'] * 10 ** (0.4 * a_lambda)
    if 'flux_err' in df.columns:
        df['flux_err'] = df['flux_err'] * 10 ** (0.4 * a_lambda)
    return Table.from_pandas(df)


def propagate_uncertainties(sed_table: Table, operations: List[str]) -> Table:
    """
    Propagate uncertainties through operations on an SED.

    Args:
        sed_table: Input Table containing SED data with uncertainties
        operations: List of operations performed on the data

    Returns:
        Table with propagated uncertainties
    """
    # Minimal placeholder: no extra propagation besides existing flux_err
    return sed_table


def create_feature_vector(sed_table: Table, features: List[str]) -> np.ndarray:
    """
    Create a feature vector from an SED for machine learning.

    Args:
        sed_table: Input Table containing SED data
        features: List of feature names to extract

    Returns:
        Numpy array containing the feature vector
    """
    df = sed_table.to_pandas()
    feat_vals: List[float] = []
    for name in features:
        if name.startswith('flux_'):
            band = name.split('flux_')[-1]
            row = df[df['band'] == band]
            if not row.empty:
                feat_vals.append(float(row['flux'].iloc[0]))
            else:
                feat_vals.append(np.nan)
        elif name == 'num_bands':
            feat_vals.append(df['band'].nunique())
        elif name == 'flux_sum':
            feat_vals.append(df['flux'].sum())
        else:
            feat_vals.append(np.nan)
    return np.array(feat_vals, dtype=float)


def preprocess_sed(
    file_path: str,
    band_mapping: Optional[Dict[str, str]] = None,
    extinction_law: str = 'ccm89',
    ebv: Optional[float] = None,
    r_v: float = 3.1,
    features: Optional[List[str]] = None
) -> Union[Table, np.ndarray]:
    """
    Complete preprocessing pipeline for stellar SEDs.

    Args:
        file_path: Path to the file containing the SED
        band_mapping: Optional mapping for band merging
        extinction_law: Extinction law to use
        ebv: Color excess for extinction correction
        r_v: Ratio of total to selective extinction
        features: List of features to extract

    Returns:
        Preprocessed Table or feature array
    """
    # Load SED
    sed_table = load_sed(file_path)

    # Apply preprocessing steps
    if band_mapping is not None:
        sed_table = merge_bands(sed_table, band_mapping)

    if ebv is not None:
        sed_table = apply_extinction_correction(sed_table, extinction_law, ebv, r_v)

    if features is not None:
        return create_feature_vector(sed_table, features)

    return sed_table
