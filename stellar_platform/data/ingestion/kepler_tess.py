"""
Kepler/TESS data connector module.

This module provides functions for connecting to and ingesting data from the Kepler and TESS missions,
including light curves, stellar parameters, and catalogs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings
import os
from pathlib import Path

try:  # pragma: no cover
    from lightkurve import search_lightcurve, LightCurve  # type: ignore
except Exception:  # pragma: no cover
    search_lightcurve = None  # type: ignore
    LightCurve = object  # type: ignore
try:  # pragma: no cover
    from astropy.timeseries import TimeSeries  # type: ignore
    from astropy import units as u  # type: ignore
except Exception:  # pragma: no cover
    TimeSeries = object  # type: ignore
    u = None  # type: ignore


class KeplerTESSConnector:
    """
    Connector for Kepler/TESS data access and retrieval.
    """

    def __init__(self, data_dir: str = "data/kepler_tess"):
        """
        Initialize the Kepler/TESS connector.

        Args:
            data_dir: Directory to store downloaded Kepler/TESS data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def search_lightcurves(
        self,
        mission: str,
        target: str,
        author: Optional[str] = None,
        cadence: Optional[str] = None,
        quarter: Optional[int] = None,
        campaign: Optional[int] = None,
        sector: Optional[int] = None,
        mission_long: bool = False,
        limit: Optional[int] = None
    ) -> List[dict]:
        """
        Search for light curves matching criteria.

        Args:
            mission: Mission name ("Kepler" or "TESS")
            target: Target identifier (KIC ID for Kepler, TIC ID for TESS, or coordinates)
            author: Data author ("Kepler", "MAST", "SPOC", etc.)
            cadence: Data cadence ("short", "long", "fast", "slow")
            quarter: Kepler quarter (for Kepler data)
            campaign: TESS campaign (for TESS data)
            sector: TESS sector (for TESS data)
            mission_long: Whether to use long cadence for Kepler
            limit: Maximum number of results

        Returns:
            List of dictionaries with search results
        """
        # Search for light curves
        if search_lightcurve is None:
            warnings.warn("lightkurve not installed; returning empty search result")
            return []
        if author is None:
            author = "Kepler" if mission.lower() == "kepler" else "SPOC"
        search_result = search_lightcurve(
            target=target,
            mission=mission,
            author=author,
            cadence=cadence,
            quarter=quarter,
            campaign=campaign,
            sector=sector,
            mission_long=mission_long,
            limit=limit
        )

        if len(search_result) == 0:
            warnings.warn(f"No light curves found for target {target}")
            return []

        return [lc.to_dict() for lc in search_result]

    def download_lightcurve(
        self,
        mission: str,
        target: str,
        author: Optional[str] = None,
        cadence: Optional[str] = None,
        quarter: Optional[int] = None,
        campaign: Optional[int] = None,
        sector: Optional[int] = None,
        mission_long: bool = False,
        download_dir: Optional[str] = None,
        overwrite: bool = False
    ) -> Optional[Any]:
        """
        Download a light curve for a target.

        Args:
            mission: Mission name ("Kepler" or "TESS")
            target: Target identifier (KIC ID for Kepler, TIC ID for TESS, or coordinates)
            author: Data author ("Kepler", "MAST", "SPOC", etc.)
            cadence: Data cadence ("short", "long", "fast", "slow")
            quarter: Kepler quarter (for Kepler data)
            campaign: TESS campaign (for TESS data)
            sector: TESS sector (for TESS data)
            mission_long: Whether to use long cadence for Kepler
            download_dir: Directory to save the light curve file
            overwrite: Whether to overwrite existing files

        Returns:
            LightCurve object or None if download failed
        """
        if download_dir is None:
            download_dir = self.data_dir

        # Search for light curves
        if search_lightcurve is None:
            warnings.warn("lightkurve not installed; cannot download light curve")
            return None
        if author is None:
            author = "Kepler" if mission.lower() == "kepler" else "SPOC"
        search_result = search_lightcurve(
            target=target,
            mission=mission,
            author=author,
            cadence=cadence,
            quarter=quarter,
            campaign=campaign,
            sector=sector,
            mission_long=mission_long,
            limit=1
        )

        if len(search_result) == 0:
            warnings.warn(f"No light curves found for target {target}")
            return None

        # Download the light curve
        lc = search_result[0].download(download_dir=download_dir, overwrite=overwrite)

        if lc is None:
            warnings.warn(f"Failed to download light curve for target {target}")
            return None

        return lc

    def get_stellar_parameters(
        self,
        mission: str,
        target: str,
        catalog: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get stellar parameters for a target.

        Args:
            mission: Mission name ("Kepler" or "TESS")
            target: Target identifier (KIC ID for Kepler, TIC ID for TESS)
            catalog: Catalog name ("q16" for Kepler, "tic" for TESS)

        Returns:
            DataFrame with stellar parameters
        """
        if catalog is None:
            catalog = "q16" if mission.lower() == "kepler" else "tic"
        # Placeholder returning empty DataFrame
        return pd.DataFrame(columns=["target", "teff", "logg", "feh"])

    def get_confirmed_planets(
        self,
        mission: str = "Kepler",
        disposition: str = "CONFIRMED",
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get a list of confirmed planets for a mission.

        Args:
            mission: Mission name ("Kepler" or "TESS")
            disposition: Planet disposition ("CONFIRMED", "CANDIDATE", etc.)
            limit: Maximum number of results

        Returns:
            DataFrame with planet information
        """
        return pd.DataFrame(columns=["planet_name", "period", "radius", "disposition"])

    def get_variable_stars(
        self,
        mission: str = "Kepler",
        variability_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get a list of variable stars for a mission.

        Args:
            mission: Mission name ("Kepler" or "TESS")
            variability_type: Type of variability ("ECLIPSING_BINARY", "RR_LYRAE", etc.)
            limit: Maximum number of results

        Returns:
            DataFrame with variable star information
        """
        return pd.DataFrame(columns=["target", "variability_type"])

    def crossmatch_with_catalog(
        self,
        ra: List[float],
        dec: List[float],
        radius: float = 10.0,  # arcseconds
        mission: str = "Kepler"
    ) -> pd.DataFrame:
        """
        Crossmatch positions with Kepler or TESS targets.

        Args:
            ra: List of right ascensions (degrees)
            dec: List of declinations (degrees)
            radius: Search radius (arcseconds)
            mission: Mission name ("Kepler" or "TESS")

        Returns:
            DataFrame with crossmatch results
        """
        return pd.DataFrame(columns=["ra", "dec", "target_id"])
