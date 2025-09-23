"""
Gaia data connector module.

This module provides functions for connecting to and ingesting data from the Gaia mission,
including astrometry, photometry, and spectroscopic information.
"""

import numpy as np
import pandas as pd
try:  # pragma: no cover
    from astroquery.gaia import Gaia  # type: ignore
except Exception:  # pragma: no cover
    Gaia = None  # type: ignore
try:  # pragma: no cover
    from astropy.io import fits  # type: ignore  
except Exception:  # pragma: no cover
    fits = None  # type: ignore
from typing import Dict, List, Optional, Tuple, Any
import warnings
import os
from pathlib import Path


class GaiaConnector:
    """
    Connector for Gaia data access and retrieval.
    """

    def __init__(self, data_dir: str = "data/gaia"):
        """
        Initialize the Gaia connector.

        Args:
            data_dir: Directory to store downloaded Gaia data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def query_source(
        self,
        source_id: int,
        release: str = "latest",
        data_release: int = 3
    ) -> pd.DataFrame:
        """
        Query a Gaia source by its source ID.

        Args:
            source_id: Gaia source ID
            release: Data release (e.g., "latest", "EDR3", "DR3")
            data_release: Data release number (2 or 3)

        Returns:
            DataFrame with source information
        """
        if Gaia is None:
            warnings.warn("astroquery.gaia not installed; returning empty DataFrame")
            return pd.DataFrame()

        query = f"""
        SELECT 
            source_id, ra, dec, parallax, parallax_error, 
            pmra, pmra_error, pmdec, pmdec_error,
            phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
            radial_velocity, radial_velocity_error,
            teff_val, logg, mh, lum_val, alphafe
        FROM gaiadr{data_release}.gaia_source
        WHERE source_id = {source_id}
        """

        # Execute the query
        try:
            job = Gaia.launch_job(query)  # type: ignore
            results = job.get_results()
        except Exception as e:
            warnings.warn(f"Gaia query failed: {e}")
            return pd.DataFrame()

        if len(results) == 0:
            warnings.warn(f"No results found for source ID {source_id}")
            return pd.DataFrame()

        return results.to_pandas()

    def query_region(
        self,
        min_ra: float,
        max_ra: float,
        min_dec: float,
        max_dec: float,
        parallax_max: Optional[float] = None,
        phot_g_min: Optional[float] = None,
        phot_g_max: Optional[float] = None,
        release: str = "latest",
        data_release: int = 3,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Query Gaia sources in a specified region.

        Args:
            min_ra: Minimum right ascension (degrees)
            max_ra: Maximum right ascension (degrees)
            min_dec: Minimum declination (degrees)
            max_dec: Maximum declination (degrees)
            parallax_max: Maximum parallax (mas) - useful for selecting nearby stars
            phot_g_min: Minimum G-band magnitude
            phot_g_max: Maximum G-band magnitude
            release: Data release (e.g., "latest", "EDR3", "DR3")
            data_release: Data release number (2 or 3)
            limit: Maximum number of results to return

        Returns:
            DataFrame with query results
        """
        query = f"""
        SELECT 
            source_id, ra, dec, parallax, parallax_error, 
            pmra, pmra_error, pmdec, pmdec_error,
            phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
            radial_velocity, radial_velocity_error,
            teff_val, logg, mh, lum_val, alphafe
        FROM gaiadr{data_release}.gaia_source
        WHERE 
            ra BETWEEN {min_ra} AND {max_ra}
            AND dec BETWEEN {min_dec} AND {max_dec}
        """

        if parallax_max is not None:
            query += f" AND parallax < {parallax_max}"

        if phot_g_min is not None:
            query += f" AND phot_g_mean_mag > {phot_g_min}"

        if phot_g_max is not None:
            query += f" AND phot_g_mean_mag < {phot_g_max}"

        if limit is not None:
            query += f" LIMIT {limit}"

        # Execute the query
        if Gaia is None:
            warnings.warn("astroquery.gaia not installed; returning empty DataFrame")
            return pd.DataFrame()
        try:
            job = Gaia.launch_job(query)  # type: ignore
            results = job.get_results()
        except Exception as e:
            warnings.warn(f"Gaia region query failed: {e}")
            return pd.DataFrame()

        if len(results) == 0:
            warnings.warn("No results found for the given query")
            return pd.DataFrame()

        return results.to_pandas()

    def query_xmatch(
        self,
        ra: List[float],
        dec: List[float],
        radius: float = 1.0,  # arcseconds
        phot_g_min: Optional[float] = None,
        phot_g_max: Optional[float] = None,
        release: str = "latest",
        data_release: int = 3
    ) -> pd.DataFrame:
        """
        Crossmatch a list of positions with Gaia sources.

        Args:
            ra: List of right ascensions (degrees)
            dec: List of declinations (degrees)
            radius: Search radius (arcseconds)
            phot_g_min: Minimum G-band magnitude
            phot_g_max: Maximum G-band magnitude
            release: Data release (e.g., "latest", "EDR3", "DR3")
            data_release: Data release number (2 or 3)

        Returns:
            DataFrame with crossmatch results
        """
        # Create a temporary table with the positions
        positions = pd.DataFrame({
            'ra': ra,
            'dec': dec,
            'index': range(len(ra))
        })

        # Save to CSV for upload
        temp_file = os.path.join(self.data_dir, "temp_crossmatch.csv")
        positions.to_csv(temp_file, index=False)

        # Build the query
        query = f"""
        SELECT 
            idx.source_id, idx.ra, idx.dec, idx.parallax, idx.parallax_error, 
            idx.pmra, idx.pmra_error, idx.pmdec, idx.pmdec_error,
            idx.phot_g_mean_mag, idx.phot_bp_mean_mag, idx.phot_rp_mean_mag,
            idx.radial_velocity, idx.radial_velocity_error,
            idx.teff_val, idx.logg, idx.mh, idx.lum_val, idx.alphafe,
            dist.angular_separation
        FROM (
            SELECT 
                index, source_id, ra, dec, parallax, parallax_error, 
                pmra, pmra_error, pmdec, pmdec_error,
                phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                radial_velocity, radial_velocity_error,
                teff_val, logg, mh, lum_val, alphafe,
                1 AS angular_separation
            FROM gaiadr{data_release}.gaia_source AS g
            JOIN (
                SELECT * FROM external.gaia_crossmatch_default AS x
                JOIN (
                    SELECT * FROM tap_upload.upload_table('{temp_file}', 'csv') AS t
                ) AS u ON x.original_index = t.index
            ) AS x ON g.source_id = x.source_id
        ) AS idx
        WHERE 1=1
        """

        if phot_g_min is not None:
            query += f" AND idx.phot_g_mean_mag > {phot_g_min}"

        if phot_g_max is not None:
            query += f" AND idx.phot_g_mean_mag < {phot_g_max}"

        # Execute the query
        if Gaia is None:
            warnings.warn("astroquery.gaia not installed; returning empty DataFrame")
            return pd.DataFrame()
        try:
            job = Gaia.launch_job(query)  # type: ignore
            results = job.get_results()
        except Exception as e:
            warnings.warn(f"Gaia crossmatch query failed: {e}")
            return pd.DataFrame()

        # Clean up temporary file
        os.remove(temp_file)

        if len(results) == 0:
            warnings.warn("No results found for the crossmatch")
            return pd.DataFrame()

        return results.to_pandas()

    def get_lightcurve(self, source_id: int, release: str = "latest", data_release: int = 3) -> pd.DataFrame:
        """
        Get the light curve for a Gaia source.

        Args:
            source_id: Gaia source ID
            release: Data release (e.g., "latest", "EDR3", "DR3")
            data_release: Data release number (2 or 3)

        Returns:
            DataFrame with light curve data
        """
        query = f"""
        SELECT 
            source_id, time, flux, flux_error, band
        FROM gaiadr{data_release}.gaia_lightcurve
        WHERE source_id = {source_id}
        ORDER BY time
        """

        # Execute the query
        if Gaia is None:
            warnings.warn("astroquery.gaia not installed; returning empty DataFrame")
            return pd.DataFrame()
        try:
            job = Gaia.launch_job(query)  # type: ignore
            results = job.get_results()
        except Exception as e:
            warnings.warn(f"Gaia lightcurve query failed: {e}")
            return pd.DataFrame()

        if len(results) == 0:
            warnings.warn(f"No light curve found for source ID {source_id}")
            return pd.DataFrame()

        return results.to_pandas()

    def get_rv_timeseries(self, source_id: int, release: str = "latest", data_release: int = 3) -> pd.DataFrame:
        """
        Get the radial velocity time series for a Gaia source.

        Args:
            source_id: Gaia source ID
            release: Data release (e.g., "latest", "EDR3", "DR3")
            data_release: Data release number (2 or 3)

        Returns:
            DataFrame with radial velocity time series data
        """
        query = f"""
        SELECT 
            source_id, time, radial_velocity, radial_velocity_error
        FROM gaiadr{data_release}.gaia_rv_time_series
        WHERE source_id = {source_id}
        ORDER BY time
        """

        # Execute the query
        if Gaia is None:
            warnings.warn("astroquery.gaia not installed; returning empty DataFrame")
            return pd.DataFrame()
        try:
            job = Gaia.launch_job(query)  # type: ignore
            results = job.get_results()
        except Exception as e:
            warnings.warn(f"Gaia RV query failed: {e}")
            return pd.DataFrame()

        if len(results) == 0:
            warnings.warn(f"No radial velocity time series found for source ID {source_id}")
            return pd.DataFrame()

        return results.to_pandas()
