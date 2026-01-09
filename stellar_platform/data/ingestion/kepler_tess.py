"""
Kepler/TESS data connector module.

This module provides functions for connecting to and ingesting data from the Kepler and TESS missions,
including light curves, stellar parameters, and catalogs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
import os
from pathlib import Path
import logging

from stellar_platform.data.schemas import LightCurve, DataProvenance, QualityFlags
from stellar_platform.data.ingestion.sync_state import get_sync_manager, SyncState

logger = logging.getLogger(__name__)

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
    def incremental_sync_lightcurves(
        self,
        mission: str = "TESS",
        min_magnitude: float = 16.0,
        batch_size: int = 50,
        max_records: Optional[int] = None,
        resume: bool = True
    ) -> Dict[str, Any]:
        """Incrementally sync light curves with resumable state tracking.
        
        Args:
            mission: Mission name ("Kepler" or "TESS")
            min_magnitude: Minimum target magnitude filter
            batch_size: Number of light curves to fetch per batch
            max_records: Maximum records to process (None = unlimited)
            resume: Whether to resume from previous sync state
        
        Returns:
            Dictionary with sync statistics
        """
        sync_manager = get_sync_manager()
        survey_name = f"{mission.upper()}"
        
        # Load or create sync state
        if resume:
            state = sync_manager.load_state(survey_name)
            if state is None:
                state = SyncState(survey=survey_name)
                logger.info(f"Starting new sync for {survey_name}")
            else:
                logger.info(f"Resuming sync for {survey_name} from {state.records_processed} records")
        else:
            state = SyncState(survey=survey_name)
            logger.info(f"Starting fresh sync for {survey_name}")
        
        stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "start_time": datetime.now().isoformat()
        }
        
        try:
            # In a real implementation, this would query MAST incrementally
            cursor = int(state.cursor or "0")
            total_processed = state.records_processed
            
            while True:
                # Check max_records limit
                if max_records and total_processed >= max_records:
                    logger.info(f"Reached max_records limit: {max_records}")
                    break
                
                # Simulate batch fetch
                logger.info(f"Fetching batch at cursor={cursor}, records={total_processed}")
                
                # Placeholder: in real implementation, query like:
                # targets = query_tic_catalog(offset=cursor, limit=batch_size, min_mag=min_magnitude)
                batch_results = []  # Would be actual query results
                
                if not batch_results:
                    logger.info("No more records to process")
                    break
                
                # Process each light curve in batch
                for target_info in batch_results:
                    try:
                        # Download light curve
                        lc_data = self.download_lightcurve(
                            mission=mission,
                            target=target_info['tic_id']
                        )
                        
                        if lc_data:
                            # Create LightCurve object with provenance
                            provenance = DataProvenance(
                                survey=survey_name,
                                retrieval_timestamp=datetime.now(),
                                query_params={"target": target_info['tic_id'], "mission": mission},
                                version=target_info.get('data_release', 'latest')
                            )
                            
                            quality = QualityFlags(
                                is_valid=True,
                                signal_to_noise=target_info.get('mean_flux', 0) / target_info.get('std_flux', 1) if target_info.get('std_flux') else None
                            )
                            
                            lightcurve = LightCurve(
                                time=lc_data['time'],
                                flux=lc_data['flux'],
                                flux_unc=lc_data.get('flux_err'),
                                band=mission,
                                provenance=provenance,
                                quality=quality
                            )
                            
                            # Save light curve (in real implementation: store to data lake)
                            stats["successful"] += 1
                        else:
                            stats["skipped"] += 1
                    
                    except Exception as e:
                        logger.error(f"Failed to process target: {e}")
                        stats["failed"] += 1
                        sync_manager.update_state(survey_name, increment_errors=True)
                    
                    stats["processed"] += 1
                    total_processed += 1
                    cursor += 1
                    
                    # Update state periodically
                    if stats["processed"] % 10 == 0:
                        sync_manager.update_state(
                            survey_name,
                            records_processed=total_processed,
                            cursor=str(cursor)
                        )
                
                # Update state after each batch
                sync_manager.update_state(
                    survey_name,
                    records_processed=total_processed,
                    cursor=str(cursor)
                )
        
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            sync_manager.update_state(survey_name, increment_errors=True)
            stats["error"] = str(e)
        
        finally:
            # Final state save
            sync_manager.update_state(
                survey_name,
                records_processed=total_processed,
                cursor=str(cursor)
            )
            stats["end_time"] = datetime.now().isoformat()
            stats["total_records"] = total_processed
        
        return stats