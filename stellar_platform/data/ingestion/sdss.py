"""
SDSS data connector module.

Provides functions for downloading spectra and querying photometry from the
Sloan Digital Sky Survey (SDSS). Includes provenance tracking and quality flags.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Any, Dict
from datetime import datetime

import pandas as pd
import numpy as np

from stellar_platform.data.schemas import Spectrum, DataProvenance, QualityFlags
from stellar_platform.data.ingestion.sync_state import get_sync_manager, SyncState

try:
    from astroquery.sdss import SDSS  # type: ignore
    from astropy import coordinates as coord
    from astropy import units as u
    from astropy.io import fits
except Exception:  # pragma: no cover - allow import without astroquery installed yet
    SDSS = None  # type: ignore
    fits = None  # type: ignore

logger = logging.getLogger(__name__)


class SDSSConnector:
    """Connector for SDSS data access.

    Methods implemented are intentionally minimal for v1.0 scope.
    Real implementation would support authentication, caching policies,
    error classification, and richer schema normalization.
    """

    def __init__(self, data_dir: str = "data/sdss"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------- Spectra ---------------------------------- #
    def download_spectrum(self, object_id: Any, force: bool = False) -> Path:
        """Download an SDSS spectrum by plate-mjd-fiber or specobjid.

        Args:
            object_id: Either a tuple (plate, mjd, fiberid) or an integer specobjid
            force: Redownload even if file present
        Returns:
            Path to local FITS file
        """
        target_path = self.data_dir / f"spectrum_{object_id}.fits"
        if target_path.exists() and not force:
            return target_path

        if SDSS is None:
            # Offline placeholder file
            logger.warning("astroquery.sdss not available; creating placeholder spectrum")
            with open(target_path, "wb") as f:
                f.write(b"")
            return target_path

        try:
            if isinstance(object_id, (list, tuple)) and len(object_id) == 3:
                plate, mjd, fiber = object_id
                sp = SDSS.get_spectra(plate=plate, mjd=mjd, fiberID=fiber)
            else:
                sp = SDSS.get_spectra(specobjid=object_id)
            if not sp:
                raise RuntimeError("No spectrum returned from SDSS")
            hdu = sp[0]
            hdu.writeto(target_path, overwrite=True)
        except Exception as e:  # pragma: no cover - network dependent
            logger.error(f"Failed to download SDSS spectrum {object_id}: {e}")
            # create empty placeholder to avoid repeated attempts
            with open(target_path, "wb") as f:
                f.write(b"")
        return target_path

    # --------------------------- Photometry -------------------------------- #
    def query_photometry(self, obj_ids: List[Any]) -> pd.DataFrame:
        """Query basic photometry for a list of object IDs.

        For placeholder mode (no astroquery), returns dummy magnitudes.
        """
        if SDSS is None:
            logger.warning("astroquery.sdss not available; returning dummy photometry")
            rows = []
            for oid in obj_ids:
                rows.append({"objid": oid, "u": 0.0, "g": 0.0, "r": 0.0, "i": 0.0, "z": 0.0})
            return pd.DataFrame(rows)

        results: List[pd.DataFrame] = []
        for oid in obj_ids:
            try:
                q = SDSS.query_sql(
                    f"""
                    SELECT top 1 objid, u, g, r, i, z
                    FROM PhotoObjAll WHERE objid = {int(oid)}
                    """
                )
                if q is not None:
                    results.append(q.to_pandas())
            except Exception as e:  # pragma: no cover
                logger.error(f"Photometry query failed for {oid}: {e}")
        if not results:
            return pd.DataFrame(columns=["objid", "u", "g", "r", "i", "z"])
        return pd.concat(results, ignore_index=True)

    # --------------------------- Crossmatch -------------------------------- #
    def query_xmatch(
        self,
        ra: List[float],
        dec: List[float],
        radius: float = 1.0,
    ) -> pd.DataFrame:
        """Crossmatch coordinates with SDSS primary objects.

        Args:
            ra: Right ascension values (deg)
            dec: Declination values (deg)
            radius: Match radius arcseconds
        Returns:
            DataFrame of matches
        """
        if SDSS is None:
            logger.warning("astroquery.sdss not available; returning empty crossmatch")
            return pd.DataFrame(columns=["ra", "dec", "objid"])

        rows: List[Dict[str, Any]] = []
        for r, d in zip(ra, dec):
            try:
                pos = coord.SkyCoord(r, d, unit=(u.deg, u.deg))
                res = SDSS.query_region(pos, radius=radius * u.arcsec, spectro=False, photo=True)
                if res is not None and len(res) > 0:
                    df = res.to_pandas()
                    df["query_ra"] = r
                    df["query_dec"] = d
                    rows.append(df)
            except Exception as e:  # pragma: no cover
                logger.error(f"Crossmatch failed at RA={r} DEC={d}: {e}")
        if not rows:
            return pd.DataFrame(columns=["query_ra", "query_dec", "objid"])
        return pd.concat(rows, ignore_index=True)

    # --------------------------- Spectrum Parsing with Provenance --------- #
    def load_spectrum_with_provenance(
        self,
        fits_path: Path,
        data_release: str = "DR17"
    ) -> Optional[Spectrum]:
        """Load SDSS spectrum FITS file and create Spectrum object with provenance.

        Args:
            fits_path: Path to SDSS FITS file
            data_release: SDSS data release version
        
        Returns:
            Spectrum object with provenance and quality flags, or None if invalid
        """
        if fits is None:
            logger.warning("astropy.io.fits not available")
            return None
        
        try:
            with fits.open(fits_path) as hdul:
                # SDSS spectra typically in first extension
                data = hdul[1].data if len(hdul) > 1 else hdul[0].data
                header = hdul[0].header
                
                # Extract wavelength and flux
                if hasattr(data, 'loglam'):
                    wavelengths = (10 ** data['loglam']).tolist()
                elif hasattr(data, 'wavelength'):
                    wavelengths = data['wavelength'].tolist()
                else:
                    logger.error("Cannot find wavelength column in FITS")
                    return None
                
                flux = data['flux'].tolist() if hasattr(data, 'flux') else []
                flux_unc = data['ivar'].tolist() if hasattr(data, 'ivar') else None
                
                # Convert inverse variance to uncertainty
                if flux_unc:
                    flux_unc = [1.0/np.sqrt(iv) if iv > 0 else 0.0 for iv in flux_unc]
                
                # Extract quality metrics
                sn_median = float(header.get('SN_MEDIAN', 0.0))
                
                # Create provenance
                provenance = DataProvenance(
                    survey="SDSS",
                    retrieval_timestamp=datetime.now(),
                    query_params={"fits_file": str(fits_path)},
                    version=data_release,
                    processing_pipeline=f"SDSS_{header.get('RUN2D', 'unknown')}"
                )
                
                # Create quality flags
                quality = QualityFlags(
                    is_valid=len(wavelengths) > 0 and len(flux) > 0,
                    signal_to_noise=sn_median if sn_median > 0 else None,
                    has_bad_pixels=False,  # Could check ZWARNING or other flags
                    flags={
                        "plate": int(header.get('PLATEID', 0)),
                        "mjd": int(header.get('MJD', 0)),
                        "fiber": int(header.get('FIBERID', 0))
                    }
                )
                
                return Spectrum(
                    wavelengths=wavelengths,
                    flux=flux,
                    flux_unc=flux_unc,
                    instrument="SDSS",
                    rest_frame=False,  # SDSS spectra are observer frame
                    provenance=provenance,
                    quality=quality
                )
        
        except Exception as e:
            logger.error(f"Failed to parse SDSS spectrum {fits_path}: {e}")
            return None

    # --------------------------- Incremental Sync ----------------------------- #
    def incremental_sync_spectra(
        self,
        min_sn: float = 5.0,
        batch_size: int = 100,
        max_records: Optional[int] = None,
        resume: bool = True
    ) -> Dict[str, Any]:
        """Incrementally sync SDSS spectra with resumable state tracking.
        
        Args:
            min_sn: Minimum signal-to-noise ratio filter
            batch_size: Number of spectra to fetch per batch
            max_records: Maximum records to process (None = unlimited)
            resume: Whether to resume from previous sync state
        
        Returns:
            Dictionary with sync statistics
        """
        sync_manager = get_sync_manager()
        survey_name = "SDSS"
        
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
            # In a real implementation, this would query SDSS incrementally
            # For now, demonstrate the pattern with placeholder logic
            cursor = state.cursor or "0"
            total_processed = state.records_processed
            
            while True:
                # Check max_records limit
                if max_records and total_processed >= max_records:
                    logger.info(f"Reached max_records limit: {max_records}")
                    break
                
                # Simulate batch fetch (in reality: query SDSS with offset/cursor)
                logger.info(f"Fetching batch at cursor={cursor}, records={total_processed}")
                
                # Placeholder: in real implementation, query like:
                # query = f"SELECT TOP {batch_size} specobjid, plate, mjd, fiberid, sn_median "
                # query += f"FROM SpecObjAll WHERE specobjid > {cursor} AND sn_median > {min_sn}"
                # results = SDSS.query_sql(query)
                
                # For demonstration, simulate empty batch (no more data)
                batch_results = []  # Would be actual query results
                
                if not batch_results:
                    logger.info("No more records to process")
                    break
                
                # Process each spectrum in batch
                for record in batch_results:
                    try:
                        # Download and parse spectrum
                        obj_id = (record['plate'], record['mjd'], record['fiberid'])
                        fits_path = self.download_spectrum(obj_id)
                        spectrum = self.load_spectrum_with_provenance(fits_path)
                        
                        if spectrum and spectrum.quality.is_valid:
                            # Save spectrum (in real implementation: store to data lake)
                            stats["successful"] += 1
                            cursor = str(record['specobjid'])
                        else:
                            stats["skipped"] += 1
                    
                    except Exception as e:
                        logger.error(f"Failed to process record: {e}")
                        stats["failed"] += 1
                        sync_manager.update_state(
                            survey_name,
                            increment_errors=True
                        )
                    
                    stats["processed"] += 1
                    total_processed += 1
                    
                    # Update state periodically (every 10 records)
                    if stats["processed"] % 10 == 0:
                        sync_manager.update_state(
                            survey_name,
                            records_processed=total_processed,
                            cursor=cursor
                        )
                
                # Update state after each batch
                sync_manager.update_state(
                    survey_name,
                    records_processed=total_processed,
                    cursor=cursor
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
                cursor=cursor
            )
            stats["end_time"] = datetime.now().isoformat()
            stats["total_records"] = total_processed
        
        return stats


__all__ = ["SDSSConnector"]
