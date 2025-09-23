"""
SDSS data connector module.

Provides functions for downloading spectra and querying photometry from the
Sloan Digital Sky Survey (SDSS). This is a lightweight placeholder
implementation using astroquery.sdss. Network calls are wrapped so that
offline usage degrades gracefully.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Any, Dict

import pandas as pd

try:
    from astroquery.sdss import SDSS  # type: ignore
    from astropy import coordinates as coord
    from astropy import units as u
except Exception:  # pragma: no cover - allow import without astroquery installed yet
    SDSS = None  # type: ignore

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


__all__ = ["SDSSConnector"]
