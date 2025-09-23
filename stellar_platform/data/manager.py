"""
Data manager module for coordinating data ingestion and preprocessing.

This module provides a unified interface for accessing and processing data from multiple
astronomical surveys, including SDSS, Gaia, Kepler, and TESS.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import os
import hashlib
from datetime import datetime
import uuid

from .ingestion.sdss import SDSSConnector
from .ingestion.gaia import GaiaConnector
from .ingestion.kepler_tess import KeplerTESSConnector
from .preprocessing.spectral import preprocess_spectrum
from .preprocessing.lightcurve import preprocess_lightcurve
from .preprocessing.sed import preprocess_sed


class DataManager:
    """
    Data manager for coordinating data ingestion and preprocessing.
    """

    def __init__(self, data_dir: str = "data", config_path: Optional[str] = None):
        """
        Initialize the data manager.

        Args:
            data_dir: Base directory for storing data
            config_path: Path to configuration file
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)

        # Initialize survey connectors
        self.surveys = {
            "sdss": SDSSConnector(str(self.data_dir / "sdss")),
            "gaia": GaiaConnector(str(self.data_dir / "gaia")),
            "kepler": KeplerTESSConnector(str(self.data_dir / "kepler")),
            "tess": KeplerTESSConnector(str(self.data_dir / "tess"))
        }

        # Initialize data tracking
        self.metadata_db = self.data_dir / "metadata.db"
        self._init_metadata_db()

    def _init_metadata_db(self):
        """
        Initialize the metadata database.
        """
        # In a real implementation, this would set up a proper database
        # For now, we'll use a simple JSON file
        if not self.metadata_db.exists():
            with open(self.metadata_db, 'w') as f:
                json.dump({"objects": {}}, f)

    def _get_object_hash(self, survey: str, object_id: Any) -> str:
        """
        Generate a hash for an object to ensure consistent identification.

        Args:
            survey: Name of the survey
            object_id: Object identifier

        Returns:
            Hash string for the object
        """
        # Create a unique identifier for the object
        object_str = f"{survey}:{object_id}"
        return hashlib.md5(object_str.encode()).hexdigest()

    def _update_metadata(self, survey: str, object_id: Any, metadata: Dict[str, Any]):
        """
        Update metadata for an object.

        Args:
            survey: Name of the survey
            object_id: Object identifier
            metadata: Metadata to update
        """
        # Load existing metadata
        with open(self.metadata_db, 'r') as f:
            db = json.load(f)

        # Get object hash
        obj_hash = self._get_object_hash(survey, object_id)

        # Update metadata
        if obj_hash not in db["objects"]:
            db["objects"][obj_hash] = {
                "survey": survey,
                "object_id": str(object_id),
                "created": datetime.now().isoformat(),
                "updated": datetime.now().isoformat(),
                "metadata": metadata
            }
        else:
            db["objects"][obj_hash].update({
                "updated": datetime.now().isoformat(),
                "metadata": metadata
            })

        # Save updated metadata
        with open(self.metadata_db, 'w') as f:
            json.dump(db, f, indent=2)

    def get_object_metadata(self, survey: str, object_id: Any) -> Dict[str, Any]:
        """
        Get metadata for an object.

        Args:
            survey: Name of the survey
            object_id: Object identifier

        Returns:
            Metadata dictionary for the object
        """
        # Load existing metadata
        with open(self.metadata_db, 'r') as f:
            db = json.load(f)

        # Get object hash
        obj_hash = self._get_object_hash(survey, object_id)

        # Return metadata if it exists
        if obj_hash in db["objects"]:
            return db["objects"][obj_hash]["metadata"]

        return {}

    def ingest_spectra(
        self,
        survey: str,
        object_ids: List[Any],
        preprocess_params: Optional[Dict[str, Any]] = None,
        force_redownload: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest spectra for a list of objects.

        Args:
            survey: Name of the survey ("sdss")
            object_ids: List of object identifiers
            preprocess_params: Parameters for preprocessing
            force_redownload: Whether to force redownload even if file exists

        Returns:
            Dictionary with ingestion results
        """
        if survey not in self.surveys:
            raise ValueError(f"Survey {survey} not supported for spectra ingestion")

        if survey != "sdss":
            raise ValueError(f"Spectra ingestion only supported for SDSS")

        if preprocess_params is None:
            preprocess_params = {}

        results = {
            "success": [],
            "failed": [],
            "metadata": []
        }

        for object_id in object_ids:
            try:
                # Check if file already exists
                file_path = self.data_dir / "sdss" / f"spectrum_{object_id}.fits"

                if not file_path.exists() or force_redownload:
                    # Download spectrum
                    file_path = self.surveys["sdss"].download_spectrum(object_id)

                # Preprocess spectrum
                processed_spectrum = preprocess_spectrum(str(file_path), **preprocess_params)

                # Update metadata
                metadata = {
                    "object_id": object_id,
                    "file_path": str(file_path),
                    "preprocess_params": preprocess_params,
                    "processed_at": datetime.now().isoformat()
                }
                self._update_metadata(survey, object_id, metadata)

                results["success"].append(object_id)
                results["metadata"].append(metadata)

            except Exception as e:
                self.logger.error(f"Failed to ingest spectrum for object {object_id}: {str(e)}")
                results["failed"].append({
                    "object_id": object_id,
                    "error": str(e)
                })

        return results

    def ingest_lightcurves(
        self,
        survey: str,
        object_ids: List[Any],
        preprocess_params: Optional[Dict[str, Any]] = None,
        force_redownload: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest light curves for a list of objects.

        Args:
            survey: Name of the survey ("kepler", "tess")
            object_ids: List of object identifiers
            preprocess_params: Parameters for preprocessing
            force_redownload: Whether to force redownload even if file exists

        Returns:
            Dictionary with ingestion results
        """
        if survey not in self.surveys:
            raise ValueError(f"Survey {survey} not supported for light curve ingestion")

        if survey not in ["kepler", "tess"]:
            raise ValueError(f"Light curve ingestion only supported for Kepler and TESS")

        if preprocess_params is None:
            preprocess_params = {}

        results = {
            "success": [],
            "failed": [],
            "metadata": []
        }

        for object_id in object_ids:
            try:
                # Check if file already exists
                file_path = self.data_dir / survey / f"lightcurve_{object_id}.fits"

                if not file_path.exists() or force_redownload:
                    # Download light curve
                    lc = self.surveys[survey].download_lightcurve(
                        mission=survey.capitalize(),
                        target=str(object_id)
                    )

                    # Save light curve
                    lc.save(str(file_path))

                # Preprocess light curve
                processed_lc = preprocess_lightcurve(str(file_path), **preprocess_params)

                # Update metadata
                metadata = {
                    "object_id": object_id,
                    "file_path": str(file_path),
                    "preprocess_params": preprocess_params,
                    "processed_at": datetime.now().isoformat()
                }
                self._update_metadata(survey, object_id, metadata)

                results["success"].append(object_id)
                results["metadata"].append(metadata)

            except Exception as e:
                self.logger.error(f"Failed to ingest light curve for object {object_id}: {str(e)}")
                results["failed"].append({
                    "object_id": object_id,
                    "error": str(e)
                })

        return results

    def ingest_photometry(
        self,
        survey: str,
        object_ids: List[Any],
        preprocess_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest photometry for a list of objects.

        Args:
            survey: Name of the survey ("gaia", "sdss")
            object_ids: List of object identifiers
            preprocess_params: Parameters for preprocessing

        Returns:
            Dictionary with ingestion results
        """
        if survey not in self.surveys:
            raise ValueError(f"Survey {survey} not supported for photometry ingestion")

        if survey not in ["gaia", "sdss"]:
            raise ValueError(f"Photometry ingestion only supported for Gaia and SDSS")

        if preprocess_params is None:
            preprocess_params = {}

        results = {
            "success": [],
            "failed": [],
            "metadata": []
        }

        for object_id in object_ids:
            try:
                # Query photometry
                if survey == "gaia":
                    photometry = self.surveys["gaia"].query_source(object_id)
                else:  # sdss
                    photometry = self.surveys["sdss"].query_photometry(
                        obj_ids=[object_id]
                    )

                # Save photometry
                file_path = self.data_dir / survey / f"photometry_{object_id}.csv"
                photometry.to_csv(file_path, index=False)

                # Preprocess photometry (create SED)
                processed_sed = preprocess_sed(str(file_path), **preprocess_params)

                # Update metadata
                metadata = {
                    "object_id": object_id,
                    "file_path": str(file_path),
                    "preprocess_params": preprocess_params,
                    "processed_at": datetime.now().isoformat()
                }
                self._update_metadata(survey, object_id, metadata)

                results["success"].append(object_id)
                results["metadata"].append(metadata)

            except Exception as e:
                self.logger.error(f"Failed to ingest photometry for object {object_id}: {str(e)}")
                results["failed"].append({
                    "object_id": object_id,
                    "error": str(e)
                })

        return results

    def crossmatch_surveys(
        self,
        survey1: str,
        object_ids1: List[Any],
        survey2: str,
        radius: float = 1.0,  # arcseconds
        return_all: bool = False
    ) -> pd.DataFrame:
        """
        Crossmatch objects between two surveys.

        Args:
            survey1: Name of the first survey
            object_ids1: List of object identifiers for the first survey
            survey2: Name of the second survey
            radius: Search radius (arcseconds)
            return_all: Whether to return all matches or just the closest

        Returns:
            DataFrame with crossmatch results
        """
        if survey1 not in self.surveys or survey2 not in self.surveys:
            raise ValueError("One or both surveys not supported")

        # Get positions for survey1 objects
        positions1 = []
        for obj_id in object_ids1:
            metadata = self.get_object_metadata(survey1, obj_id)
            if "ra" in metadata and "dec" in metadata:
                positions1.append((metadata["ra"], metadata["dec"], obj_id))

        if not positions1:
            raise ValueError("No valid positions found for survey1 objects")

        # Crossmatch with survey2
        ra_list, dec_list = zip(*[(pos[0], pos[1]) for pos in positions1])
        crossmatch = self.surveys[survey2].query_xmatch(
            ra=ra_list,
            dec=dec_list,
            radius=radius
        )

        # Add survey1 object IDs
        crossmatch["survey1_id"] = [pos[2] for pos in positions1]

        return crossmatch
