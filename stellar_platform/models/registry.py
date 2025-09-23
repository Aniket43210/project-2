"""Simple filesystem model registry for the stellar platform.

Stores model versions under a root directory with metadata JSON files.
Optionally integrates with MLflow if tracking URI is configured.
"""
from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import logging

try:  # optional mlflow
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, root: str = "registry", use_mlflow: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow and mlflow is not None

    # ---------------------- Internal helpers --------------------------- #
    def _model_dir(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _version_dir(self, name: str, version: str) -> Path:
        d = self._model_dir(name) / version
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _latest_symlink(self, name: str) -> Path:
        return self._model_dir(name) / 'latest'

    # ---------------------- Public API -------------------------------- #
    def _hash_artifact(self, path: str) -> Optional[str]:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        h = hashlib.sha256()
        try:
            with open(p, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def register(
        self,
        name: str,
        artifacts: Dict[str, str],
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a new model version.

        Args:
            name: Model family name (e.g., 'spectral_cnn')
            artifacts: Mapping label->path of produced artifact files
            metrics: Evaluation metrics for gating
            tags: Arbitrary string tags
            params: Hyperparameters
        Returns:
            version identifier (timestamp hash)
        """
        version = str(int(time.time()))
        # include short hash for uniqueness if quick successive calls
        h = hashlib.sha1(version.encode()).hexdigest()[:6]
        version = f"{version}-{h}"
        vdir = self._version_dir(name, version)

        # Copy/record artifact paths (assume already persisted)
        artifact_hashes = {label: self._hash_artifact(path) for label, path in artifacts.items()}

        metadata = {
            'name': name,
            'version': version,
            'artifacts': artifacts,
            'artifact_hashes': artifact_hashes,
            'metrics': metrics or {},
            'tags': tags or {},
            'params': params or {},
            'registered_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        # Atomic write: write to temp then rename
        tmp_fd, tmp_path = tempfile.mkstemp(prefix='meta_', dir=str(vdir))
        try:
            with open(tmp_fd, 'w') as f:  # type: ignore[arg-type]
                json.dump(metadata, f, indent=2)
            target = vdir / 'metadata.json'
            # Replace if exists
            Path(tmp_path).replace(target)
        finally:
            if Path(tmp_path).exists() and not (vdir / 'metadata.json').exists():  # pragma: no cover
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass

        # Update latest pointer
        latest = self._latest_symlink(name)
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(vdir.name)
        except OSError:  # Windows fallback: write file with version
            with open(self._model_dir(name) / 'LATEST_VERSION', 'w') as f:
                f.write(version)

        if self.use_mlflow:
            try:  # pragma: no cover
                with mlflow.start_run(run_name=f"register_{name}_{version}"):
                    if params:
                        mlflow.log_params(params)
                    if metrics:
                        mlflow.log_metrics(metrics)
                    for label, path in artifacts.items():
                        mlflow.log_artifact(path, artifact_path=label)
                    mlflow.set_tags(tags or {})
                    mlflow.set_tag('registry_version', version)
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

        return version

    def list_versions(self, name: str) -> List[str]:
        d = self._model_dir(name)
        versions = [p.name for p in d.iterdir() if p.is_dir() and p.name != 'latest']
        return sorted(versions)

    def get_metadata(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if version is None:
            version = self.get_latest_version(name)
            if version is None:
                return None
        meta_path = self._model_dir(name) / version / 'metadata.json'
        if not meta_path.exists():
            return None
        with open(meta_path, 'r') as f:
            return json.load(f)

    def get_latest_version(self, name: str) -> Optional[str]:
        latest_link = self._latest_symlink(name)
        if latest_link.is_symlink():
            return latest_link.readlink().name  # type: ignore[attr-defined]
        # Windows fallback file
        file_fallback = self._model_dir(name) / 'LATEST_VERSION'
        if file_fallback.exists():
            return file_fallback.read_text().strip()
        versions = self.list_versions(name)
        return versions[-1] if versions else None

    def promote_if_better(
        self,
        name: str,
        new_metrics: Dict[str, float],
        primary_metric: str,
        better: str = 'higher'
    ) -> bool:
        """Compare new metrics to latest and register only if better.

        Returns True if registered.
        """
        latest_version = self.get_latest_version(name)
        if latest_version:
            existing = self.get_metadata(name, latest_version)
            if existing and primary_metric in existing.get('metrics', {}):
                prev = existing['metrics'][primary_metric]
                curr = new_metrics.get(primary_metric)
                if curr is None:
                    return False
                improved = (curr > prev) if better == 'higher' else (curr < prev)
                if not improved:
                    logger.info(f"New model not better on {primary_metric}: {curr} vs {prev}")
                    return False
        # Register placeholder with no artifacts yet
        self.register(name, artifacts={}, metrics=new_metrics)
        return True


__all__ = ['ModelRegistry']
