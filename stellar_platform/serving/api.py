"""FastAPI application for model serving.

Provides batch prediction endpoints for spectral and light curve models.
Models discovered via the simple filesystem registry.
"""
from __future__ import annotations

import json
from pathlib import Path
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import time
import uuid

from ..models.registry import ModelRegistry
from ..evaluation import BaseProbCalibrator  # new import for calibration

# Lazy optional imports for heavy frameworks
tf = None  # type: ignore
try:  # pragma: no cover
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    pass

app = FastAPI(title="Stellar Platform API", version="0.1.0")

# --- CORS (initially permissive; tighten in production) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to your frontend domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static Frontend (optional) ---
_web_dir = Path(__file__).resolve().parents[2] / "web"
if _web_dir.exists():  # mount if user created web assets
    # Mount under /web to avoid shadowing API routes like /health, /predict/*
    app.mount("/web", StaticFiles(directory=str(_web_dir), html=True), name="web")

@app.get("/")
async def root_index():  # pragma: no cover
    return {"message": "Stellar API running", "static_ui": "/web/index.html", "docs": "/docs", "auth": ("required" if os.getenv("STELLAR_API_KEY") else "disabled")}
logger = logging.getLogger("stellar_api")

class _JsonFormatter(logging.Formatter):
    def format(self, record):  # type: ignore
        base = {
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'time': self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
        }
        for attr in ('request_id', 'path', 'latency_ms', 'model_family', 'model_version'):
            if hasattr(record, attr):
                base[attr] = getattr(record, attr)
        try:
            return json.dumps(base)
        except Exception:
            return super().format(record)

handler = logging.StreamHandler()
handler.setFormatter(_JsonFormatter())
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(handler)

_REQ_COUNT = 0
_CUM_LAT_MS = 0.0

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):  # pragma: no cover - integration behavior
    global _REQ_COUNT, _CUM_LAT_MS
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.time()
    try:
        response = await call_next(request)
        return response
    finally:
        latency_ms = (time.time() - start) * 1000.0
        _REQ_COUNT += 1
        _CUM_LAT_MS += latency_ms
        logger.info(
            "request",
            extra={
                'request_id': request_id,
                'path': request.url.path,
                'latency_ms': round(latency_ms, 2),
            }
        )
        # attach simple header metrics (optional)
        if 'response' in locals():
            try:
                response.headers['x-request-id'] = request_id
            except Exception:  # pragma: no cover
                pass
registry = ModelRegistry()

# --- Simple API Key auth (header: x-api-key) ---
import os
API_KEY = os.getenv("STELLAR_API_KEY")  # if unset, auth is disabled

def require_api_key(request: Request):  # pragma: no cover - simple runtime gate
    if API_KEY is None:
        return True
    provided = request.headers.get("x-api-key")
    if provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key; provide header 'x-api-key'.")
    return True

@app.get('/favicon.ico')
async def favicon():  # pragma: no cover
    return PlainTextResponse('', status_code=204)


class SpectralPredictRequest(BaseModel):
    spectra: List[List[float]]  # Each spectrum is list of flux values
    model: Optional[str] = None  # model family name
    apply_calibration: bool = False  # optionally apply stored calibrator
    top_k: Optional[int] = None  # if provided return top-k breakdown


class LightCurvePredictRequest(BaseModel):
    lightcurves: List[List[float]]  # Each light curve flux array
    model: Optional[str] = None
    apply_calibration: bool = False
    top_k: Optional[int] = None


@app.get("/health")
async def health(auth=Depends(require_api_key)):  # pragma: no cover - simple aggregation
    families = {}
    for fam_dir in registry.root.iterdir():
        if not fam_dir.is_dir():
            continue
        name = fam_dir.name
        latest = registry.get_latest_version(name)
        if latest:
            meta = registry.get_metadata(name, latest) or {}
            artifacts = meta.get('artifacts', {})
            hashes = meta.get('artifact_hashes', {})
            verified = {}
            for label, path in artifacts.items():
                try:
                    p = Path(path)
                    ok = p.exists()
                    # hash verification best-effort
                    if ok and hashes.get(label):
                        import hashlib
                        h = hashlib.sha256()
                        with open(p, 'rb') as f:
                            for chunk in iter(lambda: f.read(8192), b''):
                                h.update(chunk)
                        ok = (h.hexdigest() == hashes[label])
                    verified[label] = ok
                except Exception:
                    verified[label] = False
            families[name] = {
                'latest': latest,
                'artifact_verification': verified,
            }
    return {"status": "ok", "models": families}


@app.get("/models")
async def list_models(auth=Depends(require_api_key)):
    out: Dict[str, Any] = {}
    for family_dir in registry.root.iterdir():
        if family_dir.is_dir():
            versions = registry.list_versions(family_dir.name)
            out[family_dir.name] = versions
    return out


@app.get("/models/{family}/metadata")
async def model_metadata(family: str, version: Optional[str] = None, auth=Depends(require_api_key)):
    """Return registry metadata for a model family (optionally a specific version)."""
    v = version or registry.get_latest_version(family)
    if v is None:
        raise HTTPException(status_code=404, detail=f"No versions for family {family}")
    meta = registry.get_metadata(family, v)
    if meta is None:
        raise HTTPException(status_code=404, detail="Metadata not found")
    meta_slim = {k: meta[k] for k in meta if k not in {"artifact_hashes"}}
    return meta_slim


MAX_ITEMS = 64  # simple defensive limits
MAX_LENGTH = 10000

def _load_latest_keras_model(family: str):
    version = registry.get_latest_version(family)
    if version is None:
        raise HTTPException(status_code=404, detail=f"No versions found for model {family}")
    meta = registry.get_metadata(family, version)
    if not meta or 'artifacts' not in meta:
        raise HTTPException(status_code=500, detail=f"Metadata missing artifacts for {family}:{version}")
    # Assume artifact key 'model'
    model_path = meta['artifacts'].get('model')
    if model_path is None:
        raise HTTPException(status_code=404, detail=f"Model artifact path missing for {family}:{version}")
    p = Path(model_path)
    if not p.is_absolute():
        # resolve relative to project root (this file two levels below root)
        root = Path(__file__).resolve().parents[2]
        p = (root / p).resolve()
    if not p.exists():
        logger.error("Artifact path does not exist after resolution: %s", p)
        raise HTTPException(status_code=404, detail=f"Model artifact not found for {family}:{version}")
    # Dummy JSON artifact support
    if p.suffix.lower() == '.json':
        try:
            payload = json.loads(p.read_text())
        except Exception as e:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Failed to read dummy artifact: {e}")

        classes = int(payload.get('classes', 3))
        # Provide a lightweight object with predict() returning random probs
        class DummyModel:
            def predict(self, arr):  # type: ignore
                probs = np.random.rand(arr.shape[0], classes)
                probs /= probs.sum(axis=1, keepdims=True)
                return probs
        model = DummyModel()
    else:
        if tf is None:
            raise HTTPException(status_code=500, detail="TensorFlow not available to load Keras model and artifact is not dummy JSON.")
        try:
            model = tf.keras.models.load_model(str(p))  # type: ignore
        except Exception as e:
            logger.exception("Failed to load model %s:%s from %s", family, version, p)
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    # verify artifact hash integrity if present
    try:
        hashes = meta.get('artifact_hashes', {})
        expected_hash = hashes.get('model')
        if expected_hash:
            import hashlib
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            actual = h.hexdigest()
            if actual != expected_hash:
                logger.error("Artifact hash mismatch for %s:%s expected=%s actual=%s", family, version, expected_hash, actual)
                raise HTTPException(status_code=409, detail="Model artifact hash mismatch; refusing to serve.")
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover
        logger.exception("Artifact hash verification failed for %s:%s: %s", family, version, e)
        raise HTTPException(status_code=500, detail="Artifact hash verification failure")
    # update meta with resolved absolute path (non-persistent)
    meta['artifacts']['model_resolved'] = str(p)
    return model, meta


def _maybe_load_calibrator(meta: Dict[str, Any]):
    payload = meta.get('calibrator') if meta else None
    if not payload:
        return None
    try:
        return BaseProbCalibrator.from_dict(payload)
    except Exception:
        return None


@app.post("/predict/spectral")
async def predict_spectral(req: SpectralPredictRequest, auth=Depends(require_api_key)):
    family = req.model or 'spectral_cnn'
    model, meta = _load_latest_keras_model(family)
    arr = np.array(req.spectra, dtype=float)
    if arr.ndim != 2:
        raise HTTPException(status_code=400, detail="Spectra must be 2D array")
    if arr.shape[0] == 0:
        raise HTTPException(status_code=400, detail="No spectra provided")
    if arr.shape[0] > MAX_ITEMS:
        raise HTTPException(status_code=400, detail=f"Batch too large (max {MAX_ITEMS})")
    if arr.shape[1] > MAX_LENGTH:
        raise HTTPException(status_code=400, detail=f"Spectrum length exceeds max {MAX_LENGTH}")
    # Expect shape (N, wavelengths)
    arr = np.expand_dims(arr, -1)  # (N, L, 1)
    probs = model.predict(arr)
    if req.apply_calibration:
        calib = _maybe_load_calibrator(meta)
        if calib is not None:
            probs = calib.transform(probs)
    result: Dict[str, Any] = {
        'model': family,
        'version': meta['version'],
        'probabilities': probs.tolist(),
        'calibrated': bool(req.apply_calibration and _maybe_load_calibrator(meta) is not None)
    }
    if req.top_k:
        k = max(1, min(req.top_k, probs.shape[1]))
        top_idx = np.argsort(-probs, axis=1)[:, :k]
        top_scores = np.take_along_axis(probs, top_idx, axis=1)
        result['top_k'] = [
            [
                {'class_index': int(top_idx[i, j]), 'prob': float(top_scores[i, j])}
                for j in range(k)
            ] for i in range(probs.shape[0])
        ]
    return result


@app.post("/predict/lightcurve")
async def predict_lightcurve(req: LightCurvePredictRequest, auth=Depends(require_api_key)):
    family = req.model or 'lightcurve_transformer'
    model, meta = _load_latest_keras_model(family)
    arr = np.array(req.lightcurves, dtype=float)
    if arr.ndim != 2:
        raise HTTPException(status_code=400, detail="Light curves must be 2D array")
    if arr.shape[0] == 0:
        raise HTTPException(status_code=400, detail="No light curves provided")
    if arr.shape[0] > MAX_ITEMS:
        raise HTTPException(status_code=400, detail=f"Batch too large (max {MAX_ITEMS})")
    if arr.shape[1] > MAX_LENGTH:
        raise HTTPException(status_code=400, detail=f"Light curve length exceeds max {MAX_LENGTH}")
    arr = np.expand_dims(arr, -1)
    probs = model.predict(arr)
    if req.apply_calibration:
        calib = _maybe_load_calibrator(meta)
        if calib is not None:
            probs = calib.transform(probs)
    result: Dict[str, Any] = {
        'model': family,
        'version': meta['version'],
        'probabilities': probs.tolist(),
        'calibrated': bool(req.apply_calibration and _maybe_load_calibrator(meta) is not None)
    }
    if req.top_k:
        k = max(1, min(req.top_k, probs.shape[1]))
        top_idx = np.argsort(-probs, axis=1)[:, :k]
        top_scores = np.take_along_axis(probs, top_idx, axis=1)
        result['top_k'] = [
            [
                {'class_index': int(top_idx[i, j]), 'prob': float(top_scores[i, j])}
                for j in range(k)
            ] for i in range(probs.shape[0])
        ]
    return result


__all__ = ['app']

@app.get('/metrics')
async def metrics(auth=Depends(require_api_key)):  # pragma: no cover - textual endpoint
    avg_lat = (_CUM_LAT_MS / _REQ_COUNT) if _REQ_COUNT else 0.0
    # Minimal Prometheus exposition format
    body = [
        "# HELP stellar_requests_total Total HTTP requests handled",
        "# TYPE stellar_requests_total counter",
        f"stellar_requests_total {_REQ_COUNT}",
        "# HELP stellar_request_latency_ms_avg Average request latency in milliseconds",
        "# TYPE stellar_request_latency_ms_avg gauge",
        f"stellar_request_latency_ms_avg {avg_lat:.4f}",
    ]
    return PlainTextResponse("\n".join(body) + "\n")
