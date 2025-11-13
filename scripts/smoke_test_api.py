"""Simple smoke test for the Stellar Platform FastAPI service.

Usage:
    python scripts/smoke_test_api.py --start  # starts server in background, runs tests, then stops
    python scripts/smoke_test_api.py          # assumes server already running on host/port

Checks:
 - /health returns status ok
 - /models returns JSON (may be empty if no registry entries)
 - /predict/spectral returns probabilities shape
 - /predict/lightcurve returns probabilities shape
 - (If calibration present) calibrated flag appears when requested

Exit codes:
 0 success, 1 failure (prints first failing step)
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
import os
from typing import Optional

import urllib.request
from urllib.error import HTTPError, URLError

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
BASE = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"


def _http_json(path: str, data: Optional[dict] = None):
    url = BASE + path
    try:
        if data is not None:
            payload = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        else:
            req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as r:  # nosec B310 (simple internal smoke test)
            body = r.read().decode("utf-8")
            return json.loads(body)
    except HTTPError as e:  # pragma: no cover
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = "<no body>"
        raise RuntimeError(f"Request to {url} failed: HTTP {e.code} {e.reason} body={body}")
    except URLError as e:  # pragma: no cover
        raise RuntimeError(f"Request to {url} failed: URL error {e.reason}")
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Request to {url} failed: {e}")


def run_tests():
    # 1. Health
    health = _http_json("/health")
    assert health.get("status") == "ok", f"Health not ok: {health}"

    # 2. Models listing
    models = _http_json("/models")
    # optional: just ensure it's a dict
    assert isinstance(models, dict), "Models endpoint did not return dict"

    # Build minimal dummy inputs
    spectra = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
    lightcurves = [[1.0, 1.01, 0.99, 1.02, 1.0, 0.98]]

    failures = []
    spec_pred = {}
    spec_cal = {}
    lc_pred = {}
    sed_pred = {}
    # 3. Spectral prediction (uncalibrated)
    try:
        spec_pred = _http_json("/predict/spectral", {"spectra": spectra, "apply_calibration": False})
        if "probabilities" not in spec_pred:
            failures.append("spectral prediction missing probabilities key")
    except Exception as e:
        failures.append(f"spectral prediction error: {e}")
    # 4. Spectral prediction (calibrated)
    try:
        spec_cal = _http_json("/predict/spectral", {"spectra": spectra, "apply_calibration": True})
        if "probabilities" not in spec_cal:
            failures.append("spectral calibrated prediction missing probabilities key")
    except Exception as e:
        failures.append(f"spectral calibrated prediction error: {e}")
    # 5. Lightcurve prediction
    try:
        lc_pred = _http_json("/predict/lightcurve", {"lightcurves": lightcurves, "apply_calibration": True})
        if "probabilities" not in lc_pred:
            failures.append("lightcurve prediction missing probabilities key")
    except Exception as e:
        failures.append(f"lightcurve prediction error: {e}")
    if failures:
        raise AssertionError("; ".join(failures))

    # Optional SED test if registry has 'sed'
    try:
        models_list = list(models.keys()) if isinstance(models, dict) else []
        if 'sed' in models_list:
            sed_pred = _http_json("/predict/sed", {"features": [[1.0, 0.8, 1.1, 0.7, 0.9, 1.05]], "top_k": 3})
            if "probabilities" not in sed_pred:
                failures.append("sed prediction missing probabilities key")
    except Exception as e:  # pragma: no cover
        failures.append(f"sed prediction error: {e}")

    if failures:
        raise AssertionError("; ".join(failures))

    return {
        "health": health,
        "models": list(models.keys()),
        "spectral_shape": (len(spec_pred["probabilities"]), len(spec_pred["probabilities"][0]) if spec_pred["probabilities"] else 0),
        "lightcurve_shape": (len(lc_pred["probabilities"]), len(lc_pred["probabilities"][0]) if lc_pred["probabilities"] else 0),
        "calibrated_flag_spectral": spec_cal.get("calibrated"),
        "sed_shape": (len(sed_pred.get("probabilities", [])), len(sed_pred.get("probabilities", [[0]])[0]) if sed_pred.get("probabilities") else 0),
    }


def main():  # pragma: no cover - integration flow
    parser = argparse.ArgumentParser(description="API smoke test")
    parser.add_argument("--start", action="store_true", help="Start server in background for test")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=int, default=25)
    args = parser.parse_args()

    global BASE
    BASE = f"http://{args.host}:{args.port}"

    proc = None
    try:
        if args.start:
            env = os.environ.copy()
            # Ensure project root on PYTHONPATH
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
            cmd = [sys.executable, "-m", "uvicorn", "stellar_platform.serving.api:app", "--host", args.host, "--port", str(args.port)]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
            # Poll health endpoint instead of relying on log lines
            start_time = time.time()
            ready = False
            while time.time() - start_time < args.timeout:
                try:
                    h = _http_json("/health")
                    if h.get("status") == "ok":
                        ready = True
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            if not ready:
                raise RuntimeError("Server did not start within timeout (health check failed)")
        # Run tests
        results = run_tests()
        print("SMOKE TEST PASSED")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print("SMOKE TEST FAILED:", e)
        if proc and proc.poll() is None:
            proc.terminate()
        sys.exit(1)
    finally:
        if proc and proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    main()
