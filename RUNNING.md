# How to Run This Project (Windows-friendly)

This repo provides:
- Python library `stellar_platform` for data, models, and evaluation.
- A FastAPI server for inference with a small web UI.
- CLIs to (dummy-)train and evaluate models.
- A pytest suite.

Below are the quickest working paths on Windows (cmd.exe). Adjust paths for PowerShell if preferred.

## 1) Prerequisites
- Python 3.11 (recommended). Verify with: `python --version`
- Git (optional)

TensorFlow and PyTorch are listed in `requirements.txt`. You DO NOT need them to run tests or the demo API with dummy models. They are required only to load real `.keras` artifacts or train real models.

## 2) Create a virtual environment
```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

## 3) Install dependencies
Full install (API + UI + CLIs + tests):
```cmd
pip install -r requirements.txt
```

Minimal install (enough for tests and dummy API):
```cmd
pip install numpy==1.26.4 scipy>=1.11,<1.14 pandas==2.2.2 scikit-learn==1.4.2 astropy>=5 pytest>=6.2 fastapi>=0.78 uvicorn>=0.18 pydantic>=1.9
```

Notes:
- `healpy` is optional and may be skipped on some Windows setups.
- To silence TensorFlow oneDNN notices (optional):
  ```cmd
  set TF_ENABLE_ONEDNN_OPTS=0
  ```

## 4) Run the tests
```cmd
python -m pytest -q
```
Expected: all tests pass.

## 5) Launch the API server
From the repo root:
```cmd
python -m uvicorn stellar_platform.serving.api:app --host 127.0.0.1 --port 8000
```
Then open the docs:
- OpenAPI/Swagger: http://127.0.0.1:8000/docs
- Web UI: http://127.0.0.1:8000/web/index.html

Health check:
```cmd
curl http://127.0.0.1:8000/health
```
You should see `{ "status": "ok", ... }` with model registry details.

### API key (optional)
You can require a simple API key via env var. If not set, auth is disabled.
```cmd
set STELLAR_API_KEY=your-secret
```
When enabled, include `x-api-key: your-secret` in requests.

## 6) Try predictions
Example spectral POST (space/comma-separated vectors acceptable in the web UI):
```cmd
curl -s -X POST http://127.0.0.1:8000/predict/spectral ^
  -H "Content-Type: application/json" ^
  -d "{\"spectra\":[[0.1,0.2,0.3,0.4,0.5]],\"apply_calibration\":true,\"top_k\":3}"
```
Light curve POST:
```cmd
curl -s -X POST http://127.0.0.1:8000/predict/lightcurve ^
  -H "Content-Type: application/json" ^
  -d "{\"lightcurves\":[[1.0,1.01,0.99,1.02,1.0,0.98]],\"apply_calibration\":true}"
```
With the repo’s default registry metadata, the server will use dummy JSON artifacts and return plausible random class probabilities (optionally calibrated).

## 7) Model registry basics
Models are tracked under `registry/<family>/<version>/metadata.json`, for example:
- `registry/spectral_cnn/LATEST_VERSION`
- `registry/spectral_cnn/<version>/metadata.json`
- `registry/lightcurve_transformer/LATEST_VERSION`

Artifacts referenced in metadata can be:
- Real Keras model files (e.g., `.keras`). Requires TensorFlow to serve.
- Dummy descriptors (`*.json`) bundled in `artifacts/` for lightweight demos.

Calibration info (e.g., temperature scaling) is stored in metadata and applied by the API if `apply_calibration=true`.

## 8) CLIs (training/evaluation)
Train dummy (or real, if TF installed) and auto-register:
```cmd
python scripts\train_cli.py train-spectral --force-dummy --samples 256 --length 256 --classes 3 --output-dir artifacts
python scripts\train_cli.py train-lightcurve --force-dummy --samples 256 --length 256 --classes 3 --output-dir artifacts
```
Evaluate latest registered spectral model and produce a model card + metrics:
```cmd
python scripts\evaluate_cli.py spectral_cnn --output-dir evaluations
```
Run end-to-end API smoke test (starts server, hits endpoints, prints summary):
```cmd
python scripts\smoke_test_api.py --start
```

## 9) Web UI
Static assets live in `web/` and are mounted at `/web` by the API when present.
- Open http://127.0.0.1:8000/web/index.html
- Set API base to `http://127.0.0.1:8000`
- Use the sample inputs, click Predict.

## 10) Configuration & environment
- `STELLAR_API_KEY`: enable simple header auth (`x-api-key`).
- `TF_ENABLE_ONEDNN_OPTS=0`: disable oneDNN optimizations logs in TensorFlow (optional).
- `UVICORN` options: choose host/port via flags. For hot reload in dev: add `--reload`.
- Python path: if running tools from `scripts/`, ensure repo root is on `PYTHONPATH` (usually automatic when run from root).

## 11) Troubleshooting
- Import errors for heavy libs (tensorflow/torch): use dummy path (`--force-dummy`) or install only minimal deps.
- 404 model artifact: ensure `registry/*/LATEST_VERSION` points to a version whose metadata artifact path exists. Dummy artifacts are in `artifacts/`.
- Astropy time parsing on CSV: inputs must include columns `time, flux`.
- Windows curl: If `curl` isn’t available, use PowerShell `Invoke-WebRequest` or test via the web UI.

## 12) Development tips
- Lint/type-checks are not enforced in this template; add your preferred tools.
- To add CI, consider GitHub Actions matrix for Windows/Linux with `pytest` and an API smoke test.
- To containerize, a minimal Dockerfile would base on a Python 3.11 image and install `requirements.txt`.

---
If you want this guide merged into `README.md` or tailored for Linux/macOS, tell me and I’ll update accordingly.
