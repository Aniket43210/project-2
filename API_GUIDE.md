# API Server Guide

The Stellar Platform API is fully functional and ready to use. Due to the weight of TensorFlow imports on Windows, the server performs best when started with certain configurations.

## Quick Start

### Option 1: Using the Provided Script (Recommended)

```bash
python run_api.py
```

This script properly configures the environment variables and starts the server.

### Option 2: Direct uvicorn Command

```bash
set TF_ENABLE_ONEDNN_OPTS=0
python -m uvicorn stellar_platform.serving.api:app --host 127.0.0.1 --port 8000 --workers 1
```

### Option 3: With Reload (Development)

```bash
set TF_ENABLE_ONEDNN_OPTS=0
python -m uvicorn stellar_platform.serving.api:app --host 127.0.0.1 --port 8000 --reload
```

## Testing the API

Once the server is running at `http://127.0.0.1:8000`, you can test it with:

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

### Lightcurve Prediction
```bash
curl -X POST http://127.0.0.1:8000/predict/lightcurve \
  -H "Content-Type: application/json" \
  -d '{"lightcurves": [[1.0, 1.01, 0.99, 1.02, 1.0, 0.98]]}'
```

**Expected Response:**
```json
{
  "model": "lightcurve_transformer",
  "version": "...",
  "probabilities": [[...class probabilities...]],
  "calibrated": false
}
```

### With Calibration Applied
```bash
curl -X POST http://127.0.0.1:8000/predict/lightcurve \
  -H "Content-Type: application/json" \
  -d '{"lightcurves": [[1.0, 1.01, 0.99, 1.02, 1.0, 0.98]], "apply_calibration": true}'
```

### Spectral Prediction
```bash
curl -X POST http://127.0.0.1:8000/predict/spectral \
  -H "Content-Type: application/json" \
  -d '{"spectra": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]}'
```

### SED Prediction
```bash
curl -X POST http://127.0.0.1:8000/predict/sed \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.0, 0.8, 1.1, 0.7, 0.9, 1.05]]}'
```

### With Top-K Results
```bash
curl -X POST http://127.0.0.1:8000/predict/spectral \
  -H "Content-Type: application/json" \
  -d '{"spectra": [[0.1, 0.2, 0.3, 0.4, 0.5]], "top_k": 3}'
```

## API Documentation

Once the server is running, access the interactive API documentation at:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## Endpoints

### Predictions
- `POST /predict/spectral` - Spectral classification
- `POST /predict/lightcurve` - Light curve variability
- `POST /predict/sed` - SED-based classification
- `POST /predict/spectral/sets` - Spectral conformal prediction sets
- `POST /predict/lightcurve/sets` - Light curve conformal prediction sets
- `POST /predict/sed/sets` - SED conformal prediction sets

### Information
- `GET /health` - Health check with model registry status
- `GET /models` - List all registered models
- `GET /docs` - Swagger UI documentation
- `GET /redoc` - ReDoc documentation
- `GET /` - API root with message

## Query Parameters

### Common Parameters
- `model`: Model family name (default: uses latest)
- `apply_calibration`: Apply stored calibration (default: false)
- `top_k`: Return top-k class predictions (optional)

### Conformal Prediction Sets
- `q`: Quantile for coverage (0-1 range)

### Ensemble
- `ensemble_models`: List of additional models to ensemble
- `ensemble_weights`: Weights for ensemble averaging
- `ensemble_method`: 'prob' (default) or 'logit' averaging

## Request Format

All prediction endpoints accept JSON requests with a list of inputs:

```json
{
  "spectra": [[...float values...]],
  "apply_calibration": true,
  "top_k": 3
}
```

## Response Format

Responses include:
- `model`: Model family used
- `version`: Model version identifier
- `probabilities`: Class probabilities for each input
- `calibrated`: Whether calibration was applied
- `top_k`: (optional) Top-K predictions with scores

## Authentication

Authentication is optional and controlled by the `STELLAR_API_KEY` environment variable:

```bash
# Set API key (optional)
set STELLAR_API_KEY=your-secret-key

# Start server
python run_api.py

# In requests, include the header
curl -H "x-api-key: your-secret-key" http://127.0.0.1:8000/predict/spectral
```

## Performance Notes

- **Dummy Models**: < 10ms latency
- **Real Models**: 50-500ms depending on model size
- **Batch Size**: Currently supports up to 1000 items per batch
- **Spectrum Length**: Limited to 4096 samples
- **Light Curve Length**: Limited to 8192 samples

## Troubleshooting

### Server won't start
- Ensure Port 8000 is not in use: `netstat -ano | find ":8000"`
- Check TensorFlow is installed: `python -c "import tensorflow"`
- Try the TF optimization disable: `set TF_ENABLE_ONEDNN_OPTS=0`

### No models found
- Train models first: `python scripts/train_cli.py train-spectral --force-dummy --samples 64 --output-dir artifacts`
- Check registry: `ls registry/*/LATEST_VERSION`

### Slow responses
- Models are being loaded for the first time
- Dummy models should respond in <10ms
- Real models will be slower (TensorFlow overhead)

### Request errors
- Check input format in requests
- Ensure arrays are proper shape: spectral/LC should be 2D
- See `/docs` for exact schema

## Next Steps

1. Train models: `python scripts/train_cli.py train-lightcurve --samples 64 --output-dir artifacts`
2. Start API: `python run_api.py`
3. Visit http://127.0.0.1:8000/docs to test endpoints interactively
4. Deploy to production (Phase 1 task)
