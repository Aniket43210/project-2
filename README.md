# Multi-modal Stellar Analysis Platform

A comprehensive platform for ingesting public survey data, predicting stellar types and physical parameters, and classifying variability with calibrated uncertainties and leakage-safe evaluation.

## Scope and Phasing

- **Phase 0 (✅ Complete)**: Core infrastructure, model training, evaluation framework
- **Phase 1 (✅ Complete)**: Data pipeline API, preprocessing endpoints, web UI integration
- **Phase 2 (Next)**: Real labeled dataset integration, feature store, production training

## Data Sources

- SDSS spectroscopic DRs
- Gaia photometry
- Kepler/TESS light curves
- SIMBAD top-level classes for labels

## Key Features

- **Data Ingestion**: Incremental sync from SDSS, TESS, and Kepler with resume capability
- **Preprocessing**: Spectral normalization, telluric masking, lightcurve detrending, gap filling
- **Modeling**: CNN and Transformer architectures for spectral and lightcurve analysis
- **Evaluation**: Calibration, conformal prediction, domain shift detection, ensembling
- **Serving**: REST API with 10 endpoints covering predictions and data pipeline
- **Web UI**: Interactive interface for all features with real-time feedback

## Architecture

The platform consists of four main planes:
1. **Control Plane**: Airflow DAGs for orchestration
2. **Data Plane**: Object storage and feature store
3. **Model Plane**: Training cluster with experiment tracking
4. **Serving Plane**: REST API for batch scoring and data processing

## Quick Start

For a step-by-step Windows-friendly guide (venv, dependencies, API, web UI, CLI), see [RUNNING.md](RUNNING.md).

### Installation
```bash
pip install -r requirements.txt
```

### Start API Server
```bash
python -m uvicorn server:app --port 8000
```

### Access Web UI
```
http://localhost:8000/web/index.html
```

The web UI provides:
- **Model Predictions**: Test spectral, lightcurve, and SED models
- **Data Sync**: Incrementally sync SDSS spectra and TESS/Kepler lightcurves  
- **Preprocessing**: Apply spectral and lightcurve transformations
- **Data Loaders**: Generate training batches for model development

### Alternative Training (CLI)
```bash
# Train spectral model
python scripts/train_cli.py spectral --epochs 10 --samples 200

# Train lightcurve model  
python scripts/train_cli.py lightcurve --epochs 10 --samples 200

# Train SED model
python scripts/train_cli.py sed --epochs 10 --samples 200
```

## API Endpoints

### Prediction Endpoints

**Spectral Classification**

```bash
curl -X POST http://localhost:8000/predict/spectral \
   -H 'Content-Type: application/json' \
   -d '{"spectra": [[0.1,0.2,0.3,0.4,0.5]], "model": "spectral_cnn"}'
```

**Light Curve Variability**

```bash
curl -X POST http://localhost:8000/predict/lightcurve \
   -H 'Content-Type: application/json' \
   -d '{"lightcurves": [[1,1.01,0.99,1.02,0.98,1.0]], "model": "lightcurve_transformer"}'
```

**SED Classification**

```bash
curl -X POST http://localhost:8000/predict/sed \
   -H 'Content-Type: application/json' \
   -d '{"seds": [[0.5,0.6,0.7,0.8,0.9]], "model": "sed"}'
```

### Data Pipeline Endpoints (Phase 1 ✅)

**Sync SDSS Spectra** (incremental with resume)

```bash
curl -X POST http://localhost:8000/data/sync/spectra \
   -H 'Content-Type: application/json' \
   -d '{"max_records":10,"min_sn":5.0,"batch_size":10,"resume":true}'
```

**Sync TESS/Kepler Lightcurves**

```bash
curl -X POST http://localhost:8000/data/sync/lightcurves \
   -H 'Content-Type: application/json' \
   -d '{"survey":"TESS","max_records":5,"resume":true}'
```

**Preprocess Spectrum**

```bash
curl -X POST http://localhost:8000/data/preprocess/spectral \
   -H 'Content-Type: application/json' \
   -d '{
     "wavelength":[6800,6850,6900,6950,7000,7500,7600,7650,7700,7800],
     "flux":[1.0,1.1,1.05,1.08,1.0,0.95,1.02,1.0,0.98,1.05],
     "mask_telluric":true,
     "convert_air_to_vacuum":true
   }'
```

**Preprocess Lightcurve**

```bash
curl -X POST http://localhost:8000/data/preprocess/lightcurve \
   -H 'Content-Type: application/json' \
   -d '{
     "time":[0,1,2,3,4,5,6,7,8,9],
     "flux":[1.0,1.01,0.99,1.02,0.98,1.0,1.01,0.99,1.02,0.98],
     "apply_detrend":true,
     "fill_gaps":true
   }'
```

**Generate Training Batch**

```bash
curl -X POST http://localhost:8000/data/loaders/batch \
   -H 'Content-Type: application/json' \
   -d '{"data_type":"spectral","num_samples":16,"batch_size":16}'
```

See [API_PHASE1_FEATURES.md](API_PHASE1_FEATURES.md) for complete API documentation.

## Documentation

- [RUNNING.md](RUNNING.md) - Setup and running instructions
- [API_PHASE1_FEATURES.md](API_PHASE1_FEATURES.md) - Complete API reference
- [PHASE_1_COMPLETION.md](PHASE_1_COMPLETION.md) - Phase 1 completion report
- [COMPLETION_CHECKLIST.md](COMPLETION_CHECKLIST.md) - Project progress tracking

## Testing

Run comprehensive test suite:

```bash
# Unit tests (12 tests)
python -m pytest tests/ -v

# API endpoint tests (10 tests)
python test_api.py

# Frontend smoke tests (9 tests)
python test_frontend_smoke.py
```

**Total: 31 tests, all passing ✅**

## Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Data Ingestion | 🟢 Complete | Incremental sync with resume |
| Preprocessing | 🟢 Complete | Spectral & lightcurve pipelines |
| Data Loaders | 🟢 Complete | Training batch generation |
| Model Training | 🟢 Complete | CLI for all modalities |
| API Endpoints | 🟢 Complete | 10 endpoints operational |
| Web UI | 🟢 Complete | Full integration |
| Testing | 🟢 Complete | 31 tests passing |
| Documentation | 🟢 Complete | All features documented |

**Phase 1: ✅ Complete**  
**Phase 2: Ready to begin**

## Roadmap

### Phase 1 (✅ Complete)
- ✅ Incremental data sync endpoints
- ✅ Preprocessing API for both modalities
- ✅ Training data loader endpoints
- ✅ Web UI integration
- ✅ Comprehensive testing

### Phase 2 (Next)
- [ ] Acquire labeled datasets (SDSS + SIMBAD)
- [ ] Feature store implementation
- [ ] Real model training on labeled data
- [ ] Quality gates and model promotion
- [ ] CI/CD pipeline

### Phase 3 (Future)
- [ ] Multimodal fusion models
- [ ] Contrastive pretraining
- [ ] Production deployment
- [ ] Monitoring and alerting

## Contributing

Please read `CONTRIBUTING.md` for guidelines on how to contribute to this project.

## License

See LICENSE file for details.
