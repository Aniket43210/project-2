# Multi-modal Stellar Analysis Platform

A comprehensive platform for ingesting public survey data, predicting stellar types and physical parameters, and classifying variability with calibrated uncertainties and leakage-safe evaluation.

## Scope and Phasing

- **v1.0**: Spectral classification, light-curve variability and logg estimation, leakage-safe evaluation, model registry, batch API
- **v1.1**: SED-based classification, multi-task heads, domain-shift evaluation
- **v2.0**: Multimodal fusion, contrastive pretraining

## Data Sources

- SDSS spectroscopic DRs
- Gaia photometry
- Kepler/TESS light curves
- SIMBAD top-level classes for labels

## Key Features

- **Data Ingestion**: Connectors for major astronomical surveys
- **Preprocessing**: Spectral normalization, light-curve detrending, SED assembly
- **Modeling**: Modular trainers per modality with uncertainty estimation
- **Evaluation**: Leakage-safe splits, per-class metrics, calibration plots
- **Serving**: Batch API for catalog scoring with artifact-based versioning
- **Monitoring**: Data drift, performance regression, calibration drift detection

## Architecture

The platform consists of four main planes:
1. **Control Plane**: Airflow DAGs for orchestration
2. **Data Plane**: Object storage and feature store
3. **Model Plane**: Training cluster with experiment tracking
4. **Serving Plane**: REST/gRPC service for batch scoring

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables in `.env`
3. Initialize database: `python scripts/setup_db.py`
4. Run ingestion DAG: `airflow dags trigger ingest_stellar_data`
5. Train models: `airflow dags trigger train_stellar_models`
6. Start API server (after training at least one model): `uvicorn stellar_platform.serving.api:app --reload`

### Example Spectral Prediction Request

```bash
curl -X POST http://localhost:8000/predict/spectral \
   -H 'Content-Type: application/json' \
   -d '{"spectra": [[0.1,0.2,0.3,0.4,0.5]], "model": "spectral_cnn"}'
```

### Example Light Curve Prediction Request

```bash
curl -X POST http://localhost:8000/predict/lightcurve \
   -H 'Content-Type: application/json' \
   -d '{"lightcurves": [[1,1.01,0.99,1.02,0.98,1.0]], "model": "lightcurve_transformer"}'
```

## Documentation

See the `docs` directory for detailed documentation on:
- Data processing pipelines
- Model architectures
- Evaluation metrics
- Deployment procedures

## Contributing

Please read `CONTRIBUTING.md` for guidelines on how to contribute to this project.

## Project Structure

```
stellar_platform/
├── data/                          # Data ingestion and processing
│   ├── ingestion/                 # Survey connectors
│   │   ├── sdss.py               # SDSS data connector
│   │   ├── gaia.py               # Gaia data connector
│   │   └── kepler_tess.py       # Kepler/TESS data connector
│   ├── preprocessing/            # Data preprocessing modules
│   │   ├── spectral.py          # Spectral preprocessing
│   │   ├── lightcurve.py        # Light curve preprocessing
│   │   └── sed.py               # SED preprocessing
│   └── manager.py               # Data manager for coordinating ingestion
├── models/                       # Model architectures and training
│   ├── spectral.py              # Spectral classification models
│   ├── lightcurve.py            # Light curve variability models
│   └── sed.py                   # SED-based classification models
├── evaluation/                  # Model evaluation and metrics
│   └── metrics.py               # Evaluation metrics for stellar analysis
└── serving/                      # Model serving and API (to be implemented)
```

## Implemented Components

### Data Ingestion and Processing

1. **Survey Connectors**:
   - `SDSSConnector`: Connects to SDSS survey for spectra and photometry
   - `GaiaConnector`: Connects to Gaia mission for astrometry and photometry
   - `KeplerTESSConnector`: Connects to Kepler and TESS missions for light curves

2. **Data Preprocessing**:
   - Spectral preprocessing: Resampling, continuum normalization, rest-frame correction, bad pixel masking
   - Light curve preprocessing: Detrending, gap filling, periodogram calculation, feature extraction
   - SED preprocessing: Band merging, extinction correction, uncertainty propagation

3. **Data Manager**:
   - Coordinates ingestion from multiple surveys
   - Tracks metadata for all ingested objects
   - Handles preprocessing pipeline

### Model Architectures

1. **Spectral Models**:
   - `SpectralCNN`: 1D convolutional neural network for spectral classification
   - `SpectralTransformer`: Transformer-based architecture for spectral analysis
   - `SpectralMultiTask`: Multi-task model for classification and parameter estimation

2. **Light Curve Models**:
   - `LightCurveTransformer`: Transformer-based model for variability classification
   - `LightCurveMultiTask`: Multi-task model for classification and parameter estimation

3. **SED Models**:
   - `SEDClassifier`: Gradient boosting classifier for SED-based stellar classification
   - `SEDRegressor`: Gradient boosting regressor for parameter estimation from photometry

### Evaluation Metrics

1. **Classification Metrics**:
   - Accuracy, precision, recall, F1-score, MCC
   - AUC and PR curves for binary and multiclass problems
   - Per-class metrics and confusion matrices

2. **Regression Metrics**:
   - MSE, RMSE, MAE, R²
   - Residual analysis and prediction vs. true plots

3. **Calibration Metrics**:
   - Brier score and reliability curves
   - Calibration plots for probability estimates

### Calibration Utilities

- `TemperatureScaler`: For temperature scaling calibration
- `IsotonicCalibrator`: For isotonic regression calibration
- `expected_calibration_error`: To compute expected calibration error
- `reliability_curve`: To plot reliability curves

## Calibration Usage

After training a classifier (e.g., spectral or light curve model), you can post-hoc calibrate its probability outputs.

### Temperature Scaling

```python
from stellar_platform.evaluation import TemperatureScaler

# logits: np.ndarray shape (N, C) raw model outputs BEFORE softmax
# y_true: np.ndarray shape (N,) integer labels
scaler = TemperatureScaler().fit(logits, y_true)
calibrated_probs = scaler.transform(logits)
print('Learned temperature:', scaler.temperature)
```

### Isotonic Regression Calibration

```python
from stellar_platform.evaluation import IsotonicCalibrator

# probs: np.ndarray shape (N, C) already softmaxed probabilities
iso = IsotonicCalibrator(per_class=True).fit(probs, y_true)
calibrated_probs = iso.transform(probs)
```

### Measuring Calibration

```python
from stellar_platform.evaluation import expected_calibration_error, reliability_curve

pre_ece = expected_calibration_error(probs, y_true, n_bins=15)
post_ece = expected_calibration_error(calibrated_probs, y_true, n_bins=15)
print('ECE before:', pre_ece, 'after:', post_ece)
```

You can also plot the reliability curve:

```python
import matplotlib.pyplot as plt
centers, acc, conf, counts = reliability_curve(calibrated_probs, y_true, n_bins=15)
plt.plot([0,1],[0,1],'--',color='gray')
plt.scatter(conf, acc, s=counts, alpha=0.7)
plt.xlabel('Predicted confidence')
plt.ylabel('Empirical accuracy')
plt.title('Reliability Curve (Calibrated)')
plt.show()
```

### When to Calibrate
- After model selection on a validation set (avoid information leakage)
- Before converting probabilities into hard decisions with thresholds
- Prior to combining modalities or ensembling (consistent probability scaling)

### Persistence

Calibrators provide `to_dict()` / `from_dict()` helpers so you can store parameters alongside model artifacts.

```python
payload = scaler.to_dict()
# save payload as JSON
# later
from stellar_platform.evaluation import BaseProbCalibrator
reloaded = BaseProbCalibrator.from_dict(payload)
```

## Roadmap

### Phase 1 (Current)
- Complete implementation of v1.0 components
- Add model registry and batch API
- Implement evaluation suite with leakage-safe splits
 - Implement ingestion connectors (SDSS, Gaia, Kepler/TESS) and preprocessing

### Phase 2 (Next)
- Add SED-based classification models
- Implement multi-task heads for parameter estimation
- Add domain-shift evaluation between surveys

### Phase 3 (Future)
- Implement multimodal fusion of spectra, light curves, and SEDs
- Add contrastive pretraining on simulated/observed pairs
- Expand to imaging data in v2.0

