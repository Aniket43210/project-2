# Orchestration DAGs

This directory contains workflow definitions for the stellar platform training pipeline using two orchestration frameworks:

## Airflow DAG (`stellar_training_dag.py`)

**Apache Airflow** is a mature, production-tested workflow orchestration platform.

### Running with Airflow

1. **Install Airflow:**
   ```bash
   pip install apache-airflow>=2.4.0
   ```

2. **Initialize Airflow:**
   ```bash
   airflow db init
   ```

3. **Set AIRFLOW_HOME (optional):**
   ```bash
   set AIRFLOW_HOME=.
   ```

4. **Enable the DAG:**
   - Copy `stellar_training_dag.py` to your Airflow DAGs folder (default: `~/airflow/dags/`)
   - Or set `AIRFLOW__CORE__DAGS_FOLDER` to point to this directory

5. **Start the scheduler and webserver:**
   ```bash
   airflow scheduler
   airflow webui  # In another terminal
   ```

6. **Trigger the DAG:**
   ```bash
   airflow dags trigger stellar_training_pipeline
   ```

### DAG Structure

```
ingestion (parallel)
├── ingest_sdss_spectra
├── ingest_gaia_photometry
└── ingest_kepler_lightcurves

preprocessing (parallel)
├── preprocess_spectra
└── preprocess_lightcurves

split_data
├── train_spectral_model
├── train_lightcurve_model
└── train_sed_model

evaluate_models → promote_models → deploy_api
```

### Features

- **Idempotent**: Safe to retry failed tasks
- **Fault-tolerant**: Automatic retry with exponential backoff
- **Monitored**: Web UI for tracking DAG runs
- **Configurable**: Adjust schedule_interval, task parameters, etc.

---

## Prefect Flow (`stellar_training_prefect.py`)

**Prefect 2.x** offers a more modern, Pythonic approach to workflow orchestration.

### Running with Prefect

1. **Install Prefect:**
   ```bash
   pip install prefect>=2.0.0
   ```

2. **Start Prefect server (optional):**
   ```bash
   prefect server start
   ```

3. **Run the flow locally:**
   ```bash
   python dags/stellar_training_prefect.py
   ```

4. **Deploy to Prefect Cloud (optional):**
   ```bash
   prefect cloud login
   prefect deploy stellar_training_prefect.py
   ```

### Advantages

- **Cleaner syntax**: Task decorators are more intuitive
- **Better type support**: Native Python type hints
- **Easier testing**: Tasks can be tested independently
- **Dynamic**: Flow logic can depend on upstream task results

---

## Comparison

| Feature | Airflow | Prefect |
|---------|---------|---------|
| Maturity | Production-tested (10+ years) | Modern (2+ years, rapidly improving) |
| Complexity | Higher learning curve | Simpler, more Pythonic |
| Scalability | Excellent for large enterprises | Good for mid-size teams |
| Debugging | Web UI, extensive logging | Better error messages, UI improving |
| Cloud Integration | Broad third-party support | Prefect Cloud + integrations |

---

## Customization for Real Data

Both DAGs currently have **placeholder implementations**. To enable real ingestion & training:

1. **Implement data connectors**:
   - Fill in `ingest_sdss_spectra()` to actually query SDSS
   - Fill in `ingest_gaia_photometry()` to query Gaia
   - Fill in `ingest_kepler_lightcurves()` to query Kepler/TESS

2. **Implement preprocessing**:
   - Load downloaded data files
   - Apply normalization, detrending, feature extraction
   - Save outputs to a feature store (parquet, HDF5, etc.)

3. **Implement training tasks**:
   - Load preprocessed data
   - Train models via CLI or library calls
   - Automatically register to model registry

4. **Add quality gates**:
   - Set minimum metric thresholds in `evaluate_models()`
   - Only promote models that pass

5. **Configure scheduling**:
   - Change `schedule_interval` from `'@weekly'` to your desired cadence
   - Adjust retry logic and timeouts

---

## Example: Manual Training

Until orchestration is fully wired, you can train models manually:

```bash
# Train dummy spectral model
python scripts/train_cli.py train-spectral --force-dummy --samples 64 --output-dir artifacts

# Train with synthetic data (requires TensorFlow)
python scripts/train_cli.py train-spectral --samples 256 --length 512 --epochs 5 --output-dir artifacts

# Train light curve model
python scripts/train_cli.py train-lightcurve --force-dummy --samples 64 --output-dir artifacts

# Train SED model
python scripts/train_cli.py train-sed --samples 256 --bands 8 --output-dir artifacts
```

---

## Monitoring & Alerting

### Airflow

- **Web UI**: http://localhost:8080
- **Logs**: `$AIRFLOW_HOME/logs/`
- **Alerts**: Configure email or Slack via task config

### Prefect

- **Cloud UI**: https://app.prefect.cloud/
- **Server UI**: http://localhost:3000 (if running server)
- **Alerts**: Prefect integrations (PagerDuty, Slack, etc.)

---

## Next Steps

1. **Phase 1**: Implement real data ingestion for one modality (e.g., SDSS spectra)
2. **Phase 2**: Add preprocessing and feature store
3. **Phase 3**: Wire up model training with real labeled data
4. **Phase 4**: Deploy to production orchestrator (Airflow on Kubernetes, Prefect Cloud, etc.)
