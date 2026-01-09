// Basic helper functions
function parseMatrix(text) {
  return text.trim().split(/\n+/).map(line => {
    return line.split(/[\s,]+/).filter(x => x.length).map(Number);
  }).filter(row => row.length);
}

function pretty(obj) { return JSON.stringify(obj, null, 2); }

async function apiFetch(base, path, body, apiKey) {
  const opts = { method: body ? 'POST':'GET', headers: { 'Content-Type': 'application/json' } };
  if (apiKey) opts.headers['x-api-key'] = apiKey;
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(base.replace(/\/$/, '') + path, opts);
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HTTP ${res.status}: ${txt}`);
  }
  return await res.json();
}

function parseArray(text) {
  return text.trim().split(/[\s,]+/).filter(x => x.length).map(Number);
}

// UI wiring
window.addEventListener('DOMContentLoaded', () => {
  const apiBaseEl = document.getElementById('apiBase');
  const apiKeyEl = document.getElementById('apiKey');

  // Health
  document.getElementById('btnCheck').addEventListener('click', async () => {
    const out = document.getElementById('healthOutput');
    out.textContent = 'Checking...';
    try {
      const data = await apiFetch(apiBaseEl.value, '/health', null, apiKeyEl.value);
      out.textContent = pretty(data);
    } catch (e) { out.textContent = e.message; }
  });

  // Spectral prediction
  document.getElementById('btnPredictSpectral').addEventListener('click', async () => {
    const out = document.getElementById('spectralOutput');
    out.textContent = 'Predicting...';
    try {
      const spectra = parseMatrix(document.getElementById('spectralInput').value);
      if (!spectra.length) throw new Error('Provide at least one spectrum');
      const apply_calibration = document.getElementById('spectralCalib').checked;
      const topK = parseInt(document.getElementById('spectralTopK').value, 10) || undefined;
      const body = { spectra, apply_calibration };
      if (topK) body.top_k = topK;
      const data = await apiFetch(apiBaseEl.value, '/predict/spectral', body, apiKeyEl.value);
      out.textContent = pretty(data);
    } catch (e) { out.textContent = e.message; }
  });

  // Lightcurve prediction
  document.getElementById('btnPredictLC').addEventListener('click', async () => {
    const out = document.getElementById('lcOutput');
    out.textContent = 'Predicting...';
    try {
      const lightcurves = parseMatrix(document.getElementById('lcInput').value);
      if (!lightcurves.length) throw new Error('Provide at least one light curve');
      const apply_calibration = document.getElementById('lcCalib').checked;
      const topK = parseInt(document.getElementById('lcTopK').value, 10) || undefined;
      const body = { lightcurves, apply_calibration };
      if (topK) body.top_k = topK;
      const data = await apiFetch(apiBaseEl.value, '/predict/lightcurve', body, apiKeyEl.value);
      out.textContent = pretty(data);
    } catch (e) { out.textContent = e.message; }
  });

  // Metadata
  document.getElementById('btnMeta').addEventListener('click', async () => {
    const out = document.getElementById('metaOutput');
    out.textContent = 'Fetching...';
    try {
      const fam = document.getElementById('metaFamily').value.trim();
      const data = await apiFetch(apiBaseEl.value, `/models/${encodeURIComponent(fam)}/metadata`, null, apiKeyEl.value);
      out.textContent = pretty(data);
    } catch (e) { out.textContent = e.message; }
  });

  // --- Data Sync: Spectra ---
  const btnSyncSpectra = document.getElementById('btnSyncSpectra');
  if (btnSyncSpectra) {
    btnSyncSpectra.addEventListener('click', async () => {
      const out = document.getElementById('syncSpectraOutput');
      out.textContent = 'Syncing SDSS spectra...';
      try {
        const body = {
          max_records: parseInt(document.getElementById('syncSpecMax').value, 10) || 100,
          min_sn: parseFloat(document.getElementById('syncSpecSN').value) || 5.0,
          batch_size: parseInt(document.getElementById('syncSpecBatch').value, 10) || 100,
          resume: document.getElementById('syncSpecResume').checked
        };
        const data = await apiFetch(apiBaseEl.value, '/data/sync/spectra', body, apiKeyEl.value);
        out.textContent = pretty(data);
      } catch (e) { out.textContent = e.message; }
    });
  }

  // --- Data Sync: Lightcurves ---
  const btnSyncLightcurves = document.getElementById('btnSyncLightcurves');
  if (btnSyncLightcurves) {
    btnSyncLightcurves.addEventListener('click', async () => {
      const out = document.getElementById('syncLightcurvesOutput');
      out.textContent = 'Syncing lightcurves...';
      try {
        const body = {
          mission: document.getElementById('syncLcMission').value,
          max_records: parseInt(document.getElementById('syncLcMax').value, 10) || 100,
          resume: document.getElementById('syncLcResume').checked
        };
        const data = await apiFetch(apiBaseEl.value, '/data/sync/lightcurves', body, apiKeyEl.value);
        out.textContent = pretty(data);
      } catch (e) { out.textContent = e.message; }
    });
  }

  // --- Preprocess: Spectral ---
  const btnPreprocessSpectral = document.getElementById('btnPreprocessSpectral');
  if (btnPreprocessSpectral) {
    btnPreprocessSpectral.addEventListener('click', async () => {
      const out = document.getElementById('preSpecOutput');
      out.textContent = 'Running spectral preprocessing...';
      try {
        const wavelength = parseArray(document.getElementById('preSpecWavelength').value);
        const flux = parseArray(document.getElementById('preSpecFlux').value);
        if (!wavelength.length || !flux.length) throw new Error('Provide wavelength and flux arrays');
        if (wavelength.length !== flux.length) throw new Error('Wavelength and flux must have same length');
        const redshiftRaw = document.getElementById('preSpecRedshift').value;
        const correct_redshift = redshiftRaw ? parseFloat(redshiftRaw) : null;
        const body = {
          wavelength,
          flux,
          uncertainty: null,
          apply_continuum_normalization: document.getElementById('preSpecContinuum').checked,
          continuum_method: document.getElementById('preSpecMethod').value,
          convert_air_to_vacuum: document.getElementById('preSpecAirVac').checked,
          mask_telluric: document.getElementById('preSpecTelluric').checked,
          correct_redshift: correct_redshift
        };
        const data = await apiFetch(apiBaseEl.value, '/data/preprocess/spectral', body, apiKeyEl.value);
        out.textContent = pretty(data);
      } catch (e) { out.textContent = e.message; }
    });
  }

  // --- Preprocess: Lightcurve ---
  const btnPreprocessLightcurve = document.getElementById('btnPreprocessLightcurve');
  if (btnPreprocessLightcurve) {
    btnPreprocessLightcurve.addEventListener('click', async () => {
      const out = document.getElementById('preLcOutput');
      out.textContent = 'Running lightcurve preprocessing...';
      try {
        const time = parseArray(document.getElementById('preLcTime').value);
        const flux = parseArray(document.getElementById('preLcFlux').value);
        if (!time.length || !flux.length) throw new Error('Provide time and flux arrays');
        if (time.length !== flux.length) throw new Error('Time and flux must have same length');
        const body = {
          time,
          flux,
          uncertainty: null,
          detrend: document.getElementById('preLcDetrend').checked,
          detrend_method: document.getElementById('preLcDetrendMethod').value,
          remove_outliers: document.getElementById('preLcOutliers').checked,
          outlier_sigma: parseFloat(document.getElementById('preLcSigma').value) || 5.0,
          fill_gaps: document.getElementById('preLcFillGaps').checked,
          interpolation_method: document.getElementById('preLcInterp').value,
          find_period: document.getElementById('preLcFindPeriod').checked
        };
        const data = await apiFetch(apiBaseEl.value, '/data/preprocess/lightcurve', body, apiKeyEl.value);
        out.textContent = pretty(data);
      } catch (e) { out.textContent = e.message; }
    });
  }

  // --- Data Loaders: Batch ---
  const btnGetBatch = document.getElementById('btnGetBatch');
  if (btnGetBatch) {
    btnGetBatch.addEventListener('click', async () => {
      const out = document.getElementById('loaderOutput');
      out.textContent = 'Generating batch...';
      try {
        const body = {
          data_type: document.getElementById('loaderType').value,
          num_samples: parseInt(document.getElementById('loaderNum').value, 10) || 16,
          batch_size: parseInt(document.getElementById('loaderBatch').value, 10) || 16,
          apply_augmentation: document.getElementById('loaderAug').checked
        };
        const data = await apiFetch(apiBaseEl.value, '/data/loaders/batch', body, apiKeyEl.value);
        out.textContent = pretty(data);
      } catch (e) { out.textContent = e.message; }
    });
  }
});
