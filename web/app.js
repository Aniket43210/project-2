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
});
