"""Smoke test for frontend API integration."""
import requests
import json

BASE_URL = "http://localhost:8000"

def _test_endpoint(name, method, path, body=None):
    """Test a single endpoint (helper)."""
    try:
        url = f"{BASE_URL}{path}"
        if method == "GET":
            resp = requests.get(url, timeout=10)
        else:
            resp = requests.post(url, json=body, timeout=10)
        
        if resp.status_code in [200, 201]:
            print(f"✓ {name}: PASS ({resp.status_code})")
            return True
        else:
            print(f"✗ {name}: FAIL ({resp.status_code}) - {resp.text[:100]}")
            return False
    except Exception as e:
        print(f"✗ {name}: ERROR - {str(e)[:100]}")
        return False

def test_all_endpoints():
    """Run smoke tests for all Phase 1 endpoints."""
    print("=" * 60)
    print("FRONTEND API SMOKE TEST")
    print("=" * 60)
    
    tests = [
        ("Health Check", "GET", "/health", None),
        
        ("Sync Spectra", "POST", "/data/sync/spectra", {
            "max_records": 5,
            "min_sn": 5.0,
            "batch_size": 5,
            "resume": True
        }),
        
        ("Sync Lightcurves", "POST", "/data/sync/lightcurves", {
            "mission": "TESS",
            "max_records": 5,
            "resume": True
        }),
        
        ("Preprocess Spectral", "POST", "/data/preprocess/spectral", {
            "wavelength": [3500, 3600, 3700, 3800, 3900],
            "flux": [1.0, 1.1, 0.95, 1.05, 0.98],
            "apply_continuum_normalization": True,
            "continuum_method": "spline"
        }),
        
        ("Preprocess Lightcurve", "POST", "/data/preprocess/lightcurve", {
            "time": [2457000, 2457001, 2457002, 2457003, 2457004],
            "flux": [1.0, 1.02, 0.99, 1.01, 1.0],
            "detrend": True,
            "remove_outliers": True
        }),
        
        ("Data Loader - Spectral", "POST", "/data/loaders/batch", {
            "data_type": "spectral",
            "num_samples": 8,
            "batch_size": 8,
            "apply_augmentation": False
        }),
        
        ("Data Loader - Lightcurve", "POST", "/data/loaders/batch", {
            "data_type": "lightcurve",
            "num_samples": 8,
            "batch_size": 8,
            "apply_augmentation": False
        }),
        
        ("Predict Spectral", "POST", "/predict/spectral", {
            "spectra": [[0.1, 0.2, 0.3, 0.4, 0.5]],
            "apply_calibration": False
        }),
        
        ("Predict Lightcurve", "POST", "/predict/lightcurve", {
            "lightcurves": [[1.0, 1.01, 0.99, 1.02, 0.98]],
            "apply_calibration": False
        }),
    ]
    
    results = []
    for test_data in tests:
        results.append(_test_endpoint(*test_data))
    
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    assert passed == total, f"Expected {total} tests to pass, but only {passed} passed"

if __name__ == "__main__":
    import sys
    try:
        test_all_endpoints()
        sys.exit(0)
    except AssertionError as e:
        print(f"Test failed: {e}")
        sys.exit(1)
