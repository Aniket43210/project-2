#!/usr/bin/env python
"""Test script for the Stellar Platform API."""
import requests
import json
import time
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test the health endpoint."""
    print("Testing /health endpoint...")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✅ Status: {resp.status_code}")
        data = resp.json()
        print(f"   Models: {list(data.get('models', {}).keys())}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_lightcurve():
    """Test the lightcurve prediction endpoint."""
    print("\nTesting /predict/lightcurve endpoint...")
    try:
        payload = {
            "lightcurves": [[1.0, 1.01, 0.99, 1.02, 1.0, 0.98, 1.01, 0.99]],
            "apply_calibration": False
        }
        resp = requests.post(
            f"{BASE_URL}/predict/lightcurve",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"✅ Status: {resp.status_code}")
        data = resp.json()
        print(f"   Model: {data['model']}")
        print(f"   Version: {data['version'][:12]}...")
        print(f"   Probabilities: {[round(p, 3) for p in data['probabilities'][0]]}")
        print(f"   Calibrated: {data['calibrated']}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_spectral():
    """Test the spectral prediction endpoint."""
    print("\nTesting /predict/spectral endpoint...")
    try:
        payload = {
            "spectra": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
            "apply_calibration": True,
            "top_k": 2
        }
        resp = requests.post(
            f"{BASE_URL}/predict/spectral",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"✅ Status: {resp.status_code}")
        data = resp.json()
        print(f"   Model: {data['model']}")
        print(f"   Version: {data['version'][:12]}...")
        print(f"   Probabilities: {[round(p, 3) for p in data['probabilities'][0]]}")
        print(f"   Calibrated: {data['calibrated']}")
        if 'top_k' in data:
            print(f"   Top-2: {data['top_k'][0]}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_sed():
    """Test the SED prediction endpoint."""
    print("\nTesting /predict/sed endpoint...")
    try:
        payload = {
            "features": [[1.0, 0.8, 1.1, 0.7, 0.9, 1.05, 0.95, 1.02]],
            "apply_calibration": False
        }
        resp = requests.post(
            f"{BASE_URL}/predict/sed",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"✅ Status: {resp.status_code}")
        data = resp.json()
        print(f"   Model: {data['model']}")
        print(f"   Version: {data['version'][:12]}...")
        print(f"   Probabilities: {[round(p, 3) for p in data['probabilities'][0]]}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_sync_spectra():
    """Test SDSS spectra sync endpoint."""
    print("\nTesting /data/sync/spectra endpoint...")
    try:
        payload = {
            "max_records": 5,
            "min_sn": 5.0,
            "batch_size": 5,
            "resume": True
        }
        resp = requests.post(
            f"{BASE_URL}/data/sync/spectra",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"✅ Status: {resp.status_code}")
        data = resp.json()
        print(f"   Survey: {data['survey']}")
        print(f"   Records processed: {data['records_processed']}")
        print(f"   Status: {data['status']}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_sync_lightcurves():
    """Test TESS/Kepler lightcurves sync endpoint."""
    print("\nTesting /data/sync/lightcurves endpoint...")
    try:
        payload = {
            "mission": "TESS",
            "max_records": 5,
            "resume": True
        }
        resp = requests.post(
            f"{BASE_URL}/data/sync/lightcurves",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"✅ Status: {resp.status_code}")
        data = resp.json()
        print(f"   Survey: {data['survey']}")
        print(f"   Records processed: {data['records_processed']}")
        print(f"   Status: {data['status']}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_preprocess_spectral():
    """Test spectral preprocessing endpoint."""
    print("\nTesting /data/preprocess/spectral endpoint...")
    try:
        payload = {
            "wavelength": [3500, 3600, 3700, 3800, 3900],
            "flux": [1.0, 1.1, 0.95, 1.05, 0.98],
            "apply_continuum_normalization": True,
            "continuum_method": "spline"
        }
        resp = requests.post(
            f"{BASE_URL}/data/preprocess/spectral",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"✅ Status: {resp.status_code}")
        data = resp.json()
        print(f"   Status: {data['status']}")
        print(f"   Wavelength points: {len(data['wavelength'])}")
        print(f"   Flux range: [{min(data['flux']):.3f}, {max(data['flux']):.3f}]")
        print(f"   Continuum normalized: {data['preprocessing_applied']['continuum_normalization']}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_preprocess_lightcurve():
    """Test lightcurve preprocessing endpoint."""
    print("\nTesting /data/preprocess/lightcurve endpoint...")
    try:
        payload = {
            "time": [2457000, 2457001, 2457002, 2457003, 2457004],
            "flux": [1.0, 1.02, 0.99, 1.01, 1.0],
            "detrend": True,
            "remove_outliers": True
        }
        resp = requests.post(
            f"{BASE_URL}/data/preprocess/lightcurve",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"✅ Status: {resp.status_code}")
        data = resp.json()
        print(f"   Status: {data['status']}")
        print(f"   Length: {data['length']}")
        print(f"   Detrended: {data['preprocessing_applied']['detrending']}")
        print(f"   Outliers removed: {data['preprocessing_applied']['outlier_removal']}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_data_loader_spectral():
    """Test spectral data loader endpoint."""
    print("\nTesting /data/loaders/batch (spectral) endpoint...")
    try:
        payload = {
            "data_type": "spectral",
            "num_samples": 8,
            "batch_size": 8,
            "apply_augmentation": False
        }
        resp = requests.post(
            f"{BASE_URL}/data/loaders/batch",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"✅ Status: {resp.status_code}")
        data = resp.json()
        print(f"   Status: {data['status']}")
        print(f"   Data type: {data['data_type']}")
        print(f"   Batch shape: {data['batch_shape']}")
        print(f"   Augmentation: {data['augmentation_applied']}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_data_loader_lightcurve():
    """Test lightcurve data loader endpoint."""
    print("\nTesting /data/loaders/batch (lightcurve) endpoint...")
    try:
        payload = {
            "data_type": "lightcurve",
            "num_samples": 8,
            "batch_size": 8,
            "apply_augmentation": False
        }
        resp = requests.post(
            f"{BASE_URL}/data/loaders/batch",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"✅ Status: {resp.status_code}")
        data = resp.json()
        print(f"   Status: {data['status']}")
        print(f"   Data type: {data['data_type']}")
        print(f"   Batch shape: {data['batch_shape']}")
        print(f"   Mask shape: {data['mask_shape']}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Stellar Platform API Test Suite")
    print("=" * 60)
    
    # Give server time to start if just launched
    time.sleep(1)
    
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Light Curve Prediction", test_lightcurve()))
    results.append(("Spectral Prediction", test_spectral()))
    results.append(("SED Prediction", test_sed()))
    
    # Phase 1 endpoints
    print("\n" + "=" * 60)
    print("Phase 1: Data Pipeline Endpoints")
    print("=" * 60)
    results.append(("SDSS Spectra Sync", test_sync_spectra()))
    results.append(("TESS Lightcurves Sync", test_sync_lightcurves()))
    results.append(("Spectral Preprocessing", test_preprocess_spectral()))
    results.append(("Lightcurve Preprocessing", test_preprocess_lightcurve()))
    results.append(("Spectral Data Loader", test_data_loader_spectral()))
    results.append(("Lightcurve Data Loader", test_data_loader_lightcurve()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    all_passed = all(p for _, p in results)
    sys.exit(0 if all_passed else 1)
