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
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(p for _, p in results)
    sys.exit(0 if all_passed else 1)
