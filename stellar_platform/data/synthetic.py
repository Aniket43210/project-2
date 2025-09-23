"""Synthetic data generators for early model development.

Provides lightweight functions to create structured spectral datasets
until real ingestion pipelines are integrated.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

from .schemas import Spectrum


def generate_synthetic_spectra(n: int = 256, length: int = 512, n_classes: int = 3, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic spectra and integer class labels.

    Spectra are formed by summing a small number of gaussian absorption features
    plus noise; class determines central wavelength shifts.
    Returns (X, y) where X shape (n, length, 1).
    """
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(4000, 7000, length)  # Angstrom
    X = np.zeros((n, length, 1), dtype='float32')
    y = rng.integers(0, n_classes, size=n)
    base_centers = np.linspace(4500, 6500, n_classes)
    for i in range(n):
        cls = y[i]
        continuum = 1.0 - 0.05 * rng.random()
        spectrum = np.full(length, continuum, dtype='float32')
        # add 2-4 gaussian absorption lines
        n_lines = rng.integers(2, 5)
        for _ in range(n_lines):
            center = base_centers[cls] + rng.normal(0, 50)
            depth = rng.uniform(0.05, 0.3)
            width = rng.uniform(0.5, 2.5) * 10.0
            line = depth * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
            spectrum -= line
        noise = rng.normal(0, 0.01, size=length)
        spectrum += noise
        X[i, :, 0] = spectrum
    return X, y

def generate_synthetic_lightcurves(n: int = 256, length: int = 512, n_classes: int = 3, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic light curves with class-dependent periodic patterns.

    Each class corresponds to a different base period and amplitude modulation.
    Returns (X, y) where X shape (n, length, 1).
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, length, 1), dtype='float32')
    y = rng.integers(0, n_classes, size=n)
    base_periods = np.linspace(20, 80, n_classes)
    t = np.linspace(0, 1, length)
    for i in range(n):
        cls = y[i]
        period = base_periods[cls]
        freq = 1.0 / period
        # base sinusoid + harmonic + irregular flares
        amplitude = 1.0 + 0.1 * rng.normal()
        signal = amplitude * np.sin(2 * np.pi * freq * t)
        # add harmonic
        signal += 0.3 * amplitude * np.sin(4 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
        # occasional flare (positive spike)
        n_flares = rng.integers(0, 3)
        for _ in range(n_flares):
            center = rng.uniform(0, 1)
            width = rng.uniform(0.002, 0.01)
            height = rng.uniform(0.2, 0.8)
            flare = height * np.exp(-0.5 * ((t - center) / width) ** 2)
            signal += flare
        # noise
        noise = rng.normal(0, 0.05, size=length)
        signal += noise
        # normalize to roughly 0-1 range
        signal_min, signal_max = signal.min(), signal.max()
        if signal_max - signal_min > 1e-8:
            signal = (signal - signal_min) / (signal_max - signal_min)
        X[i, :, 0] = signal.astype('float32')
    return X, y

__all__ = ["generate_synthetic_spectra", "generate_synthetic_lightcurves"]
