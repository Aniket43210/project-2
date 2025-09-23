"""
Light curve data preprocessing module for stellar variability analysis.

This module provides functions for preprocessing astronomical light curves including:
- Detrending
- Gap filling
- Periodogram calculation
- Time-frequency feature extraction
- Segmentation for transformer models
"""

import numpy as np
import pandas as pd
from astropy.timeseries import TimeSeries
from astropy.time import Time
from astropy import units as u
from astropy.table import Table
from scipy.signal import lombscargle
from scipy.interpolate import interp1d
from typing import Tuple, Optional, Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)


def load_lightcurve(file_path: str, time_format: str = 'jd') -> TimeSeries:
    """
    Load a light curve from a file.

    Args:
        file_path: Path to the file containing the light curve
        time_format: Time format ('jd' for Julian Date, 'mjd' for Modified Julian Date)

    Returns:
        TimeSeries object containing the loaded light curve
    """
    # Simple loader: expect CSV with columns time, flux, (optional) flux_err
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        # Fallback attempt to read as ASCII
        df = pd.read_table(file_path, sep='[ ,\t]+', engine='python')
    required = {'time', 'flux'}
    if not required.issubset(df.columns):
        raise ValueError(f"Lightcurve file missing required columns {required}")
    # Interpret time column based on format
    raw_time = df['time'].values
    # If values look like small relative times (e.g., 0-1000), offset to a large JD base
    if time_format.lower() in ('jd','mjd'):
        if raw_time.max() < 1e6:  # heuristic: treat as relative days
            base_jd = 2450000.0
            if time_format.lower() == 'mjd':
                # Convert base to mjd
                base_time = Time(base_jd, format='jd')
                base_val = base_time.mjd
            else:
                base_val = base_jd
            t = Time(base_val + raw_time, format=time_format.lower())
        else:
            t = Time(raw_time, format=time_format.lower())
    else:
        # Treat times as relative days added to arbitrary JD base
        base_time = Time(2450000.0, format='jd')
        t = base_time + raw_time * u.day
    ts = TimeSeries(time=t)
    ts['flux'] = df['flux'].values * u.dimensionless_unscaled
    if 'flux_err' in df.columns:
        ts['flux_err'] = df['flux_err'].values * u.dimensionless_unscaled
    return ts


def detrend_lightcurve(
    time_series: TimeSeries,
    method: str = 'polynomial',
    order: int = 2
) -> TimeSeries:
    """
    Remove trends from a light curve.

    Args:
        time_series: Input TimeSeries object
        method: Detrending method ('polynomial', 'spline', 'lowess')
        order: Order of polynomial for polynomial detrending

    Returns:
        Detrended TimeSeries object
    """
    flux = time_series['flux'].value.astype(float)
    time = time_series['time'].value.astype(float)
    if method == 'polynomial':
        order = min(order, max(1, len(flux)//10))
        coeffs = np.polyfit(time, flux, order)
        trend = np.polyval(coeffs, time)
    elif method == 'median':
        trend = np.median(flux)
    else:
        trend = 0.0
    detrended = flux - trend
    new_ts = TimeSeries(time=time_series.time)
    new_ts['flux'] = detrended * time_series['flux'].unit
    if 'flux_err' in time_series.colnames:
        new_ts['flux_err'] = time_series['flux_err']
    return new_ts


def fill_gaps(
    time_series: TimeSeries,
    max_gap: Optional[float] = None,
    method: str = 'linear'
) -> TimeSeries:
    """
    Fill gaps in a time series.

    Args:
        time_series: Input TimeSeries object
        max_gap: Maximum gap size to fill (in time units)
        method: Interpolation method ('linear', 'cubic', 'nearest')

    Returns:
        TimeSeries object with filled gaps
    """
    time = time_series['time'].value.astype(float)
    flux = time_series['flux'].value.astype(float)
    dt = np.diff(time)
    if max_gap is None:
        max_gap = 5 * np.median(dt) if dt.size else 0
    # Identify missing segments by inserting interpolated points for big gaps
    new_time = [time[0]]
    new_flux = [flux[0]]
    for i, gap in enumerate(dt):
        if gap > max_gap and max_gap > 0:
            n_insert = int(gap / max_gap)
            t_insert = np.linspace(time[i], time[i+1], n_insert+2)[1:-1]
            if method == 'linear':
                f_insert = np.linspace(flux[i], flux[i+1], n_insert+2)[1:-1]
            else:
                f_insert = np.interp(t_insert, [time[i], time[i+1]], [flux[i], flux[i+1]])
            new_time.extend(t_insert)
            new_flux.extend(f_insert)
        new_time.append(time[i+1])
        new_flux.append(flux[i+1])
    new_ts = TimeSeries(time=new_time * u.day)
    new_ts['flux'] = np.array(new_flux) * time_series['flux'].unit
    if 'flux_err' in time_series.colnames:
        # Simple copy; real implementation would interpolate errors
        new_ts['flux_err'] = np.interp(new_time, time, time_series['flux_err'].value) * time_series['flux_err'].unit
    return new_ts


def calculate_periodogram(
    time_series: TimeSeries,
    normalization: str = 'psd',
    frequencies: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Lomb-Scargle periodogram of a time series.

    Args:
        time_series: Input TimeSeries object
        normalization: Periodogram normalization method
        frequencies: Optional frequency array to evaluate

    Returns:
        Tuple of (frequencies, power) arrays
    """
    time = time_series['time'].value.astype(float)
    flux = time_series['flux'].value.astype(float)
    time = time - time.min()
    if frequencies is None:
        # Nyquist approx for uneven sampling: use median dt
        dt = np.median(np.diff(time)) if len(time) > 1 else 1.0
        f_max = 0.5 / dt
        frequencies = np.linspace(0.01 / (time.max()+1e-9), f_max, 500)
    angular = 2 * np.pi * frequencies
    flux = flux - np.mean(flux)
    power = lombscargle(time, flux, angular, precenter=False, normalize=True)
    return frequencies, power


def extract_time_frequency_features(
    time_series: TimeSeries,
    periodogram: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Dict[str, np.ndarray]:
    """
    Extract time-frequency features from a light curve.

    Args:
        time_series: Input TimeSeries object
        periodogram: Optional precomputed periodogram

    Returns:
        Dictionary of extracted features
    """
    time = time_series['time'].value.astype(float)
    flux = time_series['flux'].value.astype(float)
    if periodogram is None:
        periodogram = calculate_periodogram(time_series)
    freq, power = periodogram
    # Basic features
    features: Dict[str, Any] = {}
    features['flux_mean'] = np.mean(flux)
    features['flux_std'] = np.std(flux)
    features['flux_skew'] = (np.mean((flux - features['flux_mean'])**3) / (features['flux_std']**3 + 1e-9)) if features['flux_std']>0 else 0
    peak_idx = np.argmax(power)
    features['peak_frequency'] = freq[peak_idx]
    features['peak_power'] = power[peak_idx]
    # Amplitude estimate
    features['amplitude_ptp'] = np.ptp(flux)
    return {k: np.array([v]) for k, v in features.items()}


def segment_lightcurve(
    time_series: TimeSeries,
    segment_length: float,
    overlap: float = 0.0
) -> List[TimeSeries]:
    """
    Segment a light curve into overlapping or non-overlapping chunks.

    Args:
        time_series: Input TimeSeries object
        segment_length: Length of each segment
        overlap: Overlap between segments (fraction of segment_length)

    Returns:
        List of TimeSeries segments
    """
    time = time_series['time'].value.astype(float)
    flux = time_series['flux'].value.astype(float)
    t_min, t_max = time.min(), time.max()
    segments: List[TimeSeries] = []
    if segment_length <= 0:
        return [time_series]
    step = segment_length * (1 - overlap)
    start = t_min
    while start < t_max - segment_length * 0.5:
        end = start + segment_length
        mask = (time >= start) & (time < end)
        if mask.sum() > 3:
            rel = time[mask] - start
            # Reconstruct Time with JD base so astropy accepts it
            base_time = Time(2450000.0, format='jd')
            ts_time = base_time + rel * u.day
            ts = TimeSeries(time=ts_time)
            ts['flux'] = flux[mask] * time_series['flux'].unit if hasattr(time_series['flux'], 'unit') else flux[mask]
            segments.append(ts)
        start += step
    if not segments:
        return [time_series]
    return segments


def preprocess_lightcurve(
    file_path: str,
    time_format: str = 'jd',
    detrend_method: str = 'polynomial',
    detrend_order: int = 2,
    max_gap: Optional[float] = None,
    fill_method: str = 'linear',
    segment_length: Optional[float] = None,
    overlap: float = 0.0
) -> Union[TimeSeries, List[TimeSeries]]:
    """
    Complete preprocessing pipeline for stellar light curves.

    Args:
        file_path: Path to the file containing the light curve
        time_format: Time format for the light curve
        detrend_method: Method for detrending
        detrend_order: Order for polynomial detrending
        max_gap: Maximum gap size to fill
        fill_method: Interpolation method for gap filling
        segment_length: Optional length for segmentation
        overlap: Overlap fraction for segmentation

    Returns:
        Preprocessed TimeSeries or list of segments
    """
    # Load light curve
    time_series = load_lightcurve(file_path, time_format)

    # Apply preprocessing steps
    time_series = detrend_lightcurve(time_series, detrend_method, detrend_order)

    if max_gap is not None:
        time_series = fill_gaps(time_series, max_gap, fill_method)

    if segment_length is not None:
        segments = segment_lightcurve(time_series, segment_length, overlap)
        return segments

    return time_series
