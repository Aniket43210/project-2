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
    order: int = 2,
    robust: bool = True
) -> TimeSeries:
    """
    Remove trends from a light curve with robust fitting options.

    Args:
        time_series: Input TimeSeries object
        method: Detrending method ('polynomial', 'spline', 'lowess', 'median')
        order: Order of polynomial for polynomial detrending or spline degree
        robust: Use robust fitting (RANSAC for polynomial, iterative sigma clipping)

    Returns:
        Detrended TimeSeries object
    """
    from scipy.interpolate import UnivariateSpline
    from scipy.signal import savgol_filter
    
    # Extract flux and time, handling both Quantity and plain array cases
    flux_col = time_series['flux']
    flux = (flux_col.value.astype(float) if hasattr(flux_col, 'value') 
            else np.array(flux_col).astype(float))
    
    time_col = time_series.time
    time = (time_col.jd.astype(float) if hasattr(time_col, 'jd') 
            else np.array(time_col).astype(float))
    
    if method == 'polynomial':
        order = min(order, max(1, len(flux)//10))
        
        if robust:
            # Robust polynomial fit with outlier rejection
            mask = np.ones(len(flux), dtype=bool)
            for _ in range(3):  # Iterative sigma clipping
                coeffs = np.polyfit(time[mask], flux[mask], order)
                trend_all = np.polyval(coeffs, time)
                residuals = flux - trend_all
                sigma = np.std(residuals[mask])
                mask = np.abs(residuals) < 3 * sigma
            trend = np.polyval(coeffs, time)
        else:
            coeffs = np.polyfit(time, flux, order)
            trend = np.polyval(coeffs, time)
    
    elif method == 'spline':
        # Spline detrending
        try:
            s = len(time) * np.var(flux) * 0.01  # Smoothing parameter
            spline = UnivariateSpline(time, flux, k=min(order, 3), s=s)
            trend = spline(time)
        except Exception:
            # Fallback to polynomial
            coeffs = np.polyfit(time, flux, 2)
            trend = np.polyval(coeffs, time)
    
    elif method == 'savgol':
        # Savitzky-Golay filter
        window = max(5, len(flux) // 20)
        if window % 2 == 0:
            window += 1
        window = min(window, len(flux) - 1)
        try:
            trend = savgol_filter(flux, window, min(order, window - 1))
        except Exception:
            trend = np.median(flux)
    
    elif method == 'median':
        trend = np.median(flux)
    
    else:
        trend = 0.0
    
    detrended = flux - trend
    new_ts = TimeSeries(time=time_series.time)
    flux_unit = getattr(time_series['flux'], 'unit', None)
    if flux_unit is not None:
        new_ts['flux'] = detrended * flux_unit
    else:
        new_ts['flux'] = detrended
    if 'flux_err' in time_series.colnames:
        new_ts['flux_err'] = time_series['flux_err']
    return new_ts


def remove_outliers(
    time_series: TimeSeries,
    sigma: float = 3.0,
    window_size: Optional[int] = None,
    method: str = 'global'
) -> TimeSeries:
    """
    Remove outliers from a light curve using sigma clipping.
    
    Args:
        time_series: Input TimeSeries object
        sigma: Sigma threshold for clipping
        window_size: Window size for local sigma clipping (None = global)
        method: 'global' (entire light curve) or 'local' (windowed)
    
    Returns:
        TimeSeries with outliers removed (masked)
    """
    # Extract flux and time, handling both Quantity and plain array cases
    flux_col = time_series['flux']
    flux = (flux_col.value.astype(float) if hasattr(flux_col, 'value') 
            else np.array(flux_col).astype(float))
    
    time_col = time_series.time
    time = (time_col.jd.astype(float) if hasattr(time_col, 'jd') 
            else np.array(time_col).astype(float))
    
    if method == 'global':
        # Global sigma clipping
        median = np.median(flux)
        mad = np.median(np.abs(flux - median))
        sigma_estimate = 1.4826 * mad  # Robust sigma estimate
        mask = np.abs(flux - median) < sigma * sigma_estimate
    
    elif method == 'local':
        # Local sigma clipping with sliding window
        if window_size is None:
            window_size = max(10, len(flux) // 50)
        
        mask = np.ones(len(flux), dtype=bool)
        half_window = window_size // 2
        
        for i in range(len(flux)):
            start = max(0, i - half_window)
            end = min(len(flux), i + half_window + 1)
            window = flux[start:end]
            
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            sigma_estimate = 1.4826 * mad
            
            if np.abs(flux[i] - median) >= sigma * sigma_estimate:
                mask[i] = False
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create new TimeSeries with only non-outlier points
    from astropy.time import Time
    new_ts = TimeSeries(time=time_series.time[mask])
    
    # Handle unit safely
    flux_unit = getattr(time_series['flux'], 'unit', None)
    if flux_unit is not None:
        new_ts['flux'] = flux[mask] * flux_unit
    else:
        new_ts['flux'] = flux[mask]
    
    if 'flux_err' in time_series.colnames:
        flux_err_unit = getattr(time_series['flux_err'], 'unit', None)
        flux_err_vals = time_series['flux_err'].value if hasattr(time_series['flux_err'], 'value') else np.array(time_series['flux_err'])
        if flux_err_unit is not None:
            new_ts['flux_err'] = flux_err_vals[mask] * flux_err_unit
        else:
            new_ts['flux_err'] = flux_err_vals[mask]
    
    return new_ts


def fill_gaps(
    time_series: TimeSeries,
    max_gap: Optional[float] = None,
    method: str = 'linear',
    interpolate_errors: bool = True
) -> TimeSeries:
    """
    Fill gaps in a time series with gap-aware interpolation.

    Args:
        time_series: Input TimeSeries object
        max_gap: Maximum gap size to fill (in time units). None = auto-detect
        method: Interpolation method ('linear', 'cubic', 'nearest', 'spline')
        interpolate_errors: Whether to also interpolate uncertainties

    Returns:
        TimeSeries object with filled gaps
    """
    from scipy.interpolate import interp1d, UnivariateSpline
    
    # Extract time and flux, handling both Quantity and plain array cases
    time_col = time_series.time
    time = (time_col.jd.astype(float) if hasattr(time_col, 'jd') 
            else np.array(time_col).astype(float))
    
    flux_col = time_series['flux']
    flux = (flux_col.value.astype(float) if hasattr(flux_col, 'value') 
            else np.array(flux_col).astype(float))
    dt = np.diff(time)
    
    if max_gap is None:
        # Auto-detect typical cadence
        max_gap = 5 * np.median(dt) if dt.size else 0
    
    # Identify gaps and fill them
    new_time = [time[0]]
    new_flux = [flux[0]]
    
    for i, gap in enumerate(dt):
        if gap > max_gap and max_gap > 0:
            # Fill this gap
            n_insert = int(gap / max_gap)
            t_insert = np.linspace(time[i], time[i+1], n_insert+2)[1:-1]
            
            if method == 'linear':
                f_insert = np.linspace(flux[i], flux[i+1], n_insert+2)[1:-1]
            
            elif method == 'cubic' or method == 'spline':
                # Use surrounding points for interpolation
                start_idx = max(0, i - 5)
                end_idx = min(len(time), i + 7)
                
                if end_idx - start_idx > 3:  # Need at least 4 points
                    try:
                        if method == 'cubic':
                            interp_func = interp1d(
                                time[start_idx:end_idx],
                                flux[start_idx:end_idx],
                                kind='cubic',
                                fill_value='extrapolate'
                            )
                        else:  # spline
                            interp_func = UnivariateSpline(
                                time[start_idx:end_idx],
                                flux[start_idx:end_idx],
                                k=3, s=0
                            )
                        f_insert = interp_func(t_insert)
                    except Exception:
                        # Fallback to linear
                        f_insert = np.linspace(flux[i], flux[i+1], n_insert+2)[1:-1]
                else:
                    f_insert = np.linspace(flux[i], flux[i+1], n_insert+2)[1:-1]
            
            elif method == 'nearest':
                # Nearest neighbor (forward fill)
                f_insert = np.full(len(t_insert), flux[i])
            
            else:
                raise ValueError(f"Unknown interpolation method: {method}")
            
            new_time.extend(t_insert)
            new_flux.extend(f_insert)
        
        new_time.append(time[i+1])
        new_flux.append(flux[i+1])
    
    # Create TimeSeries with proper time format (time is already in JD)
    from astropy.time import Time
    new_ts = TimeSeries(time=Time(new_time, format='jd'))
    flux_unit = getattr(time_series['flux'], 'unit', None)
    if flux_unit is not None:
        new_ts['flux'] = np.array(new_flux) * flux_unit
    else:
        new_ts['flux'] = np.array(new_flux)
    
    if 'flux_err' in time_series.colnames and interpolate_errors:
        # Interpolate uncertainties
        flux_err_col = time_series['flux_err']
        flux_err = (flux_err_col.value if hasattr(flux_err_col, 'value') 
                    else np.array(flux_err_col))
        if method in ['cubic', 'spline']:
            try:
                err_interp = interp1d(time, flux_err, kind='linear', fill_value='extrapolate')
                new_flux_err = err_interp(new_time)
            except Exception:
                new_flux_err = np.interp(new_time, time, flux_err)
        else:
            new_flux_err = np.interp(new_time, time, flux_err)
        
        flux_err_unit = getattr(time_series['flux_err'], 'unit', None)
        if flux_err_unit is not None:
            new_ts['flux_err'] = new_flux_err * flux_err_unit
        else:
            new_ts['flux_err'] = new_flux_err
    
    return new_ts


def calculate_periodogram(
    time_series: TimeSeries,
    normalization: str = 'psd',
    frequencies: Optional[np.ndarray] = None,
    min_period: float = 0.1,
    max_period: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Lomb-Scargle periodogram with enhanced period search.

    Args:
        time_series: Input TimeSeries object
        normalization: Periodogram normalization ('psd', 'standard', 'model')
        frequencies: Optional frequency array to evaluate
        min_period: Minimum period to search (days)
        max_period: Maximum period to search (days). None = baseline/2

    Returns:
        Tuple of (frequencies, power) arrays
    """
    time = time_series['time'].value.astype(float)
    flux = time_series['flux'].value.astype(float)
    
    # Center time at zero
    time = time - time.min()
    baseline = time.max()
    
    if frequencies is None:
        # Define frequency grid based on period range
        if max_period is None:
            max_period = baseline / 2
        
        min_freq = 1.0 / max_period
        max_freq = 1.0 / min_period
        
        # Use logarithmic spacing for better coverage
        n_freqs = min(10000, int(baseline * max_freq * 2))
        frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), n_freqs)
    
    angular = 2 * np.pi * frequencies
    
    # Subtract mean for Lomb-Scargle
    flux_centered = flux - np.mean(flux)
    
    # Calculate power
    if normalization == 'psd':
        # Power spectral density
        power = lombscargle(time, flux_centered, angular, precenter=False, normalize=False)
    else:
        # Normalized power
        power = lombscargle(time, flux_centered, angular, precenter=False, normalize=True)
    
    return frequencies, power


def find_periods(
    time_series: TimeSeries,
    n_periods: int = 5,
    min_period: float = 0.1,
    max_period: Optional[float] = None
) -> List[Dict[str, float]]:
    """
    Find dominant periods in a light curve using Lomb-Scargle periodogram.
    
    Args:
        time_series: Input TimeSeries object
        n_periods: Number of top periods to return
        min_period: Minimum period to search (days)
        max_period: Maximum period to search (days)
    
    Returns:
        List of dictionaries with 'period', 'frequency', 'power', and 'significance'
    """
    from scipy.signal import find_peaks
    
    # Calculate periodogram
    frequencies, power = calculate_periodogram(
        time_series,
        min_period=min_period,
        max_period=max_period
    )
    
    # Find peaks in the periodogram
    peaks, properties = find_peaks(power, prominence=np.std(power) * 0.5)
    
    if len(peaks) == 0:
        return []
    
    # Sort by power (descending)
    sorted_indices = np.argsort(power[peaks])[::-1]
    top_peaks = peaks[sorted_indices[:n_periods]]
    
    # Calculate false alarm probability (rough estimate)
    n_independent = len(frequencies) / 2  # Effective number of independent frequencies
    
    results = []
    for peak_idx in top_peaks:
        freq = frequencies[peak_idx]
        period = 1.0 / freq
        peak_power = power[peak_idx]
        
        # False alarm probability (Scargle 1982 approximation)
        fap = 1 - (1 - np.exp(-peak_power)) ** n_independent
        significance = 1 - fap
        
        results.append({
            'period': period,
            'frequency': freq,
            'power': peak_power,
            'significance': significance
        })
    
    return results


def extract_time_frequency_features(
    time_series: TimeSeries,
    periodogram: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Extract time-frequency features from a light curve for classification.

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
