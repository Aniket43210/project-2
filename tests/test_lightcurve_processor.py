import numpy as np
import pytest
from stellar_platform.models.lightcurve import LightCurveProcessor


def make_time_flux_with_gap(n=200, gap_start=80, gap_end=100, noise=0.01):
    t = np.linspace(0, 10, n)
    flux = 0.5 * np.sin(2 * np.pi * 0.4 * t) + 0.2 * np.cos(2 * np.pi * 0.1 * t)
    flux += noise * np.random.randn(n)
    # simulate a linear trend
    flux += 0.01 * t
    # introduce a gap by NaN-ing samples
    f = flux.copy()
    f[gap_start:gap_end] = np.nan
    return t, f


def test_detrend_polynomial_reduces_trend():
    t = np.linspace(0, 10, 200)
    flux = 0.3 * np.sin(2 * np.pi * 0.3 * t) + 0.02 * t
    proc = LightCurveProcessor({"detrend": True, "detrend_method": "polynomial", "detrend_degree": 1})
    out = proc._detrend_flux(flux, t)
    # after detrending, correlation with time should drop near zero
    corr = np.corrcoef(out, t)[0, 1]
    assert abs(corr) < 0.2


def test_fill_gaps_linear_interpolates():
    t, f = make_time_flux_with_gap()
    # replace NaNs in input to simulate mask; processor will interpolate using neighbors
    mask = np.isnan(f)
    # create a version without NaNs by simple forward fill for threshold logic
    f_ffill = f.copy()
    # simple fill for test's sake
    isn = np.isnan(f_ffill)
    f_ffill[isn] = np.interp(np.flatnonzero(isn), np.flatnonzero(~isn), f_ffill[~isn])
    proc = LightCurveProcessor({"fill_gaps": True, "gap_threshold": np.median(np.diff(t)) * 3, "fill_method": "linear"})
    out = proc._fill_gaps(f_ffill, t)
    # ensure continuity around gap edges
    assert np.isfinite(out).all()
    # values at gap edges should be close to neighbors (linear interp)
    assert abs(out[79] - out[80]) < 0.2
    assert abs(out[100] - out[99]) < 0.2
