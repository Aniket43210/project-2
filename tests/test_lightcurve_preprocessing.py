import numpy as np
import pandas as pd
from pathlib import Path
from stellar_platform.data.preprocessing.lightcurve import load_lightcurve, detrend_lightcurve, calculate_periodogram, segment_lightcurve, fill_gaps


def test_lightcurve_pipeline(tmp_path: Path):
    # Create synthetic light curve
    t = np.linspace(0,10,200)
    flux = 1 + 0.1*np.sin(2*np.pi*0.5*t) + 0.01*np.random.randn(len(t))
    df = pd.DataFrame({'time': t, 'flux': flux})
    fpath = tmp_path / 'lc.csv'
    df.to_csv(fpath, index=False)

    ts = load_lightcurve(str(fpath))
    detrended = detrend_lightcurve(ts)
    freqs, power = calculate_periodogram(detrended)
    assert len(freqs) == len(power)
    segs = segment_lightcurve(detrended, segment_length=2.0, overlap=0.5)
    assert len(segs) > 0
