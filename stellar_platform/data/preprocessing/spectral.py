"""
Spectral data preprocessing module for stellar analysis.

This module provides functions for preprocessing astronomical spectra including:
- Resampling to common wavelength grids
- Continuum normalization
- Rest-frame correction
- Masking bad pixels
- Handling missing values
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
import logging

# Optional heavy imports guarded for environments without astronomy stack
try:  # pragma: no cover
    from astropy import units as u
    from astropy.io import fits
    from astropy.nddata import StdDevUncertainty
except Exception:  # pragma: no cover
    u = None  # type: ignore
    fits = None  # type: ignore
    StdDevUncertainty = None  # type: ignore

try:  # pragma: no cover
    from specutils import Spectrum1D
    from specutils.manipulation import FluxConservingResampler
except Exception:  # pragma: no cover
    class _SimpleSpectrum:  # minimal fallback
        def __init__(self, spectral_axis, flux, uncertainty=None):
            self.spectral_axis = spectral_axis
            self.flux = flux
            self.uncertainty = uncertainty
    Spectrum1D = _SimpleSpectrum  # type: ignore
    FluxConservingResampler = None  # type: ignore

logger = logging.getLogger(__name__)


def load_spectrum(file_path: str):
    """
    Load a spectrum from a FITS file.

    Args:
        file_path: Path to the FITS file containing the spectrum

    Returns:
        Spectrum1D object containing the loaded spectrum
    """
    with fits.open(file_path, memmap=False) as hdul:  # type: ignore
        # Heuristic: look for flux+loglam (SDSS-like) or wavelength+flux
        data_hdu = None
        for h in hdul:
            if hasattr(h, 'data') and h.data is not None and len(getattr(h, 'data').shape, 0) in (1, 2):
                data_hdu = h
                break
        if data_hdu is None:
            raise ValueError(f"No data found in FITS file {file_path}")

        hdr = data_hdu.header
        flux = np.array(data_hdu.data).astype(float)
        if flux.ndim > 1:
            flux = flux[0]

        # Wavelength reconstruction
        if 'COEFF0' in hdr and 'COEFF1' in hdr:  # SDSS log10 wavelength solution
            loglam = hdr['COEFF0'] + hdr['COEFF1'] * np.arange(len(flux))
            wavelength = 10 ** loglam * u.AA
        elif 'CRVAL1' in hdr and 'CDELT1' in hdr:
            wavelength = (hdr['CRVAL1'] + hdr['CDELT1'] * np.arange(len(flux))) * u.AA
        else:
            wavelength = np.arange(len(flux)) * u.AA

        # Simple uncertainty if available
        ivar = None
        if 'IVAR' in hdul:
            try:
                ivar = np.array(hdul['IVAR'].data).astype(float)
                if ivar.ndim > 1:
                    ivar = ivar[0]
            except Exception:
                ivar = None
        uncertainty = None
        if ivar is not None:
            with np.errstate(divide='ignore'):
                sigma = np.where(ivar > 0, 1 / np.sqrt(ivar), np.inf)
            uncertainty = StdDevUncertainty(sigma * u.dimensionless_unscaled)

        if u is None:
            raise RuntimeError("Astropy not installed; cannot load spectrum")
        spectrum = Spectrum1D(spectral_axis=wavelength, flux=flux * u.Unit('1e-17 erg / (cm2 s AA)'), uncertainty=uncertainty)
        return spectrum


def resample_to_grid(spectrum, grid: np.ndarray):
    """
    Resample a spectrum to a common wavelength grid.

    Args:
        spectrum: Input spectrum to resample
        grid: Target wavelength grid

    Returns:
        Resampled Spectrum1D object
    """
    try:
        if u is None or FluxConservingResampler is None:
            return spectrum  # silently skip
        target_axis = (grid * u.AA) if not isinstance(grid, u.Quantity) else grid
        resampler = FluxConservingResampler()
        resampled_flux = resampler(spectrum, target_axis)
        return Spectrum1D(spectral_axis=target_axis, flux=resampled_flux.flux, uncertainty=resampled_flux.uncertainty)
    except Exception as e:
        logger.error(f"Resampling failed: {e}; returning original spectrum")
        return spectrum


def normalize_continuum(spectrum, method: str = 'spline', order: int = 3, sigma_clip: float = 3.0):
    """
    Normalize spectrum by continuum.

    Args:
        spectrum: Input Spectrum1D object
        method: Method for continuum fitting ('spline', 'polynomial', 'median')
        order: Order for polynomial/spline fitting
        sigma_clip: Sigma threshold for outlier rejection

    Returns:
        Continuum-normalized Spectrum1D object
    """
    from scipy.interpolate import UnivariateSpline
    from scipy.signal import medfilt
    
    flux = spectrum.flux.value
    wavelength = spectrum.spectral_axis.value
    
    # Sigma-clip outliers
    flux_median = np.median(flux)
    flux_std = np.std(flux)
    mask = np.abs(flux - flux_median) < (sigma_clip * flux_std)
    
    if method == 'spline':
        # Fit spline to continuum
        try:
            spline = UnivariateSpline(
                wavelength[mask],
                flux[mask],
                k=min(order, len(wavelength[mask]) - 1),
                s=len(wavelength) * 0.01  # Smoothing factor
            )
            continuum = spline(wavelength)
        except Exception:
            # Fallback to polynomial
            continuum = np.polyval(np.polyfit(wavelength[mask], flux[mask], order), wavelength)
    
    elif method == 'polynomial':
        coeffs = np.polyfit(wavelength[mask], flux[mask], order)
        continuum = np.polyval(coeffs, wavelength)
    
    elif method == 'median':
        # Running median filter
        window = max(3, len(flux) // 50)
        if window % 2 == 0:
            window += 1
        continuum = medfilt(flux, kernel_size=window)
    
    else:
        raise ValueError(f"Unknown continuum method: {method}")
    
    # Avoid division by zero
    continuum = np.where(continuum > 0, continuum, 1.0)
    normalized_flux = flux / continuum
    
    if u is None:
        return spectrum
    
    return Spectrum1D(
        spectral_axis=spectrum.spectral_axis,
        flux=normalized_flux * u.dimensionless_unscaled,
        uncertainty=spectrum.uncertainty
    )


def convert_air_to_vacuum(wavelength_air: np.ndarray) -> np.ndarray:
    """Convert air wavelengths to vacuum wavelengths.
    
    Uses IAU standard conversion formula (Morton 1991, ApJS, 77, 119).
    Valid for wavelengths > 2000 Angstroms.
    
    Args:
        wavelength_air: Wavelengths in air (Angstroms)
    
    Returns:
        Wavelengths in vacuum (Angstroms)
    """
    s = 1e4 / wavelength_air  # Convert to wavenumber (micron^-1)
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    return wavelength_air * n


def convert_vacuum_to_air(wavelength_vac: np.ndarray) -> np.ndarray:
    """Convert vacuum wavelengths to air wavelengths.
    
    Iterative inversion of air-to-vacuum formula.
    
    Args:
        wavelength_vac: Wavelengths in vacuum (Angstroms)
    
    Returns:
        Wavelengths in air (Angstroms)
    """
    # Iterative solution
    wl_air = wavelength_vac.copy()
    for _ in range(3):  # 3 iterations sufficient for convergence
        wl_air = wavelength_vac / (1 + 0.0000834254 + 0.02406147 / (130 - (1e4/wl_air)**2) + 
                                     0.00015998 / (38.9 - (1e4/wl_air)**2))
    return wl_air


def mask_telluric_regions(
    spectrum,
    regions: Optional[List[Tuple[float, float]]] = None
) -> np.ndarray:
    """Create mask for telluric absorption regions.
    
    Args:
        spectrum: Input Spectrum1D object or wavelength array
        regions: List of (wl_min, wl_max) tuples for telluric regions.
                 If None, uses standard optical/NIR telluric bands.
    
    Returns:
        Boolean mask array (True = good, False = telluric)
    """
    # Handle both Spectrum1D objects and plain wavelength arrays
    if isinstance(spectrum, np.ndarray):
        wavelength = spectrum
    elif hasattr(spectrum, 'spectral_axis'):
        wavelength = spectrum.spectral_axis.value
    else:
        # Assume it's a dict-like object with 'wavelength' key
        wavelength = spectrum.get('wavelength', spectrum)
    
    if regions is None:
        # Standard telluric regions in Angstroms
        regions = [
            (6860, 6960),   # B-band (O2)
            (7580, 7700),   # A-band (O2)
            (9300, 9600),   # Water vapor
            (11100, 11500), # Water vapor
            (13200, 14500)  # Water vapor
        ]
    
    mask = np.ones(len(wavelength), dtype=bool)
    for wl_min, wl_max in regions:
        mask &= ~((wavelength >= wl_min) & (wavelength <= wl_max))
    
    return mask


def correct_redshift(spectrum, z: float):
    """Shift spectrum to rest frame.
    
    Args:
        spectrum: Input Spectrum1D object or dict with 'wavelength' key
        z: Redshift
    
    Returns:
        Rest-frame Spectrum1D object or dict
    """
    # Handle dict representation
    if isinstance(spectrum, dict):
        rest_wavelength = spectrum['wavelength'] / (1 + z)
        return {
            'wavelength': rest_wavelength,
            'flux': spectrum['flux'],
            'uncertainty': spectrum.get('uncertainty')
        }
    
    # Handle Spectrum1D object
    rest_wavelength = spectrum.spectral_axis / (1 + z)
    
    if u is None:
        return spectrum
    
    return Spectrum1D(
        spectral_axis=rest_wavelength,
        flux=spectrum.flux,
        uncertainty=spectrum.uncertainty
    )
    """
    Normalize the continuum of a spectrum.

    Args:
        spectrum: Input spectrum to normalize
        method: Continuum fitting method ('spline', 'polynomial', 'median')

    Returns:
        Normalized Spectrum1D object
    """
    flux = spectrum.flux.value
    wave = spectrum.spectral_axis.to(u.AA).value

    if method == 'median':
        continuum = np.median(flux)
    else:
        # Low-order polynomial fit as default robust fallback
        order = 3
        try:
            # Mask strong absorption/emission using sigma clipping
            mask = np.isfinite(flux)
            for _ in range(2):
                resid = flux[mask] - np.median(flux[mask])
                sigma = np.std(resid)
                new_mask = np.abs(resid) < 3 * sigma
                temp = np.where(mask)[0]
                mask[temp[~new_mask]] = False
            coeffs = np.polyfit(wave[mask], flux[mask], order)
            continuum = np.polyval(coeffs, wave)
        except Exception:
            continuum = np.median(flux)

    norm_flux = flux / continuum
    if u is None:
        return spectrum
    return Spectrum1D(spectral_axis=spectrum.spectral_axis, flux=norm_flux * spectrum.flux.unit, uncertainty=spectrum.uncertainty)


def apply_redshift_correction(spectrum, redshift: float):
    """
    Apply redshift correction to shift spectrum to rest frame.

    Args:
        spectrum: Input spectrum to correct
        redshift: Redshift value (z)

    Returns:
        Rest-frame corrected Spectrum1D object
    """
    if redshift == 0:
        return spectrum
    # Shift to rest frame: lambda_rest = lambda_obs / (1+z)
    rest_axis = spectrum.spectral_axis / (1 + redshift)
    return Spectrum1D(spectral_axis=rest_axis, flux=spectrum.flux, uncertainty=spectrum.uncertainty)


def mask_bad_pixels(spectrum, mask_threshold: float = 3.0):
    """
    Mask bad pixels in a spectrum based on sigma clipping.

    Args:
        spectrum: Input spectrum to process
        mask_threshold: Threshold for sigma clipping

    Returns:
        Spectrum1D object with masked bad pixels
    """
    flux = spectrum.flux.value
    finite = np.isfinite(flux)
    good_flux = flux[finite]
    if good_flux.size < 10:
        return spectrum
    med = np.median(good_flux)
    std = np.std(good_flux)
    mask = np.abs(flux - med) < mask_threshold * std
    # Replace bad pixels via linear interpolation
    bad_idx = np.where(~mask)[0]
    if bad_idx.size > 0:
        good_idx = np.where(mask)[0]
        interp = np.interp(bad_idx, good_idx, flux[good_idx])
        flux_corr = flux.copy()
        flux_corr[bad_idx] = interp
    else:
        flux_corr = flux
    return Spectrum1D(spectral_axis=spectrum.spectral_axis, flux=flux_corr * spectrum.flux.unit, uncertainty=spectrum.uncertainty)


def preprocess_spectrum(
    file_path: str,
    grid: Optional[np.ndarray] = None,
    continuum_method: str = 'spline',
    redshift: Optional[float] = None,
    mask_threshold: float = 3.0
) -> object:
    """
    Complete preprocessing pipeline for stellar spectra.

    Args:
        file_path: Path to the FITS file containing the spectrum
        grid: Target wavelength grid for resampling (optional)
        continuum_method: Method for continuum normalization
        redshift: Redshift value for correction (optional)
        mask_threshold: Threshold for bad pixel masking

    Returns:
        Fully preprocessed Spectrum1D object
    """
    # Load spectrum
    spectrum = load_spectrum(file_path)

    # Apply preprocessing steps
    if mask_threshold is not None:
        spectrum = mask_bad_pixels(spectrum, mask_threshold)

    if continuum_method is not None:
        spectrum = normalize_continuum(spectrum, continuum_method)

    if redshift is not None:
        spectrum = apply_redshift_correction(spectrum, redshift)

    if grid is not None:
        spectrum = resample_to_grid(spectrum, grid)

    return spectrum
