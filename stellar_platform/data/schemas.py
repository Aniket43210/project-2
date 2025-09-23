"""Pydantic data schemas for canonical stellar platform objects.

Defines strongly-typed structures for spectra, light curves, and unified source objects.
These schemas standardize input/output across ingestion, preprocessing, training, and serving.

Design goals:
- Minimal required fields for early phase
- Validation of array lengths > 0
- Support optional uncertainty arrays
- Leave room for metadata provenance extension
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

class Spectrum(BaseModel):
    wavelengths: List[float] = Field(..., description="Monotonic wavelength (or frequency) grid")
    flux: List[float] = Field(..., description="Flux values aligned to wavelengths")
    flux_unc: Optional[List[float]] = Field(None, description="Optional 1-sigma uncertainties")
    instrument: Optional[str] = Field(None, description="Instrument or survey identifier")
    rest_frame: bool = Field(False, description="Whether wavelengths are in rest frame")

    @field_validator('wavelengths')
    @classmethod
    def _w_non_empty(cls, v: List[float]):
        if len(v) == 0:
            raise ValueError("wavelengths must be non-empty")
        return v

    @field_validator('flux')
    @classmethod
    def _f_non_empty(cls, v: List[float]):
        if len(v) == 0:
            raise ValueError("flux must be non-empty")
        return v

    @model_validator(mode='after')
    def _flux_unc_check(self):  # type: ignore
        if self.flux_unc is not None and len(self.flux_unc) != len(self.flux):
            raise ValueError("flux_unc length must match flux length")
        return self

    @field_validator('wavelengths')
    @classmethod
    def _monotonic(cls, v: List[float]):
        if any(v[i] >= v[i+1] for i in range(len(v)-1)):
            raise ValueError("wavelengths must be strictly increasing")
        return v

class LightCurve(BaseModel):
    time: List[float] = Field(..., description="Time values (e.g., JD) sorted ascending")
    flux: List[float] = Field(..., description="Flux or relative intensity values")
    flux_unc: Optional[List[float]] = Field(None, description="Optional uncertainties aligned with flux")
    band: Optional[str] = Field(None, description="Photometric band or passband code")

    @field_validator('time')
    @classmethod
    def _t_non_empty(cls, v: List[float]):
        if len(v) == 0:
            raise ValueError("time must be non-empty")
        return v

    @field_validator('flux')
    @classmethod
    def _lc_flux_non_empty(cls, v: List[float]):
        if len(v) == 0:
            raise ValueError("flux must be non-empty")
        return v

    @field_validator('time')
    @classmethod
    def _ascending(cls, v: List[float]):
        if any(v[i] >= v[i+1] for i in range(len(v)-1)):
            raise ValueError("time must be strictly increasing")
        return v

    @model_validator(mode='after')
    def _lc_flux_unc_check(self):  # type: ignore
        if self.flux_unc is not None and len(self.flux_unc) != len(self.flux):
            raise ValueError("flux_unc length must match flux length")
        return self

class SourceObject(BaseModel):
    source_id: str = Field(..., description="Unique canonical identifier for the object")
    ra: Optional[float] = Field(None, ge=0.0, le=360.0, description="Right Ascension degrees")
    dec: Optional[float] = Field(None, ge=-90.0, le=90.0, description="Declination degrees")
    spectra: Optional[List[Spectrum]] = Field(None, description="List of associated spectra")
    lightcurves: Optional[List[LightCurve]] = Field(None, description="List of associated light curves")
    meta: dict = Field(default_factory=dict, description="Arbitrary supplemental metadata")

    @model_validator(mode='after')
    def _list_non_empty(self):  # type: ignore
        if self.spectra is not None and len(self.spectra) == 0:
            raise ValueError("spectra list cannot be empty if provided")
        if self.lightcurves is not None and len(self.lightcurves) == 0:
            raise ValueError("lightcurves list cannot be empty if provided")
        return self

__all__ = [
    'Spectrum', 'LightCurve', 'SourceObject'
]
