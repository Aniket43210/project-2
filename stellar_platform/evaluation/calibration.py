"""Calibration utilities for classification model outputs.

Provides:
- BaseProbCalibrator: interface for all calibrators.
- TemperatureScaler: single-parameter softmax temperature scaling.
- IsotonicCalibrator: per-class (or binary) isotonic regression.
- reliability_curve / expected_calibration_error helpers.

Design goals:
- Minimal dependencies (only numpy / sklearn).
- Safe fallback if sklearn IsotonicRegression is unavailable.
- Works with either logits or already-softmaxed probabilities.
- Supports persistence via to_dict / from_dict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

try:  # optional sklearn
    from sklearn.isotonic import IsotonicRegression  # type: ignore
except Exception:  # pragma: no cover
    IsotonicRegression = None  # type: ignore

ArrayLike = np.ndarray


def _softmax(logits: ArrayLike, axis: int = -1) -> ArrayLike:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _is_probabilities(x: ArrayLike, atol: float = 1e-6) -> bool:
    if x.ndim < 1:
        return False
    s = np.sum(x, axis=-1)
    return np.allclose(s, 1.0, atol=atol) and np.all(x >= -atol) and np.all(x <= 1 + atol)


def reliability_curve(probs: ArrayLike, y_true: ArrayLike, n_bins: int = 15) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Compute reliability curve (confidence bin accuracies).

    Returns (bin_centers, bin_accuracy, bin_confidence, counts)
    counts = number of samples falling into each bin (shape n_bins)
    """
    probs = np.asarray(probs)
    if probs.ndim == 2:  # multi-class -> use max prob
        confidences = probs.max(axis=1)
        pred = probs.argmax(axis=1)
    else:  # binary (shape (N,))
        confidences = probs
        pred = (probs >= 0.5).astype(int)

    y_true = np.asarray(y_true)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bins) - 1
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            bin_acc[b] = np.nan
            bin_conf[b] = np.nan
            continue
        counts[b] = mask.sum()
        bin_acc[b] = (pred[mask] == y_true[mask]).mean()
        bin_conf[b] = confidences[mask].mean()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, bin_acc, bin_conf, counts


def expected_calibration_error(probs: ArrayLike, y_true: ArrayLike, n_bins: int = 15) -> float:
    """Compute the (weighted) Expected Calibration Error.

    Standard definition: sum_b (n_b / N) * |acc_b - conf_b| over non-empty bins.
    """
    _, acc, conf, counts = reliability_curve(probs, y_true, n_bins=n_bins)
    mask = counts > 0
    if counts[mask].sum() == 0:
        return np.nan
    weights = counts[mask] / counts[mask].sum()
    return float(np.sum(weights * np.abs(acc[mask] - conf[mask])))


@dataclass
class BaseProbCalibrator:
    """Abstract base class for probability calibrators."""

    def fit(self, y_prob: ArrayLike, y_true: ArrayLike) -> "BaseProbCalibrator":  # pragma: no cover - interface
        raise NotImplementedError

    def transform(self, y_prob: ArrayLike) -> ArrayLike:  # pragma: no cover - interface
        raise NotImplementedError

    def fit_transform(self, y_prob: ArrayLike, y_true: ArrayLike) -> ArrayLike:
        return self.fit(y_prob, y_true).transform(y_prob)

    # persistence helpers
    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BaseProbCalibrator":
        t = payload.get("type")
        if t == "TemperatureScaler":
            return TemperatureScaler(temperature=payload.get("temperature", 1.0))
        if t == "IsotonicCalibrator":
            return IsotonicCalibrator(payload.get("per_class", False))
        raise ValueError(f"Unknown calibrator type: {t}")


@dataclass
class TemperatureScaler(BaseProbCalibrator):
    temperature: float = 1.0
    t_min: float = 0.05
    t_max: float = 5.0
    grid_points: int = 100

    def fit(self, y_prob_or_logits: ArrayLike, y_true: ArrayLike) -> "TemperatureScaler":
        y_true = np.asarray(y_true)
        logits = y_prob_or_logits
        if _is_probabilities(logits):  # convert to logits (avoid numerical issues with clipping)
            eps = 1e-12
            logits = np.log(np.clip(logits, eps, 1 - eps))
        else:
            logits = np.asarray(logits)
        # Grid search temperatures to minimize NLL (robust & deterministic)
        temps = np.linspace(self.t_min, self.t_max, self.grid_points)
        best_t = self.temperature
        best_nll = np.inf
        # Precompute correct-class logits for efficiency
        for t in temps:
            scaled = logits / t
            p = _softmax(scaled, axis=1)
            # Negative log likelihood
            nll = -np.log(p[np.arange(p.shape[0]), y_true] + 1e-12).mean()
            if nll < best_nll:
                best_nll = nll
                best_t = t
        self.temperature = float(best_t)
        return self

    def transform(self, y_prob_or_logits: ArrayLike) -> ArrayLike:
        x = np.asarray(y_prob_or_logits)
        if _is_probabilities(x):
            # Convert to logits approximately then apply temperature; alternative is raise warning.
            eps = 1e-12
            logits = np.log(np.clip(x, eps, 1 - eps))
        else:
            logits = x
        scaled = logits / self.temperature
        if scaled.ndim == 1:
            return _softmax(scaled[None, :], axis=1)[0]
        return _softmax(scaled, axis=1)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["temperature"] = float(self.temperature)
        return d


@dataclass
class IsotonicCalibrator(BaseProbCalibrator):
    per_class: bool = False
    _iso_models: Optional[Any] = None

    def fit(self, y_prob: ArrayLike, y_true: ArrayLike) -> "IsotonicCalibrator":
        y_prob = np.asarray(y_prob)
        y_true = np.asarray(y_true)
        if not _is_probabilities(y_prob):
            raise ValueError("IsotonicCalibrator requires probabilities (softmaxed). Provide model.predict_proba output.")
        if IsotonicRegression is None:
            raise RuntimeError("sklearn IsotonicRegression not available. Install scikit-learn to use this calibrator.")
        n_classes = y_prob.shape[1]
        if n_classes == 2 and not self.per_class:
            # Calibrate positive class probability only
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(y_prob[:, 1], (y_true == 1).astype(int))
            self._iso_models = iso
            return self
        if self.per_class:
            models = []
            for c in range(n_classes):
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(y_prob[:, c], (y_true == c).astype(int))
                models.append(iso)
            self._iso_models = models
            return self
        # One-vs-rest on argmax predicted class (simpler approach)
        models = []
        pred = y_prob.argmax(axis=1)
        for c in range(n_classes):
            mask = pred == c
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(y_prob[mask, c], (y_true[mask] == c).astype(int))
            models.append(iso)
        self._iso_models = models
        return self

    def transform(self, y_prob: ArrayLike) -> ArrayLike:
        if self._iso_models is None:
            raise RuntimeError("Calibrator not fit.")
        y_prob = np.asarray(y_prob)
        if not _is_probabilities(y_prob):
            raise ValueError("Input must be probabilities.")
        if isinstance(self._iso_models, IsotonicRegression):  # binary
            p1_cal = self._iso_models.transform(y_prob[:, 1])
            p1_cal = np.clip(p1_cal, 1e-6, 1 - 1e-6)
            p0 = 1 - p1_cal
            return np.vstack([p0, p1_cal]).T
        # list of per-class models
        out = []
        for c, iso in enumerate(self._iso_models):
            pc = iso.transform(y_prob[:, c])
            out.append(pc)
        out = np.vstack(out).T
        # renormalize
        s = out.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return out / s


__all__ = [
    "BaseProbCalibrator",
    "TemperatureScaler",
    "IsotonicCalibrator",
    "reliability_curve",
    "expected_calibration_error",
]
