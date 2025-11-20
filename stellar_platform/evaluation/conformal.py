"""Conformal prediction utilities for classification (split conformal sets).

Implements simple split conformal prediction sets for multiclass problems
using nonconformity scores s = 1 - p_true on a calibration set.

APIs:
- compute_conformal_threshold(y_cal, probs_cal, alpha): float q_hat
- conformal_prediction_sets(probs, q): list of index arrays per row
- empirical_coverage(y_true, probs, q): float coverage

Dependencies: numpy only.
"""
from __future__ import annotations

import numpy as np
from typing import List, Sequence, Tuple


def compute_conformal_threshold(
    y_cal: np.ndarray,
    probs_cal: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Compute split conformal threshold q_hat from calibration set.

    Uses nonconformity scores s_i = 1 - p_i[y_i]. Finite-sample valid threshold
    is the k-th order statistic with k = ceil((n+1)*(1-alpha)).
    """
    y_cal = np.asarray(y_cal)
    probs_cal = np.asarray(probs_cal)
    if probs_cal.ndim != 2:
        raise ValueError("probs_cal must be 2D (N, C)")
    if y_cal.shape[0] != probs_cal.shape[0]:
        raise ValueError("y_cal and probs_cal must have matching first dimension")
    n = probs_cal.shape[0]
    if n == 0:
        raise ValueError("Calibration set is empty")
    scores = 1.0 - probs_cal[np.arange(n), y_cal]
    # Finite-sample rank
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = np.clip(k, 1, n)
    sorted_scores = np.sort(scores)
    q_hat = float(sorted_scores[k - 1])
    return q_hat


def conformal_prediction_sets(probs: np.ndarray, q: float) -> List[np.ndarray]:
    """Produce prediction sets for each row given threshold q.

    A label c is included if 1 - p_c <= q  <=> p_c >= 1 - q.
    """
    probs = np.asarray(probs)
    if probs.ndim != 2:
        raise ValueError("probs must be 2D (N, C)")
    cutoff = 1.0 - float(q)
    sets: List[np.ndarray] = []
    for row in probs:
        idx = np.flatnonzero(row >= cutoff)
        # Safety: ensure non-empty set (include argmax if empty)
        if idx.size == 0:
            idx = np.array([int(np.argmax(row))])
        sets.append(idx)
    return sets


def empirical_coverage(
    y_true: np.ndarray,
    probs: np.ndarray,
    q: float,
) -> float:
    """Compute empirical coverage of conformal sets on a dataset."""
    y_true = np.asarray(y_true)
    sets = conformal_prediction_sets(probs, q)
    covered = 0
    for yi, S in zip(y_true, sets):
        if int(yi) in S:
            covered += 1
    return float(covered / len(sets))


__all__ = [
    "compute_conformal_threshold",
    "conformal_prediction_sets",
    "empirical_coverage",
]
