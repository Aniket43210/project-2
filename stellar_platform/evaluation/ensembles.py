"""Simple ensembling utilities for combining classifier probabilities.

Supports probability averaging and logit-space averaging with optional
weights. Keeps dependencies to numpy only.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np


def _to_logits(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p)


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def average_probs(
    probs_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Weighted average of probability distributions with renormalization."""
    if not probs_list:
        raise ValueError("probs_list cannot be empty")
    shapes = {tuple(p.shape) for p in probs_list}
    if len(shapes) != 1:
        raise ValueError("All probability arrays must have the same shape")
    stacked = np.stack(probs_list, axis=0)  # (M, N, C)
    if weights is None:
        w = np.ones((stacked.shape[0],), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != stacked.shape[0]:
            raise ValueError("weights length must match number of models")
    w = w / (w.sum() if w.sum() != 0 else 1.0)
    weighted = (w[:, None, None] * stacked).sum(axis=0)
    # Renormalize per row
    s = weighted.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return weighted / s


def logit_average(
    probs_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Average in logit space then softmax back to probabilities."""
    if not probs_list:
        raise ValueError("probs_list cannot be empty")
    shapes = {tuple(p.shape) for p in probs_list}
    if len(shapes) != 1:
        raise ValueError("All probability arrays must have the same shape")
    logits = np.stack([_to_logits(p) for p in probs_list], axis=0)  # (M, N, C)
    if weights is None:
        w = np.ones((logits.shape[0],), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != logits.shape[0]:
            raise ValueError("weights length must match number of models")
    w = w / (w.sum() if w.sum() != 0 else 1.0)
    avg_logits = (w[:, None, None] * logits).sum(axis=0)
    return _softmax(avg_logits, axis=1)


__all__ = [
    "average_probs",
    "logit_average",
]
