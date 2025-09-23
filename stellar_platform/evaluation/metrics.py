"""Evaluation metric utilities for the stellar platform.

Provides:
- Classification: macro/micro F1, precision, recall, MCC, PR-AUC per class
- Regression: MAE, MSE, RMSE, R2
- Calibration: Brier score, reliability curve
- Confusion matrix helper

Only uses scikit-learn + numpy to keep dependencies minimal.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from sklearn.metrics import (
    precision_recall_fscore_support,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    brier_score_loss,
)
import math


@dataclass
class ClassificationReport:
    macro_f1: float
    macro_precision: float
    macro_recall: float
    mcc: float
    per_class: Dict[str, Dict[str, float]]
    roc_auc_macro: Optional[float]
    pr_auc_macro: Optional[float]


def classification_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> ClassificationReport:
    """Compute classification metrics given probabilities.

    Args:
        y_true: shape (N,) integer encoded labels
        y_pred_probs: shape (N, C) probabilities
        class_names: optional class name list length C
    """
    if class_names is None:
        class_names = [str(i) for i in range(y_pred_probs.shape[1])]
    y_pred = np.argmax(y_pred_probs, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = 0.0

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }

    # AUC metrics (one-vs-rest) if > 1 class and probs valid
    roc_auc_macro = None
    pr_auc_macro = None
    if y_pred_probs.shape[1] > 1 and len(np.unique(y_true)) > 1:
        try:
            roc_auc_macro = float(roc_auc_score(y_true, y_pred_probs, multi_class='ovr'))
        except Exception:
            roc_auc_macro = None
        # PR AUC macro average manually
        pr_scores = []
        for c in range(y_pred_probs.shape[1]):
            y_true_bin = (y_true == c).astype(int)
            try:
                pr_scores.append(average_precision_score(y_true_bin, y_pred_probs[:, c]))
            except Exception:
                pass
        if pr_scores:
            pr_auc_macro = float(np.mean(pr_scores))

    return ClassificationReport(
        macro_f1=macro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        mcc=mcc,
        per_class=per_class,
        roc_auc_macro=roc_auc_macro,
        pr_auc_macro=pr_auc_macro,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'mse': float(mse),
        'rmse': rmse,
        'mae': float(mae),
        'r2': float(r2)
    }


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    method: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute reliability (calibration) curve for binary probabilities.

    Args:
        y_true: Binary ground truth (0/1)
        y_prob: Predicted probabilities for the positive class
        n_bins: Number of bins
        method: 'uniform' -> fixed-width bins; 'quantile' -> equal-frequency bins

    Returns:
        bin_centers, mean_predicted, fraction_positives, counts
    """
    y_prob = np.asarray(y_prob)
    if method not in {"uniform", "quantile"}:
        raise ValueError("method must be 'uniform' or 'quantile'")
    if method == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:  # quantile
        # Use unique quantiles to avoid duplicate edges causing empty bins
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.unique(np.quantile(y_prob, quantiles))
        # Ensure last edge is exactly 1.0
        if bins[-1] < 1.0:
            bins[-1] = 1.0
        # If not enough unique bins (all probs similar), fallback to uniform
        if len(bins) <= 2:
            bins = np.linspace(0.0, 1.0, min(n_bins, 3) + 1)
    # Digitize (rightmost edge inclusive for last bin)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    n_effective_bins = len(bins) - 1
    mean_pred = []
    frac_pos = []
    counts = []
    centers = []
    for b in range(n_effective_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue  # skip empty bin entirely
        counts.append(int(mask.sum()))
        mean_pred.append(float(np.mean(y_prob[mask])))
        frac_pos.append(float(np.mean(y_true[mask])))
        centers.append(0.5 * (bins[b] + bins[b + 1]))
    return np.array(centers), np.array(mean_pred), np.array(frac_pos), np.array(counts)


def calibration_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    focus_class: int = 0,
    n_bins: int = 10,
    method: str = "quantile",
) -> Dict[str, Any]:
    """Calibration metrics for a focus class.

    Returns dict with brier score, reliability curve (without empty bins) and counts.
    """
    probs = y_pred_probs[:, focus_class]
    y_bin = (y_true == focus_class).astype(int)
    brier = brier_score_loss(y_bin, probs)
    centers, mean_pred, frac_pos, counts = reliability_curve(y_bin, probs, n_bins=n_bins, method=method)
    return {
        'brier': float(brier),
        'reliability_centers': centers.tolist(),
        'reliability_mean_pred': mean_pred.tolist(),
        'reliability_fraction_pos': frac_pos.tolist(),
        'reliability_counts': counts.tolist(),
        'binning_method': method,
    }


def confusion_matrix_dict(y_true: np.ndarray, y_pred_probs: np.ndarray, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    y_pred = np.argmax(y_pred_probs, axis=1)
    if class_names is None:
        class_names = [str(i) for i in range(y_pred_probs.shape[1])]
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    return {
        'matrix': cm.tolist(),
        'labels': class_names
    }


__all__ = [
    'classification_metrics',
    'regression_metrics',
    'calibration_metrics',
    'reliability_curve',
    'confusion_matrix_dict',
    'ClassificationReport',
]


def per_class_calibration(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    n_bins: int = 10,
    method: str = "quantile",
) -> Dict[str, Any]:
    """Compute calibration curves for every class.

    Returns dict keyed by class index with reliability data.
    """
    n_classes = y_pred_probs.shape[1]
    out: Dict[str, Any] = {}
    for c in range(n_classes):
        cal = calibration_metrics(y_true, y_pred_probs, focus_class=c, n_bins=n_bins, method=method)
        out[str(c)] = cal
    return out


def per_class_ece(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    n_bins: int = 10,
    method: str = "quantile",
) -> Dict[str, float]:
    """Compute per-class expected calibration error.

    ECE per class over chosen binning; weighted by bin counts.
    """
    n_classes = y_pred_probs.shape[1]
    results: Dict[str, float] = {}
    for c in range(n_classes):
        probs = y_pred_probs[:, c]
        y_bin = (y_true == c).astype(int)
        centers, mean_pred, frac_pos, counts = reliability_curve(y_bin, probs, n_bins=n_bins, method=method)
        total = counts.sum()
        if total == 0:
            results[str(c)] = 0.0
            continue
        ece = 0.0
        for mp, fp, ct in zip(mean_pred, frac_pos, counts):
            ece += (ct / total) * abs(mp - fp)
        results[str(c)] = float(ece)
    return results


__all__.extend(['per_class_calibration', 'per_class_ece'])
