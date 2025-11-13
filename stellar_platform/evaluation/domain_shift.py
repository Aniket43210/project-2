"""Domain-shift evaluation utilities.

Provides helpers to compare model behavior between a source domain and a target
domain (e.g., two surveys), focusing on differences in accuracy, macro-F1 and
calibration.

Only depends on numpy and scikit-learn. Designed to work with already-computed
probabilities and labels.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import numpy as np
from .metrics import classification_metrics
from .calibration import expected_calibration_error


def _safe_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    try:
        val = expected_calibration_error(probs, y, n_bins=n_bins)
        return float(val) if np.isfinite(val) else float("nan")
    except Exception:
        return float("nan")


def domain_shift_report(
    y_source: np.ndarray,
    probs_source: np.ndarray,
    y_target: np.ndarray,
    probs_target: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Summarize performance/calibration gaps between two domains.

    Args:
        y_source: Labels for source domain (shape (N_src,))
        probs_source: Probabilities for source domain (shape (N_src, C))
        y_target: Labels for target domain (shape (N_tgt,))
        probs_target: Probabilities for target domain (shape (N_tgt, C))
        n_bins: Number of bins for ECE

    Returns:
        Dict with per-domain metrics and gaps (target - source).
    """
    # Classification metrics
    rep_src = classification_metrics(y_source, probs_source)
    rep_tgt = classification_metrics(y_target, probs_target)
    # Accuracy
    acc_src = float((np.argmax(probs_source, axis=1) == y_source).mean())
    acc_tgt = float((np.argmax(probs_target, axis=1) == y_target).mean())
    # Calibration
    ece_src = _safe_ece(probs_source, y_source, n_bins=n_bins)
    ece_tgt = _safe_ece(probs_target, y_target, n_bins=n_bins)

    report: Dict[str, Any] = {
        "source": {
            "accuracy": acc_src,
            "macro_f1": rep_src.macro_f1,
            "macro_precision": rep_src.macro_precision,
            "macro_recall": rep_src.macro_recall,
            "mcc": rep_src.mcc,
            "ece": ece_src,
        },
        "target": {
            "accuracy": acc_tgt,
            "macro_f1": rep_tgt.macro_f1,
            "macro_precision": rep_tgt.macro_precision,
            "macro_recall": rep_tgt.macro_recall,
            "mcc": rep_tgt.mcc,
            "ece": ece_tgt,
        },
        "gap": {  # target - source
            "accuracy": acc_tgt - acc_src,
            "macro_f1": rep_tgt.macro_f1 - rep_src.macro_f1,
            "macro_precision": rep_tgt.macro_precision - rep_src.macro_precision,
            "macro_recall": rep_tgt.macro_recall - rep_src.macro_recall,
            "mcc": rep_tgt.mcc - rep_src.mcc,
            "ece": (ece_tgt - ece_src) if (np.isfinite(ece_tgt) and np.isfinite(ece_src)) else float("nan"),
        },
        "n_bins": int(n_bins),
        "n_source": int(len(y_source)),
        "n_target": int(len(y_target)),
    }
    return report


__all__ = ["domain_shift_report"]
