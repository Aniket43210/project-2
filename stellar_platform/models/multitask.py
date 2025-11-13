"""Lightweight multi-task wrapper (classification + regression) using scikit-learn.

Intended for environments where TensorFlow isn't available during tests.
Provides a simple interface:
- fit(X, y_class, y_reg)
- predict_proba(X) -> class probabilities
- predict_regression(X) -> regression targets

Backed by GradientBoostingClassifier and GradientBoostingRegressor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@dataclass
class MultiTaskConfig:
    classifier_params: Optional[Dict[str, Any]] = None
    regressor_params: Optional[Dict[str, Any]] = None


class MultiTaskHead:
    def __init__(self, n_classes: int, config: Optional[MultiTaskConfig] = None):
        self.n_classes = int(n_classes)
        self.config = config or MultiTaskConfig()
        cls_params = self.config.classifier_params or {}
        reg_params = self.config.regressor_params or {}
        # Simple pipelines with scaling then model
        self._clf = Pipeline([
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingClassifier(random_state=42, **cls_params)),
        ])
        self._reg = Pipeline([
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingRegressor(random_state=42, **reg_params)),
        ])

    def fit(self, X: np.ndarray, y_class: np.ndarray, y_reg: np.ndarray):
        X = np.asarray(X)
        self._clf.fit(X, y_class)
        self._reg.fit(X, y_reg)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        p = self._clf.predict_proba(X)
        # Some sklearn classifiers return list for multiclass; normalize to (N, C)
        if isinstance(p, list):
            # concatenate per-class probabilities
            p = np.column_stack([pi[:, 1] if pi.shape[1] == 2 else pi.max(axis=1) for pi in p])
            # renormalize rows
            s = p.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            p = p / s
        return p

    def predict_regression(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        y = self._reg.predict(X)
        return np.asarray(y)


__all__ = ["MultiTaskHead", "MultiTaskConfig"]
