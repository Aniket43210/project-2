import numpy as np
from stellar_platform.evaluation.metrics import classification_metrics, regression_metrics, calibration_metrics

def test_classification_metrics_basic():
    y_true = np.array([0,1,0,1])
    y_probs = np.array([[0.8,0.2],[0.1,0.9],[0.7,0.3],[0.2,0.8]])
    report = classification_metrics(y_true, y_probs)
    assert 0.0 <= report.macro_f1 <= 1.0
    assert '0' in report.per_class


def test_regression_metrics_basic():
    y_true = np.array([1.0,2.0,3.0])
    y_pred = np.array([1.1,1.9,3.2])
    metrics = regression_metrics(y_true, y_pred)
    assert 'mse' in metrics and metrics['mse'] >= 0


def test_calibration_metrics_basic():
    y_true = np.array([0,1,0,1,1])
    y_probs = np.array([[0.8,0.2],[0.2,0.8],[0.6,0.4],[0.3,0.7],[0.1,0.9]])
    cal = calibration_metrics(y_true, y_probs, focus_class=1)
    assert 'brier' in cal
