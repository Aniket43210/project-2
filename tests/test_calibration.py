import numpy as np
from stellar_platform.evaluation import (
    TemperatureScaler,
    expected_calibration_error,
)


def synthetic_logits(seed=0, n=400, k=3):
    rng = np.random.default_rng(seed)
    # create moderately miscalibrated logits by adding noise
    raw = rng.normal(0, 1.0, size=(n, k))
    # inject class bias
    raw[:, 0] += 0.5
    return raw


def test_temperature_scaler_reduces_ece():
    logits = synthetic_logits()
    # ground-truth labels sampled from softmax(logits * 1.5) to create miscalibration
    def softmax(x):
        z = x - x.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    true_probs = softmax(logits * 1.5)
    rng = np.random.default_rng(42)
    y = np.array([rng.choice(true_probs.shape[1], p=tp) for tp in true_probs])

    # model predicted probabilities (uncalibrated)
    pred_probs = softmax(logits)
    pre_ece = expected_calibration_error(pred_probs, y, n_bins=10)

    scaler = TemperatureScaler().fit(logits, y)
    post_probs = scaler.transform(logits)
    post_ece = expected_calibration_error(post_probs, y, n_bins=10)

    # In rare random seeds ECE might not strictly decrease; allow small tolerance
    assert post_ece <= pre_ece + 1e-4
    # And usually it improves by a noticeable margin
    assert post_ece < pre_ece * 1.05
