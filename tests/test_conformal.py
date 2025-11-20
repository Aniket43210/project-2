import numpy as np
from stellar_platform.evaluation import (
    compute_conformal_threshold,
    conformal_prediction_sets,
    empirical_coverage,
)


def softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def synthetic_probs(n: int, k: int, seed: int = 0, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.0, size=(n, k)) * scale
    return softmax(logits)


def test_conformal_basic_coverage():
    k = 4
    p_cal = synthetic_probs(600, k, seed=1, scale=1.2)
    # sample labels according to true probs for calibration
    rng = np.random.default_rng(10)
    y_cal = np.array([rng.choice(k, p=p) for p in p_cal])
    alpha = 0.1
    q = compute_conformal_threshold(y_cal, p_cal, alpha=alpha)

    p_test = synthetic_probs(400, k, seed=2, scale=1.2)
    y_test = np.array([rng.choice(k, p=p) for p in p_test])
    cov = empirical_coverage(y_test, p_test, q)
    # finite-sample coverage >= 1-alpha approximately; allow small randomness
    assert cov >= 0.85

    sets = conformal_prediction_sets(p_test[:5], q)
    assert len(sets) == 5
    for s in sets:
        assert s.ndim == 1 and s.size >= 1
