import numpy as np
from stellar_platform.evaluation import average_probs, logit_average


def softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def test_ensemble_shapes_and_normalization():
    rng = np.random.default_rng(0)
    logits1 = rng.normal(0, 1.0, size=(50, 3))
    logits2 = rng.normal(0, 1.0, size=(50, 3))
    p1 = softmax(logits1)
    p2 = softmax(logits2)

    pav = average_probs([p1, p2])
    plog = logit_average([p1, p2])
    assert pav.shape == p1.shape == plog.shape
    np.testing.assert_allclose(pav.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(plog.sum(axis=1), 1.0, atol=1e-6)
