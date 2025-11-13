import numpy as np
from stellar_platform.models.multitask import MultiTaskHead

def test_multitask_basic_fit_predict():
    rng = np.random.default_rng(0)
    n, f, k = 200, 6, 3
    X = rng.normal(size=(n, f))
    # class label determined by linear rule on two features
    scores = np.stack([
        0.8*X[:,0] - 0.2*X[:,1],
        -0.3*X[:,0] + 0.7*X[:,1],
        0.1*X[:,0] + 0.1*X[:,1],
    ], axis=1)
    y_class = scores.argmax(axis=1)
    # regression target correlates with feature 2
    y_reg = 2.0*X[:,2] + 0.1*rng.normal(size=n)

    head = MultiTaskHead(n_classes=k)
    head.fit(X, y_class, y_reg)

    p = head.predict_proba(X[:5])
    yhat = head.predict_regression(X[:5])

    assert p.shape == (5, k)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert yhat.shape == (5,)
