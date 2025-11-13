import numpy as np
from stellar_platform.evaluation.domain_shift import domain_shift_report


def make_probs(n: int, k: int, bias: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.0, size=(n, k))
    logits[:, 0] += bias  # bias class 0 for source/target difference
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def test_domain_shift_report_shapes_and_gaps():
    n, k = 300, 3
    # Source: moderately confident for class 0
    p_src = make_probs(n, k, bias=0.6, seed=123)
    y_src = np.argmax(p_src, axis=1)
    # Target: less biased/confident
    p_tgt = make_probs(n, k, bias=0.1, seed=456)
    y_tgt = np.argmax(p_tgt, axis=1)

    rep = domain_shift_report(y_src, p_src, y_tgt, p_tgt, n_bins=8)

    assert 'source' in rep and 'target' in rep and 'gap' in rep
    for key in ('accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'mcc'):
        assert key in rep['source'] and key in rep['target'] and key in rep['gap']
    # basic sanity: counts present and positive
    assert rep['n_source'] == n and rep['n_target'] == n
    # gap values should be finite (ece may be nan in degenerate cases)
    for key in ('accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'mcc'):
        assert np.isfinite(rep['gap'][key])
