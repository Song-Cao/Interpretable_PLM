from __future__ import annotations

import numpy as np

from crossfit.gate1 import bootstrap_drift_gap, linear_cka_by_layer, tokenwise_relative_drift


def test_gate1_drift_and_cka_preserve_layer_axis():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(4, 3, 5, 6)).astype(np.float32)
    adapted = base.copy()
    adapted[:, 1] += 0.2
    drift = tokenwise_relative_drift(base, adapted)
    assert drift.shape == (4, 3)
    assert np.all(drift[:, 0] == 0.0)
    np.testing.assert_allclose(linear_cka_by_layer(base, base), 1.0, atol=1e-6)


def test_bootstrap_gap_detects_uniformly_larger_real_drift():
    real = np.ones((20, 4))
    shuffled = np.zeros((20, 4))
    lower, upper = bootstrap_drift_gap(real, shuffled, samples=50, seed=2)
    np.testing.assert_allclose(lower, 1.0)
    np.testing.assert_allclose(upper, 1.0)
