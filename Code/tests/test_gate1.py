from __future__ import annotations

import numpy as np

from crossfit.gate1 import (
    activation_norms_by_layer,
    bootstrap_drift_gap,
    linear_cka_by_layer,
    paired_task_gain_test,
    select_analysis_layer,
    tokenwise_relative_drift,
)


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


def test_activation_norm_ratio_uses_mean_token_norm_per_layer():
    base = np.ones((2, 3, 4, 2), dtype=np.float32)
    adapted = base * np.asarray([1.0, 2.0, 0.5], dtype=np.float32)[None, :, None, None]
    base_norm, adapted_norm, ratio = activation_norms_by_layer(base, adapted)
    np.testing.assert_allclose(adapted_norm / base_norm, ratio)
    np.testing.assert_allclose(ratio, [1.0, 2.0, 0.5])


def test_paired_task_gain_test_reports_fold_level_inference():
    result = paired_task_gain_test(np.asarray([0.016562, 0.040247, -0.029163, 0.080856, -0.013211]))
    assert result["degrees_of_freedom"] == 4
    np.testing.assert_allclose(result["t_statistic"], 0.97505, rtol=1e-4)
    np.testing.assert_allclose(result["p_value_two_sided"], 0.38475, rtol=1e-4)


def test_layer_selection_uses_cka_midnetwork_and_earliest_near_maximum():
    gap = np.asarray([9.0, 0.1, 0.2, 0.91, 1.0, 1.2, 8.0])
    cka = np.asarray([0.9, 0.9, 0.9, 0.8, 0.8, 0.4, 0.9])
    selected, eligible, near_max, threshold = select_analysis_layer(
        gap, cka, cka_floor=0.5, exclude_first=2, exclude_last=1,
        near_max_fraction=0.9,
    )
    assert selected == 3
    assert eligible.tolist() == [False, False, True, True, True, False, False]
    assert near_max.tolist() == [False, False, False, True, True, False, False]
    np.testing.assert_allclose(threshold, 0.9)
