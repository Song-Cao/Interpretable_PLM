from __future__ import annotations

import numpy as np
import pytest

from crossfit.metrics import (
    gated_test_pairs,
    pair_metrics,
    pair_metrics_by_hamming,
    sample_pair_indices,
    variant_metrics,
)


def test_variant_and_pair_metrics_are_exact_for_perfect_predictions():
    target = np.array([0.0, 3.0, 1.0, 2.0])
    metrics = variant_metrics(target, target, [2])
    assert metrics["variant_spearman"] == pytest.approx(1.0)
    assert metrics["variant_pearson"] == pytest.approx(1.0)
    assert metrics["top_2_precision"] == 1.0
    assert metrics["ndcg_2"] == pytest.approx(1.0)
    u, v = sample_pair_indices(4, 100, 5)
    paired = pair_metrics(target, target, u, v)
    assert paired["pair_spearman"] == pytest.approx(1.0)
    assert paired["pair_rmse"] == 0.0
    assert paired["pair_sign_accuracy"] == 1.0


def test_pair_sampling_is_deterministic_and_respects_budget():
    first = sample_pair_indices(100, 50, 7)
    second = sample_pair_indices(100, 50, 7)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert len(first[0]) == 50
    assert np.all(first[0] < first[1])


def test_test_pair_gate_and_hamming_strata_do_not_filter_distance():
    target = np.array([0.0, 0.1, 1.0, 2.0])
    censor = ["none", "none", "floor", "floor"]
    u, v = gated_test_pairs(target, censor, margin_sd=0.25, max_pairs=100, seed=1)
    assert list(zip(u, v)) == [(0, 2), (0, 3), (1, 2), (1, 3)]
    rows = pair_metrics_by_hamming(
        target,
        target,
        ["WT", "A1C", "A1D:B2C", "A1E:B2D:C3F"],
        u,
        v,
    )
    assert {row["hamming_distance"] for row in rows} == {2, 3}
    assert sum(row["n_pairs"] for row in rows) == len(u)
