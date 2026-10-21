from __future__ import annotations

import numpy as np
import pytest

from crossfit.evaluation import (
    MondrianConformal,
    combine_deep_ensemble,
    coverage_by_stratum,
    interval_coverage,
    ndcg,
    score_metrics,
    top_k_precision,
)


def test_direct_score_metrics_and_wild_type_precision_are_exact():
    target = np.array([0.0, 3.0, 1.0, 2.0])
    metrics = score_metrics(target, target, top_k=[2], wild_type_fitness=1.5)
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["mse"] == 0.0
    assert metrics["ndcg"] == pytest.approx(1.0)
    assert metrics["top_2_precision"] == 1.0
    assert metrics["test_hit_rate"] == 0.5
    assert top_k_precision(target, target, k=2, wild_type_fitness=1.5) == 1.0


def test_external_metrics_do_not_improvise_precision_without_wild_type():
    metrics = score_metrics([0.0, 1.0], [0.0, 1.0])
    assert set(metrics) == {"spearman", "pearson", "mse", "ndcg"}
    with pytest.raises(ValueError, match="wild-type"):
        score_metrics([0.0, 1.0], [0.0, 1.0], top_k=[1])


def test_ndcg_uses_full_direct_ranking():
    assert ndcg([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]) == pytest.approx(1.0)
    assert ndcg([2.0, 1.0, 0.0], [0.0, 1.0, 2.0]) < 1.0


def test_deep_ensemble_requires_exactly_five_members():
    values = np.arange(15, dtype=float).reshape(5, 3)
    combined = combine_deep_ensemble(values)
    np.testing.assert_allclose(combined.mean, values.mean(axis=0))
    with pytest.raises(ValueError):
        combine_deep_ensemble(values[:4])


def test_mondrian_conformal_has_nominal_coverage_in_every_stratum():
    alpha = 0.1
    calibration_size = 999
    evaluation_size = 1000
    calibration_prediction = []
    calibration_target = []
    calibration_count = []
    calibration_unseen = []
    evaluation_prediction = []
    evaluation_target = []
    evaluation_count = []
    evaluation_unseen = []
    for count in (1, 2):
        for unseen in (False, True):
            scale = count + int(unseen)
            calibration_prediction.extend(np.zeros(calibration_size))
            calibration_target.extend(np.linspace(0.0, scale, calibration_size))
            calibration_count.extend([count] * calibration_size)
            calibration_unseen.extend([unseen] * calibration_size)
            evaluation_prediction.extend(np.zeros(evaluation_size))
            evaluation_target.extend(np.linspace(0.0, scale, evaluation_size))
            evaluation_count.extend([count] * evaluation_size)
            evaluation_unseen.extend([unseen] * evaluation_size)

    calibrator = MondrianConformal(alpha).fit(
        calibration_prediction, calibration_target, calibration_count, calibration_unseen
    )
    interval = calibrator.predict_interval(
        evaluation_prediction, evaluation_count, evaluation_unseen
    )
    coverage = coverage_by_stratum(
        interval, evaluation_target, evaluation_count, evaluation_unseen
    )
    assert all(value == pytest.approx(0.9, abs=0.003) for value in coverage.values())
    assert interval_coverage(interval, evaluation_target) == pytest.approx(0.9, abs=0.003)
