from __future__ import annotations

import numpy as np
import pytest

from crossfit.evaluation import (
    MondrianConformal,
    aggregate_pairwise_scores,
    coverage_by_stratum,
)


def test_rank_aggregation_recovers_known_scores_with_wild_type_anchor():
    scores = np.array([0.0, 1.5, -0.5, 2.0])
    u = np.array([0, 0, 1, 1, 2, 2])
    v = np.array([1, 2, 2, 3, 3, 0])
    result = aggregate_pairwise_scores(
        len(scores), u, v, scores[u] - scores[v], wild_type_index=0
    )
    np.testing.assert_allclose(result.scores, scores, atol=1e-9)
    assert result.scores[0] == 0.0
    assert result.residual_norm < 1e-9


def test_rank_aggregation_rejects_disconnected_comparison_graph():
    with pytest.raises(ValueError, match="connected"):
        aggregate_pairwise_scores(4, [0, 2], [1, 3], [1.0, 1.0], wild_type_index=0)


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
        calibration_prediction,
        calibration_target,
        calibration_count,
        calibration_unseen,
    )
    interval = calibrator.predict_interval(
        evaluation_prediction, evaluation_count, evaluation_unseen
    )
    coverage = coverage_by_stratum(
        interval, evaluation_target, evaluation_count, evaluation_unseen
    )
    assert set(coverage) == {(1, False), (1, True), (2, False), (2, True)}
    assert all(value == pytest.approx(0.9, abs=0.003) for value in coverage.values())


def test_mondrian_conformal_rejects_uncalibrated_stratum():
    calibrator = MondrianConformal(0.1).fit([0.0] * 10, [0.0] * 10, [1] * 10, [False] * 10)
    with pytest.raises(KeyError, match="no calibration"):
        calibrator.predict_interval([0.0], [2], [False])
