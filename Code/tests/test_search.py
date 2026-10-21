from __future__ import annotations

import numpy as np

from crossfit.search import boundary_hyperparameters, sample_configurations, training_pool_folds


def test_cv_uses_only_training_pool_and_covers_it_five_times():
    pool = np.arange(10, 110)
    forbidden_test = set(range(10))
    folds = training_pool_folds(pool, 5, 17)
    validation_seen = []
    for train, validation in folds:
        assert not set(train) & set(validation)
        assert not (set(train) | set(validation)) & forbidden_test
        assert set(train) | set(validation) == set(pool)
        validation_seen.extend(validation.tolist())
    assert sorted(validation_seen) == pool.tolist()


def test_random_search_is_repeatable_unique_and_includes_registered_bounds():
    space = {
        "penalty": {
            "distribution": "log_uniform",
            "low": 1e-4,
            "high": 1e4,
            "include": [1e-4, 1e3, 1e4],
        },
        "batch": [32, 64, 128],
    }
    first = sample_configurations(space, 30, 9)
    second = sample_configurations(space, 30, 9)
    assert first == second
    assert len({repr(sorted(value.items())) for value in first}) == 30
    assert {1e-4, 1e3, 1e4} <= {value["penalty"] for value in first}


def test_boundary_reporting_catches_both_kinds_of_space():
    space = {
        "learning_rate": [1e-4, 1e-3],
        "penalty": {"distribution": "log_uniform", "low": 1e-4, "high": 1e4},
    }
    assert boundary_hyperparameters(
        {"learning_rate": 1e-3, "penalty": 1e-4}, space
    ) == {"learning_rate": "upper", "penalty": "lower"}
