from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crossfit.search import (
    boundary_hyperparameters,
    load_search_state,
    sample_configurations,
    training_pool_folds,
    write_search_state,
)


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


def test_search_state_round_trip_contains_only_completed_unique_folds(tmp_path):
    path = tmp_path / "arm_partial.csv"
    frame = pd.DataFrame([
        {"arm": "arm", "configuration": 0, "fold": 0,
         "validation_spearman": 0.2, "best_epoch": 3},
        {"arm": "arm", "configuration": 0, "fold": 1,
         "validation_spearman": 0.3, "best_epoch": 4},
    ])
    write_search_state(frame, path)
    assert load_search_state(path, "arm", 2, 5) == frame.to_dict("records")
    assert not path.with_suffix(".csv.tmp").exists()


def test_search_state_rejects_duplicate_or_mismatched_rows(tmp_path):
    path = tmp_path / "bad.csv"
    duplicate = pd.DataFrame([
        {"arm": "arm", "configuration": 0, "fold": 0,
         "validation_spearman": 0.2, "best_epoch": 3},
        {"arm": "arm", "configuration": 0, "fold": 0,
         "validation_spearman": 0.3, "best_epoch": 4},
    ])
    write_search_state(duplicate, path)
    with pytest.raises(ValueError, match="duplicate"):
        load_search_state(path, "arm", 2, 5)

    write_search_state(duplicate.iloc[:1].assign(arm="other"), path)
    with pytest.raises(ValueError, match="another arm"):
        load_search_state(path, "arm", 2, 5)
