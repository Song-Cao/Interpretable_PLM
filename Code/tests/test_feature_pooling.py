from __future__ import annotations

import numpy as np

from crossfit.feature_pooling import pool_codes, select_feature_set


def test_pool_codes_keeps_feature_axis_and_supports_registered_operators():
    codes = np.array([[[0, 2], [3, 0]], [[1, 0], [2, 4]]], dtype=np.float32)
    assert np.array_equal(pool_codes(codes, "sum"), [[3, 2], [3, 4]])
    assert np.array_equal(pool_codes(codes, "max"), [[3, 2], [2, 4]])
    assert np.array_equal(pool_codes(codes, "binarized"), [[1, 1], [1, 1]])


def test_feature_set_masks_are_distinct_and_dead_never_enters_all_live():
    labels = np.array(["dead", "emergent_candidate", "evolution_only", "rotated", "conserved"])
    config = {
        "objective_shifted_classes": ["emergent_candidate", "evolution_only"],
        "shifted_plus_conserved_classes": ["emergent_candidate", "evolution_only", "conserved"],
    }
    assert select_feature_set(labels, "all_live", config).tolist() == [False, True, True, True, True]
    assert select_feature_set(labels, "objective_shifted", config).tolist() == [False, True, True, False, False]
    assert select_feature_set(labels, "shifted_plus_conserved", config).tolist() == [False, True, True, False, True]
