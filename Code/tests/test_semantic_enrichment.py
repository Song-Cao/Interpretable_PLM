import numpy as np

from crossfit.semantic_enrichment import (
    benjamini_hochberg_adjusted,
    blocked_class_permutation,
    profile_enrichment,
    top_position_shell_statistic,
)


def test_profile_enrichment_preserves_feature_and_track_axes():
    codes = np.asarray([[4.0, 1.0], [2.0, 1.0], [1.0, 4.0], [1.0, 2.0]])
    names, scores = profile_enrichment(codes, {"left": [1, 1, 0, 0], "right": [0, 0, 1, 1]})
    assert names == ["left", "right"]
    assert scores.shape == (2, 2)
    assert scores[0, 0] > 0 and scores[1, 0] < 0


def test_blocked_permutation_detects_replicated_class_profile():
    labels = np.asarray(["rotated"] * 12 + ["conserved"] * 12)
    scores = np.column_stack((np.r_[np.ones(12) * 4, np.zeros(12)], np.ones(24)))
    observed, pvalues, by_seed = blocked_class_permutation(
        [scores, scores], [labels, labels], target="rotated", comparison="conserved",
        permutations=999, seed=4,
    )
    assert observed[0] > 1.0 and pvalues[0] <= 0.01
    assert by_seed.shape == (2, 2)


def test_bh_adjustment_is_monotone_in_rank():
    adjusted = benjamini_hochberg_adjusted(np.asarray([0.01, 0.04, 0.02, 0.8]))
    assert np.allclose(adjusted[:3], [0.04, 0.05333333333333334, 0.04])
    assert adjusted[3] == 0.8


def test_top_position_shell_test_uses_dictionary_seed_blocks():
    labels = np.asarray(["rotated"] * 10 + ["conserved"] * 10)
    codes = np.zeros((4, 20))
    codes[0, :10] = 2.0
    codes[3, 10:] = 2.0
    statistic, pvalue, rows = top_position_shell_statistic(
        [codes, codes], [labels, labels], np.asarray(["cat", "first", "second", "distal"]),
        target="rotated", comparison="conserved", permutations=999, seed=8,
    )
    assert statistic == 1.0 and pvalue <= 0.01
    assert sum(row["count"] for row in rows) == 40
