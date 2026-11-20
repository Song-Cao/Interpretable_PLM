import numpy as np

from crossfit.feature_atlas import (
    activation_weighted_profiles,
    empirical_reference_qvalues,
    feature_seed_replication,
    paired_label_permutation,
    select_profile_subtypes,
    subtype_effect_test,
)


def test_activation_weighted_profiles_are_fractions_for_binary_tracks():
    codes = np.asarray([[3.0, 0.0], [1.0, 1.0], [0.0, 3.0]])
    names, values = activation_weighted_profiles(
        codes, {"left": np.asarray([1, 1, 0]), "right": np.asarray([0, 0, 1])}
    )
    assert names == ["left", "right"]
    assert np.allclose(values[:, 0] + values[:, 1], 1.0)
    assert values[0, 0] > values[1, 0]


def test_empirical_reference_qvalues_flag_extreme_profile():
    profiles = np.asarray([[0.0], [0.1], [-0.1], [0.05], [8.0]])
    labels = np.asarray(["conserved"] * 4 + ["rotated"])
    pvalues, qvalues = empirical_reference_qvalues(
        profiles, labels, reference_class="conserved"
    )
    assert pvalues.shape == profiles.shape
    assert pvalues[-1, 0] <= pvalues[0, 0]
    assert qvalues[-1, 0] <= 1.0


def test_feature_seed_replication_matches_permuted_profiles():
    first = np.asarray([[3.0, 0.0, 1.0], [0.0, 3.0, 1.0]])
    second = first[:, [1, 2, 0]]
    labels_first = np.asarray(["rotated", "conserved", "dead"])
    labels_second = np.asarray(["conserved", "dead", "rotated"])
    replication, rows = feature_seed_replication(
        [first, second], [labels_first, labels_second],
        live_masks=[labels_first != "dead", labels_second != "dead"], minimum_cosine=0.9,
    )
    assert np.allclose(replication, 1.0)
    assert all(row["class_match"] for row in rows)


def test_subtype_selection_and_effect_test_detect_separated_groups():
    rng = np.random.default_rng(2)
    profiles = np.r_[rng.normal(-3, 0.1, (12, 3)), rng.normal(3, 0.1, (12, 3))]
    labels, rows, selected = select_profile_subtypes(
        profiles, candidates=[2, 3], seed=8, n_init=10
    )
    assert selected == 2
    assert len(rows) == 2
    effects = np.where(labels == labels[0], 0.0, 2.0)
    result = subtype_effect_test(effects, labels, permutations=999, seed=4)
    assert result["permutation_p_value"] <= 0.01


def test_paired_label_permutation_reports_direction_and_interval():
    inside = np.linspace(2.0, 3.0, 20)
    outside = np.linspace(0.0, 1.0, 20)
    result = paired_label_permutation(
        inside, outside, permutations=999, seed=5,
        bootstrap_samples=500, bootstrap_seed=6,
    )
    assert result["paired_mean_difference"] == 2.0
    assert result["bootstrap_ci_lower"] > 0.0
    assert result["permutation_p_value"] <= 0.01
