import numpy as np
import pytest

from crossfit.patching_analysis import (
    clustered_mann_whitney,
    quantile_rows,
    shell_activation_fractions,
    spatial_effect_correlations,
)


def test_quantile_rows_preserve_population_and_sampling_unit():
    rows = quantile_rows(
        [0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 1.0],
        population="candidate", sampling_unit="feature",
    )
    assert [row["value"] for row in rows] == pytest.approx([0.0, 1.5, 3.0])
    assert {row["population"] for row in rows} == {"candidate"}
    assert {row["sampling_unit"] for row in rows} == {"feature"}
    assert {row["n"] for row in rows} == {4}


def test_spatial_effect_correlations_detect_ordered_profile():
    effect = np.arange(12, dtype=float)
    rows = spatial_effect_correlations(
        effect,
        {"ordered": effect.copy(), "reversed": effect[::-1].copy()},
        permutations=999,
        seed=31,
        fdr_q=0.05,
    )
    by_name = {row["shell"]: row for row in rows}
    assert by_name["ordered"]["spearman_rho"] == pytest.approx(1.0)
    assert by_name["reversed"]["spearman_rho"] == pytest.approx(-1.0)
    assert by_name["ordered"]["permutation_p_value"] <= 0.01
    assert by_name["reversed"]["permutation_p_value"] <= 0.01


def test_spatial_effect_correlations_reject_misaligned_profile():
    with pytest.raises(ValueError, match="not aligned"):
        spatial_effect_correlations(
            np.arange(5, dtype=float), {"short": np.arange(4, dtype=float)},
            permutations=10, seed=1, fdr_q=0.05,
        )


def test_shell_activation_fractions_use_activation_mass():
    fractions = shell_activation_fractions(
        {"small": np.array([2.0, 1.0]), "large": np.array([1.0, 1.0])},
        {"small": 1, "large": 3},
    )
    assert fractions["small"] == pytest.approx([0.4, 0.25])
    assert fractions["large"] == pytest.approx([0.6, 0.75])
    assert fractions["small"] + fractions["large"] == pytest.approx([1.0, 1.0])


def test_clustered_mann_whitney_reports_positive_rank_biserial():
    rotated = np.array([[3.0, 4.0, 5.0], [2.0, 3.0, 4.0]])
    control = np.array([0.1, 0.2, 0.3])
    result = clustered_mann_whitney(
        rotated, control, bootstrap_samples=100, seed=8, confidence_level=0.95
    )
    assert result["rank_biserial"] == pytest.approx(1.0)
    assert result["bootstrap_ci_lower"] == pytest.approx(1.0)
    assert result["rotated_feature_variant_observations"] == 6
