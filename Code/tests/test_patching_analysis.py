import numpy as np
import pytest

from crossfit.patching_analysis import quantile_rows, spatial_effect_correlations


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
