import numpy as np
import pandas as pd
import pytest

from crossfit.structural_domains import (
    aggregate_structure_axes,
    assign_structural_domains,
    overlap_counts,
)


def test_assign_structural_domains_requires_complete_nonoverlapping_partition():
    frame = assign_structural_domains(
        5,
        {"left": {"campaign_ranges": [[2, 3]]}, "right": {"campaign_ranges": [[4, 5]]}},
        [1],
    )
    assert frame["structural_domain"].tolist() == ["unassigned", "left", "left", "right", "right"]
    with pytest.raises(ValueError, match="multiple"):
        assign_structural_domains(
            4,
            {"left": {"campaign_ranges": [[1, 3]]}, "right": {"campaign_ranges": [[3, 4]]}},
            [],
        )


def test_aggregate_structure_axes_preserves_missing_coordinates():
    values = pd.DataFrame({
        "campaign_position": [1, 1, 2, 2],
        "relative_sasa": [np.nan, np.nan, 0.1, 0.3],
        "interface_contact": [False, False, True, False],
    })
    result = aggregate_structure_axes(
        values, interface_min_fraction=0.5, core_surface_rsa_cutoff=0.25
    ).set_index("campaign_position")
    assert result.loc[1, "core_surface"] == "unassigned"
    assert result.loc[2, "core_surface"] == "core"
    assert bool(result.loc[2, "dimer_interface"])


def test_overlap_counts_keeps_axis_categories():
    frame = pd.DataFrame({
        "domain": ["a", "a", "b"], "shell": ["near", "far", "near"]
    })
    result = overlap_counts(frame, "domain")
    assert result.loc["a", "near"] == 1
    assert result.loc["b", "far"] == 0
