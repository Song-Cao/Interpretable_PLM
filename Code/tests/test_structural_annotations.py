from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crossfit.datasets import wt_sequence
from crossfit.structural_annotations import (
    align_campaign_sequence,
    assign_functional_shells,
    biological_assembly_polymer_chains,
    canonical_polymer_sequence,
    mutation_responsive_codes,
    validate_residue_tracks,
)
from crossfit.utils import REPO_ROOT, load_config


def test_ired_numbering_maps_exactly_to_5ocm() -> None:
    config = load_config(REPO_ROOT / "Code/config/step8.yaml")
    path = REPO_ROOT / config["structure"]["deposited_mmcif"]
    campaign = wt_sequence("IRED")
    structure = canonical_polymer_sequence(path)
    alignment = align_campaign_sequence(campaign, structure)
    assert len(campaign) == 290
    assert len(structure) == 291
    assert alignment.identity == 1.0
    assert alignment.campaign_coverage == 1.0
    assert not alignment.mismatches
    assert not alignment.campaign_gaps
    assert alignment.mapping == {position: position for position in range(1, 291)}
    assert structure[-1] == "A"


def test_ired_supplied_tracks_match_both_sequences() -> None:
    config = load_config(REPO_ROOT / "Code/config/step8.yaml")
    campaign = wt_sequence("IRED")
    path = REPO_ROOT / config["structure"]["deposited_mmcif"]
    structure = canonical_polymer_sequence(path)
    tracks = validate_residue_tracks(campaign, config["annotation_tracks"])
    assert len(tracks) == 13
    assert all(
        structure[int(row.campaign_position) - 1] == row.campaign_residue
        for row in tracks.itertuples(index=False)
    )
    assert campaign[177] == "W"
    assert campaign[243] == "H"
    assert campaign[207] == "W"
    assert campaign[237] == "H"


def test_5ocm_assembly_metadata_is_three_biological_dimers() -> None:
    config = load_config(REPO_ROOT / "Code/config/step8.yaml")
    path = REPO_ROOT / config["structure"]["deposited_mmcif"]
    assert biological_assembly_polymer_chains(path) == {
        "1": ("A", "F"), "2": ("B", "C"), "3": ("D", "E")
    }


def test_functional_shell_precedence() -> None:
    distances = pd.DataFrame({
        "campaign_position": [1, 2, 3, 4, 5],
        "minimum_catalytic_distance_angstrom": [0.0, 4.9, 7.9, 9.0, 9.0],
    })
    result = assign_functional_shells(
        "ACDEF", distances, [1], [1, 2, 4],
        first_shell_cutoff=5.0, second_shell_cutoff=8.0,
    )
    assert result["shell"].tolist() == [
        "catalytic", "first_shell", "second_shell",
        "distal_functional_network", "generic_distal",
    ]


def test_mutation_responsive_codes_are_mutant_minus_wild_type() -> None:
    wild = np.array([[1.0, 3.0], [2.0, 4.0]])
    mutant = np.array([[1.5, 2.0], [5.0, 4.0]])
    np.testing.assert_allclose(
        mutation_responsive_codes(mutant, wild),
        np.array([[0.5, -1.0], [3.0, 0.0]]),
    )
    with pytest.raises(ValueError):
        mutation_responsive_codes(mutant[0], wild)
