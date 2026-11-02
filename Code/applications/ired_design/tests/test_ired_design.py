"""Application-local unit tests; no CROSSFIT result artifacts are touched."""
from __future__ import annotations

import numpy as np
import pandas as pd

from Code.applications.ired_design.lineage import expand, score_round
from Code.applications.ired_design.mutations import add, all_singles, apply, canonical, parse_set
from Code.applications.ired_design.oracles import oracle_lookup


class _Scorer:
    def predict_members(self, sequences, mutation_sets):
        values = np.arange(len(sequences), dtype=float)
        return np.stack((values, values + 1.0))


def test_mutation_round_trip_and_single_count():
    wild_type = "ACD"
    assert canonical("D3A-A1C") == "A1C-D3A"
    assert parse_set("WT") == ()
    assert apply(wild_type, "A1C-D3A") == "CCA"
    assert len(all_singles(wild_type)) == 3 * 19
    assert add("A1C", "A1D") is None


def test_expansion_and_lineage_schema():
    expanded = expand(["A1C", "A1D"], ["C2A"])
    assert expanded["mutation_set_canonical"].tolist() == ["A1C-C2A", "A1D-C2A"]
    scored = score_round(
        expanded, model="test", founder="A1C", round_number=1, wild_type="ACD",
        scorer=_Scorer(), survivors=1, known_fitness={"A1D-C2A": 2.0},
    )
    assert scored["selected_for_next_round"].sum() == 1
    assert scored.iloc[0]["mutation_set_canonical"] == "A1D-C2A"
    assert scored.iloc[0]["parent_variant_id"] == "A1D"
    assert bool(scored.iloc[0]["in_curated_dataset"])


def test_oracle_lookup_converts_zero_based_positions():
    frame = pd.DataFrame({
        "position": [0, 240], "wildtype": ["M", "T"],
        "mutation": ["A", "A"], "ddG_pred": [0.5, -0.25],
    })
    assert oracle_lookup(frame) == {"M1A": 0.5, "T241A": -0.25}
