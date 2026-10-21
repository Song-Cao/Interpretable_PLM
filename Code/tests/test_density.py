from __future__ import annotations

import numpy as np

from crossfit.density import DPLLDensityFeature, changed_positions


def test_dpll_uses_variant_context_and_reuses_wild_type_position_terms():
    calls = []

    def masked(sequence, position, residue):
        calls.append((sequence, position, residue))
        return -0.1 * (position + 1) + (0.5 if sequence[position] == residue else -9.0)

    encoder = DPLLDensityFeature("ACDE", masked)
    first = encoder.score("AWDE")
    second = encoder.score("AYDE")
    assert first == 0.0
    assert second == 0.0
    assert calls.count(("ACDE", 1, "C")) == 1
    assert ("AWDE", 1, "W") in calls
    assert ("AYDE", 1, "Y") in calls
    assert encoder.score("AWDE") == first
    assert len(calls) == 3


def test_dpll_sums_only_mutated_positions():
    def masked(sequence, position, residue):
        return float(ord(residue) + 10 * position)

    encoder = DPLLDensityFeature("ACDE", masked)
    observed = encoder.transform(["ACDE", "AWDY"])
    expected = (ord("W") - ord("C")) + (ord("Y") - ord("E"))
    np.testing.assert_allclose(observed, [0.0, expected])
    assert changed_positions("ACDE", "AWDY") == (1, 3)
