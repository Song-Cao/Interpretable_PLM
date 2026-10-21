"""ESM-C pseudo-log-likelihood density feature for augmented baselines."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .representations import AMINO_ACIDS


MaskedLogProbability = Callable[[str, int, str], float]


def changed_positions(wild_type: str, variant: str) -> tuple[int, ...]:
    """Return zero-based substituted positions after validating aligned sequences."""
    if len(variant) != len(wild_type):
        raise ValueError("variant and wild type must have identical length")
    invalid = (set(wild_type) | set(variant)) - set(AMINO_ACIDS)
    if invalid:
        raise ValueError(f"non-canonical residues: {sorted(invalid)}")
    return tuple(index for index, (left, right) in enumerate(zip(wild_type, variant)) if left != right)


class DPLLDensityFeature:
    """Cache mutated-position dPLL with reusable wild-type terms by position."""

    def __init__(self, wild_type: str, masked_log_probability: MaskedLogProbability) -> None:
        if not wild_type or set(wild_type) - set(AMINO_ACIDS):
            raise ValueError("wild type must contain only canonical amino acids")
        self.wild_type = wild_type
        self.masked_log_probability = masked_log_probability
        self.wild_type_terms: dict[int, float] = {}
        self.variant_values: dict[str, float] = {wild_type: 0.0}

    def _wild_type_term(self, position: int) -> float:
        if position not in self.wild_type_terms:
            value = self.masked_log_probability(
                self.wild_type, position, self.wild_type[position]
            )
            if not np.isfinite(value):
                raise ValueError("masked log-probability must be finite")
            self.wild_type_terms[position] = float(value)
        return self.wild_type_terms[position]

    def score(self, variant: str) -> float:
        """Compute one scalar using variant context only at substituted positions."""
        if variant in self.variant_values:
            return self.variant_values[variant]
        total = 0.0
        for position in changed_positions(self.wild_type, variant):
            variant_term = self.masked_log_probability(variant, position, variant[position])
            if not np.isfinite(variant_term):
                raise ValueError("masked log-probability must be finite")
            total += float(variant_term) - self._wild_type_term(position)
        self.variant_values[variant] = total
        return total

    def transform(self, variants: Sequence[str]) -> np.ndarray:
        """Return one cached dPLL scalar per variant."""
        return np.asarray([self.score(str(variant)) for variant in variants], dtype=np.float64)

    def save(self, path: str | Path) -> None:
        """Save variant scalars; the wild-type position cache remains reproducible."""
        frame = pd.DataFrame(
            {"aa_seq": list(self.variant_values), "dpll": list(self.variant_values.values())}
        )
        frame.to_csv(path, index=False)
