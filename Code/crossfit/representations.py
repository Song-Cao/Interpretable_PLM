"""CPU-side Step 2 sequence and precomputed-embedding representations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .qc import STANDARD_AA, TOKEN_RE

GEORGIEV_CHANNELS = 19


def _mutation_tokens(wild_type: str, mutation: str) -> list[tuple[int, str]]:
    """Parse and validate one curated substitution string."""
    if mutation == "WT":
        return []
    parsed: list[tuple[int, str]] = []
    seen_positions: set[int] = set()
    for token in str(mutation).split(":"):
        match = TOKEN_RE.match(token)
        if match is None:
            raise ValueError(f"invalid mutation token: {token}")
        source, position, target = match.group(1), int(match.group(2)), match.group(3)
        if source not in STANDARD_AA or target not in STANDARD_AA:
            raise ValueError(f"non-canonical residue in mutation token: {token}")
        if position < 1 or position > len(wild_type):
            raise ValueError(f"mutation position outside wild type: {token}")
        if wild_type[position - 1] != source:
            raise ValueError(f"mutation reference disagrees with wild type: {token}")
        if position in seen_positions:
            raise ValueError(f"multiple substitutions at one position: {mutation}")
        if source == target:
            raise ValueError(f"synonymous token must be stripped before encoding: {token}")
        seen_positions.add(position)
        parsed.append((position, target))
    return parsed


def load_georgiev_table(path: str | Path) -> pd.DataFrame:
    """Load the frozen standardized Georgiev 20 by 19 descriptor table."""
    table = pd.read_csv(path).set_index("amino_acid")
    if table.shape != (len(STANDARD_AA), GEORGIEV_CHANNELS):
        raise ValueError(f"expected a 20 x 19 Georgiev table, found {table.shape}")
    if set(table.index) != STANDARD_AA:
        raise ValueError("Georgiev table must contain exactly the canonical amino acids")
    if not np.isfinite(table.to_numpy(dtype=float)).all():
        raise ValueError("Georgiev table contains a non-finite value")
    return table.sort_index()


@dataclass(frozen=True)
class MutationVocabulary:
    """Observed assay-specific (position, mutant amino acid) indicator columns."""

    features: tuple[tuple[int, str], ...]

    @classmethod
    def from_mutations(
        cls, wild_type: str, mutations: Sequence[str]
    ) -> "MutationVocabulary":
        observed = {
            feature
            for mutation in mutations
            for feature in _mutation_tokens(wild_type, str(mutation))
        }
        return cls(tuple(sorted(observed)))

    @property
    def index(self) -> dict[tuple[int, str], int]:
        return {feature: idx for idx, feature in enumerate(self.features)}


def mutation_indicator(
    wild_type: str,
    mutations: Sequence[str],
    vocabulary: MutationVocabulary,
) -> sparse.csr_matrix:
    """Return the sparse assay-specific mutation-indicator matrix."""
    feature_index = vocabulary.index
    rows: list[int] = []
    cols: list[int] = []
    for row, mutation in enumerate(mutations):
        for feature in _mutation_tokens(wild_type, str(mutation)):
            if feature not in feature_index:
                raise ValueError(f"mutation feature absent from vocabulary: {feature}")
            rows.append(row)
            cols.append(feature_index[feature])
    values = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix(
        (values, (rows, cols)), shape=(len(mutations), len(vocabulary.features))
    )


def aaindex_flat(
    wild_type: str,
    mutations: Sequence[str],
    georgiev_table: pd.DataFrame,
) -> sparse.csr_matrix:
    """Return sparse flattened L x 19 Georgiev difference representations."""
    if any(amino_acid not in STANDARD_AA for amino_acid in wild_type):
        invalid = sorted(set(wild_type) - STANDARD_AA)
        raise ValueError(f"non-canonical residues in wild type: {invalid}")
    channels = georgiev_table.shape[1]
    if channels != GEORGIEV_CHANNELS:
        raise ValueError(f"Georgiev table must have {GEORGIEV_CHANNELS} channels")
    values_by_amino_acid = {
        amino_acid: georgiev_table.loc[amino_acid].to_numpy(dtype=np.float64)
        for amino_acid in georgiev_table.index
    }
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row, mutation in enumerate(mutations):
        for position, target in _mutation_tokens(wild_type, str(mutation)):
            source = wild_type[position - 1]
            difference = values_by_amino_acid[target] - values_by_amino_acid[source]
            offset = (position - 1) * channels
            rows.extend([row] * channels)
            cols.extend(range(offset, offset + channels))
            data.extend(difference.tolist())
    encoded = sparse.csr_matrix(
        (np.asarray(data), (rows, cols)),
        shape=(len(mutations), len(wild_type) * channels),
    )
    encoded.eliminate_zeros()
    return encoded


def aaindex_positional(
    wild_type: str,
    mutations: Sequence[str],
    georgiev_table: pd.DataFrame,
) -> np.ndarray:
    """Return the same Georgiev differences unflattened as N x L x 19."""
    flat = aaindex_flat(wild_type, mutations, georgiev_table)
    return flat.toarray().reshape(len(mutations), len(wild_type), GEORGIEV_CHANNELS)


def esm_residue(activations: np.ndarray) -> np.ndarray:
    """Validate and return precomputed per-residue activations."""
    array = np.asarray(activations)
    if array.ndim not in {2, 3}:
        raise ValueError("ESM residue activations must have shape L x d or N x L x d")
    if not np.isfinite(array).all():
        raise ValueError("ESM residue activations contain non-finite values")
    return array


def esm_mean(activations: np.ndarray) -> np.ndarray:
    """Mean-pool precomputed ESM activations over all residues."""
    return esm_residue(activations).mean(axis=-2)


def esm_mutpool(
    activations: np.ndarray,
    mutations: str | Sequence[str],
) -> np.ndarray:
    """Mean-pool precomputed ESM activations over mutated positions only."""
    array = esm_residue(activations)
    single = array.ndim == 2
    if single:
        array = array[None, ...]
        mutation_list = [str(mutations)]
    else:
        if isinstance(mutations, str):
            raise ValueError("batched activations require one mutation string per row")
        mutation_list = [str(mutation) for mutation in mutations]
    if len(mutation_list) != len(array):
        raise ValueError("mutation strings must align with activation rows")

    pooled: list[np.ndarray] = []
    for row, mutation in zip(array, mutation_list):
        positions: list[int] = []
        if mutation != "WT":
            for token in mutation.split(":"):
                match = TOKEN_RE.match(token)
                if match is None:
                    raise ValueError(f"invalid mutation token: {token}")
                position = int(match.group(2))
                if position < 1 or position > row.shape[0]:
                    raise ValueError(f"mutation position outside activation tensor: {token}")
                positions.append(position - 1)
        if not positions:
            raise ValueError("mutated-position pooling is undefined for wild type")
        pooled.append(row[positions].mean(axis=0))
    result = np.stack(pooled)
    return result[0] if single else result


def pair_representation(left, right, fusion: str = "early"):
    """Apply the fixed early-subtraction rule or return a late-fusion control pair."""
    if fusion == "early":
        return left - right
    if fusion == "late":
        return left, right
    raise ValueError("fusion must be 'early' or 'late'")
