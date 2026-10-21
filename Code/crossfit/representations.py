"""Published sequence encodings and precomputed ESM-C representations."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .qc import TOKEN_RE


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
GEORGIEV_CHANNELS = 19


def _validated_sequences(sequences: Sequence[str]) -> list[str]:
    values = [str(sequence) for sequence in sequences]
    if not values:
        raise ValueError("at least one sequence is required")
    lengths = {len(sequence) for sequence in values}
    if len(lengths) != 1:
        raise ValueError("all sequences in one dataset must have the same length")
    invalid = sorted(set("".join(values)) - set(AMINO_ACIDS))
    if invalid:
        raise ValueError(f"non-canonical residues: {invalid}")
    return values


def load_georgiev_table(path: str | Path) -> pd.DataFrame:
    """Load the frozen, standardized Georgiev 20 by 19 descriptor table."""
    table = pd.read_csv(path).set_index("amino_acid")
    if table.shape != (len(AMINO_ACIDS), GEORGIEV_CHANNELS):
        raise ValueError(f"expected a 20 x 19 Georgiev table, found {table.shape}")
    if set(table.index) != set(AMINO_ACIDS):
        raise ValueError("Georgiev table must contain exactly the canonical amino acids")
    values = table.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Georgiev table contains a non-finite value")
    if not np.allclose(values.mean(axis=0), 0.0, atol=1e-10):
        raise ValueError("Georgiev columns are not centered across amino acids")
    if not np.allclose(values.var(axis=0), 1.0, atol=1e-10):
        raise ValueError("Georgiev columns do not have unit population variance")
    return table.loc[list(AMINO_ACIDS)]


def onehot_positional(sequences: Sequence[str]) -> np.ndarray:
    """Encode every residue as the published dense N by L by 20 binary matrix."""
    values = _validated_sequences(sequences)
    lookup = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
    encoded = np.zeros((len(values), len(values[0]), len(AMINO_ACIDS)), dtype=np.float32)
    for row, sequence in enumerate(values):
        positions = np.arange(len(sequence))
        encoded[row, positions, [lookup[residue] for residue in sequence]] = 1.0
    return encoded


def onehot(sequences: Sequence[str]) -> np.ndarray:
    """Flatten the whole-sequence N by L by 20 encoding without dropping a column."""
    encoded = onehot_positional(sequences)
    return encoded.reshape(len(encoded), -1)


def aaindex_flat(sequences: Sequence[str], georgiev_table: pd.DataFrame) -> np.ndarray:
    """Return dense absolute Georgiev values at every position, flattened to 19L."""
    values = _validated_sequences(sequences)
    if georgiev_table.shape != (len(AMINO_ACIDS), GEORGIEV_CHANNELS):
        raise ValueError("Georgiev table must have shape 20 by 19")
    lookup = {
        amino_acid: georgiev_table.loc[amino_acid].to_numpy(dtype=np.float32)
        for amino_acid in AMINO_ACIDS
    }
    encoded = np.stack([[lookup[residue] for residue in sequence] for sequence in values])
    return encoded.reshape(len(values), -1)


def esm_residue(activations: np.ndarray) -> np.ndarray:
    """Validate and return precomputed full per-residue ESM-C activations."""
    array = np.asarray(activations)
    if array.ndim not in {2, 3}:
        raise ValueError("ESM residue activations must have shape L x d or N x L x d")
    if not np.isfinite(array).all():
        raise ValueError("ESM residue activations contain non-finite values")
    return array


def esm_residue_delta(variant: np.ndarray, wild_type: np.ndarray) -> np.ndarray:
    """Subtract wild-type ESM-C residues from aligned variant residue activations."""
    variant_array = esm_residue(variant)
    wild_type_array = esm_residue(wild_type)
    if wild_type_array.ndim != 2 or variant_array.shape[-2:] != wild_type_array.shape:
        raise ValueError("wild-type activations must be one L x d tensor aligned to variants")
    return variant_array - wild_type_array


def esm_mean(activations: np.ndarray) -> np.ndarray:
    """Mean-pool precomputed ESM-C activations over every residue."""
    return esm_residue(activations).mean(axis=-2)


def _mutation_positions(mutation: str, length: int) -> list[int]:
    if mutation == "WT":
        return []
    positions: list[int] = []
    for token in mutation.split(":"):
        match = TOKEN_RE.match(token)
        if match is None:
            raise ValueError(f"invalid mutation token: {token}")
        position = int(match.group(2))
        if position < 1 or position > length:
            raise ValueError(f"mutation position outside activation tensor: {token}")
        positions.append(position - 1)
    return positions


def esm_mutpool(activations: np.ndarray, mutations: str | Sequence[str]) -> np.ndarray:
    """Diagnostic only: mean-pool ESM-C activations over mutated positions."""
    array = esm_residue(activations)
    single = array.ndim == 2
    rows = array[None, ...] if single else array
    mutation_list = [str(mutations)] if single else [str(value) for value in mutations]
    if len(mutation_list) != len(rows):
        raise ValueError("mutation strings must align with activation rows")
    pooled = []
    for row, mutation in zip(rows, mutation_list):
        positions = _mutation_positions(mutation, len(row))
        if not positions:
            raise ValueError("mutated-position pooling is undefined for wild type")
        pooled.append(row[positions].mean(axis=0))
    result = np.stack(pooled)
    return result[0] if single else result


def encode_representation(name: str, **inputs: object) -> np.ndarray:
    """Select a v10 representation by its config name."""
    if name == "onehot":
        return onehot(inputs["sequences"])
    if name == "onehot_positional":
        return onehot_positional(inputs["sequences"])
    if name == "aaindex_flat":
        return aaindex_flat(inputs["sequences"], inputs["georgiev_table"])
    if name == "esm_residue":
        return esm_residue(inputs["activations"])
    if name == "esm_residue_delta":
        return esm_residue_delta(inputs["activations"], inputs["wild_type_activations"])
    if name == "esm_mean":
        return esm_mean(inputs["activations"])
    if name == "esm_mutpool":
        return esm_mutpool(inputs["activations"], inputs["mutations"])
    raise KeyError(f"unknown representation: {name}")
