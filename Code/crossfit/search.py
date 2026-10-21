"""Leakage-safe cross-validation and random hyperparameter search."""
from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def training_pool_folds(
    training_indices: Sequence[int], n_folds: int, seed: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Partition only the supplied training pool into deterministic CV folds."""
    indices = np.asarray(training_indices, dtype=np.int64)
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        raise ValueError("training indices must be a unique one-dimensional sequence")
    if n_folds < 2 or len(indices) < n_folds:
        raise ValueError("cross-validation requires at least one row per fold")
    shuffled = np.random.default_rng(seed).permutation(indices)
    validation_folds = np.array_split(shuffled, n_folds)
    output = []
    for validation in validation_folds:
        selected = ~np.isin(shuffled, validation)
        output.append((np.sort(shuffled[selected]), np.sort(validation)))
    return tuple(output)


def _draw_value(specification: Any, rng: np.random.Generator) -> Any:
    if isinstance(specification, list):
        value = specification[int(rng.integers(len(specification)))]
        return list(value) if isinstance(value, list) else value
    if isinstance(specification, Mapping):
        if specification.get("distribution") != "log_uniform":
            raise ValueError(f"unsupported search distribution: {specification}")
        low, high = float(specification["low"]), float(specification["high"])
        if not 0 < low < high:
            raise ValueError("log-uniform bounds must satisfy 0 < low < high")
        return float(math.exp(rng.uniform(math.log(low), math.log(high))))
    raise TypeError(f"search parameter must be a list or distribution mapping: {specification}")


def sample_configurations(
    space: Mapping[str, Any], count: int, seed: int
) -> list[dict[str, Any]]:
    """Sample unique configurations, including registered boundary values first."""
    if count < 1 or not space:
        raise ValueError("random search needs a positive count and non-empty space")
    rng = np.random.default_rng(seed)
    required: list[dict[str, Any]] = []
    for name, specification in space.items():
        if isinstance(specification, Mapping):
            for value in specification.get("include", []):
                config = {key: _draw_value(spec, rng) for key, spec in space.items()}
                config[name] = value
                required.append(config)

    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(config: dict[str, Any]) -> None:
        identity = repr(sorted(config.items()))
        if identity not in seen and len(output) < count:
            output.append(config)
            seen.add(identity)

    for config in required:
        add(config)
    attempts = 0
    while len(output) < count and attempts < count * 1000:
        add({name: _draw_value(specification, rng) for name, specification in space.items()})
        attempts += 1
    if len(output) < count:
        raise ValueError(f"search space has fewer than {count} unique configurations")
    return output


def boundary_hyperparameters(
    selected: Mapping[str, Any], space: Mapping[str, Any]
) -> dict[str, str]:
    """Identify selected numeric values on a registered search boundary."""
    hits: dict[str, str] = {}
    for name, value in selected.items():
        specification = space.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if isinstance(specification, list):
            numeric = [item for item in specification if isinstance(item, (int, float))]
            if numeric and value == min(numeric):
                hits[name] = "lower"
            if numeric and value == max(numeric):
                hits[name] = "upper" if name not in hits else "both"
        elif isinstance(specification, Mapping):
            if np.isclose(value, float(specification["low"])):
                hits[name] = "lower"
            if np.isclose(value, float(specification["high"])):
                hits[name] = "upper"
    return hits
