"""Residue-level, class-level semantic enrichment statistics."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def profile_enrichment(
    residue_codes: np.ndarray, tracks: Mapping[str, np.ndarray]
) -> tuple[list[str], np.ndarray]:
    """Return per-feature log2 enrichment for each residue annotation track."""
    values = np.asarray(residue_codes, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("residue codes must be a finite nonnegative position-by-feature matrix")
    names, columns = [], []
    positive = values[values > 0]
    epsilon = (float(positive.mean()) if len(positive) else 1.0) * 1e-6
    for name, raw_mask in tracks.items():
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != (values.shape[0],) or not mask.any() or mask.all():
            raise ValueError(f"track {name} must select a strict nonempty residue subset")
        inside = values[mask].mean(axis=0)
        outside = values[~mask].mean(axis=0)
        columns.append(np.log2((inside + epsilon) / (outside + epsilon)))
        names.append(str(name))
    return names, np.stack(columns, axis=1)


def _standardize(values: np.ndarray) -> np.ndarray:
    center = np.mean(values, axis=0, keepdims=True)
    scale = np.std(values, axis=0, ddof=1, keepdims=True)
    return (values - center) / np.where(scale > 0, scale, 1.0)


def blocked_class_permutation(
    scores_by_seed: Sequence[np.ndarray],
    labels_by_seed: Sequence[np.ndarray],
    *,
    target: str,
    comparison: str,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare two classes while permuting feature labels within dictionary seed."""
    if len(scores_by_seed) != len(labels_by_seed) or not scores_by_seed:
        raise ValueError("scores and labels require the same nonzero seed count")
    rng = np.random.default_rng(seed)
    prepared = []
    observed_by_seed = []
    for scores, labels in zip(scores_by_seed, labels_by_seed):
        values = np.asarray(scores, dtype=float)
        groups = np.asarray(labels).astype(str)
        keep = (groups == target) | (groups == comparison)
        values, groups = _standardize(values[keep]), groups[keep]
        if not (groups == target).any() or not (groups == comparison).any():
            raise ValueError("both comparison classes must be populated in every seed")
        prepared.append((values, groups))
        observed_by_seed.append(
            values[groups == target].mean(axis=0) - values[groups == comparison].mean(axis=0)
        )
    observed_seeds = np.stack(observed_by_seed)
    observed = observed_seeds.mean(axis=0)
    null = np.empty((permutations, len(observed)), dtype=float)
    for draw in range(permutations):
        effects = []
        for values, groups in prepared:
            shuffled = rng.permutation(groups)
            effects.append(
                values[shuffled == target].mean(axis=0)
                - values[shuffled == comparison].mean(axis=0)
            )
        null[draw] = np.mean(effects, axis=0)
    pvalues = (1.0 + (np.abs(null) >= np.abs(observed)).sum(axis=0)) / (permutations + 1.0)
    return observed, pvalues, observed_seeds


def benjamini_hochberg_adjusted(pvalues: np.ndarray) -> np.ndarray:
    """Return monotone Benjamini-Hochberg adjusted p-values."""
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be a one-dimensional unit-interval array")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def top_position_shell_statistic(
    residue_codes_by_seed: Sequence[np.ndarray],
    labels_by_seed: Sequence[np.ndarray],
    shells: np.ndarray,
    *,
    target: str,
    comparison: str,
    permutations: int,
    seed: int,
) -> tuple[float, float, list[dict[str, object]]]:
    """Test target-versus-comparison concentration of top positions across shells."""
    categories = np.unique(np.asarray(shells).astype(str))
    rng = np.random.default_rng(seed)
    blocks = []
    rows = []
    for seed_index, (codes, labels) in enumerate(zip(residue_codes_by_seed, labels_by_seed)):
        values = np.asarray(codes, dtype=float)
        groups = np.asarray(labels).astype(str)
        keep = (groups == target) | (groups == comparison)
        top_shell = np.asarray(shells).astype(str)[np.argmax(values[:, keep], axis=0)]
        groups = groups[keep]
        blocks.append((top_shell, groups))
        for group in (target, comparison):
            for category in categories:
                rows.append({
                    "seed_index": seed_index, "class": group, "shell": category,
                    "count": int(((groups == group) & (top_shell == category)).sum()),
                })

    def statistic(parts: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
        differences = []
        for positions, groups in parts:
            left = np.asarray([(positions[groups == target] == item).mean() for item in categories])
            right = np.asarray([(positions[groups == comparison] == item).mean() for item in categories])
            differences.append(0.5 * np.abs(left - right).sum())
        return float(np.mean(differences))

    observed = statistic(blocks)
    exceed = 0
    for _ in range(permutations):
        shuffled = [(positions, rng.permutation(groups)) for positions, groups in blocks]
        exceed += statistic(shuffled) >= observed
    return observed, (exceed + 1.0) / (permutations + 1.0), rows
