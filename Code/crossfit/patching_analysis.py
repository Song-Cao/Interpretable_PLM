"""Summarize causal patching effects and relate them to residue-shell profiles."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from scipy.stats import spearmanr

from crossfit.semantic_enrichment import benjamini_hochberg_adjusted


def quantile_rows(
    values: Iterable[float], quantiles: Iterable[float], *, population: str,
    sampling_unit: str,
) -> list[dict[str, float | int | str]]:
    """Return named quantiles for one finite one-dimensional distribution."""
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("quantile values must be a non-empty finite vector")
    levels = np.asarray(list(quantiles), dtype=float)
    if np.any((levels < 0.0) | (levels > 1.0)):
        raise ValueError("quantile levels must lie in [0, 1]")
    estimates = np.quantile(array, levels)
    return [
        {
            "population": population,
            "sampling_unit": sampling_unit,
            "n": int(len(array)),
            "quantile": float(level),
            "value": float(value),
        }
        for level, value in zip(levels, estimates, strict=True)
    ]


def spatial_effect_correlations(
    absolute_effect: np.ndarray,
    shell_profiles: Mapping[str, np.ndarray],
    *,
    permutations: int,
    seed: int,
    fdr_q: float,
) -> list[dict[str, float | int | str | bool]]:
    """Test Spearman association between feature effects and shell activations."""
    effect = np.asarray(absolute_effect, dtype=float)
    if effect.ndim != 1 or len(effect) < 3 or not np.all(np.isfinite(effect)):
        raise ValueError("absolute effects must be a finite vector with at least three features")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    prepared: dict[str, np.ndarray] = {}
    observed = []
    for name, values in shell_profiles.items():
        profile = np.asarray(values, dtype=float)
        if profile.shape != effect.shape or not np.all(np.isfinite(profile)):
            raise ValueError(f"shell profile {name!r} is not aligned and finite")
        prepared[name] = profile
        statistic = float(spearmanr(effect, profile).statistic)
        if not np.isfinite(statistic):
            raise ValueError(f"shell profile {name!r} gives undefined Spearman correlation")
        observed.append(statistic)
    rng = np.random.default_rng(seed)
    exceedances = np.zeros(len(prepared), dtype=int)
    for _ in range(permutations):
        permuted = rng.permutation(effect)
        for index, profile in enumerate(prepared.values()):
            statistic = float(spearmanr(permuted, profile).statistic)
            exceedances[index] += abs(statistic) >= abs(observed[index])
    pvalues = (exceedances + 1.0) / (permutations + 1.0)
    adjusted = benjamini_hochberg_adjusted(pvalues)
    return [
        {
            "shell": name,
            "features": int(len(effect)),
            "spearman_rho": float(observed[index]),
            "permutation_p_value": float(pvalues[index]),
            "fdr_q_value": float(adjusted[index]),
            "fdr_significant": bool(adjusted[index] <= fdr_q),
            "permutations": int(permutations),
            "permutation_seed": int(seed),
        }
        for index, name in enumerate(prepared)
    ]
