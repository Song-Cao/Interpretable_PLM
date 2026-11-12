"""Summarize causal patching effects and relate them to residue-shell profiles."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

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
    """Test Spearman association between feature effects and shell profiles."""
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


def shell_activation_fractions(
    shell_means: Mapping[str, np.ndarray], position_counts: Mapping[str, int]
) -> dict[str, np.ndarray]:
    """Convert per-shell mean activations into fractions of total activation mass."""
    names = list(shell_means)
    if set(names) != set(position_counts):
        raise ValueError("shell means and position counts must have identical keys")
    values = [np.asarray(shell_means[name], dtype=float) for name in names]
    if not values or any(value.ndim != 1 for value in values):
        raise ValueError("shell means must be non-empty one-dimensional vectors")
    if any(value.shape != values[0].shape for value in values):
        raise ValueError("shell means must share one feature axis")
    if any(not np.all(np.isfinite(value)) or np.any(value < 0.0) for value in values):
        raise ValueError("shell means must be finite and non-negative")
    masses = np.stack([
        value * int(position_counts[name]) for name, value in zip(names, values, strict=True)
    ])
    total = masses.sum(axis=0)
    if np.any(total <= 0.0):
        raise ValueError("every feature must have positive total activation")
    return {name: masses[index] / total for index, name in enumerate(names)}


def _rank_biserial(left: np.ndarray, right: np.ndarray) -> float:
    """Return rank-biserial effect size, positive when left tends to be larger."""
    ordered = np.sort(np.asarray(right, dtype=float))
    values = np.asarray(left, dtype=float)
    lower = np.searchsorted(ordered, values, side="left")
    upper = np.searchsorted(ordered, values, side="right")
    u_statistic = np.sum(lower + 0.5 * (upper - lower))
    return float(2.0 * u_statistic / (len(values) * len(ordered)) - 1.0)


def clustered_mann_whitney(
    rotated_effects: np.ndarray,
    control_effects: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float | int]:
    """Compare absolute effects with a feature-and-variant clustered bootstrap."""
    rotated = np.abs(np.asarray(rotated_effects, dtype=float))
    control = np.abs(np.asarray(control_effects, dtype=float))
    if rotated.ndim != 2 or control.ndim != 1 or rotated.shape[1] != len(control):
        raise ValueError("rotated effects must be feature-by-variant and align to control variants")
    if not np.all(np.isfinite(rotated)) or not np.all(np.isfinite(control)):
        raise ValueError("effect arrays must be finite")
    result = mannwhitneyu(rotated.ravel(), control, alternative="two-sided", method="asymptotic")
    rank_biserial = _rank_biserial(rotated.ravel(), control)
    rng = np.random.default_rng(seed)
    draws = np.empty(bootstrap_samples, dtype=float)
    for draw in range(bootstrap_samples):
        feature_indices = rng.integers(0, rotated.shape[0], rotated.shape[0])
        variant_indices = rng.integers(0, rotated.shape[1], rotated.shape[1])
        sampled_rotated = rotated[np.ix_(feature_indices, variant_indices)].ravel()
        sampled_control = control[variant_indices]
        draws[draw] = _rank_biserial(sampled_rotated, sampled_control)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "rotated_features": int(rotated.shape[0]),
        "variants": int(rotated.shape[1]),
        "rotated_feature_variant_observations": int(rotated.size),
        "control_variant_observations": int(len(control)),
        "mann_whitney_u": float(result.statistic),
        "mann_whitney_p_value": float(result.pvalue),
        "rank_biserial": rank_biserial,
        "bootstrap_ci_lower": float(np.quantile(draws, alpha)),
        "bootstrap_ci_upper": float(np.quantile(draws, 1.0 - alpha)),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
    }
