"""Statistics used to build a residue-level crosscoder feature atlas."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .semantic_enrichment import benjamini_hochberg_adjusted


def activation_weighted_profiles(
    residue_codes: np.ndarray,
    tracks: Mapping[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    """Return activation-weighted track means for every feature."""
    codes = np.asarray(residue_codes, dtype=float)
    if codes.ndim != 2 or np.any(codes < 0) or not np.isfinite(codes).all():
        raise ValueError("residue codes must be a finite nonnegative position-by-feature matrix")
    total = codes.sum(axis=0)
    if np.any(total <= 0):
        raise ValueError("every requested feature must have positive activation mass")
    names, columns = [], []
    for name, raw in tracks.items():
        values = np.asarray(raw, dtype=float)
        if values.shape != (codes.shape[0],) or not np.isfinite(values).all():
            raise ValueError(f"track {name!r} is not aligned and finite")
        names.append(str(name))
        columns.append((codes * values[:, None]).sum(axis=0) / total)
    return names, np.stack(columns, axis=1)


def empirical_reference_qvalues(
    profiles: np.ndarray,
    labels: np.ndarray,
    *,
    reference_class: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare feature profiles with a reference class and adjust all tests jointly."""
    values = np.asarray(profiles, dtype=float)
    groups = np.asarray(labels).astype(str)
    if values.ndim != 2 or len(groups) != len(values) or not np.isfinite(values).all():
        raise ValueError("profiles and labels must be aligned and finite")
    reference = values[groups == reference_class]
    if len(reference) < 2:
        raise ValueError("reference class requires at least two features")
    center = np.median(reference, axis=0)
    reference_distance = np.abs(reference - center)
    observed_distance = np.abs(values - center)
    pvalues = np.empty_like(values)
    for column in range(values.shape[1]):
        pvalues[:, column] = (
            1.0
            + (reference_distance[:, column, None] >= observed_distance[None, :, column]).sum(axis=0)
        ) / (len(reference) + 1.0)
    qvalues = benjamini_hochberg_adjusted(pvalues.ravel()).reshape(values.shape)
    return pvalues, qvalues


def feature_seed_replication(
    profiles_by_seed: Sequence[np.ndarray],
    labels_by_seed: Sequence[np.ndarray],
    *,
    live_masks: Sequence[np.ndarray],
    minimum_cosine: float,
) -> tuple[np.ndarray, list[dict[str, float | int | str | bool]]]:
    """Match live features by residue profile and return class replication fractions."""
    if not profiles_by_seed or len(profiles_by_seed) != len(labels_by_seed):
        raise ValueError("profiles and labels require the same nonzero seed count")
    if len(live_masks) != len(profiles_by_seed):
        raise ValueError("one live mask is required per seed")
    primary_ids = np.flatnonzero(np.asarray(live_masks[0], dtype=bool))
    primary = np.asarray(profiles_by_seed[0], dtype=float)[:, primary_ids].T
    primary_norm = np.linalg.norm(primary, axis=1)
    replication = np.ones(len(primary_ids), dtype=float)
    denominator = np.ones(len(primary_ids), dtype=float)
    rows: list[dict[str, float | int | str | bool]] = []
    for seed_index in range(1, len(profiles_by_seed)):
        candidate_ids = np.flatnonzero(np.asarray(live_masks[seed_index], dtype=bool))
        candidate = np.asarray(profiles_by_seed[seed_index], dtype=float)[:, candidate_ids].T
        candidate_norm = np.linalg.norm(candidate, axis=1)
        cosine = (primary @ candidate.T) / np.maximum(
            primary_norm[:, None] * candidate_norm[None, :], np.finfo(float).tiny
        )
        left, right = linear_sum_assignment(-cosine)
        mapping = {int(a): int(b) for a, b in zip(left, right, strict=True)}
        for primary_offset, feature_id in enumerate(primary_ids):
            matched_offset = mapping.get(primary_offset)
            if matched_offset is None:
                rows.append({
                    "primary_feature": int(feature_id), "seed_index": seed_index,
                    "matched_feature": -1, "profile_cosine": np.nan,
                    "class_match": False, "accepted_match": False,
                })
                continue
            matched_id = int(candidate_ids[matched_offset])
            score = float(cosine[primary_offset, matched_offset])
            accepted = score >= minimum_cosine
            class_match = bool(
                accepted
                and str(labels_by_seed[0][feature_id]) == str(labels_by_seed[seed_index][matched_id])
            )
            if accepted:
                denominator[primary_offset] += 1.0
                replication[primary_offset] += float(class_match)
            rows.append({
                "primary_feature": int(feature_id), "seed_index": seed_index,
                "matched_feature": matched_id, "profile_cosine": score,
                "class_match": class_match, "accepted_match": accepted,
            })
    return replication / denominator, rows


def select_profile_subtypes(
    profiles: np.ndarray,
    *,
    candidates: Sequence[int],
    seed: int,
    n_init: int,
) -> tuple[np.ndarray, list[dict[str, float | int]], int]:
    """Select a K-means subtype count by silhouette on annotation profiles."""
    values = np.asarray(profiles, dtype=float)
    if values.ndim != 2 or len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("subtype profiles must be a finite matrix with at least four rows")
    variable = np.std(values, axis=0) > 0
    if not variable.any():
        raise ValueError("subtype profiles have no variable annotation")
    scaled = StandardScaler().fit_transform(values[:, variable])
    rows, fits = [], {}
    for k in map(int, candidates):
        if k < 2 or k >= len(values):
            continue
        model = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = model.fit_predict(scaled)
        score = float(silhouette_score(scaled, labels))
        rows.append({"clusters": k, "silhouette": score, "inertia": float(model.inertia_)})
        fits[k] = labels
    if not rows:
        raise ValueError("no admissible subtype cluster count")
    selected = int(max(rows, key=lambda row: (row["silhouette"], -row["clusters"]))["clusters"])
    return fits[selected], rows, selected


def paired_label_permutation(
    inside: np.ndarray,
    outside: np.ndarray,
    *,
    permutations: int,
    seed: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, float | int]:
    """Compare paired response magnitudes by random within-variant label swaps."""
    left, right = np.asarray(inside, dtype=float), np.asarray(outside, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("paired response vectors must align and contain at least two variants")
    difference = left - right
    observed = float(difference.mean())
    rng = np.random.default_rng(seed)
    batch = 1000
    exceed = 0
    completed = 0
    while completed < permutations:
        count = min(batch, permutations - completed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(difference)))
        permuted = (signs * difference).mean(axis=1)
        exceed += int((np.abs(permuted) >= abs(observed)).sum())
        completed += count
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    indices = bootstrap_rng.integers(0, len(difference), size=(bootstrap_samples, len(difference)))
    draws = difference[indices].mean(axis=1)
    return {
        "variants": int(len(left)), "inside_mean": float(left.mean()),
        "outside_mean": float(right.mean()), "paired_mean_difference": observed,
        "bootstrap_ci_lower": float(np.quantile(draws, 0.025)),
        "bootstrap_ci_upper": float(np.quantile(draws, 0.975)),
        "permutation_p_value": float((exceed + 1.0) / (permutations + 1.0)),
        "permutations": int(permutations), "permutation_seed": int(seed),
        "bootstrap_samples": int(bootstrap_samples), "bootstrap_seed": int(bootstrap_seed),
    }


def subtype_effect_test(
    effect: np.ndarray,
    subtype: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, float | int]:
    """Test between-subtype differences in causal effect by label permutation."""
    values = np.asarray(effect, dtype=float)
    groups = np.asarray(subtype, dtype=int)
    keep = np.isfinite(values)
    values, groups = values[keep], groups[keep]
    categories = np.unique(groups)
    if len(categories) < 2 or any((groups == category).sum() < 2 for category in categories):
        raise ValueError("at least two subtypes with two causal features each are required")

    def statistic(labels: np.ndarray) -> float:
        overall = values.mean()
        return float(sum(
            (labels == category).sum() * (values[labels == category].mean() - overall) ** 2
            for category in categories
        ))

    observed = statistic(groups)
    rng = np.random.default_rng(seed)
    exceed = sum(statistic(rng.permutation(groups)) >= observed for _ in range(permutations))
    return {
        "features_with_causal_effect": int(len(values)), "subtypes": int(len(categories)),
        "between_subtype_statistic": observed,
        "permutation_p_value": float((exceed + 1.0) / (permutations + 1.0)),
        "permutations": int(permutations), "permutation_seed": int(seed),
    }


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return Spearman correlation and a percentile variant bootstrap interval."""
    left, right = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    if len(left) < 3:
        return np.nan, np.nan, np.nan
    observed = float(spearmanr(left, right).statistic)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        indices = rng.integers(0, len(left), len(left))
        value = float(spearmanr(left[indices], right[indices]).statistic)
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return observed, np.nan, np.nan
    return observed, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
