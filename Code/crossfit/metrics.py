"""Per-variant and pairwise metrics for Step 2 predictors."""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
from scipy import stats

from .pair_analysis import mutation_map, variant_hamming_distance


def _correlation(left: np.ndarray, right: np.ndarray, method: str) -> float:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    result = stats.spearmanr(left, right) if method == "spearman" else stats.pearsonr(left, right)
    return float(result.statistic)


def variant_metrics(prediction: np.ndarray, target: np.ndarray, top_k: Sequence[int]) -> dict[str, float]:
    """Compute full-test variant metrics without pooling assays."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 1:
        raise ValueError("predictions and targets must be aligned vectors")
    output = {
        "variant_spearman": _correlation(prediction, target, "spearman"),
        "variant_pearson": _correlation(prediction, target, "pearson"),
    }
    predicted_order = np.argsort(-prediction, kind="stable")
    true_order = np.argsort(-target, kind="stable")
    relevance = target - target.min()
    for requested_k in top_k:
        k = min(int(requested_k), len(target))
        predicted_top = predicted_order[:k]
        true_top = set(true_order[:k].tolist())
        output[f"top_{requested_k}_precision"] = len(set(predicted_top.tolist()) & true_top) / k
        discount = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float(np.sum(relevance[predicted_top] * discount))
        ideal = float(np.sum(relevance[true_order[:k]] * discount))
        output[f"ndcg_{requested_k}"] = dcg / ideal if ideal > 0 else float("nan")
    return output


def sample_pair_indices(n_variants: int, max_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return all pairs when feasible, otherwise a deterministic uniform sample."""
    if n_variants < 2 or max_pairs < 1:
        raise ValueError("pair sampling requires at least two variants and a positive budget")
    total = n_variants * (n_variants - 1) // 2
    if total <= max_pairs:
        pairs = np.asarray(tuple(itertools.combinations(range(n_variants), 2)), dtype=np.int64)
        return pairs[:, 0], pairs[:, 1]
    rng = np.random.default_rng(seed)
    selected: set[tuple[int, int]] = set()
    while len(selected) < max_pairs:
        needed = max_pairs - len(selected)
        left = rng.integers(0, n_variants, size=needed * 2)
        right = rng.integers(0, n_variants, size=needed * 2)
        for u, v in zip(left, right):
            if u == v:
                continue
            selected.add((int(min(u, v)), int(max(u, v))))
            if len(selected) == max_pairs:
                break
    pairs = np.asarray(sorted(selected), dtype=np.int64)
    return pairs[:, 0], pairs[:, 1]


def gated_test_pairs(
    target: np.ndarray,
    censored: Sequence[str],
    *,
    margin_sd: float,
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample test pairs, then apply the fixed censor and margin gates."""
    target = np.asarray(target, dtype=np.float64)
    u, v = sample_pair_indices(len(target), max_pairs, seed)
    censor = np.asarray(censored, dtype=str)
    same_bound = (censor[u] == censor[v]) & (censor[u] != "none")
    distinguishable = np.abs(target[u] - target[v]) > margin_sd
    keep = ~same_bound & distinguishable
    return u[keep], v[keep]


def pair_metrics(
    score: np.ndarray,
    target: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> dict[str, float]:
    """Compute metrics on predicted and observed pair differences."""
    predicted = np.asarray(score)[u] - np.asarray(score)[v]
    observed = np.asarray(target)[u] - np.asarray(target)[v]
    if len(observed) == 0:
        return {
            "pair_spearman": float("nan"),
            "pair_pearson": float("nan"),
            "pair_rmse": float("nan"),
            "pair_sign_accuracy": float("nan"),
        }
    return {
        "pair_spearman": _correlation(predicted, observed, "spearman"),
        "pair_pearson": _correlation(predicted, observed, "pearson"),
        "pair_rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "pair_sign_accuracy": float(np.mean(np.sign(predicted) == np.sign(observed))),
    }


def pair_metrics_by_hamming(
    score: np.ndarray,
    target: np.ndarray,
    mutations: Sequence[str],
    u: np.ndarray,
    v: np.ndarray,
) -> list[dict[str, float | int]]:
    """Stratify pair metrics by exact variant-to-variant Hamming distance."""
    parsed = [mutation_map(value) for value in mutations]
    distances = np.asarray(
        [variant_hamming_distance(parsed[left], parsed[right]) for left, right in zip(u, v)]
    )
    rows = []
    for distance in np.unique(distances):
        selected = distances == distance
        rows.append(
            {
                "hamming_distance": int(distance),
                "n_pairs": int(selected.sum()),
                **pair_metrics(score, target, u[selected], v[selected]),
            }
        )
    return rows
