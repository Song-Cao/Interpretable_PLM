"""Direct-score metrics, deep ensembles, and Mondrian conformal intervals."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _aligned(prediction: Iterable[float], target: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    prediction_array = np.asarray(tuple(prediction), dtype=np.float64)
    target_array = np.asarray(tuple(target), dtype=np.float64)
    if prediction_array.ndim != 1 or target_array.shape != prediction_array.shape or not len(target_array):
        raise ValueError("predictions and targets must be aligned non-empty vectors")
    if not np.isfinite(prediction_array).all() or not np.isfinite(target_array).all():
        raise ValueError("predictions and targets must be finite")
    return prediction_array, target_array


def ndcg(prediction: Iterable[float], target: Iterable[float], k: int | None = None) -> float:
    """Compute NDCG from direct scores, shifting relevance to be non-negative."""
    prediction_array, target_array = _aligned(prediction, target)
    limit = len(target_array) if k is None else min(int(k), len(target_array))
    if limit < 1:
        raise ValueError("NDCG cutoff must be positive")
    relevance = target_array - target_array.min()
    discount = np.log2(np.arange(2, limit + 2))

    def gain(order: np.ndarray) -> float:
        return float(np.sum((np.exp2(relevance[order[:limit]]) - 1.0) / discount))

    observed = gain(np.argsort(-prediction_array, kind="stable"))
    ideal = gain(np.argsort(-target_array, kind="stable"))
    return 1.0 if ideal == 0.0 else observed / ideal


def top_k_precision(
    prediction: Iterable[float], target: Iterable[float], *, k: int, wild_type_fitness: float
) -> float:
    """Return the fraction of the k predicted best variants that improve on wild type."""
    prediction_array, target_array = _aligned(prediction, target)
    limit = min(int(k), len(target_array))
    if limit < 1 or not np.isfinite(wild_type_fitness):
        raise ValueError("top-k precision needs positive k and a finite wild-type cutoff")
    selected = np.argsort(-prediction_array, kind="stable")[:limit]
    return float(np.mean(target_array[selected] > wild_type_fitness))


def score_metrics(
    prediction: Iterable[float],
    target: Iterable[float],
    *,
    top_k: Iterable[int] = (),
    wild_type_fitness: float | None = None,
) -> dict[str, float]:
    """Compute direct-score metrics, adding precision only when a WT cutoff is supplied."""
    prediction_array, target_array = _aligned(prediction, target)
    result = {
        "spearman": float(spearmanr(prediction_array, target_array).statistic),
        "pearson": float(pearsonr(prediction_array, target_array).statistic),
        "mse": float(np.mean((prediction_array - target_array) ** 2)),
        "ndcg": ndcg(prediction_array, target_array),
    }
    if wild_type_fitness is not None:
        result["test_hit_rate"] = float(np.mean(target_array > wild_type_fitness))
        for k in top_k:
            result[f"top_{int(k)}_precision"] = top_k_precision(
                prediction_array, target_array, k=int(k), wild_type_fitness=wild_type_fitness
            )
    elif tuple(top_k):
        raise ValueError("top-k precision cannot be computed without a wild-type cutoff")
    return result


@dataclass(frozen=True)
class EnsemblePrediction:
    """Deep-ensemble mean and epistemic spread for each variant."""

    mean: np.ndarray
    standard_deviation: np.ndarray


def combine_deep_ensemble(member_predictions: np.ndarray, members: int = 5) -> EnsemblePrediction:
    """Combine exactly M independently initialized model predictions."""
    values = np.asarray(member_predictions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != members or not np.isfinite(values).all():
        raise ValueError(f"deep ensemble requires a finite {members} by N array")
    return EnsemblePrediction(values.mean(axis=0), values.std(axis=0, ddof=1))


@dataclass(frozen=True)
class ConformalInterval:
    """Lower and upper prediction bounds from Mondrian calibration."""

    lower: np.ndarray
    upper: np.ndarray


class MondrianConformal:
    """Split conformal intervals stratified by mutation count and position novelty."""

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        self.alpha = float(alpha)
        self.quantiles_: dict[tuple[int, bool], float] = {}
        self.counts_: dict[tuple[int, bool], int] = {}

    @staticmethod
    def _strata(
        mutation_count: Iterable[int], unseen_position: Iterable[bool]
    ) -> list[tuple[int, bool]]:
        count = np.asarray(tuple(mutation_count))
        unseen = np.asarray(tuple(unseen_position))
        if count.ndim != 1 or unseen.shape != count.shape:
            raise ValueError("mutation count and unseen-position labels must be aligned")
        if not np.issubdtype(count.dtype, np.integer) or np.any(count < 0):
            raise ValueError("mutation counts must be non-negative integers")
        return [(int(value), bool(novel)) for value, novel in zip(count, unseen)]

    def fit(
        self,
        prediction: Iterable[float],
        target: Iterable[float],
        mutation_count: Iterable[int],
        unseen_position: Iterable[bool],
    ) -> "MondrianConformal":
        """Fit finite-sample residual quantiles independently in every stratum."""
        prediction_array, target_array = _aligned(prediction, target)
        strata = self._strata(mutation_count, unseen_position)
        if len(strata) != len(prediction_array):
            raise ValueError("calibration values and strata must align")
        residual = np.abs(target_array - prediction_array)
        grouped: dict[tuple[int, bool], list[float]] = {}
        for key, value in zip(strata, residual):
            grouped.setdefault(key, []).append(float(value))
        self.quantiles_.clear()
        self.counts_.clear()
        for key, values in grouped.items():
            ordered = np.sort(np.asarray(values))
            rank = math.ceil((len(ordered) + 1) * (1.0 - self.alpha))
            self.quantiles_[key] = float("inf") if rank > len(ordered) else float(ordered[rank - 1])
            self.counts_[key] = len(ordered)
        return self

    def predict_interval(
        self,
        prediction: Iterable[float],
        mutation_count: Iterable[int],
        unseen_position: Iterable[bool],
    ) -> ConformalInterval:
        """Return intervals and reject strata absent from calibration."""
        if not self.quantiles_:
            raise RuntimeError("fit must be called before interval prediction")
        prediction_array = np.asarray(tuple(prediction), dtype=np.float64)
        strata = self._strata(mutation_count, unseen_position)
        if prediction_array.ndim != 1 or len(strata) != len(prediction_array):
            raise ValueError("prediction and stratum metadata must align")
        missing = sorted(set(strata) - set(self.quantiles_))
        if missing:
            raise KeyError(f"no calibration examples for strata {missing}")
        radius = np.asarray([self.quantiles_[key] for key in strata])
        return ConformalInterval(prediction_array - radius, prediction_array + radius)


def coverage_by_stratum(
    interval: ConformalInterval,
    target: Iterable[float],
    mutation_count: Iterable[int],
    unseen_position: Iterable[bool],
) -> dict[tuple[int, bool], float]:
    """Calculate empirical interval coverage separately for every Mondrian stratum."""
    target_array = np.asarray(tuple(target), dtype=np.float64)
    strata = MondrianConformal._strata(mutation_count, unseen_position)
    if interval.lower.shape != target_array.shape or interval.upper.shape != target_array.shape:
        raise ValueError("intervals and targets must align")
    hit = (interval.lower <= target_array) & (target_array <= interval.upper)
    return {
        key: float(hit[np.asarray([value == key for value in strata])].mean())
        for key in sorted(set(strata))
    }


def interval_coverage(interval: ConformalInterval, target: Iterable[float]) -> float:
    """Calculate overall empirical coverage alongside Mondrian-stratum coverage."""
    target_array = np.asarray(tuple(target), dtype=np.float64)
    if interval.lower.shape != target_array.shape or interval.upper.shape != target_array.shape:
        raise ValueError("intervals and targets must align")
    return float(np.mean((interval.lower <= target_array) & (target_array <= interval.upper)))
