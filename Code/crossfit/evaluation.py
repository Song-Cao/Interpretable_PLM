"""Rank aggregation and Mondrian conformal calibration for Step 2."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import lsqr


@dataclass(frozen=True)
class RankAggregationResult:
    """Anchored scores and least-squares solver diagnostics."""

    scores: np.ndarray
    residual_norm: float
    iterations: int


def aggregate_pairwise_scores(
    n_variants: int,
    pair_u: Iterable[int],
    pair_v: Iterable[int],
    predicted_difference: Iterable[float],
    *,
    wild_type_index: int,
    atol: float = 1e-10,
    btol: float = 1e-10,
    iteration_limit: int | None = None,
) -> RankAggregationResult:
    """Solve the anchored graph-Laplacian least-squares score problem."""
    if n_variants < 2 or not 0 <= wild_type_index < n_variants:
        raise ValueError("variant count and wild-type index are inconsistent")
    u = np.asarray(tuple(pair_u), dtype=np.int64)
    v = np.asarray(tuple(pair_v), dtype=np.int64)
    difference = np.asarray(tuple(predicted_difference), dtype=np.float64)
    if u.ndim != 1 or v.shape != u.shape or difference.shape != u.shape or len(u) == 0:
        raise ValueError("pair endpoints and predicted differences must be non-empty and aligned")
    if np.any(u < 0) or np.any(v < 0) or np.any(u >= n_variants) or np.any(v >= n_variants):
        raise ValueError("pair endpoint lies outside the variant range")
    if np.any(u == v) or not np.isfinite(difference).all():
        raise ValueError("pairs must join distinct variants and have finite differences")

    edge_rows = np.repeat(np.arange(len(u)), 2)
    incidence = sparse.coo_matrix(
        (np.tile((1.0, -1.0), len(u)), (edge_rows, np.column_stack((u, v)).ravel())),
        shape=(len(u), n_variants),
    ).tocsr()
    adjacency = sparse.coo_matrix(
        (np.ones(2 * len(u)), (np.concatenate((u, v)), np.concatenate((v, u)))),
        shape=(n_variants, n_variants),
    ).tocsr()
    n_components, labels = csgraph.connected_components(adjacency, directed=False)
    if n_components != 1:
        sizes = np.bincount(labels).tolist()
        raise ValueError(f"comparison graph must be connected to the wild type; components={sizes}")

    free = np.arange(n_variants) != wild_type_index
    solution = lsqr(
        incidence[:, free],
        difference,
        atol=atol,
        btol=btol,
        iter_lim=iteration_limit,
    )
    scores = np.zeros(n_variants, dtype=np.float64)
    scores[free] = solution[0]
    return RankAggregationResult(scores, float(solution[3]), int(solution[2]))


@dataclass(frozen=True)
class ConformalInterval:
    """Lower and upper prediction bounds from Mondrian calibration."""

    lower: np.ndarray
    upper: np.ndarray


class MondrianConformal:
    """Split conformal intervals stratified by mutation count and novelty."""

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
        return [(int(c), bool(novel)) for c, novel in zip(count, unseen)]

    def fit(
        self,
        prediction: Iterable[float],
        target: Iterable[float],
        mutation_count: Iterable[int],
        unseen_position: Iterable[bool],
    ) -> "MondrianConformal":
        """Fit finite-sample residual quantiles independently in every stratum."""
        prediction_array = np.asarray(tuple(prediction), dtype=np.float64)
        target_array = np.asarray(tuple(target), dtype=np.float64)
        strata = self._strata(mutation_count, unseen_position)
        if prediction_array.ndim != 1 or target_array.shape != prediction_array.shape:
            raise ValueError("predictions and targets must be aligned one-dimensional arrays")
        if len(strata) != len(prediction_array) or not np.isfinite(prediction_array).all() \
                or not np.isfinite(target_array).all():
            raise ValueError("calibration inputs must be aligned and finite")
        if len(prediction_array) == 0:
            raise ValueError("calibration data cannot be empty")

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
        """Return intervals, rejecting strata absent from calibration."""
        if not self.quantiles_:
            raise RuntimeError("fit must be called before interval prediction")
        prediction_array = np.asarray(tuple(prediction), dtype=np.float64)
        strata = self._strata(mutation_count, unseen_position)
        if prediction_array.ndim != 1 or len(strata) != len(prediction_array):
            raise ValueError("prediction and stratum metadata must be aligned")
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
    """Calculate empirical interval coverage separately for each stratum."""
    target_array = np.asarray(tuple(target), dtype=np.float64)
    strata = MondrianConformal._strata(mutation_count, unseen_position)
    if interval.lower.shape != target_array.shape or interval.upper.shape != target_array.shape:
        raise ValueError("intervals and targets must be aligned")
    hit = (interval.lower <= target_array) & (target_array <= interval.upper)
    output: dict[tuple[int, bool], float] = {}
    for key in sorted(set(strata)):
        selected = np.asarray([value == key for value in strata])
        output[key] = float(hit[selected].mean())
    return output
