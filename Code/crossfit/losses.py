"""Pair construction and the Step 2 composite fitness objective."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

CENSOR_NONE = 0
CENSOR_FLOOR = 1
CENSOR_CEILING = 2
_CENSOR_CODES = {
    "none": CENSOR_NONE,
    "floor": CENSOR_FLOOR,
    "ceiling": CENSOR_CEILING,
}


@dataclass(frozen=True)
class PairBatch:
    """A batch of within-assay, distance-filtered, margin-gated pairs."""

    u: torch.Tensor
    v: torch.Tensor
    assay_index: torch.Tensor
    target_difference: torch.Tensor
    order_sign: torch.Tensor
    censor_sign: torch.Tensor
    sigma_u: torch.Tensor
    sigma_v: torch.Tensor
    censor_u: torch.Tensor
    censor_v: torch.Tensor

    def __len__(self) -> int:
        return int(self.u.numel())


@dataclass(frozen=True)
class CompositeLossOutput:
    """The scalar objective and its auditable component values."""

    total: torch.Tensor
    rank: torch.Tensor
    magnitude: torch.Tensor
    censoring: torch.Tensor
    regularization: torch.Tensor
    n_pairs: int
    n_censor_constraints: int


def hamming_distance(left: str, right: str) -> int:
    """Return the Hamming distance between equal-length amino-acid sequences."""
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal-length sequences")
    return sum(a != b for a, b in zip(left, right))


def encode_censoring(censored: Sequence[str], device: torch.device | None = None) -> torch.Tensor:
    """Map the curated censoring labels to integer codes."""
    unknown = sorted(set(censored) - set(_CENSOR_CODES))
    if unknown:
        raise ValueError(f"unknown censoring labels: {unknown}")
    return torch.tensor([_CENSOR_CODES[value] for value in censored], dtype=torch.int8, device=device)


def censor_constraint_sign(censor_u: int, censor_v: int) -> int:
    """Return +1 for an upper-bound constraint, -1 for lower, and 0 for none.

    Equal-bound double-censored pairs carry no difference constraint. Opposite
    bounds reinforce one another and retain the same sign as either bound alone.
    """
    valid = {CENSOR_NONE, CENSOR_FLOOR, CENSOR_CEILING}
    if censor_u not in valid or censor_v not in valid:
        raise ValueError("unknown censoring code")
    if censor_u == censor_v:
        return 0
    upper = censor_u == CENSOR_FLOOR or censor_v == CENSOR_CEILING
    lower = censor_u == CENSOR_CEILING or censor_v == CENSOR_FLOOR
    if upper == lower:
        raise ValueError("incompatible censoring constraint")
    return 1 if upper else -1


def construct_pairs(
    targets: torch.Tensor,
    assay_index: torch.Tensor,
    sequences: Sequence[str],
    censored: Sequence[str],
    sigma_v: torch.Tensor,
    *,
    margin_sd: float,
    max_hamming_distance: int,
) -> PairBatch:
    """Construct all valid pairs within a batch.

    Pairs must share an assay, lie within the configured Hamming distance, and
    have an observed target difference whose magnitude is strictly greater than
    the within-assay-SD margin. Rows are never removed merely for being censored.
    """
    targets = torch.as_tensor(targets)
    assay_index = torch.as_tensor(assay_index, device=targets.device)
    sigma_v = torch.as_tensor(sigma_v, dtype=targets.dtype, device=targets.device)
    n = int(targets.numel())
    if targets.ndim != 1 or assay_index.shape != targets.shape or sigma_v.shape != targets.shape:
        raise ValueError("targets, assay_index, and sigma_v must be aligned one-dimensional tensors")
    if len(sequences) != n or len(censored) != n:
        raise ValueError("sequence and censoring labels must align with targets")
    if margin_sd < 0 or max_hamming_distance < 0:
        raise ValueError("pair thresholds must be non-negative")
    if not torch.isfinite(targets).all():
        raise ValueError("pair targets must be finite")
    finite_sigma = torch.isfinite(sigma_v)
    if torch.any(sigma_v[finite_sigma] <= 0):
        raise ValueError("available sigma_v values must be positive")

    censor_codes = encode_censoring(censored, targets.device)
    kept_u: list[int] = []
    kept_v: list[int] = []
    censor_signs: list[int] = []
    for u, v in itertools.combinations(range(n), 2):
        if int(assay_index[u]) != int(assay_index[v]):
            continue
        if hamming_distance(sequences[u], sequences[v]) > max_hamming_distance:
            continue
        cu, cv = int(censor_codes[u]), int(censor_codes[v])
        if cu == cv and cu != CENSOR_NONE:
            continue
        if abs(float(targets[u] - targets[v])) <= margin_sd:
            continue
        kept_u.append(u)
        kept_v.append(v)
        censor_signs.append(censor_constraint_sign(cu, cv))

    u_tensor = torch.tensor(kept_u, dtype=torch.long, device=targets.device)
    v_tensor = torch.tensor(kept_v, dtype=torch.long, device=targets.device)
    difference = targets[u_tensor] - targets[v_tensor]
    return PairBatch(
        u=u_tensor,
        v=v_tensor,
        assay_index=assay_index[u_tensor].to(dtype=torch.long),
        target_difference=difference,
        order_sign=torch.sign(difference),
        censor_sign=torch.tensor(censor_signs, dtype=torch.int8, device=targets.device),
        sigma_u=sigma_v[u_tensor],
        sigma_v=sigma_v[v_tensor],
        censor_u=censor_codes[u_tensor],
        censor_v=censor_codes[v_tensor],
    )


class LearnedResidualScale(nn.Module):
    """A positive learned residual scale for each assay."""

    def __init__(self, n_assays: int, initial: float, floor: float) -> None:
        super().__init__()
        if n_assays < 1:
            raise ValueError("n_assays must be positive")
        if floor <= 0 or initial <= floor:
            raise ValueError("residual-scale initial value must exceed its positive floor")
        unconstrained = math.log(math.expm1(initial - floor))
        self.raw_scale = nn.Parameter(torch.full((n_assays,), unconstrained))
        self.floor = float(floor)

    def forward(self, assay_index: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.raw_scale[assay_index]) + self.floor


class CompositeFitnessLoss(nn.Module):
    """Compute the single ranking + magnitude + censoring + L1 objective.

    Optimizer weight decay remains an optimizer setting. ``regularization`` is
    the separately weighted L1 penalty on readout parameters specified by the
    plan.
    """

    def __init__(
        self,
        *,
        n_assays: int,
        prediction_mode: str,
        temperature: float,
        magnitude_mode: str,
        huber_delta: float,
        residual_scale_initial: float,
        residual_scale_floor: float,
        rank_weight: float,
        magnitude_weight: float,
        censoring_weight: float,
        regularization_weight: float,
        use_sigma_magnitude_weights: bool = False,
        min_sigma_coverage: float | None = None,
        sigma_coverage_by_assay: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if prediction_mode not in {"difference", "score"}:
            raise ValueError("prediction_mode must be 'difference' or 'score'")
        if magnitude_mode not in {"huber", "squared"}:
            raise ValueError("magnitude_mode must be 'huber' or 'squared'")
        if temperature <= 0 or huber_delta <= 0:
            raise ValueError("temperature and Huber delta must be positive")
        weights = (rank_weight, magnitude_weight, censoring_weight, regularization_weight)
        if any(weight < 0 for weight in weights):
            raise ValueError("loss weights must be non-negative")
        if use_sigma_magnitude_weights and min_sigma_coverage is None:
            raise ValueError("a high sigma-coverage threshold is required when weighting is enabled")
        coverage_threshold = 0.0 if min_sigma_coverage is None else float(min_sigma_coverage)
        if not 0.0 <= coverage_threshold <= 1.0:
            raise ValueError("minimum sigma coverage must lie in [0, 1]")
        if use_sigma_magnitude_weights and sigma_coverage_by_assay is None:
            raise ValueError("sigma coverage is required when magnitude weighting is enabled")
        if sigma_coverage_by_assay is None:
            coverage = torch.empty(0, dtype=torch.float32)
        else:
            coverage = torch.as_tensor(sigma_coverage_by_assay, dtype=torch.float32)
            if coverage.shape != (n_assays,):
                raise ValueError("sigma coverage must contain one value per assay")
            if not torch.isfinite(coverage).all() or torch.any((coverage < 0) | (coverage > 1)):
                raise ValueError("sigma coverage values must be finite and lie in [0, 1]")
        self.prediction_mode = prediction_mode
        self.temperature = float(temperature)
        self.magnitude_mode = magnitude_mode
        self.huber_delta = float(huber_delta)
        self.rank_weight = float(rank_weight)
        self.magnitude_weight = float(magnitude_weight)
        self.censoring_weight = float(censoring_weight)
        self.regularization_weight = float(regularization_weight)
        self.use_sigma_magnitude_weights = bool(use_sigma_magnitude_weights)
        self.min_sigma_coverage = coverage_threshold
        self.register_buffer("sigma_coverage_by_assay", coverage)
        self.residual_scale = LearnedResidualScale(
            n_assays, residual_scale_initial, residual_scale_floor
        )

    def _magnitude_term(self, predicted_difference: torch.Tensor, pairs: PairBatch) -> torch.Tensor:
        if self.magnitude_mode == "huber":
            per_pair = F.huber_loss(
                predicted_difference,
                pairs.target_difference,
                reduction="none",
                delta=self.huber_delta,
            )
        else:
            per_pair = (predicted_difference - pairs.target_difference).square()

        pair_weight = torch.ones_like(per_pair)
        if self.use_sigma_magnitude_weights:
            used_assays = torch.unique(pairs.assay_index)
            coverage = self.sigma_coverage_by_assay[used_assays]
            if torch.any(coverage < self.min_sigma_coverage):
                failed = used_assays[coverage < self.min_sigma_coverage].tolist()
                raise ValueError(
                    "sigma magnitude weighting requires configured coverage in every used "
                    f"assay; below threshold for assay indices {failed}"
                )
            have_variance = (
                torch.isfinite(pairs.sigma_u)
                & torch.isfinite(pairs.sigma_v)
                & (pairs.sigma_u > 0)
                & (pairs.sigma_v > 0)
            )
            raw_weight = torch.ones_like(per_pair)
            raw_weight[have_variance] = 1.0 / (
                pairs.sigma_u[have_variance].square() + pairs.sigma_v[have_variance].square()
            )
            for assay in used_assays:
                selected = pairs.assay_index == assay
                pair_weight[selected] = raw_weight[selected] / raw_weight[selected].mean()
        return (pair_weight * per_pair).sum()

    def _difference_censoring(
        self, predicted_difference: torch.Tensor, pairs: PairBatch
    ) -> tuple[torch.Tensor, int]:
        constrained = pairs.censor_sign != 0
        if not torch.any(constrained):
            return predicted_difference.sum() * 0.0, 0
        scale = self.residual_scale(pairs.assay_index[constrained])
        sign = pairs.censor_sign[constrained].to(dtype=predicted_difference.dtype)
        z = sign * (
            pairs.target_difference[constrained] - predicted_difference[constrained]
        ) / scale
        return -torch.special.log_ndtr(z).sum(), int(constrained.sum())

    def _score_censoring(
        self,
        scores: torch.Tensor,
        pairs: PairBatch,
        variant_targets: torch.Tensor,
        variant_assay_index: torch.Tensor,
        variant_censoring: Sequence[str],
    ) -> tuple[torch.Tensor, int]:
        if scores.ndim != 1 or variant_targets.shape != scores.shape:
            raise ValueError("scores and variant targets must be aligned one-dimensional tensors")
        if variant_assay_index.shape != scores.shape or len(variant_censoring) != scores.numel():
            raise ValueError("score-head censoring metadata must align with scores")
        used = torch.unique(torch.cat((pairs.u, pairs.v)))
        mu = scores[used]
        observed = variant_targets[used]
        assay = variant_assay_index[used].to(dtype=torch.long)
        scale = self.residual_scale(assay)
        codes = encode_censoring(variant_censoring, scores.device)[used]

        per_variant = torch.empty_like(mu)
        uncensored = codes == CENSOR_NONE
        floor = codes == CENSOR_FLOOR
        ceiling = codes == CENSOR_CEILING
        standardized = (observed - mu) / scale
        per_variant[uncensored] = (
            torch.log(scale[uncensored])
            + 0.5 * standardized[uncensored].square()
            + 0.5 * math.log(2.0 * math.pi)
        )
        per_variant[floor] = -torch.special.log_ndtr(standardized[floor])
        per_variant[ceiling] = -torch.special.log_ndtr(-standardized[ceiling])
        return per_variant.sum(), int(floor.sum() + ceiling.sum())

    def forward(
        self,
        prediction: torch.Tensor,
        pairs: PairBatch,
        *,
        readout_parameters: Iterable[torch.Tensor] = (),
        variant_targets: torch.Tensor | None = None,
        variant_assay_index: torch.Tensor | None = None,
        variant_censoring: Sequence[str] | None = None,
    ) -> CompositeLossOutput:
        """Return one scalar objective plus component diagnostics."""
        if len(pairs) == 0:
            raise ValueError("the margin gate produced no training pairs")
        prediction = torch.as_tensor(prediction)
        if self.prediction_mode == "difference":
            if prediction.ndim != 1 or prediction.numel() != len(pairs):
                raise ValueError("difference predictions must have one value per pair")
            predicted_difference = prediction
        else:
            if prediction.ndim != 1:
                raise ValueError("score predictions must be one-dimensional")
            predicted_difference = prediction[pairs.u] - prediction[pairs.v]

        rank = F.softplus(
            -pairs.order_sign.to(dtype=predicted_difference.dtype)
            * predicted_difference
            / self.temperature
        ).sum()
        magnitude = self._magnitude_term(predicted_difference, pairs)

        if self.prediction_mode == "difference":
            censoring, n_constraints = self._difference_censoring(predicted_difference, pairs)
        else:
            if variant_targets is None or variant_assay_index is None or variant_censoring is None:
                raise ValueError("score mode requires variant targets, assay indices, and censoring")
            censoring, n_constraints = self._score_censoring(
                prediction,
                pairs,
                torch.as_tensor(variant_targets, dtype=prediction.dtype, device=prediction.device),
                torch.as_tensor(variant_assay_index, device=prediction.device),
                variant_censoring,
            )

        parameters = tuple(readout_parameters)
        regularization = (
            torch.stack([parameter.abs().sum() for parameter in parameters]).sum()
            if parameters
            else predicted_difference.sum() * 0.0
        )
        total = (
            self.rank_weight * rank
            + self.magnitude_weight * magnitude
            + self.censoring_weight * censoring
            + self.regularization_weight * regularization
        )
        return CompositeLossOutput(
            total=total,
            rank=rank,
            magnitude=magnitude,
            censoring=censoring,
            regularization=regularization,
            n_pairs=len(pairs),
            n_censor_constraints=n_constraints,
        )
