"""Composite objective for direct per-variant fitness prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


_CENSOR_CODES = {"none": 0, "floor": 1, "ceiling": 2}


def encode_censoring(labels: Sequence[str], device: torch.device | None = None) -> torch.Tensor:
    """Encode none, floor, and ceiling labels as integer tensor values."""
    try:
        values = [_CENSOR_CODES[str(label)] for label in labels]
    except KeyError as error:
        raise ValueError(f"unknown censoring label: {error.args[0]}") from error
    return torch.tensor(values, dtype=torch.long, device=device)


def in_batch_rank_indices(target: torch.Tensor, margin: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return upper-triangle batch pairs whose observed values clear the fixed margin."""
    target = torch.as_tensor(target)
    if target.ndim != 1 or not torch.isfinite(target).all():
        raise ValueError("targets must be a finite one-dimensional tensor")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    indices = torch.triu_indices(len(target), len(target), offset=1, device=target.device)
    keep = (target[indices[0]] - target[indices[1]]).abs() > margin
    return indices[0, keep], indices[1, keep]


class LearnedResidualScale(nn.Module):
    """One positive residual scale learned jointly with one dataset predictor."""

    def __init__(self, initial: float, floor: float) -> None:
        super().__init__()
        if initial <= floor or floor <= 0:
            raise ValueError("residual scale requires initial > floor > 0")
        self.floor = float(floor)
        raw = torch.log(torch.expm1(torch.tensor(float(initial - floor))))
        self.raw_scale = nn.Parameter(raw)

    def forward(self) -> torch.Tensor:
        return F.softplus(self.raw_scale) + self.floor


@dataclass(frozen=True)
class CompositeLossOutput:
    """Weighted objective and unweighted component diagnostics."""

    total: torch.Tensor
    ranking: torch.Tensor
    magnitude: torch.Tensor
    censoring: torch.Tensor
    regularization: torch.Tensor
    n_rank_pairs: int
    n_uncensored: int
    n_floor_censored: int
    n_ceiling_censored: int


class CompositeFitnessLoss(nn.Module):
    """Combine ranking, magnitude, Tobit likelihood, and L2 regularization."""

    def __init__(
        self,
        *,
        margin_sd: float,
        temperature: float,
        magnitude_mode: str,
        huber_delta: float,
        residual_scale_initial: float,
        residual_scale_floor: float,
        rank_weight: float,
        magnitude_weight: float,
        censoring_weight: float,
        regularization_weight: float,
    ) -> None:
        super().__init__()
        if margin_sd < 0 or temperature <= 0 or huber_delta <= 0:
            raise ValueError("margin must be non-negative and temperature/delta positive")
        if magnitude_mode not in {"huber", "squared"}:
            raise ValueError("magnitude_mode must be 'huber' or 'squared'")
        weights = (rank_weight, magnitude_weight, censoring_weight, regularization_weight)
        if any(weight < 0 for weight in weights):
            raise ValueError("objective weights must be non-negative")
        self.margin_sd = float(margin_sd)
        self.temperature = float(temperature)
        self.magnitude_mode = magnitude_mode
        self.huber_delta = float(huber_delta)
        self.rank_weight = float(rank_weight)
        self.magnitude_weight = float(magnitude_weight)
        self.censoring_weight = float(censoring_weight)
        self.regularization_weight = float(regularization_weight)
        self.residual_scale = LearnedResidualScale(residual_scale_initial, residual_scale_floor)

    @staticmethod
    def _l2(parameters: Iterable[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        values = [parameter.square().sum() for parameter in parameters if parameter.requires_grad]
        return torch.stack(values).sum() if values else reference.new_zeros(())

    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        censoring: Sequence[str] | torch.Tensor,
        *,
        regularized_parameters: Iterable[torch.Tensor] = (),
    ) -> CompositeLossOutput:
        """Evaluate the single Step 2 objective on one dataset batch."""
        scores = torch.as_tensor(scores)
        targets = torch.as_tensor(targets, dtype=scores.dtype, device=scores.device)
        if scores.ndim != 1 or targets.shape != scores.shape or len(scores) == 0:
            raise ValueError("scores and targets must be aligned non-empty vectors")
        if not torch.isfinite(scores).all() or not torch.isfinite(targets).all():
            raise ValueError("scores and targets must be finite")
        codes = (
            encode_censoring(censoring, scores.device)
            if not isinstance(censoring, torch.Tensor)
            else censoring.to(device=scores.device, dtype=torch.long)
        )
        allowed = torch.tensor([0, 1, 2], device=scores.device)
        if codes.shape != scores.shape or not torch.isin(codes, allowed).all():
            raise ValueError("censoring codes must align and belong to {0, 1, 2}")

        left, right = in_batch_rank_indices(targets, self.margin_sd)
        if len(left):
            direction = torch.sign(targets[left] - targets[right])
            predicted_difference = scores[left] - scores[right]
            ranking = F.softplus(-direction * predicted_difference / self.temperature).mean()
        else:
            ranking = scores.new_zeros(())

        uncensored = codes == 0
        if uncensored.any():
            if self.magnitude_mode == "huber":
                magnitude = F.huber_loss(
                    scores[uncensored], targets[uncensored],
                    delta=self.huber_delta, reduction="mean",
                )
            else:
                magnitude = F.mse_loss(scores[uncensored], targets[uncensored])
        else:
            magnitude = scores.new_zeros(())

        sigma = self.residual_scale()
        terms: list[torch.Tensor] = []
        if uncensored.any():
            residual = (targets[uncensored] - scores[uncensored]) / sigma
            normalizer = 0.5 * torch.log(scores.new_tensor(2.0 * torch.pi))
            terms.append(0.5 * residual.square() + torch.log(sigma) + normalizer)
        floor = codes == 1
        if floor.any():
            terms.append(-torch.special.log_ndtr((targets[floor] - scores[floor]) / sigma))
        ceiling = codes == 2
        if ceiling.any():
            terms.append(-torch.special.log_ndtr((scores[ceiling] - targets[ceiling]) / sigma))
        censor_loss = torch.cat([term.reshape(-1) for term in terms]).mean()

        regularization = self._l2(regularized_parameters, scores)
        total = (
            self.rank_weight * ranking
            + self.magnitude_weight * magnitude
            + self.censoring_weight * censor_loss
            + self.regularization_weight * regularization
        )
        return CompositeLossOutput(
            total=total,
            ranking=ranking,
            magnitude=magnitude,
            censoring=censor_loss,
            regularization=regularization,
            n_rank_pairs=len(left),
            n_uncensored=int(uncensored.sum()),
            n_floor_censored=int(floor.sum()),
            n_ceiling_censored=int(ceiling.sum()),
        )
