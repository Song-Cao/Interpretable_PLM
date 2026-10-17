"""Optimizer-step contracts shared by Step 2 predictor trainers."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OptimizerStepResult:
    """Diagnostics for one cross-assay accumulated optimizer step."""

    assay_ids: tuple[str, ...]
    mean_loss: float
    gradient_norm: float | None


def step_across_assays(
    optimizer: torch.optim.Optimizer,
    assay_losses: Iterable[tuple[str, torch.Tensor]],
    parameters: Iterable[torch.nn.Parameter],
    *,
    min_assays: int,
    gradient_clip_norm: float | None,
) -> OptimizerStepResult:
    """Average gradients from distinct single-assay batches, then step once."""
    losses = tuple(assay_losses)
    if min_assays < 2 or len(losses) < min_assays:
        raise ValueError("an optimizer step requires several single-assay batches")
    assay_ids = tuple(assay_id for assay_id, _ in losses)
    if len(set(assay_ids)) != len(assay_ids):
        raise ValueError("each accumulated batch must come from a distinct assay")
    if any(loss.ndim != 0 or not torch.isfinite(loss) for _, loss in losses):
        raise ValueError("each assay loss must be a finite scalar tensor")
    trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not trainable:
        raise ValueError("no trainable parameters were supplied")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient clip norm must be positive when configured")

    optimizer.zero_grad(set_to_none=True)
    divisor = float(len(losses))
    for _, loss in losses:
        (loss / divisor).backward()
    gradient_norm = None
    if gradient_clip_norm is not None:
        norm = torch.nn.utils.clip_grad_norm_(trainable, gradient_clip_norm)
        gradient_norm = float(norm.detach())
    optimizer.step()
    return OptimizerStepResult(
        assay_ids=assay_ids,
        mean_loss=float(torch.stack([loss.detach() for _, loss in losses]).mean()),
        gradient_norm=gradient_norm,
    )
