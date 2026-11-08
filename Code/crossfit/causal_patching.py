"""Error-preserving sparse-feature patches inside an adapted representation."""
from __future__ import annotations

import torch

from .crosscoder import CrossCoder


@torch.inference_mode()
def error_preserving_fit_patch(
    model: CrossCoder,
    evo: torch.Tensor,
    fit: torch.Tensor,
    *,
    ablated_feature: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return patched fit states and codes while preserving the original fit residual."""
    if evo.shape != fit.shape or evo.ndim != 2:
        raise ValueError("patch inputs must be aligned position-by-channel matrices")
    code = model.encode(evo, fit)
    reconstruction = model.fit_decoder(code)
    residual = fit - reconstruction
    patched_code = code.clone()
    if ablated_feature is not None:
        if ablated_feature < 0 or ablated_feature >= model.features:
            raise IndexError("ablated feature is outside the dictionary")
        patched_code[:, ablated_feature] = 0.0
    return model.fit_decoder(patched_code) + residual, code


def standardized_patch_effect(
    patched_prediction: torch.Tensor,
    baseline_prediction: torch.Tensor,
    baseline_standard_deviation: float,
) -> torch.Tensor:
    """Express a prediction change in fixed baseline-prediction SD units."""
    if baseline_standard_deviation <= 0:
        raise ValueError("baseline prediction standard deviation must be positive")
    return (patched_prediction - baseline_prediction) / float(baseline_standard_deviation)
