"""Checkpointed training for aligned BatchTopK crosscoders."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

from .crosscoder import CrossCoder, reconstruction_loss


def fit_crosscoder(
    model: CrossCoder,
    evo: torch.Tensor,
    fit: torch.Tensor,
    training: dict,
    *,
    seed: int,
    device: str | torch.device,
) -> list[float]:
    """Fit one crosscoder and return its complete epoch-loss trace."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    trace: list[float] = []
    for epoch in range(int(training["epochs"])):
        order = torch.randperm(len(evo), generator=generator, device=device)
        losses = []
        for start in range(0, len(order), int(training["batch_size"])):
            indices = order[start : start + int(training["batch_size"])]
            left = evo.index_select(0, indices)
            right = fit.index_select(0, indices)
            _, left_hat, right_hat = model(left, right)
            loss = reconstruction_loss(left, right, left_hat, right_hat)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        trace.append(float(np.mean(losses)))
    return trace


def load_or_fit_crosscoder(
    path: str | Path,
    *,
    dimension: int,
    width_multiplier: int,
    k: int,
    decoder_bias: bool,
    evo: torch.Tensor,
    fit: torch.Tensor,
    training: dict,
    seed: int,
    device: str | torch.device,
) -> tuple[CrossCoder, list[float], bool]:
    """Load a complete matching checkpoint or train atomically from initialization."""
    path = Path(path)
    if path.exists():
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        expected = (int(width_multiplier), int(seed), bool(decoder_bias), int(k))
        observed = (
            int(artifact["width_multiplier"]), int(artifact["seed"]),
            bool(artifact["decoder_bias"]), int(artifact["nominal_k"]),
        )
        if observed != expected:
            raise AssertionError(f"checkpoint identity mismatch: {path}")
        model = CrossCoder(dimension, width_multiplier, k, decoder_bias=decoder_bias)
        model.load_state_dict(artifact["state_dict"])
        return model.to(device), list(artifact["trace"]), True

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = CrossCoder(dimension, width_multiplier, k, decoder_bias=decoder_bias)
    trace = fit_crosscoder(model, evo, fit, training, seed=seed, device=device)
    artifact = {
        "format": "crossfit_crosscoder_v1", "width_multiplier": int(width_multiplier),
        "seed": int(seed), "nominal_k": int(k), "decoder_bias": bool(decoder_bias),
        "state_dict": model.state_dict(), "trace": trace,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(artifact, temporary)
    os.replace(temporary, path)
    return model, trace, False
