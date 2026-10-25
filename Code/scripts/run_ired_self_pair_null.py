#!/usr/bin/env python3
"""Train the mandatory IRED self-pair null and verify decoder symmetry."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.activation_cache import load_cached_pair
from crossfit.crosscoder import CrossCoder, crosscoder_diagnostics, decoder_delta, reconstruction_loss
from crossfit.utils import REPO_ROOT, load_config


def _train(model, left, right, config, device):
    train = config["training"]
    seed = int(train["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train["learning_rate"]),
        weight_decay=float(train["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    trace = []
    for epoch in range(int(train["epochs"])):
        losses = []
        order = rng.permutation(len(left))
        for start in range(0, len(left), int(train["batch_size"])):
            indices = order[start : start + int(train["batch_size"])]
            evo = torch.as_tensor(left[indices], dtype=torch.float32, device=device)
            fit = torch.as_tensor(right[indices], dtype=torch.float32, device=device)
            code, evo_hat, fit_hat = model(evo, fit)
            loss = reconstruction_loss(evo, fit, evo_hat, fit_hat)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        trace.append(float(np.mean(losses)))
    return trace


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("self-pair crosscoder requires the GPU pod")
    started = time.monotonic()
    config = load_config(REPO_ROOT / "Code/config/step6.yaml")
    report_path = REPO_ROOT / config["paths"]["report"]
    checkpoint_path = REPO_ROOT / config["paths"]["checkpoint"]
    if report_path.exists() or checkpoint_path.exists():
        raise FileExistsError("refusing to overwrite a self-pair result")
    evo, _ = load_cached_pair(REPO_ROOT / config["paths"]["cache"])
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = CrossCoder(
        int(config["dimension"]), int(config["width_multiplier"]),
        int(config["batch_topk"]["k"]),
    )
    trace = _train(model, evo, evo, config, "cuda")
    delta = decoder_delta(model).detach().cpu().numpy()
    maximum_deviation = float(np.max(np.abs(delta - 0.5)))
    tolerance = float(config["training"]["delta_tolerance"])
    passed = maximum_deviation <= tolerance
    diagnostics = crosscoder_diagnostics(
        model, evo, evo, batch_size=int(config["training"]["batch_size"]), device="cuda"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save({"state_dict": model.state_dict(), "config": config, "trace": trace}, temporary)
    os.replace(temporary, checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    result = {
        "dataset": str(config["dataset"]),
        "pair": "h_evo_h_evo",
        "width_multiplier": int(config["width_multiplier"]),
        "nominal_k": int(config["batch_topk"]["k"]),
        "k_reasoning": str(config["batch_topk"]["reasoning"]),
        "positions": len(evo),
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "maximum_absolute_delta_deviation_from_half": maximum_deviation,
        "numerical_tolerance": tolerance,
        "passed": passed,
        "realized_l0": diagnostics.realized_l0,
        "final_reconstruction_loss": trace[-1],
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
        "checkpoint_sha256": digest,
        "gpu_wall_hours": (time.monotonic() - started) / 3600.0,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("self-pair Delta is not 0.5 to the registered numerical tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
