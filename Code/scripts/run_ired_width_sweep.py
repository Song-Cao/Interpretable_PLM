#!/usr/bin/env python3
"""Train the fixed-k IRED crosscoder width sweep after the self-pair check."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.activation_cache import load_cached_pair
from crossfit.crosscoder import CrossCoder, crosscoder_diagnostics, reconstruction_loss
from crossfit.utils import REPO_ROOT, load_config


def _fit(model, evo, fit, train, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train["learning_rate"]),
        weight_decay=float(train["weight_decay"]),
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    trace = []
    for epoch in range(int(train["epochs"])):
        order = torch.randperm(len(evo), generator=generator, device=device)
        losses = []
        for start in range(0, len(order), int(train["batch_size"])):
            indices = order[start : start + int(train["batch_size"])]
            left = evo.index_select(0, indices)
            right = fit.index_select(0, indices)
            code, left_hat, right_hat = model(left, right)
            loss = reconstruction_loss(left, right, left_hat, right_hat)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        trace.append(float(np.mean(losses)))
        print(json.dumps({"width": model.width_multiplier, "seed": seed, "epoch": epoch + 1, "loss": trace[-1]}), flush=True)
    return trace


def _complete_checkpoint(path, config, width, seed, evo, fit, device):
    if path.exists():
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact["width_multiplier"] != width or artifact["seed"] != seed:
            raise AssertionError(f"checkpoint identity mismatch: {path}")
        model = CrossCoder(int(config["dimension"]), width, int(config["batch_topk"]["k"]))
        model.load_state_dict(artifact["state_dict"])
        model.to(device)
        return model, list(artifact["trace"]), True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = CrossCoder(int(config["dimension"]), width, int(config["batch_topk"]["k"]))
    trace = _fit(model, evo, fit, config["training"], seed, device)
    artifact = {
        "format": "crossfit_crosscoder_v1", "width_multiplier": width, "seed": seed,
        "nominal_k": int(config["batch_topk"]["k"]), "state_dict": model.state_dict(),
        "trace": trace,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(artifact, temporary)
    os.replace(temporary, path)
    return model, trace, False


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("crosscoder width sweep requires the GPU pod")
    started = time.monotonic()
    config = load_config(REPO_ROOT / "Code/config/step7.yaml")
    self_pair = json.loads((REPO_ROOT / "Reports/step6/self_pair_null.json").read_text())
    if not self_pair.get("passed"):
        raise RuntimeError("the self-pair null did not pass; real-pair training is forbidden")
    evo_array, fit_array = load_cached_pair(REPO_ROOT / config["paths"]["cache"])
    # The 1.99-GB float32 pair fits comfortably on the 80-GB A100. Keeping it
    # resident avoids making a host-side fancy-indexed copy for every batch.
    evo = torch.as_tensor(evo_array, dtype=torch.float32, device="cuda")
    fit = torch.as_tensor(fit_array, dtype=torch.float32, device="cuda")
    del evo_array, fit_array
    rows = []
    for width in map(int, config["width_multipliers"]):
        for seed in map(int, config["training"]["seeds"]):
            path = REPO_ROOT / config["paths"]["checkpoints"] / f"w{width}_seed{seed}.pt"
            run_started = time.monotonic()
            model, trace, resumed = _complete_checkpoint(path, config, width, seed, evo, fit, "cuda")
            diagnostics = crosscoder_diagnostics(
                model, evo, fit, batch_size=int(config["training"]["evaluation_batch_size"]),
                device="cuda",
            )
            rows.append({
                "dataset": str(config["dataset"]), "width_multiplier": width, "seed": seed,
                "features": diagnostics.features, "positions": len(evo),
                "samples_per_feature": len(evo) / diagnostics.features,
                "nominal_k": int(config["batch_topk"]["k"]),
                "realized_l0": diagnostics.realized_l0,
                "dead_feature_fraction": diagnostics.dead_feature_fraction,
                "active_features": diagnostics.active_features,
                "participation_ratio_effective_rank": diagnostics.participation_ratio,
                "participation_ratio_definition": "trace(cov(z))^2 / trace(cov(z)^2)",
                "evo_fraction_variance_explained": diagnostics.evo_fraction_variance_explained,
                "fit_fraction_variance_explained": diagnostics.fit_fraction_variance_explained,
                "final_reconstruction_loss": trace[-1], "epochs": len(trace),
                "checkpoint": str(path.relative_to(REPO_ROOT)),
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "loaded_complete_checkpoint": resumed,
                "run_wall_hours": (time.monotonic() - run_started) / 3600.0,
            })
            print(json.dumps(rows[-1]), flush=True)
            del model
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    expected = len(config["width_multipliers"]) * len(config["training"]["seeds"])
    if len(frame) != expected:
        raise AssertionError("width sweep is incomplete; refusing to report partial results")
    report_dir = REPO_ROOT / config["paths"]["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report_dir / "width_sweep_runs.csv", index=False)
    metrics = [
        "dead_feature_fraction", "participation_ratio_effective_rank",
        "evo_fraction_variance_explained", "fit_fraction_variance_explained", "realized_l0",
    ]
    summary = frame.groupby(["width_multiplier", "features", "positions", "samples_per_feature", "nominal_k"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(report_dir / "width_sweep_summary.csv", index=False)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.errorbar(
        summary["samples_per_feature"], summary["dead_feature_fraction_mean"],
        yerr=summary["dead_feature_fraction_std"], marker="o", capsize=3,
    )
    for row in summary.itertuples():
        axis.annotate(f"w={row.width_multiplier}", (row.samples_per_feature, row.dead_feature_fraction_mean))
    axis.set(xlabel="position-samples per feature", ylabel="dead-feature fraction")
    axis.invert_xaxis()
    figure.tight_layout()
    figure.savefig(report_dir / "dead_fraction_by_samples_per_feature.png", dpi=180)
    plt.close(figure)
    run_manifest = {
        "status": "complete", "dataset": str(config["dataset"]),
        "width_multipliers": list(map(int, config["width_multipliers"])),
        "nominal_k": int(config["batch_topk"]["k"]),
        "k_reasoning": str(config["batch_topk"]["reasoning"]),
        "seeds": list(map(int, config["training"]["seeds"])),
        "gpu_wall_hours": (time.monotonic() - started) / 3600.0,
        "partial_results_reported": False,
    }
    (report_dir / "width_sweep_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
