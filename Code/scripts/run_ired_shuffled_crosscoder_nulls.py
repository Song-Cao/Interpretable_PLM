#!/usr/bin/env python3
"""Train IRED shuffled-adapter crosscoders used only for null calibration."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.activation_cache import load_cached_model
from crossfit.crosscoder import crosscoder_diagnostics
from crossfit.crosscoder_training import load_or_fit_crosscoder
from crossfit.utils import REPO_ROOT, load_config


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("shuffled crosscoder null training requires the GPU pod")
    started = time.monotonic()
    config = load_config(REPO_ROOT / "Code/config/step6.yaml")
    step7 = load_config(REPO_ROOT / "Code/config/step7.yaml")
    widths = list(map(int, step7["diagnostics"]["selected_working_widths"]))
    seeds = list(map(int, config["calibration"]["shuffled_crosscoder_seeds"]))
    evo_array = load_cached_model(REPO_ROOT / config["paths"]["step4_cache"], "evo")
    shuffled_array = load_cached_model(REPO_ROOT / config["paths"]["shuffled_cache"], "shuffled")
    if evo_array.shape != shuffled_array.shape:
        raise AssertionError("evo and shuffled activation pools are not aligned")
    evo = torch.as_tensor(evo_array, dtype=torch.float32, device="cuda")
    shuffled = torch.as_tensor(shuffled_array, dtype=torch.float32, device="cuda")
    rows = []
    for width in widths:
        for seed in seeds:
            path = (
                REPO_ROOT / config["paths"]["shuffled_checkpoints"]
                / f"w{width}_seed{seed}_nobias.pt"
            )
            run_started = time.monotonic()
            model, trace, resumed = load_or_fit_crosscoder(
                path, dimension=int(config["dimension"]), width_multiplier=width,
                k=int(config["batch_topk"]["k"]), decoder_bias=bool(config["decoder_bias"]),
                evo=evo, fit=shuffled, training=config["training"], seed=seed, device="cuda",
            )
            diagnostics = crosscoder_diagnostics(
                model, evo, shuffled,
                batch_size=int(step7["training"]["evaluation_batch_size"]), device="cuda",
            )
            rows.append({
                "dataset": str(config["dataset"]), "null": "shuffled_label",
                "width_multiplier": width, "seed": seed, "features": model.features,
                "positions": len(evo), "nominal_k": model.k,
                "dead_feature_fraction": diagnostics.dead_feature_fraction,
                "participation_ratio_effective_rank": diagnostics.participation_ratio,
                "evo_fraction_variance_explained": diagnostics.evo_fraction_variance_explained,
                "shuffled_fraction_variance_explained": diagnostics.fit_fraction_variance_explained,
                "realized_l0": diagnostics.realized_l0,
                "final_reconstruction_loss": trace[-1], "epochs": len(trace),
                "checkpoint": str(path.relative_to(REPO_ROOT)),
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "loaded_complete_checkpoint": resumed,
                "run_wall_hours": (time.monotonic() - run_started) / 3600.0,
            })
            print(json.dumps(rows[-1]), flush=True)
            del model
            torch.cuda.empty_cache()
    expected = len(widths) * len(seeds)
    if len(rows) != expected:
        raise AssertionError("null crosscoder sweep incomplete; refusing partial report")
    output = REPO_ROOT / config["paths"]["calibration_report_dir"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "shuffled_crosscoder_runs.csv", index=False)
    manifest = {
        "status": "complete", "dataset": str(config["dataset"]),
        "width_multipliers": widths, "seeds": seeds,
        "gpu_wall_hours": (time.monotonic() - started) / 3600.0,
        "partial_results_reported": False,
    }
    (output / "shuffled_crosscoder_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
