#!/usr/bin/env python3
"""Ensure five complete real-pair crosscoder seeds at the two IRED working widths."""
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

from crossfit.activation_cache import load_cached_pair
from crossfit.crosscoder import crosscoder_diagnostics
from crossfit.crosscoder_training import load_or_fit_crosscoder
from crossfit.utils import REPO_ROOT, load_config


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("five-seed crosscoder completion requires the GPU pod")
    config = load_config(REPO_ROOT / "Code/config/step7.yaml")
    self_pair = json.loads((REPO_ROOT / "Reports/step6/self_pair_null.json").read_text())
    if not self_pair.get("passed"):
        raise RuntimeError("self-pair null failed; real-pair training is forbidden")
    evo_array, fit_array = load_cached_pair(REPO_ROOT / config["paths"]["cache"])
    evo = torch.as_tensor(evo_array, dtype=torch.float32, device="cuda")
    fit = torch.as_tensor(fit_array, dtype=torch.float32, device="cuda")
    del evo_array, fit_array
    started = time.monotonic()
    rows = []
    for width in map(int, config["diagnostics"]["selected_working_widths"]):
        for seed in map(int, config["training"]["seeds"]):
            path = REPO_ROOT / config["paths"]["checkpoints"] / f"w{width}_seed{seed}_nobias.pt"
            run_started = time.monotonic()
            model, trace, resumed = load_or_fit_crosscoder(
                path,
                dimension=int(config["dimension"]), width_multiplier=width,
                k=int(config["batch_topk"]["k"]), decoder_bias=bool(config["decoder_bias"]),
                evo=evo, fit=fit, training=config["training"], seed=seed, device="cuda",
            )
            diagnostics = crosscoder_diagnostics(
                model, evo, fit,
                batch_size=int(config["training"]["evaluation_batch_size"]), device="cuda",
            )
            rows.append({
                "dataset": str(config["dataset"]), "width_multiplier": width, "seed": seed,
                "features": diagnostics.features, "positions": len(evo),
                "nominal_k": int(config["batch_topk"]["k"]),
                "realized_l0": diagnostics.realized_l0,
                "dead_feature_fraction": diagnostics.dead_feature_fraction,
                "participation_ratio_effective_rank": diagnostics.participation_ratio,
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
    expected = len(config["diagnostics"]["selected_working_widths"]) * len(config["training"]["seeds"])
    if len(frame) != expected:
        raise AssertionError("five-seed crosscoder set is incomplete")
    report = REPO_ROOT / config["paths"]["report_dir"]
    frame.to_csv(report / "v17_taxonomy_crosscoder_runs.csv", index=False)
    manifest = {
        "status": "complete", "plan": "v17", "dataset": str(config["dataset"]),
        "widths": list(map(int, config["diagnostics"]["selected_working_widths"])),
        "seeds": list(map(int, config["training"]["seeds"])),
        "gpu_wall_hours": (time.monotonic() - started) / 3600.0,
        "partial_results_reported": False,
    }
    (report / "v17_taxonomy_crosscoder_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
