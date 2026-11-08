#!/usr/bin/env python3
"""Complete the w=1 matched-drift null replication needed for empirical BH resolution."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.activation_cache import load_cached_model
from crossfit.crosscoder import crosscoder_diagnostics
from crossfit.crosscoder_training import load_or_fit_crosscoder
from crossfit.utils import REPO_ROOT, load_config


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("null-resolution crosscoders require the GPU pod")
    step6 = load_config(REPO_ROOT / "Code/config/step6.yaml")
    step7 = load_config(REPO_ROOT / "Code/config/step7.yaml")
    audit = step6["calibration"]["resolution_audit"]
    width = int(audit["width_multiplier"])
    seeds = list(map(int, audit["seeds"]))
    cache = REPO_ROOT / step6["paths"]["drift_matched_cache"]
    evo_array = load_cached_model(cache, "evo")
    matched_array = load_cached_model(cache, "drift_matched")
    if evo_array.shape != matched_array.shape:
        raise AssertionError("drift-matched null activation arrays are not aligned")
    evo = torch.as_tensor(evo_array, dtype=torch.float32, device="cuda")
    matched = torch.as_tensor(matched_array, dtype=torch.float32, device="cuda")
    started = time.monotonic()
    rows = []
    destination = REPO_ROOT / step6["paths"]["drift_matched_crosscoder_checkpoints"]
    for seed in seeds:
        path = destination / f"w{width}_seed{seed}_nobias.pt"
        run_started = time.monotonic()
        model, trace, resumed = load_or_fit_crosscoder(
            path, dimension=int(step6["dimension"]), width_multiplier=width,
            k=int(step6["batch_topk"]["k"]), decoder_bias=bool(step6["decoder_bias"]),
            evo=evo, fit=matched, training=step7["training"], seed=seed, device="cuda",
        )
        diagnostics = crosscoder_diagnostics(
            model, evo, matched,
            batch_size=int(step7["training"]["evaluation_batch_size"]), device="cuda",
        )
        row = {
            "dataset": "IRED", "width_multiplier": width, "seed": seed,
            "features": diagnostics.features, "active_features": diagnostics.active_features,
            "dead_feature_fraction": diagnostics.dead_feature_fraction,
            "participation_ratio_effective_rank": diagnostics.participation_ratio,
            "evo_fraction_variance_explained": diagnostics.evo_fraction_variance_explained,
            "matched_fraction_variance_explained": diagnostics.fit_fraction_variance_explained,
            "realized_l0": diagnostics.realized_l0, "epochs": len(trace),
            "checkpoint": str(path.relative_to(REPO_ROOT)),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "loaded_complete_checkpoint": resumed,
            "run_wall_hours": (time.monotonic() - run_started) / 3600.0,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()
    report = REPO_ROOT / step6["paths"]["resolution_audit_report_dir"]
    report.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report / "null_crosscoder_runs.csv", index=False)
    manifest = {
        "status": "complete", "dataset": "IRED", "width_multiplier": width,
        "requested_total_null_seeds": int(audit["requested_total_null_seeds"]),
        "realized_total_null_seeds": len(seeds), "seeds": seeds,
        "reason_for_extension": str(audit["reasoning"]),
        "gpu_wall_hours": (time.monotonic() - started) / 3600.0,
        "partial_results_reported": False,
    }
    (report / "null_crosscoder_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
