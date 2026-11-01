#!/usr/bin/env python3
"""Extract IRED plate base/adapted activations at the frozen analysis layer."""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.adaptation import SequenceFitnessAdapter, build_lora_adapter, load_trainable_state
from crossfit.datasets import load_plate_dataset
from crossfit.esmc import extract_residue_layer, load_esmc
from crossfit.persistence import sha256_file
from crossfit.utils import REPO_ROOT, load_config


def _backbone(step3: dict):
    return load_esmc(
        str(step3["backbone"]["name"]), "cuda",
        repository=str(step3["backbone"]["repository"]),
        expected_revision=str(step3["backbone"]["revision"]),
    )


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("plate activation extraction requires the GPU pod")
    started = time.monotonic()
    config = load_config(REPO_ROOT / "Code/config/step9.yaml")
    step3 = load_config(REPO_ROOT / "Code/config/step3.yaml")
    destination = REPO_ROOT / config["paths"]["plate_cache"]
    temporary = destination.with_name(destination.name + ".incomplete")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite plate cache: {destination}")
    if temporary.exists():
        raise FileExistsError(f"inspect incomplete plate cache before removal: {temporary}")
    temporary.mkdir(parents=True)
    plate = load_plate_dataset(str(config["dataset"]))
    sequences = plate["aa_seq"].astype(str).tolist()

    base = _backbone(step3)
    base_tokens = base._tokenize(sequences[:1]).detach().cpu()
    base_values = extract_residue_layer(base, sequences, layer=int(config["analysis_layer"]), batch_size=4)
    np.save(temporary / "base.npy", base_values.astype(np.float16))
    del base
    gc.collect()
    torch.cuda.empty_cache()

    checkpoint = REPO_ROOT / config["paths"]["step3_adapter"]
    verification = json.loads(
        (REPO_ROOT / config["paths"]["gate1_checkpoint_verification"]).read_text()
    )
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != verification["real"]["sha256"]:
        raise AssertionError("real adapter hash differs from Gate 1 verification")
    adapted_backbone = _backbone(step3)
    adapted = build_lora_adapter(
        adapted_backbone, rank=int(step3["adapter"]["rank"]),
        alpha=int(step3["adapter"]["alpha"]), dropout=float(step3["adapter"]["dropout"]),
    )
    wrapped = SequenceFitnessAdapter(adapted, int(config["dimension"])).to("cuda")
    load_trainable_state(wrapped, checkpoint)
    wrapped.eval()
    fit_tokens = wrapped.backbone._tokenize(sequences[:1]).detach().cpu()
    if not torch.equal(base_tokens, fit_tokens):
        raise AssertionError("plate token alignment differs between base and adapted models")
    fit_values = extract_residue_layer(
        wrapped.backbone, sequences, layer=int(config["analysis_layer"]), batch_size=4
    )
    np.save(temporary / "fit.npy", fit_values.astype(np.float16))
    if base_values.shape != fit_values.shape:
        raise AssertionError("plate activation tensors are not aligned")
    manifest = {
        "status": "complete", "dataset": str(config["dataset"]),
        "condition": "external_plate_evaluation_only", "variants": len(plate),
        "shape": list(base_values.shape), "analysis_layer": int(config["analysis_layer"]),
        "adapter_checkpoint_sha256": checkpoint_hash,
        "gpu_wall_hours": (time.monotonic() - started) / 3600.0,
        "used_for_training_selection_or_calibration": False,
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, destination)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
