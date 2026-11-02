#!/usr/bin/env python3
"""Retrain the selected LoRA adapter and linear head on all curated IRED variants."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from applications.ired_design.training import campaign
from crossfit.adaptation import SequenceFitnessAdapter, build_lora_adapter, fit_adapter, save_trainable_state
from crossfit.esmc import load_esmc
from crossfit.utils import REPO_ROOT, load_config


def main() -> int:
    config = load_config(REPO_ROOT / "Code/config/ired_design.yaml")
    step2 = load_config(REPO_ROOT / config["training"]["objective_config"])
    frame, _ = campaign(config)
    sequences = frame["mutant_sequence"].astype(str).tolist()
    target = frame["y"].to_numpy(np.float32)
    censoring = frame["censored"].astype(str).to_numpy()
    settings = dict(config["training"]["selected"]["adapter_linear"])
    epochs = int(settings.pop("epochs"))
    destination = REPO_ROOT / config["paths"]["models"] / "adapter_linear"
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = []
    started = time.monotonic()
    for seed in map(int, config["training"]["seeds"]):
        path = destination / f"seed_{seed}.pt"
        if path.exists():
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            if artifact.get("training_rows") != 1182 or artifact.get("epochs") != epochs:
                raise RuntimeError(f"refusing incompatible completed adapter: {path}")
        else:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            backbone = load_esmc(
                str(config["backbone"]["name"]), "cuda",
                repository=str(config["backbone"]["repository"]),
                expected_revision=str(config["backbone"]["revision"]),
            )
            adapted = build_lora_adapter(
                backbone, rank=int(settings["rank"]), alpha=int(settings["alpha"]),
                dropout=float(settings["dropout"]),
            )
            model = SequenceFitnessAdapter(adapted, int(backbone.embed.embedding_dim))
            fit = fit_adapter(
                model, sequences, target, censoring, backbone_tokenize=backbone._tokenize,
                objective_config=step2["objective"], adapter_config=settings,
                epochs=epochs, seed=seed, device="cuda",
            )
            temporary = path.with_suffix(".incomplete.pt")
            save_trainable_state(fit.model, temporary)
            state = torch.load(temporary, map_location="cpu", weights_only=True)
            state.update({
                "application_format": "ired_design_generation_adapter_v1",
                "training_rows": 1182, "epochs": epochs,
                "optimizer_steps": fit.optimizer_steps, "seed": seed,
            })
            torch.save(state, temporary)
            temporary.replace(path)
            del fit, model, adapted, backbone
            torch.cuda.empty_cache()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts.append({"seed": seed, "path": str(path.relative_to(REPO_ROOT)), "sha256": digest})
    manifest = {
        "status": "complete", "training_rows": 1182, "epochs": epochs,
        "epoch_policy": "rounded median best epoch from the registered five CV folds",
        "gpu_hours": (time.monotonic() - started) / 3600.0, "artifacts": artifacts,
    }
    target_path = REPO_ROOT / config["paths"]["reports"] / "adapter_generation_manifest.json"
    target_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
