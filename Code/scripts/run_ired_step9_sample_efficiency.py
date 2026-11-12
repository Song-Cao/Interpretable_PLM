#!/usr/bin/env python3
"""Measure IRED headline-phi sample efficiency after architecture selection."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.cache_provenance import verify_ired_esmc_cache
from crossfit.datasets import get_split, load_assay
from crossfit.feature_pooling import select_feature_set
from crossfit.models import build_model
from crossfit.persistence import prediction_frame, write_prediction_csv
from crossfit.training import fit_torch_fixed_epochs, predict_torch, seed_everything
from crossfit.utils import REPO_ROOT, load_config


def _headline_features(config: dict, frame: pd.DataFrame, arm: str) -> np.ndarray:
    parts = arm.split("_")
    width, operator, feature_set = int(parts[1][1:]), parts[2], "_".join(parts[3:])
    root = REPO_ROOT / config["paths"]["phi_cache"] / f"w{width}"
    rows = np.load(REPO_ROOT / config["paths"]["phi_cache"] / "campaign_rows.npy")
    lookup = {int(row): index for index, row in enumerate(rows)}
    indices = np.asarray([lookup[int(row)] for row in frame["variant_row"]], dtype=int)
    values = np.asarray(np.load(root / f"campaign_{operator}.npy", mmap_mode="r")[indices], dtype=np.float32)
    labels = pd.read_csv(root / "taxonomy.csv")["provisional_class"].to_numpy(str)
    return values[:, select_feature_set(labels, feature_set, config)]


def main() -> int:
    started = time.monotonic()
    config = load_config(REPO_ROOT / "Code/config/step9.yaml")
    step3 = load_config(REPO_ROOT / "Code/config/step3.yaml")
    provenance = load_config(REPO_ROOT / config["paths"]["provenance_config"])
    reference_verification = verify_ired_esmc_cache(
        REPO_ROOT / config["paths"]["phi_cache"] / "manifest.json",
        step3=step3,
        verification=provenance["verification"],
        adapter_checkpoint=REPO_ROOT / config["paths"]["step3_adapter"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    step2 = load_config(REPO_ROOT / "Code/config/step2.yaml")
    report = REPO_ROOT / config["paths"]["reports_step9"]
    results = pd.read_csv(report / "ired_phi_feature_set_results.csv")
    complete = results.loc[results["status"] == "complete"]
    selected = complete.sort_values("validation_spearman", ascending=False).iloc[0]
    arm = str(selected["arm"])
    epochs = int(selected["selected_epochs"])
    shortlist = pd.read_csv(report / "phi_architecture_shortlist.csv")
    architecture = shortlist.sort_values("validation_spearman", ascending=False).iloc[0]
    if architecture["family"] != "mlp":
        raise AssertionError("sample-efficiency implementation expected the selected MLP family")
    parameters = json.loads(architecture["selected_configuration"])
    model_parameters = {
        key: value for key, value in parameters.items()
        if key not in {"learning_rate", "weight_decay", "batch_size"}
    }
    frame = load_assay("IRED", with_sequences=True)
    features = _headline_features(config, frame, arm)
    target = frame["y"].to_numpy(np.float32)
    censoring = frame["censored"].astype(str).to_numpy()
    train_pool, test = get_split("IRED", "pg_kfold", fold=int(config["evaluation"]["primary_test_fold"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, summary = [], []
    model_root = REPO_ROOT / config["paths"]["models"] / "sample_efficiency" / arm
    for size in map(int, config["sample_efficiency"]["sizes"]):
        if size > len(train_pool):
            continue
        for seed in map(int, config["sample_efficiency"]["seeds"]):
            rng = np.random.default_rng(seed)
            train = np.sort(rng.choice(train_pool, size=size, replace=False))
            checkpoint = model_root / f"n{size}_seed{seed}.pt"
            if checkpoint.exists():
                artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = build_model("mlp", (features.shape[1],), model_parameters).to(device)
                model.load_state_dict(artifact["state_dict"])
                from crossfit.training import FeatureNormalizer
                normalizer = FeatureNormalizer(
                    np.asarray(artifact["normalizer_mean"]), np.asarray(artifact["normalizer_scale"])
                )
            else:
                seed_everything(seed)
                fitted = fit_torch_fixed_epochs(
                    build_model("mlp", (features.shape[1],), model_parameters),
                    features[train], target[train], censoring[train],
                    objective_config=step2["objective"], optimization_config=config["optimization"],
                    parameters=parameters, epochs=epochs, seed=seed, device=device,
                )
                model, normalizer = fitted.model, fitted.normalizer
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(".pt.tmp")
                torch.save({
                    "state_dict": model.state_dict(), "normalizer_mean": normalizer.mean,
                    "normalizer_scale": normalizer.scale, "epochs": epochs, "size": size,
                    "seed": seed, "configuration": parameters,
                }, temporary)
                os.replace(temporary, checkpoint)
            prediction = predict_torch(model, normalizer, features[test], device=device)
            rows.append(prediction_frame(
                dataset="IRED", step=9, arm=f"{arm}_sample_efficiency", config_id=size,
                fold="final", seed=seed, condition="pg_kfold_test",
                variant_rows=frame.iloc[test]["variant_row"], targets=target[test],
                predictions=prediction,
            ))
            summary.append({
                "arm": arm, "training_size": size, "seed": seed,
                "test_spearman": float(spearmanr(prediction, target[test]).statistic),
                "epochs": epochs, "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            })
    write_prediction_csv(pd.concat(rows, ignore_index=True), report / "sample_efficiency_predictions.csv")
    frame_summary = pd.DataFrame(summary)
    aggregate = frame_summary.groupby("training_size", as_index=False)["test_spearman"].agg(["mean", "std"]).reset_index()
    frame_summary.to_csv(report / "sample_efficiency_runs.csv", index=False)
    aggregate.to_csv(report / "sample_efficiency_summary.csv", index=False)
    manifest = {
        "representation_layer": int(config["analysis_layer"]),
        "cache_reference_verification": reference_verification,
        "status": "complete", "arm": arm, "epochs": epochs, "seeds_per_size": 10,
        "gpu_wall_hours": (time.monotonic() - started) / 3600.0,
        "selection_used_low_n": False,
    }
    (report / "sample_efficiency_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
