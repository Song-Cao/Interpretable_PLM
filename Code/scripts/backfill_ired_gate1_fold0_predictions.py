#!/usr/bin/env python3
"""Backfill predictions from surviving Gate 1 fold-0 checkpoints without training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.adaptation import (
    SequenceFitnessAdapter,
    build_lora_adapter,
    load_trainable_state,
    predict_adapter,
)
from crossfit.datasets import get_split, load_assay
from crossfit.esmc import load_esmc
from crossfit.persistence import prediction_frame, sha256_file, write_prediction_csv
from crossfit.search import training_pool_folds
from crossfit.utils import REPO_ROOT, load_config


def _model(config: dict, device: str, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    backbone = load_esmc(
        str(config["backbone"]["name"]), device,
        repository=str(config["backbone"]["repository"]),
        expected_revision=str(config["backbone"]["revision"]),
    )
    adapted = build_lora_adapter(
        backbone, rank=int(config["adapter"]["rank"]),
        alpha=int(config["adapter"]["alpha"]),
        dropout=float(config["adapter"]["dropout"]),
    )
    model = SequenceFitnessAdapter(adapted, int(backbone.embed.embedding_dim)).to(device)
    return model, backbone._tokenize


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint inference backfill requires the GPU pod")
    device = "cuda"
    step2 = load_config(REPO_ROOT / "Code/config/step2.yaml")
    config = load_config(REPO_ROOT / "Code/config/step3.yaml")
    frame = load_assay("IRED", with_sequences=True)
    target = frame["y"].to_numpy(float)
    train_pool, _ = get_split(
        "IRED", "pg_kfold", fold=int(step2["evaluation"]["primary_test_fold"])
    )
    folds = training_pool_folds(
        train_pool, int(config["selection"]["cv_folds"]), int(config["selection"]["cv_seed"])
    )
    fold = int(config["gate1"]["final_adapter_cv_fold"])
    validation = folds[fold][1]
    seed = int(config["adapter"]["seed"]) + fold
    checkpoint_dir = REPO_ROOT / config["paths"]["checkpoints"]
    verification = json.loads(
        (REPO_ROOT / config["paths"]["reports"] / "gate1_checkpoint_verification.json").read_text()
    )
    rows = []
    for arm, filename, verification_key in (
        ("real_adapter", "real_adapter.pt", "real"),
        ("shuffled_adapter", "shuffled_adapter.pt", "shuffled"),
    ):
        checkpoint = checkpoint_dir / filename
        if sha256_file(checkpoint) != verification[verification_key]["sha256"]:
            raise AssertionError(f"checkpoint hash mismatch: {checkpoint}")
        model, tokenize = _model(config, device, seed)
        load_trainable_state(model, checkpoint)
        prediction = predict_adapter(
            model, frame.iloc[validation]["mutant_sequence"].astype(str).tolist(),
            backbone_tokenize=tokenize, batch_size=int(config["adapter"]["batch_size"]),
            device=device,
        )
        rows.append(prediction_frame(
            dataset="IRED", step=3, arm=arm, config_id="rank16_lr1e-4",
            fold=fold, seed=seed, condition="cv_validation_checkpoint_backfill",
            variant_rows=frame.iloc[validation]["variant_row"],
            targets=target[validation], predictions=prediction,
        ))
        model.to("cpu")
        del model, tokenize, prediction
        torch.cuda.empty_cache()
    output = REPO_ROOT / config["paths"]["reports"] / "rank16_lr1e-4_fold0_checkpoint_backfill_predictions.csv"
    write_prediction_csv(pd.concat(rows, ignore_index=True), output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
