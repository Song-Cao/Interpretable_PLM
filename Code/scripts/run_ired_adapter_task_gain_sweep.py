#!/usr/bin/env python3
"""Run the bounded IRED adapter task-gain sweep without changing the analysis layer."""
from __future__ import annotations

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

from crossfit.adaptation import (
    SequenceFitnessAdapter,
    build_lora_adapter,
    fit_adapter_early_stopping,
    predict_adapter,
    save_trainable_state,
)
from crossfit.datasets import get_split, load_assay
from crossfit.esmc import load_esmc
from crossfit.gate1 import paired_task_gain_test
from crossfit.persistence import (
    prediction_frame,
    upsert_checkpoint_manifest,
    write_prediction_csv,
)
from crossfit.search import training_pool_folds
from crossfit.utils import REPO_ROOT, load_config


def _new_adapter(config: dict, rank: int, device: str, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    backbone = load_esmc(
        str(config["backbone"]["name"]), device,
        repository=str(config["backbone"]["repository"]),
        expected_revision=str(config["backbone"]["revision"]),
    )
    adapted = build_lora_adapter(
        backbone, rank=rank, alpha=int(config["adapter"]["alpha"]),
        dropout=float(config["adapter"]["dropout"]),
    )
    return SequenceFitnessAdapter(adapted, int(backbone.embed.embedding_dim)), backbone._tokenize


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _parameters(config: dict, learning_rate: float) -> dict:
    adapter = config["adapter"]
    return {
        "learning_rate": learning_rate,
        "weight_decay": adapter["weight_decay"],
        "batch_size": adapter["batch_size"],
        "gradient_accumulation_steps": adapter["gradient_accumulation_steps"],
        "gradient_clip_norm": adapter["gradient_clip_norm"],
    }


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("the adapter task-gain sweep requires the GPU pod")
    step2 = load_config(REPO_ROOT / "Code/config/step2.yaml")
    config = load_config(REPO_ROOT / "Code/config/step3.yaml")
    frame = load_assay("IRED", with_sequences=True)
    sequences = frame["mutant_sequence"].astype(str).tolist()
    target = frame["y"].to_numpy(np.float32)
    censoring = frame["censored"].astype(str).to_numpy()
    train_pool, _ = get_split(
        "IRED", "pg_kfold", fold=int(step2["evaluation"]["primary_test_fold"])
    )
    folds = training_pool_folds(
        train_pool, int(config["selection"]["cv_folds"]),
        int(config["selection"]["cv_seed"]),
    )
    authoritative = pd.read_csv(REPO_ROOT / config["paths"]["reports"] / "gate1_task_gain_cv.csv")
    if authoritative["fold"].tolist() != list(range(len(folds))):
        raise AssertionError("authoritative frozen-head folds do not match the registered CV folds")

    report_dir = REPO_ROOT / config["paths"]["reports"] / "task_gain_sweep"
    partial_dir = REPO_ROOT / config["paths"]["cache"] / "task_gain_sweep"
    report_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    persistence = config["persistence"]
    checkpoint_root = REPO_ROOT / config["paths"]["checkpoints"] / "task_gain_sweep"
    checkpoint_manifest = REPO_ROOT / persistence["checkpoint_manifest"]
    summary = [{
        "configuration": "rank16_lr1e-4",
        "rank": int(config["adapter"]["rank"]),
        "learning_rate": float(config["adapter"]["learning_rate"]),
        **paired_task_gain_test(authoritative["task_gain"].to_numpy()),
        "complete": True,
        "source": "authoritative_gate1_cv",
    }]
    total_started = time.perf_counter()
    for candidate in config["task_gain_sweep"]["configurations"]:
        name = str(candidate["name"])
        rank = int(candidate["rank"])
        learning_rate = float(candidate["learning_rate"])
        final_path = report_dir / f"{name}_cv.csv"
        partial_path = partial_dir / f"{name}_partial.csv"
        prediction_path = report_dir / f"{name}_oof_predictions.csv"
        if final_path.exists():
            rows = pd.read_csv(final_path).to_dict("records")
            if len(rows) != len(folds):
                raise AssertionError(f"incomplete final result: {final_path}")
        elif partial_path.exists():
            rows = pd.read_csv(partial_path).to_dict("records")
        else:
            rows = []
        valid_completed = set()
        for row in rows:
            fold = int(row["fold"])
            checkpoint = checkpoint_root / name / f"fold_{fold}" / "real_adapter.pt"
            if prediction_path.exists():
                persisted = pd.read_csv(prediction_path)
                has_predictions = (
                    (persisted["config_id"].astype(str) == name)
                    & (persisted["fold"].astype(int) == fold)
                    & (persisted["arm"].astype(str) == "real_adapter")
                ).sum() == len(folds[fold][1])
            else:
                has_predictions = False
            if has_predictions and (
                not bool(persistence["save_sweep_checkpoints"]) or checkpoint.is_file()
            ):
                valid_completed.add(fold)
        rows = [row for row in rows if int(row["fold"]) in valid_completed]
        completed = {int(row["fold"]) for row in rows}
        candidate_started = time.perf_counter()
        for fold_index, (train, validation) in enumerate(folds):
            if fold_index in completed:
                continue
            seed = int(config["adapter"]["seed"]) + fold_index
            model, tokenize = _new_adapter(config, rank, device, seed)
            fit = fit_adapter_early_stopping(
                model,
                [sequences[index] for index in train], target[train], censoring[train],
                [sequences[index] for index in validation], target[validation], censoring[validation],
                backbone_tokenize=tokenize, objective_config=step2["objective"],
                adapter_config=_parameters(config, learning_rate),
                max_epochs=int(config["adapter"]["max_epochs"]),
                patience=int(config["adapter"]["early_stopping_patience"]),
                min_delta=float(config["adapter"]["early_stopping_min_delta"]),
                scheduler_factor=float(config["adapter"]["scheduler_factor"]),
                scheduler_patience=int(config["adapter"]["scheduler_patience"]),
                seed=seed, device=device,
            )
            prediction = predict_adapter(
                fit.model, [sequences[index] for index in validation],
                backbone_tokenize=tokenize, batch_size=int(config["adapter"]["batch_size"]),
                device=device,
            )
            held_out = prediction_frame(
                dataset="IRED", step=3, arm="real_adapter", config_id=name,
                fold=fold_index, seed=seed, condition="cv_validation",
                variant_rows=frame.iloc[validation]["variant_row"],
                targets=target[validation], predictions=prediction,
            )
            existing_predictions = (
                pd.read_csv(prediction_path) if prediction_path.exists() else pd.DataFrame()
            )
            if not existing_predictions.empty:
                existing_predictions = existing_predictions.loc[
                    existing_predictions["fold"].astype(int) != fold_index
                ]
            combined_predictions = pd.concat(
                [existing_predictions, held_out], ignore_index=True
            )
            write_prediction_csv(combined_predictions, prediction_path)
            if bool(persistence["save_sweep_checkpoints"]):
                checkpoint = checkpoint_root / name / f"fold_{fold_index}" / "real_adapter.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                save_trainable_state(fit.model, checkpoint)
                upsert_checkpoint_manifest(
                    checkpoint_manifest, dataset="IRED", step=3, arm="real_adapter",
                    config_id=name, fold=fold_index, seed=seed,
                    condition="cv_validation", checkpoint=checkpoint,
                )
            adapted = float(spearmanr(prediction, target[validation]).statistic)
            frozen = float(authoritative.loc[authoritative["fold"] == fold_index, "frozen_spearman"].iloc[0])
            rows.append({
                "configuration": name,
                "rank": rank,
                "learning_rate": learning_rate,
                "fold": fold_index,
                "frozen_spearman": frozen,
                "adapted_spearman": adapted,
                "task_gain": adapted - frozen,
                "adapter_best_epoch": fit.best_epoch + 1,
                "adapter_optimizer_steps": fit.best_optimizer_steps,
                "adapter_executed_optimizer_steps": fit.executed_optimizer_steps,
                "prediction_file": f"Reports/step3/task_gain_sweep/{name}_oof_predictions.csv",
                "checkpoint_manifest": str(persistence["checkpoint_manifest"]),
            })
            rows.sort(key=lambda row: int(row["fold"]))
            _atomic_csv(pd.DataFrame(rows), partial_path)
            print(json.dumps(rows[-1]), flush=True)
            fit.model.to("cpu")
            del prediction, fit, model, tokenize
            torch.cuda.empty_cache()
        completed_frame = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
        if completed_frame["fold"].tolist() != list(range(len(folds))):
            raise AssertionError(f"configuration {name} did not complete every fold")
        _atomic_csv(completed_frame, final_path)
        partial_path.unlink(missing_ok=True)
        test = paired_task_gain_test(completed_frame["task_gain"].to_numpy())
        summary.append({
            "configuration": name,
            "rank": rank,
            "learning_rate": learning_rate,
            **test,
            "complete": True,
            "source": "bounded_sweep",
            "elapsed_gpu_hours": (time.perf_counter() - candidate_started) / 3600.0,
        })
    summary_frame = pd.DataFrame(summary)
    alpha = float(config["gate1"]["task_gain_alpha"])
    summary_frame["task_gain_established"] = (
        (summary_frame["mean"] > 0.0) & (summary_frame["p_value_two_sided"] < alpha)
    )
    _atomic_csv(summary_frame, report_dir / "summary.csv")
    timing = {
        "total_gpu_hours": (time.perf_counter() - total_started) / 3600.0,
        "completed_configurations": int(len(summary_frame) - 1),
        "baseline_reused": True,
    }
    (report_dir / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(summary_frame.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
