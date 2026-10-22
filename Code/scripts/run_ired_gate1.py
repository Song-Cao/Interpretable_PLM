#!/usr/bin/env python3
"""Adapt ESM-C on IRED and evaluate Gate 1 through analysis-layer selection."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.adaptation import (
    SequenceFitnessAdapter,
    build_lora_adapter,
    fit_adapter,
    predict_adapter,
    save_trainable_state,
)
from crossfit.datasets import get_split, load_assay
from crossfit.esmc import extract_residue_embeddings, load_esmc
from crossfit.gate1 import bootstrap_drift_gap, linear_cka_by_layer, tokenwise_relative_drift
from crossfit.models import RidgeHead
from crossfit.representations import esm_mean
from crossfit.search import training_pool_folds
from crossfit.training import fit_torch_fixed_epochs, predict_torch
from crossfit.utils import REPO_ROOT, load_config


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmark-only", action="store_true")
    return parser.parse_args()


def _new_adapter(config: dict, device: str) -> tuple[SequenceFitnessAdapter, object]:
    backbone = load_esmc(
        str(config["backbone"]["name"]), device,
        repository=str(config["backbone"]["repository"]),
        expected_revision=str(config["backbone"]["revision"]),
    )
    adapted = build_lora_adapter(
        backbone,
        rank=int(config["adapter"]["rank"]),
        alpha=int(config["adapter"]["alpha"]),
        dropout=float(config["adapter"]["dropout"]),
    )
    model = SequenceFitnessAdapter(adapted, int(backbone.embed.embedding_dim))
    return model, backbone._tokenize


def _adapter_parameters(config: dict) -> dict:
    settings = config["adapter"]
    return {
        "learning_rate": settings["learning_rate"],
        "weight_decay": settings["weight_decay"],
        "batch_size": settings["batch_size"],
        "gradient_accumulation_steps": settings["gradient_accumulation_steps"],
        "gradient_clip_norm": settings["gradient_clip_norm"],
    }


def _head_parameters(config: dict) -> dict:
    settings = config["adapter"]
    return {
        "learning_rate": settings["learning_rate"],
        "weight_decay": settings["weight_decay"],
        "batch_size": int(settings["batch_size"]) * int(settings["gradient_accumulation_steps"]),
    }


def _clear(*values: object) -> None:
    for value in values:
        del value
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _extract_layers(model: SequenceFitnessAdapter, sequences: list[str], batch_size: int) -> np.ndarray:
    result = extract_residue_embeddings(
        model.backbone, sequences, batch_size=batch_size, all_layers=True
    )
    if result.layers is None:
        raise AssertionError("all-layer extraction returned no layers")
    return result.layers


def main() -> int:
    args = _arguments()
    step2 = load_config(REPO_ROOT / "Code" / "config" / "step2.yaml")
    config = load_config(REPO_ROOT / "Code" / "config" / "step3.yaml")
    frame = load_assay("IRED", with_sequences=True)
    sequences = frame["mutant_sequence"].astype(str).tolist()
    target = frame["y"].to_numpy(np.float32)
    censoring = frame["censored"].astype(str).to_numpy()
    train_pool, _ = get_split("IRED", "pg_kfold", fold=int(step2["evaluation"]["primary_test_fold"]))
    folds = training_pool_folds(
        train_pool, int(config["selection"]["cv_folds"]), int(config["selection"]["cv_seed"])
    )
    adapter_parameters = _adapter_parameters(config)
    epochs = int(config["adapter"]["max_epochs"])

    report_dir = REPO_ROOT / config["paths"]["reports"]
    report_dir.mkdir(parents=True, exist_ok=True)
    benchmark_model, tokenize = _new_adapter(config, args.device)
    benchmark_train, _ = folds[0]
    started = time.perf_counter()
    benchmark = fit_adapter(
        benchmark_model,
        [sequences[index] for index in benchmark_train],
        target[benchmark_train],
        censoring[benchmark_train],
        backbone_tokenize=tokenize,
        objective_config=step2["objective"],
        adapter_config=adapter_parameters,
        epochs=1,
        seed=int(config["adapter"]["seed"]),
        device=args.device,
    )
    epoch_seconds = time.perf_counter() - started
    # Five CV adapters plus the matched real and shuffled full-training adapters.
    projected_hours = epoch_seconds * epochs * 7 / 3600.0
    timing = {
        "one_cv_epoch_seconds": epoch_seconds,
        "projected_gate1_gpu_hours": projected_hours,
        "epochs_per_adapter": epochs,
        "adapter_fits": 7,
        "benchmark_optimizer_steps": benchmark.optimizer_steps,
    }
    (report_dir / "gate1_timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing, indent=2), flush=True)
    _clear(benchmark, benchmark_model)
    if args.benchmark_only:
        return 0
    budget_limit = float(config["budget"]["estimated_gpu_hours"]) * (
        1.0 + float(config["budget"]["estimate_tolerance_fraction"])
    )
    if projected_hours > budget_limit:
        raise RuntimeError(
            f"projected Gate 1 cost {projected_hours:.2f} GPU-hours exceeds the v12 "
            f"Step 3 limit {budget_limit:.2f}; no measurement was weakened"
        )

    cache_dir = REPO_ROOT / step2["paths"]["representations"]
    rows = np.load(cache_dir / "variant_rows.npy")
    row_to_cache = {int(value): index for index, value in enumerate(rows)}
    feature_index = np.asarray([row_to_cache[int(value)] for value in frame["variant_row"]])
    base_residue = np.asarray(
        np.load(cache_dir / "variant_residue.npy", mmap_mode="r")[feature_index], dtype=np.float32
    )
    base_mean = esm_mean(base_residue)
    del base_residue

    cv_rows = []
    for fold_index, (train, validation) in enumerate(folds):
        head_fit = fit_torch_fixed_epochs(
            RidgeHead(base_mean.shape[1]),
            base_mean[train], target[train], censoring[train],
            objective_config=step2["objective"],
            optimization_config={"gradient_clip_norm": config["adapter"]["gradient_clip_norm"]},
            parameters=_head_parameters(config), epochs=epochs,
            seed=int(config["adapter"]["seed"]) + fold_index, device=args.device,
        )
        frozen_prediction = predict_torch(
            head_fit.model, head_fit.normalizer, base_mean[validation], device=args.device
        )
        adapter_model, tokenize = _new_adapter(config, args.device)
        real_fit = fit_adapter(
            adapter_model,
            [sequences[index] for index in train], target[train], censoring[train],
            backbone_tokenize=tokenize,
            objective_config=step2["objective"], adapter_config=adapter_parameters,
            epochs=epochs, seed=int(config["adapter"]["seed"]) + fold_index, device=args.device,
        )
        real_prediction = predict_adapter(
            real_fit.model, [sequences[index] for index in validation],
            backbone_tokenize=tokenize, batch_size=int(config["adapter"]["batch_size"]),
            device=args.device,
        )
        frozen_score = float(spearmanr(frozen_prediction, target[validation]).statistic)
        real_score = float(spearmanr(real_prediction, target[validation]).statistic)
        cv_rows.append({
            "fold": fold_index,
            "frozen_spearman": frozen_score,
            "adapted_spearman": real_score,
            "task_gain": real_score - frozen_score,
            "adapter_optimizer_steps": real_fit.optimizer_steps,
        })
        print(json.dumps(cv_rows[-1]), flush=True)
        _clear(head_fit, real_fit, adapter_model)
    cv_frame = pd.DataFrame(cv_rows)
    cv_frame.to_csv(report_dir / "gate1_task_gain_cv.csv", index=False)

    checkpoint_dir = REPO_ROOT / config["paths"]["checkpoints"]
    if checkpoint_dir.exists():
        raise FileExistsError(f"refusing to overwrite Gate 1 checkpoints: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True)
    real_model, real_tokenize = _new_adapter(config, args.device)
    real_fit = fit_adapter(
        real_model, [sequences[index] for index in train_pool], target[train_pool], censoring[train_pool],
        backbone_tokenize=real_tokenize,
        objective_config=step2["objective"], adapter_config=adapter_parameters,
        epochs=epochs, seed=int(config["adapter"]["seed"]), device=args.device,
    )
    save_trainable_state(real_fit.model, checkpoint_dir / "real_adapter.pt")

    reserve_frame = load_assay("IRED", with_sequences=True, include_reserve=True)
    reserve_frame = reserve_frame.loc[reserve_frame["is_layer_freeze_reserve"]].reset_index(drop=True)
    reserve_sequences = reserve_frame["mutant_sequence"].astype(str).tolist()
    adapted_layers = _extract_layers(
        real_fit.model, reserve_sequences, int(config["extraction"]["reserve_batch_size"])
    )
    shuffled_rng = np.random.default_rng(int(config["adapter"]["shuffled_label_seed"]))
    permutation = shuffled_rng.permutation(len(train_pool))
    shuffled_target = target[train_pool][permutation]
    shuffled_censoring = censoring[train_pool][permutation]
    shuffled_model, shuffled_tokenize = _new_adapter(config, args.device)
    shuffled_fit = fit_adapter(
        shuffled_model, [sequences[index] for index in train_pool], shuffled_target, shuffled_censoring,
        backbone_tokenize=shuffled_tokenize,
        objective_config=step2["objective"], adapter_config=adapter_parameters,
        epochs=epochs, seed=int(config["adapter"]["seed"]), device=args.device,
    )
    if shuffled_fit.optimizer_steps != real_fit.optimizer_steps:
        raise AssertionError("real and shuffled adapters took different optimizer step counts")
    save_trainable_state(shuffled_fit.model, checkpoint_dir / "shuffled_adapter.pt")
    shuffled_layers = _extract_layers(
        shuffled_fit.model, reserve_sequences, int(config["extraction"]["reserve_batch_size"])
    )

    base_layers = np.asarray(np.load(cache_dir / "reserve_all_layers_base.npy"), dtype=np.float32)
    reserve_rows = np.load(cache_dir / "reserve_rows.npy")
    if not np.array_equal(reserve_rows, reserve_frame["variant_row"].to_numpy(int)):
        raise AssertionError("reserve row alignment changed between extraction and Gate 1")
    base_tokens = load_esmc(
        str(config["backbone"]["name"]), args.device,
        repository=str(config["backbone"]["repository"]),
        expected_revision=str(config["backbone"]["revision"]),
    )._tokenize(reserve_sequences[:1])
    real_tokens = real_tokenize(reserve_sequences[:1])
    shuffled_tokens = shuffled_tokenize(reserve_sequences[:1])
    if not torch.equal(base_tokens, real_tokens) or not torch.equal(base_tokens, shuffled_tokens):
        raise AssertionError("ESM-C token alignment differs between base, real, and shuffled models")
    if base_layers.shape != adapted_layers.shape or base_layers.shape != shuffled_layers.shape:
        raise AssertionError("reserve activation shapes are not aligned")
    magnitude_floor = 10.0 * np.finfo(np.float32).eps * np.sqrt(base_layers.shape[-1])
    maximum_change = float(np.linalg.norm(adapted_layers - base_layers, axis=-1).max())
    if maximum_change <= magnitude_floor:
        raise AssertionError("adapted activation change does not exceed the numerical magnitude floor")

    real_drift = tokenwise_relative_drift(base_layers, adapted_layers)
    shuffled_drift = tokenwise_relative_drift(base_layers, shuffled_layers)
    real_cka = linear_cka_by_layer(base_layers, adapted_layers, device=args.device)
    shuffled_cka = linear_cka_by_layer(base_layers, shuffled_layers, device=args.device)
    lower, upper = bootstrap_drift_gap(
        real_drift, shuffled_drift,
        samples=int(config["gate1"]["bootstrap_samples"]),
        seed=int(config["gate1"]["bootstrap_seed"]),
        confidence_level=float(config["gate1"]["confidence_level"]),
    )
    drift_gap = real_drift.mean(axis=0) - shuffled_drift.mean(axis=0)
    selected_layer = int(np.argmax(drift_gap))
    layer_frame = pd.DataFrame({
        "layer": np.arange(base_layers.shape[1]),
        "real_relative_drift": real_drift.mean(axis=0),
        "shuffled_relative_drift": shuffled_drift.mean(axis=0),
        "drift_gap": drift_gap,
        "drift_gap_lower": lower,
        "drift_gap_upper": upper,
        "real_cka": real_cka,
        "shuffled_cka": shuffled_cka,
    })
    layer_frame.to_csv(report_dir / "gate1_by_layer.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(layer_frame["layer"], layer_frame["real_relative_drift"], label="real")
    axes[0].plot(layer_frame["layer"], layer_frame["shuffled_relative_drift"], label="shuffled")
    axes[0].axvline(selected_layer, color="black", linestyle="--")
    axes[0].set(xlabel="layer", ylabel="relative drift")
    axes[0].legend()
    axes[1].plot(layer_frame["layer"], layer_frame["real_cka"], label="real")
    axes[1].plot(layer_frame["layer"], layer_frame["shuffled_cka"], label="shuffled")
    axes[1].axvline(selected_layer, color="black", linestyle="--")
    axes[1].set(xlabel="layer", ylabel="linear CKA")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(report_dir / "gate1_drift_by_depth.png", dpi=180)
    plt.close(figure)

    mean_gain = float(cv_frame["task_gain"].mean())
    gate_pass = mean_gain > 0.0 and float(lower[selected_layer]) > 0.0
    result = {
        "dataset": "IRED",
        "task_gain_mean": mean_gain,
        "frozen_spearman_mean": float(cv_frame["frozen_spearman"].mean()),
        "adapted_spearman_mean": float(cv_frame["adapted_spearman"].mean()),
        "selected_layer": selected_layer,
        "layer_selection_rule": "maximum mean real-minus-shuffled relative drift on the reserve",
        "selected_real_relative_drift": float(real_drift[:, selected_layer].mean()),
        "selected_shuffled_relative_drift": float(shuffled_drift[:, selected_layer].mean()),
        "selected_drift_gap_95_ci": [float(lower[selected_layer]), float(upper[selected_layer])],
        "selected_real_cka": float(real_cka[selected_layer]),
        "selected_shuffled_cka": float(shuffled_cka[selected_layer]),
        "maximum_activation_change": maximum_change,
        "numerical_magnitude_floor": magnitude_floor,
        "optimizer_steps_real": real_fit.optimizer_steps,
        "optimizer_steps_shuffled": shuffled_fit.optimizer_steps,
        "gate1_pass": bool(gate_pass),
        "gpu_projection_hours": projected_hours,
    }
    (report_dir / "gate1_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not gate_pass:
        return 2

    selected_cache = REPO_ROOT / config["paths"]["cache"]
    if selected_cache.exists():
        raise FileExistsError(f"refusing to overwrite selected-layer cache: {selected_cache}")
    selected_cache.mkdir(parents=True)
    np.save(selected_cache / "reserve_all_layers_adapted.npy", adapted_layers.astype(np.float16))
    np.save(selected_cache / "reserve_all_layers_shuffled.npy", shuffled_layers.astype(np.float16))
    full_frame = load_assay("IRED", with_sequences=True, include_reserve=True)
    full_sequences = full_frame["mutant_sequence"].astype(str).tolist()
    base_model = load_esmc(
        str(config["backbone"]["name"]), args.device,
        repository=str(config["backbone"]["repository"]),
        expected_revision=str(config["backbone"]["revision"]),
    )
    base_selected = extract_residue_embeddings(
        base_model, full_sequences, batch_size=int(config["extraction"]["full_dataset_batch_size"]),
        all_layers=True,
    ).layers[:, selected_layer]
    adapted_selected = _extract_layers(
        real_fit.model, full_sequences, int(config["extraction"]["full_dataset_batch_size"])
    )[:, selected_layer]
    np.save(selected_cache / "full_selected_layer_base.npy", base_selected.astype(np.float16))
    np.save(selected_cache / "full_selected_layer_adapted.npy", adapted_selected.astype(np.float16))
    np.save(selected_cache / "full_variant_rows.npy", full_frame["variant_row"].to_numpy(np.int64))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
