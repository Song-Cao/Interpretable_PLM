#!/usr/bin/env python3
"""Adapt ESM-C on IRED and evaluate Gate 1 through analysis-layer selection."""
from __future__ import annotations

import argparse
import json
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
    fit_adapter_early_stopping,
    fit_adapter_for_steps,
    predict_adapter,
    save_trainable_state,
)
from crossfit.datasets import get_split, load_assay
from crossfit.esmc import extract_residue_embeddings, extract_residue_layer, load_esmc
from crossfit.gate1 import (
    activation_norms_by_layer,
    bootstrap_drift_gap,
    linear_cka_by_layer,
    paired_task_gain_test,
    select_analysis_layer,
    tokenwise_relative_drift,
)
from crossfit.models import RidgeHead
from crossfit.representations import esm_mean
from crossfit.search import training_pool_folds
from crossfit.training import fit_torch_predictor, predict_torch
from crossfit.utils import REPO_ROOT, load_config


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmark-only", action="store_true")
    return parser.parse_args()


def _new_adapter(config: dict, device: str, seed: int) -> tuple[SequenceFitnessAdapter, object]:
    # Model construction initializes both LoRA A matrices and the readout. Seed
    # before construction so a complete restart recreates the measured model.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    if config["adapter"]["scheduler_metric"] != "validation_spearman":
        raise ValueError("Step 3 scheduling must monitor validation Spearman")
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
    benchmark_model, tokenize = _new_adapter(
        config, args.device, int(config["adapter"]["seed"])
    )
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
    # Five early-stopped CV adapters; one is the final real adapter. The
    # shuffled control then replays that adapter's exact best-step LR schedule.
    projected_hours = epoch_seconds * epochs * 6 / 3600.0
    timing = {
        "one_cv_epoch_seconds": epoch_seconds,
        "projected_gate1_gpu_hours": projected_hours,
        "epochs_per_adapter": epochs,
        "adapter_fits_upper_bound": 6,
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
    final_fold = int(config["gate1"]["final_adapter_cv_fold"])
    if final_fold < 0 or final_fold >= len(folds):
        raise ValueError("gate1.final_adapter_cv_fold is outside the CV range")
    real_fit = None
    real_tokenize = None
    real_train = None
    for fold_index, (train, validation) in enumerate(folds):
        fit_seed = int(config["adapter"]["seed"]) + fold_index
        np.random.seed(fit_seed)
        torch.manual_seed(fit_seed)
        torch.cuda.manual_seed_all(fit_seed)
        head_fit = fit_torch_predictor(
            RidgeHead(base_mean.shape[1]),
            base_mean[train], target[train], censoring[train],
            base_mean[validation], target[validation], censoring[validation],
            objective_config=step2["objective"], optimization_config=config["adapter"],
            parameters=_head_parameters(config), seed=fit_seed, device=args.device,
        )
        frozen_prediction = predict_torch(
            head_fit.model, head_fit.normalizer, base_mean[validation], device=args.device
        )
        adapter_model, tokenize = _new_adapter(config, args.device, fit_seed)
        fold_fit = fit_adapter_early_stopping(
            adapter_model,
            [sequences[index] for index in train], target[train], censoring[train],
            [sequences[index] for index in validation], target[validation], censoring[validation],
            backbone_tokenize=tokenize,
            objective_config=step2["objective"], adapter_config=adapter_parameters,
            max_epochs=epochs,
            patience=int(config["adapter"]["early_stopping_patience"]),
            min_delta=float(config["adapter"]["early_stopping_min_delta"]),
            scheduler_factor=float(config["adapter"]["scheduler_factor"]),
            scheduler_patience=int(config["adapter"]["scheduler_patience"]),
            seed=fit_seed, device=args.device,
        )
        real_prediction = predict_adapter(
            fold_fit.model, [sequences[index] for index in validation],
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
            "adapter_best_epoch": fold_fit.best_epoch + 1,
            "adapter_optimizer_steps": fold_fit.best_optimizer_steps,
            "adapter_executed_optimizer_steps": fold_fit.executed_optimizer_steps,
            "frozen_head_best_epoch": head_fit.best_epoch + 1,
        })
        print(json.dumps(cv_rows[-1]), flush=True)
        if fold_index == final_fold:
            real_fit = fold_fit
            real_fit.model.to("cpu")
            real_tokenize = tokenize
            real_train = train.copy()
            _clear(head_fit)
        else:
            _clear(head_fit, fold_fit, adapter_model)
    cv_frame = pd.DataFrame(cv_rows)
    cv_frame.to_csv(report_dir / "gate1_task_gain_cv.csv", index=False)
    if real_fit is None or real_tokenize is None or real_train is None:
        raise AssertionError("the configured final adapter CV fold was not retained")
    real_fit.model.to(args.device)

    checkpoint_dir = REPO_ROOT / config["paths"]["checkpoints"]
    if checkpoint_dir.exists():
        raise FileExistsError(f"refusing to overwrite Gate 1 checkpoints: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True)
    save_trainable_state(real_fit.model, checkpoint_dir / "real_adapter.pt")

    reserve_frame = load_assay("IRED", with_sequences=True, include_reserve=True)
    reserve_frame = reserve_frame.loc[reserve_frame["is_layer_freeze_reserve"]].reset_index(drop=True)
    reserve_sequences = reserve_frame["mutant_sequence"].astype(str).tolist()
    adapted_layers = _extract_layers(
        real_fit.model, reserve_sequences, int(config["extraction"]["reserve_batch_size"])
    )
    shuffled_rng = np.random.default_rng(int(config["adapter"]["shuffled_label_seed"]))
    permutation = shuffled_rng.permutation(len(real_train))
    shuffled_target = target[real_train][permutation]
    shuffled_censoring = censoring[real_train][permutation]
    shuffled_model, shuffled_tokenize = _new_adapter(
        config, args.device, int(config["adapter"]["seed"]) + final_fold
    )
    shuffled_fit = fit_adapter_for_steps(
        shuffled_model, [sequences[index] for index in real_train],
        shuffled_target, shuffled_censoring,
        backbone_tokenize=shuffled_tokenize,
        objective_config=step2["objective"], adapter_config=adapter_parameters,
        optimizer_steps=real_fit.best_optimizer_steps,
        learning_rates=real_fit.learning_rates_to_best,
        seed=int(config["adapter"]["seed"]) + final_fold, device=args.device,
    )
    if shuffled_fit.optimizer_steps != real_fit.best_optimizer_steps:
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
    base_norm, real_norm, norm_ratio = activation_norms_by_layer(base_layers, adapted_layers)
    lower, upper = bootstrap_drift_gap(
        real_drift, shuffled_drift,
        samples=int(config["gate1"]["bootstrap_samples"]),
        seed=int(config["gate1"]["bootstrap_seed"]),
        confidence_level=float(config["gate1"]["confidence_level"]),
    )
    drift_gap = real_drift.mean(axis=0) - shuffled_drift.mean(axis=0)
    gate = config["gate1"]
    selected_layer, eligible, near_max, near_max_threshold = select_analysis_layer(
        drift_gap, real_cka,
        cka_floor=float(gate["real_cka_floor"]),
        exclude_first=int(gate["exclude_first_layers"]),
        exclude_last=int(gate["exclude_last_layers"]),
        near_max_fraction=float(gate["near_maximum_drift_gap_fraction"]),
    )
    layer_frame = pd.DataFrame({
        "layer": np.arange(base_layers.shape[1]),
        "real_relative_drift": real_drift.mean(axis=0),
        "shuffled_relative_drift": shuffled_drift.mean(axis=0),
        "drift_gap": drift_gap,
        "drift_gap_lower": lower,
        "drift_gap_upper": upper,
        "real_cka": real_cka,
        "shuffled_cka": shuffled_cka,
        "base_mean_activation_norm": base_norm,
        "real_mean_activation_norm": real_norm,
        "activation_norm_ratio": norm_ratio,
        "selection_eligible": eligible,
        "within_near_maximum": near_max,
        "selected": np.arange(base_layers.shape[1]) == selected_layer,
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

    task_test = paired_task_gain_test(cv_frame["task_gain"].to_numpy())
    mean_gain = float(task_test["mean"])
    task_gain_established = (
        mean_gain > 0.0
        and float(task_test["p_value_two_sided"]) < float(gate["task_gain_alpha"])
    )
    gate_pass = task_gain_established and float(lower[selected_layer]) > 0.0
    sensitivity = []
    for offset in map(int, gate["sensitivity_offsets"]):
        layer = selected_layer + offset
        if layer < 0 or layer >= len(layer_frame):
            sensitivity.append({"offset": offset, "available": False})
        else:
            sensitivity.append({
                "offset": offset, "available": True, "layer": layer,
                "drift_gap": float(drift_gap[layer]),
                "drift_gap_95_ci": [float(lower[layer]), float(upper[layer])],
                "real_cka": float(real_cka[layer]),
                "activation_norm_ratio": float(norm_ratio[layer]),
            })
    drift_ratio = float(
        real_drift[:, selected_layer].mean() / shuffled_drift[:, selected_layer].mean()
    )
    ratio_window = list(map(float, gate["drift_ratio_window"]))
    result = {
        "dataset": "IRED",
        "task_gain_mean": mean_gain,
        "frozen_spearman_mean": float(cv_frame["frozen_spearman"].mean()),
        "adapted_spearman_mean": float(cv_frame["adapted_spearman"].mean()),
        "selected_layer": selected_layer,
        "task_gain_test": {
            "method": str(gate["task_gain_test"]),
            "alpha": float(gate["task_gain_alpha"]),
            **task_test,
            "established": bool(task_gain_established),
        },
        "layer_selection_rule": "earliest eligible layer within 90% of the maximum drift gap",
        "layer_selection_parameters": {
            "real_cka_floor": float(gate["real_cka_floor"]),
            "exclude_first_layers": int(gate["exclude_first_layers"]),
            "exclude_last_layers": int(gate["exclude_last_layers"]),
            "near_maximum_drift_gap_fraction": float(gate["near_maximum_drift_gap_fraction"]),
            "near_maximum_threshold": near_max_threshold,
        },
        "selected_real_relative_drift": float(real_drift[:, selected_layer].mean()),
        "selected_shuffled_relative_drift": float(shuffled_drift[:, selected_layer].mean()),
        "selected_real_to_shuffled_drift_ratio": drift_ratio,
        "matched_drift_null_required": not ratio_window[0] <= drift_ratio <= ratio_window[1],
        "selected_drift_gap_95_ci": [float(lower[selected_layer]), float(upper[selected_layer])],
        "selected_real_cka": float(real_cka[selected_layer]),
        "selected_shuffled_cka": float(shuffled_cka[selected_layer]),
        "selected_activation_norm_ratio": float(norm_ratio[selected_layer]),
        "sensitivity": sensitivity,
        "maximum_activation_change": maximum_change,
        "numerical_magnitude_floor": magnitude_floor,
        "stopping_metric": "validation_spearman",
        "final_adapter_cv_fold": final_fold,
        "optimizer_steps_real": real_fit.best_optimizer_steps,
        "optimizer_steps_real_executed_before_restore": real_fit.executed_optimizer_steps,
        "optimizer_steps_shuffled": shuffled_fit.optimizer_steps,
        "matched_learning_rate_schedule": True,
        "gate1_pass": bool(gate_pass),
        "gate_failure_reasons": ([] if task_gain_established else ["task_gain_not_established"]),
        "gpu_projection_hours": projected_hours,
    }
    if not gate_pass:
        result["run_complete"] = True
        (report_dir / "gate1_report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
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
    base_selected = extract_residue_layer(
        base_model, full_sequences, batch_size=int(config["extraction"]["full_dataset_batch_size"]),
        layer=selected_layer,
    )
    adapted_selected = extract_residue_layer(
        real_fit.model.backbone, full_sequences,
        batch_size=int(config["extraction"]["full_dataset_batch_size"]), layer=selected_layer,
    )
    np.save(selected_cache / "full_selected_layer_base.npy", base_selected.astype(np.float16))
    np.save(selected_cache / "full_selected_layer_adapted.npy", adapted_selected.astype(np.float16))
    np.save(selected_cache / "full_variant_rows.npy", full_frame["variant_row"].to_numpy(np.int64))
    result["run_complete"] = True
    result["selected_cache_shapes"] = {
        "reserve_all_layers_adapted": list(adapted_layers.shape),
        "reserve_all_layers_shuffled": list(shuffled_layers.shape),
        "full_selected_layer_base": list(base_selected.shape),
        "full_selected_layer_adapted": list(adapted_selected.shape),
    }
    (report_dir / "gate1_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
