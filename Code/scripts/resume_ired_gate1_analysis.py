#!/usr/bin/env python3
"""Resume Gate 1 analysis from complete, verified real and shuffled checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.adaptation import SequenceFitnessAdapter, build_lora_adapter, load_trainable_state
from crossfit.datasets import load_assay
from crossfit.esmc import extract_residue_embeddings, extract_residue_layer, load_esmc
from crossfit.gate1 import (
    activation_norms_by_layer,
    bootstrap_drift_gap,
    linear_cka_by_layer,
    paired_task_gain_test,
    select_analysis_layer,
    tokenwise_relative_drift,
)
from crossfit.utils import REPO_ROOT, load_config


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_adapter(config: dict, label: str, device: str) -> tuple[SequenceFitnessAdapter, Path]:
    backbone = load_esmc(
        str(config["backbone"]["name"]), device,
        repository=str(config["backbone"]["repository"]),
        expected_revision=str(config["backbone"]["revision"]),
    )
    model = SequenceFitnessAdapter(
        build_lora_adapter(
            backbone, rank=int(config["adapter"]["rank"]),
            alpha=int(config["adapter"]["alpha"]),
            dropout=float(config["adapter"]["dropout"]),
        ),
        int(backbone.embed.embedding_dim),
    ).to(device)
    path = REPO_ROOT / config["paths"]["checkpoints"] / f"{label}_adapter.pt"
    load_trainable_state(model, path)
    model.eval()
    return model, path


def _checkpoint_record(path: Path) -> dict[str, int | str]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    parameters = artifact["parameters"]
    return {
        "bytes": path.stat().st_size,
        "parameter_tensors": len(parameters),
        "lora_parameter_tensors": sum("lora_" in name for name in parameters),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    args = _arguments()
    step2 = load_config(REPO_ROOT / "Code/config/step2.yaml")
    config = load_config(REPO_ROOT / "Code/config/step3.yaml")
    report_dir = REPO_ROOT / config["paths"]["reports"]
    cv_path = report_dir / "gate1_task_gain_cv.csv"
    training_path = report_dir / "gate1_training_complete.json"
    outputs = (
        report_dir / "gate1_report.json",
        report_dir / "gate1_by_layer.csv",
        report_dir / "gate1_drift_by_depth.png",
        report_dir / "gate1_checkpoint_verification.json",
        REPO_ROOT / config["paths"]["cache"],
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite an active Gate 1 analysis artifact")
    cv = pd.read_csv(cv_path)
    training = json.loads(training_path.read_text())
    final_fold = int(config["gate1"]["final_adapter_cv_fold"])
    final_row = cv.loc[cv["fold"] == final_fold]
    if len(final_row) != 1:
        raise AssertionError("the configured final fold is not unique in the completed CV table")
    real_steps = int(final_row.iloc[0]["adapter_optimizer_steps"])
    executed_steps = int(final_row.iloc[0]["adapter_executed_optimizer_steps"])
    if training["optimizer_steps_real"] != real_steps or training["optimizer_steps_shuffled"] != real_steps:
        raise AssertionError("the completed real and shuffled training step counts are not matched")

    reserve = load_assay("IRED", with_sequences=True, include_reserve=True)
    reserve = reserve.loc[reserve["is_layer_freeze_reserve"]].reset_index(drop=True)
    sequences = reserve["mutant_sequence"].astype(str).tolist()
    real_model, real_path = _load_adapter(config, "real", args.device)
    shuffled_model, shuffled_path = _load_adapter(config, "shuffled", args.device)
    batch_size = int(config["extraction"]["reserve_batch_size"])
    real_layers = extract_residue_embeddings(
        real_model.backbone, sequences, batch_size=batch_size, all_layers=True
    ).layers
    shuffled_layers = extract_residue_embeddings(
        shuffled_model.backbone, sequences, batch_size=batch_size, all_layers=True
    ).layers

    step2_cache = REPO_ROOT / step2["paths"]["representations"]
    base_layers = np.asarray(np.load(step2_cache / "reserve_all_layers_base.npy"), dtype=np.float32)
    reserve_rows = np.load(step2_cache / "reserve_rows.npy")
    if not np.array_equal(reserve_rows, reserve["variant_row"].to_numpy(int)):
        raise AssertionError("reserve row alignment changed between extraction and resumed analysis")
    if base_layers.shape != real_layers.shape or base_layers.shape != shuffled_layers.shape:
        raise AssertionError("base, real, and shuffled reserve activation shapes differ")
    base_tokens = load_esmc(
        str(config["backbone"]["name"]), args.device,
        repository=str(config["backbone"]["repository"]),
        expected_revision=str(config["backbone"]["revision"]),
    )._tokenize(sequences[:1])
    if not torch.equal(base_tokens, real_model.backbone._tokenize(sequences[:1])):
        raise AssertionError("real adapter token alignment differs from the base model")
    if not torch.equal(base_tokens, shuffled_model.backbone._tokenize(sequences[:1])):
        raise AssertionError("shuffled adapter token alignment differs from the base model")

    magnitude_floor = 10.0 * np.finfo(np.float32).eps * np.sqrt(base_layers.shape[-1])
    maximum_change = float(np.linalg.norm(real_layers - base_layers, axis=-1).max())
    if maximum_change <= magnitude_floor:
        raise AssertionError("adapted activation change does not exceed the numerical floor")
    real_drift = tokenwise_relative_drift(base_layers, real_layers)
    shuffled_drift = tokenwise_relative_drift(base_layers, shuffled_layers)
    real_cka = linear_cka_by_layer(base_layers, real_layers, device=args.device)
    shuffled_cka = linear_cka_by_layer(base_layers, shuffled_layers, device=args.device)
    base_norm, real_norm, norm_ratio = activation_norms_by_layer(base_layers, real_layers)
    lower, upper = bootstrap_drift_gap(
        real_drift, shuffled_drift,
        samples=int(config["gate1"]["bootstrap_samples"]),
        seed=int(config["gate1"]["bootstrap_seed"]),
        confidence_level=float(config["gate1"]["confidence_level"]),
    )
    gap = real_drift.mean(axis=0) - shuffled_drift.mean(axis=0)
    gate = config["gate1"]
    selected_layer, eligible, near_max, near_max_threshold = select_analysis_layer(
        gap, real_cka,
        cka_floor=float(gate["real_cka_floor"]),
        exclude_first=int(gate["exclude_first_layers"]),
        exclude_last=int(gate["exclude_last_layers"]),
        near_max_fraction=float(gate["near_maximum_drift_gap_fraction"]),
    )
    layer_frame = pd.DataFrame({
        "layer": np.arange(base_layers.shape[1]),
        "real_relative_drift": real_drift.mean(axis=0),
        "shuffled_relative_drift": shuffled_drift.mean(axis=0),
        "drift_gap": gap,
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

    task_test = paired_task_gain_test(cv["task_gain"].to_numpy())
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
                "drift_gap": float(gap[layer]),
                "drift_gap_95_ci": [float(lower[layer]), float(upper[layer])],
                "real_cka": float(real_cka[layer]),
                "activation_norm_ratio": float(norm_ratio[layer]),
            })
    drift_ratio = float(real_drift[:, selected_layer].mean() / shuffled_drift[:, selected_layer].mean())
    ratio_window = list(map(float, gate["drift_ratio_window"]))
    result = {
        "dataset": "IRED",
        "task_gain_mean": mean_gain,
        "frozen_spearman_mean": float(cv["frozen_spearman"].mean()),
        "adapted_spearman_mean": float(cv["adapted_spearman"].mean()),
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
        "optimizer_steps_real": real_steps,
        "optimizer_steps_real_executed_before_restore": executed_steps,
        "optimizer_steps_shuffled": real_steps,
        "matched_learning_rate_schedule": True,
        "resumed_from_complete_checkpoints": True,
        "gate1_pass": bool(gate_pass),
        "gate_failure_reasons": ([] if task_gain_established else ["task_gain_not_established"]),
    }
    if not gate_pass:
        result["run_complete"] = True
        (report_dir / "gate1_report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return 2

    selected_cache = REPO_ROOT / config["paths"]["cache"]
    selected_cache.mkdir(parents=True)
    np.save(selected_cache / "reserve_all_layers_adapted.npy", real_layers.astype(np.float16))
    np.save(selected_cache / "reserve_all_layers_shuffled.npy", shuffled_layers.astype(np.float16))
    full = load_assay("IRED", with_sequences=True, include_reserve=True)
    full_sequences = full["mutant_sequence"].astype(str).tolist()
    base_model = load_esmc(
        str(config["backbone"]["name"]), args.device,
        repository=str(config["backbone"]["repository"]),
        expected_revision=str(config["backbone"]["revision"]),
    )
    base_selected = extract_residue_layer(
        base_model, full_sequences,
        batch_size=int(config["extraction"]["full_dataset_batch_size"]), layer=selected_layer,
    )
    real_selected = extract_residue_layer(
        real_model.backbone, full_sequences,
        batch_size=int(config["extraction"]["full_dataset_batch_size"]), layer=selected_layer,
    )
    np.save(selected_cache / "full_selected_layer_base.npy", base_selected.astype(np.float16))
    np.save(selected_cache / "full_selected_layer_adapted.npy", real_selected.astype(np.float16))
    np.save(selected_cache / "full_variant_rows.npy", full["variant_row"].to_numpy(np.int64))
    result["run_complete"] = True
    result["selected_cache_shapes"] = {
        "reserve_all_layers_adapted": list(real_layers.shape),
        "reserve_all_layers_shuffled": list(shuffled_layers.shape),
        "full_selected_layer_base": list(base_selected.shape),
        "full_selected_layer_adapted": list(real_selected.shape),
    }
    (report_dir / "gate1_report.json").write_text(json.dumps(result, indent=2) + "\n")
    verification = {
        "training_restart": "authoritative training completed before resumed eval-mode analysis",
        "optimizer_steps_each": real_steps,
        "real": _checkpoint_record(real_path),
        "shuffled": _checkpoint_record(shuffled_path),
        "reserve_extraction_mode": "eval",
    }
    (report_dir / "gate1_checkpoint_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
