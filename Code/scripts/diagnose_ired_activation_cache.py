#!/usr/bin/env python3
"""Compare persisted IRED activations with single-sequence re-extraction."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.adaptation import SequenceFitnessAdapter, build_lora_adapter, load_trainable_state
from crossfit.datasets import load_assay, wt_sequence
from crossfit.esmc import extract_residue_embeddings, load_esmc
from crossfit.utils import REPO_ROOT, load_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sharded_row(directory: Path, row_index: int) -> np.ndarray:
    offset = int(row_index)
    for path in sorted(directory.glob("shard_*.npy")):
        shard = np.load(path, mmap_mode="r")
        if offset < len(shard):
            return np.asarray(shard[offset])
        offset -= len(shard)
    raise IndexError(f"row {row_index} is outside {directory}")


def _quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    return {
        f"q{int(probability * 100):02d}": float(value)
        for probability, value in zip(probabilities, np.quantile(values, probabilities))
    }


def _comparison(cached: np.ndarray, current: np.ndarray) -> dict:
    if cached.shape != current.shape:
        raise ValueError(f"activation shapes differ: {cached.shape} versus {current.shape}")
    cached_values = cached.astype(np.float64, copy=False)
    current_values = current.astype(np.float64, copy=False)
    old = cached_values.reshape(-1)
    new = current_values.reshape(-1)
    difference = new - old
    magnitude = np.abs(old)
    denominator_floor = max(float(np.quantile(magnitude, 0.01)), np.finfo(np.float16).tiny)
    ratio_mask = magnitude >= denominator_floor
    ratio = new[ratio_mask] / old[ratio_mask]
    slope = float(np.dot(old, new) / np.dot(old, old))
    residual_after_scale = new - slope * old

    edges = np.quantile(magnitude, np.linspace(0.0, 1.0, 11))
    magnitude_bins = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (magnitude >= lower) & (
            magnitude <= upper if index == len(edges) - 2 else magnitude < upper
        )
        magnitude_bins.append({
            "bin": index,
            "cached_abs_lower": float(lower),
            "cached_abs_upper": float(upper),
            "count": int(selected.sum()),
            "mean_absolute_difference": float(np.mean(np.abs(difference[selected]))),
            "difference_sd": float(np.std(difference[selected], ddof=1)),
        })

    result = {
        "element_count": int(old.size),
        "pearson": float(pearsonr(old, new).statistic),
        "spearman": float(spearmanr(old, new).statistic),
        "cached_quantiles": _quantiles(old),
        "current_quantiles": _quantiles(new),
        "difference_quantiles": _quantiles(difference),
        "absolute_difference_quantiles": _quantiles(np.abs(difference)),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "difference_vs_cached_absolute_magnitude_pearson": float(
            pearsonr(np.abs(difference), magnitude).statistic
        ),
        "ratio_denominator_absolute_floor": denominator_floor,
        "ratio_element_count": int(ratio.size),
        "ratio_quantiles": _quantiles(ratio),
        "through_origin_scale": slope,
        "residual_after_scale_absolute_quantiles": _quantiles(np.abs(residual_after_scale)),
        "magnitude_bins": magnitude_bins,
    }
    if cached_values.ndim == 2:
        position_correlations = []
        position_slopes = []
        position_intercepts = []
        residuals = []
        for cached_position, current_position in zip(cached_values, current_values):
            slope, intercept = np.polyfit(cached_position, current_position, 1)
            position_correlations.append(pearsonr(cached_position, current_position).statistic)
            position_slopes.append(slope)
            position_intercepts.append(intercept)
            residuals.append(current_position - (slope * cached_position + intercept))
        residual = np.concatenate(residuals)
        result["per_position_affine_alignment"] = {
            "pearson_quantiles": _quantiles(np.asarray(position_correlations)),
            "slope_quantiles": _quantiles(np.asarray(position_slopes)),
            "intercept_quantiles": _quantiles(np.asarray(position_intercepts)),
            "residual_absolute_quantiles": _quantiles(np.abs(residual)),
            "residual_rmse": float(np.sqrt(np.mean(residual ** 2))),
        }
    return result


def _layer_alignment(cached: np.ndarray, current: np.ndarray, selected_layer: int) -> dict:
    if cached.shape != current.shape or cached.ndim != 3:
        raise ValueError("layer alignment expects matching layer-position-channel arrays")
    same_depth = []
    for depth in range(cached.shape[0]):
        old = cached[depth].astype(np.float64, copy=False).reshape(-1)
        new = current[depth].astype(np.float64, copy=False).reshape(-1)
        same_depth.append({
            "depth": depth,
            "pearson": float(pearsonr(old, new).statistic),
            "spearman": float(spearmanr(old, new).statistic),
        })
    selected = cached[selected_layer].astype(np.float64, copy=False).reshape(-1)
    cross_depth = []
    for depth in range(current.shape[0]):
        candidate = current[depth].astype(np.float64, copy=False).reshape(-1)
        cross_depth.append({
            "current_depth": depth,
            "pearson_to_cached_selected_depth": float(pearsonr(selected, candidate).statistic),
        })
    return {
        "same_depth": same_depth,
        "cached_selected_depth_against_every_current_depth": cross_depth,
        "best_current_depth": max(
            cross_depth, key=lambda row: row["pearson_to_cached_selected_depth"]
        ),
    }


def _stored_layer_against_current_depths(cached: np.ndarray, current: np.ndarray) -> dict:
    old = cached.astype(np.float64, copy=False).reshape(-1)
    rows = []
    for depth in range(current.shape[0]):
        candidate = current[depth].astype(np.float64, copy=False).reshape(-1)
        rows.append({
            "current_depth": depth,
            "pearson": float(pearsonr(old, candidate).statistic),
            "spearman": float(spearmanr(old, candidate).statistic),
            "through_origin_scale": float(np.dot(old, candidate) / np.dot(old, old)),
            "maximum_absolute_difference": float(np.max(np.abs(candidate - old))),
        })
    return {
        "by_current_depth": rows,
        "best_current_depth_by_pearson": max(rows, key=lambda row: row["pearson"]),
    }


def _runtime() -> dict:
    import esm
    import peft

    return {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "esm": esm.__version__,
        "peft": peft.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "model_parameter_dtype": "bfloat16",
        "extraction_return_dtype": "float32",
        "autocast_enabled": bool(torch.is_autocast_enabled("cuda")),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "requested_attention_implementation": "flash_attention_2",
        "flash_attn_installed": importlib.util.find_spec("flash_attn") is not None,
        "xformers_installed": importlib.util.find_spec("xformers") is not None,
        "transformer_engine_installed": importlib.util.find_spec("transformer_engine") is not None,
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("cache diagnosis requires the GPU pod")
    step3 = load_config(REPO_ROOT / "Code/config/step3.yaml")
    step8 = load_config(REPO_ROOT / "Code/config/step8.yaml")
    cache = REPO_ROOT / step3["paths"]["cache"]
    checkpoint = REPO_ROOT / step8["paths"]["step3_adapter"]
    rows = np.load(cache / "full_variant_rows.npy")
    frame = load_assay("IRED", with_sequences=True, include_reserve=True)
    if not np.array_equal(rows, frame["variant_row"].to_numpy(np.int64)):
        raise AssertionError("cached campaign rows no longer align with the curated loader")

    variant_index = 0
    variant = frame.iloc[variant_index]
    repository = str(step3["backbone"]["repository"])
    revision = str(step3["backbone"]["revision"])
    base = load_esmc(
        str(step3["backbone"]["name"]), "cuda",
        repository=repository, expected_revision=revision,
    )
    adapted_backbone = load_esmc(
        str(step3["backbone"]["name"]), "cuda",
        repository=repository, expected_revision=revision,
    )
    adapted = SequenceFitnessAdapter(
        build_lora_adapter(
            adapted_backbone,
            rank=int(step3["adapter"]["rank"]),
            alpha=int(step3["adapter"]["alpha"]),
            dropout=float(step3["adapter"]["dropout"]),
        ),
        960,
    ).cuda()
    load_trainable_state(adapted, checkpoint)
    adapted.eval()

    analysis_layer = int(step8["analysis_layer"])
    current_variant_base_layers = extract_residue_embeddings(
        base, [str(variant["mutant_sequence"])], batch_size=1, all_layers=True,
    ).layers[0]
    current_variant_layers = extract_residue_embeddings(
        adapted.backbone, [str(variant["mutant_sequence"])], batch_size=1, all_layers=True,
    ).layers[0]
    current_variant_base = current_variant_base_layers[analysis_layer]
    current_variant = current_variant_layers[analysis_layer]
    cached_base_file = cache / "full_selected_layer_base.npy"
    cached_variant_base = np.load(cached_base_file, mmap_mode="r")[variant_index]
    cached_variant_file = cache / "full_selected_layer_adapted.npy"
    cached_variant = np.load(cached_variant_file, mmap_mode="r")[variant_index]

    wt = wt_sequence("IRED")
    current_wt_final = extract_residue_embeddings(
        base, [wt], batch_size=1, all_layers=False,
    ).final[0]
    cached_wt_file = REPO_ROOT / "Cache/step2/IRED/representations/wild_type_residue.npy"
    cached_wt_final = np.load(cached_wt_file, mmap_mode="r")

    reserve = frame.loc[frame["is_layer_freeze_reserve"]].reset_index(drop=True)
    reserve_rows = np.load(REPO_ROOT / "Cache/step2/IRED/representations/reserve_rows.npy")
    if not np.array_equal(reserve_rows, reserve["variant_row"].to_numpy(np.int64)):
        raise AssertionError("reserve cache and curated rows do not align")
    current_reserve_layers = extract_residue_embeddings(
        base, [str(reserve.iloc[0]["mutant_sequence"])], batch_size=1, all_layers=True,
    ).layers[0]
    cached_reserve_layers = np.load(
        REPO_ROOT / "Cache/step2/IRED/representations/reserve_all_layers_base.npy",
        mmap_mode="r",
    )[0]

    step4 = json.loads((REPO_ROOT / step8["paths"]["step4_manifest"]).read_text())
    normalized_scalar = float(step4["models"]["fit"]["normalization_scalar"])
    step4_root = REPO_ROOT / "Cache/step4/IRED"
    step4_rows = np.load(step4_root / "variant_rows.npy")
    step4_row = int(step4_rows[0])
    step4_frame_index = int(np.flatnonzero(rows == step4_row)[0])
    step4_variant = frame.iloc[step4_frame_index]
    current_step4_base = extract_residue_embeddings(
        base, [str(step4_variant["mutant_sequence"])], batch_size=1, all_layers=True,
    ).layers[0, analysis_layer]
    current_step4_fit = extract_residue_embeddings(
        adapted.backbone, [str(step4_variant["mutant_sequence"])], batch_size=1, all_layers=True,
    ).layers[0, analysis_layer]
    cached_step4_base_normalized = _sharded_row(step4_root / "evo", 0)
    cached_step4_fit_normalized = _sharded_row(step4_root / "fit", 0)
    evo_scalar = float(step4["models"]["evo"]["normalization_scalar"])
    fit_scalar = float(step4["models"]["fit"]["normalization_scalar"])
    result = {
        "status": "complete_diagnosis_only",
        "diagnosis": "superseded_step3_files_are_depth29_not_depth26",
        "downstream_action": "stop_without_reextraction_or_recomputation",
        "authoritative_step4_depth26_cache_valid": True,
        "dataset": "IRED",
        "analysis_layer": analysis_layer,
        "model_repository": repository,
        "model_revision": revision,
        "adapter_checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "adapter_sha256": _sha256(checkpoint),
        "runtime": _runtime(),
        "cache_provenance": {
            "device": "NVIDIA A100",
            "torch": "not recorded",
            "cuda_runtime": "not recorded",
            "esm": str(step3["backbone"]["esm_package"]),
            "model_parameter_dtype": "bfloat16",
            "extraction_intermediate_dtype": "float32",
            "persisted_dtype": str(cached_variant.dtype),
            "autocast_enabled": "not recorded; extraction code did not enter autocast",
            "matmul_allow_tf32": "not recorded",
            "cudnn_allow_tf32": "not recorded",
            "float32_matmul_precision": "not recorded",
            "requested_attention_implementation": "flash_attention_2 (ESMC default)",
            "realized_attention_kernel": "not recorded",
            "transformer_engine_installed": "not recorded",
        },
        "cache_conventions": {
            "step3_selected_layer_file": str(cached_variant_file.relative_to(REPO_ROOT)),
            "step3_selected_layer_is_raw": True,
            "step3_selected_layer_normalization_applied": False,
            "step4_crosscoder_scalar": normalized_scalar,
            "step4_normalized_value_definition": "step3_raw_activation * scalar",
            "current_reextraction_is_raw": True,
            "current_reextraction_normalization_applied": False,
            "wild_type_layer26_persisted": False,
            "wild_type_available_cache": str(cached_wt_file.relative_to(REPO_ROOT)),
            "wild_type_available_cache_layer": "final model embedding after final normalization",
        },
        "wild_type_final_layer_base_comparison": _comparison(cached_wt_final, current_wt_final),
        "base_all_layer_alignment": {
            "variant_row": int(reserve.iloc[0]["variant_row"]),
            "mutations": str(reserve.iloc[0]["mutations"]),
            "statistics": _layer_alignment(
                cached_reserve_layers, current_reserve_layers, analysis_layer,
            ),
        },
        "direct_layer26_base_comparison": {
            "variant_row": int(variant["variant_row"]),
            "mutations": str(variant["mutations"]),
            "cached_file": str(cached_base_file.relative_to(REPO_ROOT)),
            "cached_dtype": str(cached_variant_base.dtype),
            "current_return_dtype": str(current_variant_base.dtype),
            "statistics": _comparison(cached_variant_base, current_variant_base),
            "stored_file_against_every_current_depth": _stored_layer_against_current_depths(
                cached_variant_base, current_variant_base_layers,
            ),
        },
        "direct_layer26_adapted_comparison": {
            "substitute_reason": (
                "No wild-type layer-26 activation was persisted; use one exactly aligned cached "
                "campaign sequence for the direct selected-layer test."
            ),
            "variant_row": int(variant["variant_row"]),
            "mutations": str(variant["mutations"]),
            "sequence_length": len(str(variant["mutant_sequence"])),
            "cached_dtype": str(cached_variant.dtype),
            "current_return_dtype": str(current_variant.dtype),
            "statistics": _comparison(cached_variant, current_variant),
            "stored_file_against_every_current_depth": _stored_layer_against_current_depths(
                cached_variant, current_variant_layers,
            ),
        },
        "authoritative_step4_depth26_comparison": {
            "variant_row": step4_row,
            "mutations": str(step4_variant["mutations"]),
            "cache_is_normalized": True,
            "evo_normalization_scalar": evo_scalar,
            "fit_normalization_scalar": fit_scalar,
            "base_after_reversing_fixed_scalar": _comparison(
                cached_step4_base_normalized.astype(np.float32) / evo_scalar,
                current_step4_base,
            ),
            "fit_after_reversing_fixed_scalar": _comparison(
                cached_step4_fit_normalized.astype(np.float32) / fit_scalar,
                current_step4_fit,
            ),
        },
        "within_runtime_adaptation_comparisons": {
            "cached_fit_vs_cached_base": _comparison(cached_variant_base, cached_variant),
            "current_fit_vs_current_base": _comparison(current_variant_base, current_variant),
        },
    }
    report = REPO_ROOT / "Reports/step8/v17_activation_cache_diagnosis.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
