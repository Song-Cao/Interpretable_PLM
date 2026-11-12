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
from crossfit.esmc import extract_residue_embeddings, extract_residue_layer, load_esmc
from crossfit.utils import REPO_ROOT, load_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    return {
        f"q{int(probability * 100):02d}": float(value)
        for probability, value in zip(probabilities, np.quantile(values, probabilities))
    }


def _comparison(cached: np.ndarray, current: np.ndarray) -> dict:
    if cached.shape != current.shape:
        raise ValueError(f"activation shapes differ: {cached.shape} versus {current.shape}")
    old = cached.astype(np.float64, copy=False).reshape(-1)
    new = current.astype(np.float64, copy=False).reshape(-1)
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

    return {
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
    current_variant_base = extract_residue_layer(
        base, [str(variant["mutant_sequence"])],
        layer=analysis_layer, batch_size=1,
    )[0]
    current_variant = extract_residue_layer(
        adapted.backbone, [str(variant["mutant_sequence"])],
        layer=analysis_layer, batch_size=1,
    )[0]
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

    step4 = json.loads((REPO_ROOT / step8["paths"]["step4_manifest"]).read_text())
    normalized_scalar = float(step4["models"]["fit"]["normalization_scalar"])
    result = {
        "status": "complete_diagnosis_only",
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
        "direct_layer26_base_comparison": {
            "variant_row": int(variant["variant_row"]),
            "mutations": str(variant["mutations"]),
            "cached_file": str(cached_base_file.relative_to(REPO_ROOT)),
            "cached_dtype": str(cached_variant_base.dtype),
            "current_return_dtype": str(current_variant_base.dtype),
            "statistics": _comparison(cached_variant_base, current_variant_base),
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
