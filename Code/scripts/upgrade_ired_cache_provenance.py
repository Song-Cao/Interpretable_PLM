#!/usr/bin/env python3
"""Add the v18 provenance contract to active historical IRED activation caches."""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.adaptation import SequenceFitnessAdapter, build_lora_adapter, load_trainable_state
from crossfit.cache_provenance import runtime_provenance, sha256_file, write_reference_block
from crossfit.datasets import wt_sequence
from crossfit.esmc import extract_residue_embeddings, extract_residue_layer, load_esmc
from crossfit.utils import REPO_ROOT, load_config


def _backbone(step3: dict):
    return load_esmc(
        str(step3["backbone"]["name"]), "cuda",
        repository=str(step3["backbone"]["repository"]),
        expected_revision=str(step3["backbone"]["revision"]),
    )


def _adapter(step3: dict, checkpoint: Path, dimension: int):
    backbone = _backbone(step3)
    lora = build_lora_adapter(
        backbone,
        rank=int(step3["adapter"]["rank"]),
        alpha=int(step3["adapter"]["alpha"]),
        dropout=float(step3["adapter"]["dropout"]),
    )
    wrapped = SequenceFitnessAdapter(lora, dimension).cuda()
    load_trainable_state(wrapped, checkpoint)
    return wrapped.eval()


def _historical(
    *, dataset: str, split: str, layer: int | str, units: str,
    references: dict, revision: str, storage_dtype: str,
) -> dict:
    return {
        "provenance_contract_version": 1,
        "dataset": dataset,
        "split": split,
        "layer_index": layer,
        "device_class": "NVIDIA A100 (historical cache-producing run)",
        "storage_dtype": storage_dtype,
        "compute_dtype": "not_recorded_in_historical_run",
        "tf32": {
            "matmul_allow_tf32": "not_recorded_in_historical_run",
            "cudnn_allow_tf32": "not_recorded_in_historical_run",
            "float32_matmul_precision": "not_recorded_in_historical_run",
        },
        "attention_kernel_settings": {"realized": "not_recorded_in_historical_run"},
        "backbone_revision": revision,
        "activation_units": units,
        "reference_blocks": references,
        "historical_provenance_completeness": "partial_fields_recovered_retroactively",
    }


def _write(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("cache provenance upgrade requires a GPU reference pass")
    step3 = load_config(REPO_ROOT / "Code/config/step3.yaml")
    step2 = load_config(REPO_ROOT / "Code/config/step2.yaml")
    step4 = load_config(REPO_ROOT / "Code/config/step4.yaml")
    step6 = load_config(REPO_ROOT / "Code/config/step6.yaml")
    provenance = load_config(REPO_ROOT / "Code/config/cache_provenance.yaml")
    sequence = wt_sequence("IRED")
    revision = str(step3["backbone"]["revision"])
    layer = int(step4["analysis_layer"])
    dimension = int(step4["embedding_dimension"])
    requested = str(provenance["runtime"]["requested_attention_implementation"])
    reference_runtime = runtime_provenance(requested)

    base = _backbone(step3)
    final_base = extract_residue_embeddings(base, [sequence], batch_size=1).final[0]
    all_base_batch = extract_residue_embeddings(base, [sequence], batch_size=1, all_layers=True)
    all_base = all_base_batch.layers[0]
    depth26_base = extract_residue_layer(base, [sequence], layer=layer, batch_size=1)[0]
    del base
    gc.collect()
    torch.cuda.empty_cache()

    real_checkpoint = REPO_ROOT / step4["paths"]["adapter_checkpoint"]
    real = _adapter(step3, real_checkpoint, dimension)
    all_fit_batch = extract_residue_embeddings(real.backbone, [sequence], batch_size=1, all_layers=True)
    all_fit = all_fit_batch.layers[0]
    depth26_fit = extract_residue_layer(real.backbone, [sequence], layer=layer, batch_size=1)[0]
    del real
    gc.collect()
    torch.cuda.empty_cache()

    shuffled_checkpoint = REPO_ROOT / step6["paths"]["shuffled_adapter_checkpoint"]
    shuffled = _adapter(step3, shuffled_checkpoint, dimension)
    all_shuffled_batch = extract_residue_embeddings(
        shuffled.backbone, [sequence], batch_size=1, all_layers=True
    )
    all_shuffled = all_shuffled_batch.layers[0]
    depth26_shuffled = extract_residue_layer(
        shuffled.backbone, [sequence], layer=layer, batch_size=1
    )[0]
    del shuffled
    gc.collect()
    torch.cuda.empty_cache()

    matched_checkpoint = REPO_ROOT / step6["paths"]["drift_matched_adapter_checkpoint"]
    matched = _adapter(step3, matched_checkpoint, dimension)
    depth26_matched = extract_residue_layer(
        matched.backbone, [sequence], layer=layer, batch_size=1
    )[0]
    del matched
    gc.collect()
    torch.cuda.empty_cache()

    step2_dir = REPO_ROOT / step2["paths"]["representations"]
    step2_path = step2_dir / "metadata.json"
    step2_old = json.loads(step2_path.read_text())
    step2_reference = write_reference_block(step2_dir, "base", final_base, activation_units="raw")
    step2_manifest = {
        **step2_old,
        **_historical(
            dataset="IRED", split="campaign_all_plus_external_plate_and_reserve",
            layer="final", units="raw", references={"base": step2_reference},
            revision=revision, storage_dtype="float16",
        ),
        "pre_v18_manifest_sha256": sha256_file(step2_path),
        "reference_extraction_runtime": reference_runtime,
        "dimension": dimension,
    }
    _write(step2_path, step2_manifest)

    step3_dir = REPO_ROOT / step3["paths"]["cache"]
    step3_references = {
        "base": write_reference_block(step3_dir, "base", all_base, activation_units="raw"),
        "fit": write_reference_block(step3_dir, "fit", all_fit, activation_units="raw"),
        "shuffled": write_reference_block(
            step3_dir, "shuffled", all_shuffled, activation_units="raw"
        ),
    }
    step3_manifest = {
        "status": "complete",
        **_historical(
            dataset="IRED", split="layer_freeze_reserve_only", layer="all", units="raw",
            references=step3_references, revision=revision, storage_dtype="float16",
        ),
        "dimension": dimension,
        "reserve_variants": int(
            np.load(step2_dir / "reserve_all_layers_base.npy", mmap_mode="r").shape[0]
        ),
        "reference_extraction_runtime": reference_runtime,
        "payload": {
            "reserve_all_layers_base.npy": {
                "path": "Cache/step2/IRED/representations/reserve_all_layers_base.npy",
                "sha256": sha256_file(step2_dir / "reserve_all_layers_base.npy"),
            },
            "reserve_all_layers_adapted.npy": {
                "path": "Cache/step3/IRED/reserve_all_layers_adapted.npy",
                "sha256": sha256_file(step3_dir / "reserve_all_layers_adapted.npy"),
            },
            "reserve_all_layers_shuffled.npy": {
                "path": "Cache/step3/IRED/reserve_all_layers_shuffled.npy",
                "sha256": sha256_file(step3_dir / "reserve_all_layers_shuffled.npy"),
            },
        },
    }
    _write(step3_dir / "manifest.json", step3_manifest)

    step4_dir = REPO_ROOT / step4["paths"]["cache"]
    step4_path = step4_dir / "manifest.json"
    step4_old = json.loads(step4_path.read_text())
    evo_scalar = float(step4_old["models"]["evo"]["normalization_scalar"])
    fit_scalar = float(step4_old["models"]["fit"]["normalization_scalar"])
    step4_references = {
        "base": write_reference_block(
            step4_dir, "base", depth26_base * evo_scalar,
            activation_units="normalized", normalization_scalar=evo_scalar,
        ),
        "fit": write_reference_block(
            step4_dir, "fit", depth26_fit * fit_scalar,
            activation_units="normalized", normalization_scalar=fit_scalar,
        ),
    }
    step4_manifest = {
        **step4_old,
        **_historical(
            dataset="IRED", split="pg_kfold_fold_0_training_only", layer=layer,
            units="normalized", references=step4_references, revision=revision,
            storage_dtype="float16",
        ),
        "pre_v18_manifest_sha256": sha256_file(step4_path),
        "reference_extraction_runtime": reference_runtime,
    }
    _write(step4_path, step4_manifest)

    for name, cache_relative, role, values, checkpoint in (
        (
            "shuffled", step6["paths"]["shuffled_cache"], "shuffled",
            depth26_shuffled, shuffled_checkpoint,
        ),
        (
            "drift_matched", step6["paths"]["drift_matched_cache"], "drift_matched",
            depth26_matched, matched_checkpoint,
        ),
    ):
        cache_dir = REPO_ROOT / cache_relative
        manifest_path = cache_dir / "manifest.json"
        old = json.loads(manifest_path.read_text())
        scalar_key = "shuffled" if name == "shuffled" else "drift_matched"
        scalar = float(old["models"][scalar_key]["normalization_scalar"])
        reference = write_reference_block(
            cache_dir, role, values * scalar,
            activation_units="normalized", normalization_scalar=scalar,
        )
        manifest = {
            **old,
            **_historical(
                dataset="IRED", split="pg_kfold_fold_0_training_only", layer=layer,
                units="normalized", references={role: reference}, revision=revision,
                storage_dtype="float16",
            ),
            "dimension": dimension,
            "pre_v18_manifest_sha256": sha256_file(manifest_path),
            "reference_extraction_runtime": reference_runtime,
            "adapter_checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
            "adapter_checkpoint_sha256": sha256_file(checkpoint),
        }
        _write(manifest_path, manifest)

    result = {
        "status": "complete",
        "manifests": [
            str(step2_path.relative_to(REPO_ROOT)),
            str((step3_dir / "manifest.json").relative_to(REPO_ROOT)),
            str(step4_path.relative_to(REPO_ROOT)),
            str((REPO_ROOT / step6["paths"]["shuffled_cache"] / "manifest.json").relative_to(REPO_ROOT)),
            str((REPO_ROOT / step6["paths"]["drift_matched_cache"] / "manifest.json").relative_to(REPO_ROOT)),
        ],
        "reference_extraction_runtime": reference_runtime,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
