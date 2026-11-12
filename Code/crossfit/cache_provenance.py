"""Record and verify provenance for activation-derived caches."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


REQUIRED_FIELDS = (
    "provenance_contract_version",
    "dataset",
    "split",
    "layer_index",
    "device_class",
    "storage_dtype",
    "compute_dtype",
    "tf32",
    "attention_kernel_settings",
    "backbone_revision",
    "activation_units",
    "reference_blocks",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_provenance(requested_attention: str) -> dict[str, object]:
    """Describe the torch and CUDA settings that affect activation extraction."""
    cuda = torch.cuda.is_available()
    device_class = torch.cuda.get_device_name(0) if cuda else "cpu"
    tf32 = {
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if cuda else False,
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32) if cuda else False,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    attention = {"requested": requested_attention}
    if cuda:
        for key, attribute in (
            ("flash_sdp_enabled", "flash_sdp_enabled"),
            ("memory_efficient_sdp_enabled", "mem_efficient_sdp_enabled"),
            ("math_sdp_enabled", "math_sdp_enabled"),
            ("cudnn_sdp_enabled", "cudnn_sdp_enabled"),
        ):
            function = getattr(torch.backends.cuda, attribute, None)
            attention[key] = bool(function()) if function is not None else "unavailable"
    return {
        "device_class": device_class,
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda) if cuda else None,
        "tf32": tf32,
        "attention_kernel_settings": attention,
    }


def write_reference_block(
    cache_dir: Path,
    role: str,
    values: np.ndarray,
    *,
    activation_units: str,
    normalization_scalar: float = 1.0,
) -> dict[str, object]:
    """Persist one canonical reference block and return its manifest record."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"reference_wild_type_{role}.npy"
    temporary = path.with_suffix(".npy.tmp")
    array = np.asarray(values, dtype=np.float32)
    with temporary.open("wb") as stream:
        np.save(stream, array)
    os.replace(temporary, path)
    return {
        "sequence_role": "wild_type",
        "file": path.name,
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "activation_units": activation_units,
        "normalization_scalar": float(normalization_scalar),
    }


def copy_reference_blocks(source_manifest: Path, destination: Path) -> dict[str, dict[str, object]]:
    """Copy canonical blocks into a derived cache and return updated records."""
    source = validate_manifest(source_manifest)
    destination.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, object]] = {}
    for role, value in source["reference_blocks"].items():
        record = dict(value)
        source_path = source_manifest.parent / str(record["file"])
        target_path = destination / str(record["file"])
        shutil.copy2(source_path, target_path)
        if sha256_file(target_path) != record["sha256"]:
            raise AssertionError(f"copied reference checksum failed: {target_path}")
        records[str(role)] = record
    return records


def manifest_provenance(
    *,
    contract_version: int,
    dataset: str,
    split: str,
    layer_index: int | str,
    storage_dtype: str,
    compute_dtype: str,
    backbone_revision: str,
    activation_units: str,
    reference_blocks: Mapping[str, Mapping[str, object]],
    requested_attention: str,
) -> dict[str, object]:
    """Build the required provenance section for a cache manifest."""
    runtime = runtime_provenance(requested_attention)
    return {
        "provenance_contract_version": int(contract_version),
        "dataset": dataset,
        "split": split,
        "layer_index": layer_index,
        "device_class": runtime["device_class"],
        "storage_dtype": storage_dtype,
        "compute_dtype": compute_dtype,
        "tf32": runtime["tf32"],
        "attention_kernel_settings": runtime["attention_kernel_settings"],
        "torch_version": runtime["torch_version"],
        "cuda_version": runtime["cuda_version"],
        "backbone_revision": backbone_revision,
        "activation_units": activation_units,
        "reference_blocks": dict(reference_blocks),
    }


def validate_manifest(manifest_path: Path) -> dict[str, object]:
    """Load a provenance manifest and require all v18 fields."""
    manifest = json.loads(manifest_path.read_text())
    missing = [field for field in REQUIRED_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"cache provenance is incomplete at {manifest_path}: {missing}")
    references = manifest["reference_blocks"]
    if not isinstance(references, dict) or not references:
        raise ValueError(f"cache has no canonical reference blocks: {manifest_path}")
    return manifest


def _agreement(reference: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    expected = np.asarray(reference, dtype=np.float64).ravel()
    actual = np.asarray(observed, dtype=np.float64).ravel()
    if expected.shape != actual.shape:
        raise AssertionError(f"reference shape {expected.shape} differs from fresh shape {actual.shape}")
    centered_expected = expected - expected.mean()
    centered_actual = actual - actual.mean()
    denominator = np.linalg.norm(centered_expected) * np.linalg.norm(centered_actual)
    pearson = float(np.dot(centered_expected, centered_actual) / denominator)
    relative_rmse = float(
        np.sqrt(np.mean((actual - expected) ** 2))
        / max(np.sqrt(np.mean(expected ** 2)), np.finfo(np.float64).eps)
    )
    scale = float(np.dot(actual, expected) / np.dot(expected, expected))
    return {
        "pearson": pearson,
        "relative_rmse": relative_rmse,
        "through_origin_scale": scale,
        "maximum_absolute_difference": float(np.max(np.abs(actual - expected))),
    }


def verify_reference_blocks(
    manifest_path: Path,
    fresh_blocks: Mapping[str, np.ndarray],
    verification: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Verify checksums and fresh forward-pass blocks against one manifest."""
    manifest = validate_manifest(manifest_path)
    results: dict[str, dict[str, float]] = {}
    for role, fresh in fresh_blocks.items():
        if role not in manifest["reference_blocks"]:
            raise KeyError(f"reference role {role!r} is absent from {manifest_path}")
        record = manifest["reference_blocks"][role]
        path = manifest_path.parent / str(record["file"])
        if sha256_file(path) != record["sha256"]:
            raise AssertionError(f"canonical reference checksum failed: {path}")
        reference = np.load(path)
        observed = np.asarray(fresh, dtype=np.float32) * float(record["normalization_scalar"])
        metrics = _agreement(reference, observed)
        if metrics["pearson"] < float(verification["minimum_pearson"]):
            raise AssertionError(f"cache reference Pearson failed for {role}: {metrics}")
        if metrics["relative_rmse"] > float(verification["maximum_relative_rmse"]):
            raise AssertionError(f"cache reference relative RMSE failed for {role}: {metrics}")
        if not (
            float(verification["minimum_through_origin_scale"])
            <= metrics["through_origin_scale"]
            <= float(verification["maximum_through_origin_scale"])
        ):
            raise AssertionError(f"cache reference scale failed for {role}: {metrics}")
        results[role] = metrics
    return results


def verify_ired_esmc_cache(
    manifest_path: Path,
    *,
    step3: Mapping[str, object],
    verification: Mapping[str, float],
    adapter_checkpoint: Path | None,
    adapter_checkpoints: Mapping[str, Path] | None = None,
    device: str = "cuda",
) -> dict[str, dict[str, float]]:
    """Re-extract IRED wild type and verify an active ESM-C cache."""
    from crossfit.adaptation import (
        SequenceFitnessAdapter,
        build_lora_adapter,
        load_trainable_state,
    )
    from crossfit.datasets import wt_sequence
    from crossfit.esmc import extract_residue_embeddings, extract_residue_layer, load_esmc

    manifest = validate_manifest(manifest_path)
    if str(manifest["backbone_revision"]) != str(step3["backbone"]["revision"]):
        raise AssertionError("cache backbone revision differs from the configured revision")
    sequence = wt_sequence(str(manifest["dataset"]))
    layer = manifest["layer_index"]

    def extract(model: torch.nn.Module) -> np.ndarray:
        if layer == "final":
            return extract_residue_embeddings(model, [sequence], batch_size=1).final[0]
        if layer == "all":
            result = extract_residue_embeddings(model, [sequence], batch_size=1, all_layers=True)
            if result.layers is None:
                raise AssertionError("all-layer extraction did not return hidden states")
            return result.layers[0]
        return extract_residue_layer(model, [sequence], layer=int(layer), batch_size=1)[0]

    backbone = step3["backbone"]
    base = load_esmc(
        str(backbone["name"]), device,
        repository=str(backbone["repository"]),
        expected_revision=str(backbone["revision"]),
    )
    fresh = {"base": extract(base)}
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checkpoints = dict(adapter_checkpoints or {})
    if adapter_checkpoint is not None:
        checkpoints.setdefault("fit", adapter_checkpoint)
    for role in manifest["reference_blocks"]:
        if role == "base":
            continue
        if role not in checkpoints:
            raise ValueError(f"reference role {role!r} requires an adapter checkpoint")
        if role == "fit" and "adapter_checkpoint_sha256" in manifest:
            if sha256_file(checkpoints[role]) != manifest["adapter_checkpoint_sha256"]:
                raise AssertionError("cache adapter checkpoint differs from the manifest")
        adapted = load_esmc(
            str(backbone["name"]), device,
            repository=str(backbone["repository"]),
            expected_revision=str(backbone["revision"]),
        )
        lora = build_lora_adapter(
            adapted,
            rank=int(step3["adapter"]["rank"]),
            alpha=int(step3["adapter"]["alpha"]),
            dropout=float(step3["adapter"]["dropout"]),
        )
        wrapped = SequenceFitnessAdapter(lora, int(manifest["dimension"])).to(device)
        load_trainable_state(wrapped, checkpoints[role])
        wrapped.eval()
        fresh[role] = extract(wrapped.backbone)
        del wrapped
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return verify_reference_blocks(manifest_path, fresh, verification)
