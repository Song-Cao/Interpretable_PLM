#!/usr/bin/env python3
"""Cache residue-level IRED mutant-minus-wild-type crosscoder responses."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.adaptation import SequenceFitnessAdapter, build_lora_adapter, load_trainable_state
from crossfit.crosscoder import CrossCoder
from crossfit.datasets import load_assay, wt_sequence
from crossfit.esmc import extract_residue_layer, load_esmc
from crossfit.persistence import sha256_file
from crossfit.structural_annotations import mutation_responsive_codes
from crossfit.utils import REPO_ROOT, load_config


def _backbone(step3: dict):
    return load_esmc(
        str(step3["backbone"]["name"]), "cuda",
        repository=str(step3["backbone"]["repository"]),
        expected_revision=str(step3["backbone"]["revision"]),
    )


def _extract_wild_type(config: dict, step3: dict) -> tuple[np.ndarray, np.ndarray, str]:
    sequence = wt_sequence(str(config["dataset"]))
    base = _backbone(step3)
    base_tokens = base._tokenize([sequence]).detach().cpu()
    evo = extract_residue_layer(
        base, [sequence], layer=int(config["analysis_layer"]), batch_size=1
    )[0]
    del base
    gc.collect()
    torch.cuda.empty_cache()

    checkpoint = REPO_ROOT / config["paths"]["step3_adapter"]
    verification = json.loads(
        (REPO_ROOT / config["paths"]["gate1_checkpoint_verification"]).read_text()
    )
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != verification["real"]["sha256"]:
        raise AssertionError("real adapter hash differs from Gate 1 verification")
    adapted_backbone = _backbone(step3)
    adapted = build_lora_adapter(
        adapted_backbone, rank=int(step3["adapter"]["rank"]),
        alpha=int(step3["adapter"]["alpha"]), dropout=float(step3["adapter"]["dropout"]),
    )
    wrapped = SequenceFitnessAdapter(adapted, int(config["dimension"])).to("cuda")
    load_trainable_state(wrapped, checkpoint)
    wrapped.eval()
    fit_tokens = wrapped.backbone._tokenize([sequence]).detach().cpu()
    if not torch.equal(base_tokens, fit_tokens):
        raise AssertionError("wild-type token alignment differs between base and adapted models")
    fit = extract_residue_layer(
        wrapped.backbone, [sequence], layer=int(config["analysis_layer"]), batch_size=1
    )[0]
    return evo, fit, checkpoint_hash


@torch.inference_mode()
def _cache_width(
    model: CrossCoder,
    evo: np.ndarray,
    fit: np.ndarray,
    wild_evo: np.ndarray,
    wild_fit: np.ndarray,
    destination: Path,
) -> dict:
    device = torch.device("cuda")
    model.to(device).eval()
    wt_left = torch.as_tensor(wild_evo, dtype=torch.float32, device=device)
    wt_right = torch.as_tensor(wild_fit, dtype=torch.float32, device=device)
    wild_code = model.encode(wt_left, wt_right).cpu().numpy().astype(np.float32)
    residue_sum = np.zeros_like(wild_code, dtype=np.float64)
    row_blocks: list[np.ndarray] = []
    column_blocks: list[np.ndarray] = []
    value_blocks: list[np.ndarray] = []
    length = wild_code.shape[0]
    for variant in range(len(evo)):
        left = torch.as_tensor(np.asarray(evo[variant]), dtype=torch.float32, device=device)
        right = torch.as_tensor(np.asarray(fit[variant]), dtype=torch.float32, device=device)
        code = model.encode(left, right).cpu().numpy().astype(np.float32)
        residue_sum += code
        delta = mutation_responsive_codes(code, wild_code)
        row, column = np.nonzero(delta)
        row_blocks.append(row.astype(np.int64) + variant * length)
        column_blocks.append(column.astype(np.int32))
        value_blocks.append(delta[row, column].astype(np.float32))
    rows = np.concatenate(row_blocks)
    columns = np.concatenate(column_blocks)
    values = np.concatenate(value_blocks)
    matrix = sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(evo) * length, model.features), dtype=np.float32
    )
    sparse.save_npz(destination / "delta_z.npz", matrix, compressed=True)
    np.save(destination / "wild_type_codes.npy", wild_code)
    residue_mean = (residue_sum / len(evo)).astype(np.float32)
    np.save(destination / "residue_mean_codes.npy", residue_mean)
    return {
        "shape": list(matrix.shape), "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        "wild_type_shape": list(wild_code.shape),
        "residue_mean_shape": list(residue_mean.shape),
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("delta-z caching requires the GPU pod for wild-type extraction and coding")
    started = time.monotonic()
    config = load_config(REPO_ROOT / "Code/config/step8.yaml")
    step3 = load_config(REPO_ROOT / "Code/config/step3.yaml")
    destination = REPO_ROOT / config["paths"]["delta_z_cache"]
    temporary = destination.with_name(destination.name + ".incomplete")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite delta-z cache: {destination}")
    if temporary.exists():
        raise FileExistsError(f"inspect incomplete delta-z cache before removal: {temporary}")
    temporary.mkdir(parents=True)

    step4 = json.loads((REPO_ROOT / config["paths"]["step4_manifest"]).read_text())
    evo_scalar = float(step4["models"]["evo"]["normalization_scalar"])
    fit_scalar = float(step4["models"]["fit"]["normalization_scalar"])
    evo = np.load(REPO_ROOT / config["paths"]["campaign_base"], mmap_mode="r")
    fit = np.load(REPO_ROOT / config["paths"]["campaign_fit"], mmap_mode="r")
    rows = np.load(REPO_ROOT / config["paths"]["campaign_rows"])
    if evo.shape != fit.shape or len(rows) != len(evo):
        raise AssertionError("campaign activations and variant rows are not aligned")
    frame = load_assay(str(config["dataset"]), with_sequences=True, include_reserve=True)
    expected_rows = frame["variant_row"].to_numpy(np.int64)
    if not np.array_equal(rows.astype(np.int64), expected_rows):
        raise AssertionError("campaign activation rows differ from the curated assay order")
    np.save(temporary / "variant_rows.npy", rows)

    wild_evo, wild_fit, adapter_hash = _extract_wild_type(config, step3)
    wild_evo = wild_evo.astype(np.float32) * evo_scalar
    wild_fit = wild_fit.astype(np.float32) * fit_scalar
    manifest = {
        "status": "complete", "dataset": str(config["dataset"]),
        "definition": "z_j(x_mutant) - z_j(x_WT)",
        "batch_semantics": str(config["delta_z"]["batch_semantics"]),
        "reasoning": str(config["delta_z"]["reasoning"]),
        "analysis_layer": int(config["analysis_layer"]), "variants": int(len(evo)),
        "positions": int(evo.shape[1]), "adapter_checkpoint_sha256": adapter_hash,
        "widths": {}, "enrichment_tests_run": False,
    }
    for width in map(int, config["width_multipliers"]):
        seed = int(config["primary_crosscoder_seed"])
        checkpoint = (
            REPO_ROOT / config["paths"]["crosscoder_checkpoints"]
            / f"w{width}_seed{seed}_nobias.pt"
        )
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = CrossCoder(
            int(config["dimension"]), width, int(artifact["nominal_k"]),
            decoder_bias=bool(artifact["decoder_bias"]),
        )
        model.load_state_dict(artifact["state_dict"])
        width_dir = temporary / f"w{width}"
        width_dir.mkdir()
        details = _cache_width(
            model, evo, fit, wild_evo, wild_fit, width_dir
        )
        details["crosscoder_checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        manifest["widths"][str(width)] = details
    manifest["gpu_wall_hours"] = (time.monotonic() - started) / 3600.0
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, destination)
    report = REPO_ROOT / config["paths"]["reports"]
    report.mkdir(parents=True, exist_ok=True)
    (report / "ired_delta_z_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
