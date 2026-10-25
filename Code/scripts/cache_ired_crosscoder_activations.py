#!/usr/bin/env python3
"""Cache normalized IRED depth-26 base/adapted activations for crosscoding."""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.activation_cache import normalization_scalar, normalize_shards_in_place, shard_sha256
from crossfit.adaptation import SequenceFitnessAdapter, build_lora_adapter, load_trainable_state
from crossfit.datasets import get_split, load_assay
from crossfit.esmc import extract_residue_layer, load_esmc
from crossfit.persistence import sha256_file
from crossfit.utils import REPO_ROOT, load_config


def _load_backbone(step3: dict, device: str):
    return load_esmc(
        str(step3["backbone"]["name"]), device,
        repository=str(step3["backbone"]["repository"]),
        expected_revision=str(step3["backbone"]["revision"]),
    )


def _extract_model(
    model: torch.nn.Module,
    sequences: list[str],
    output_dir: Path,
    *,
    layer: int,
    batch_size: int,
    variants_per_shard: int,
    dimension: int,
) -> tuple[list[Path], float, float]:
    output_dir.mkdir(parents=True)
    paths: list[Path] = []
    sum_norms = 0.0
    positions = 0
    for shard_index, start in enumerate(range(0, len(sequences), variants_per_shard)):
        block = extract_residue_layer(
            model, sequences[start : start + variants_per_shard],
            layer=layer, batch_size=batch_size,
        )
        if block.ndim != 3 or block.shape[-1] != dimension:
            raise AssertionError(f"unexpected extracted activation shape {block.shape}")
        sum_norms += float(np.linalg.norm(block, axis=-1).sum())
        positions += int(np.prod(block.shape[:-1]))
        path = output_dir / f"shard_{shard_index:04d}.npy"
        np.save(path, block.astype(np.float16))
        paths.append(path)
    scalar = normalization_scalar(sum_norms, positions, dimension)
    realized = normalize_shards_in_place(paths, scalar)
    return paths, scalar, realized


def _model_record(paths: list[Path], scalar: float, realized: float, raw_mean_norm: float) -> dict:
    return {
        "raw_mean_norm": raw_mean_norm,
        "normalization_scalar": scalar,
        "normalized_mean_norm": realized,
        "shards": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": shard_sha256(path)}
            for path in paths
        ],
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("Step 4 activation extraction requires the GPU pod")
    started = time.monotonic()
    config = load_config(REPO_ROOT / "Code/config/step4.yaml")
    step3 = load_config(REPO_ROOT / "Code/config/step3.yaml")
    cache_dir = REPO_ROOT / config["paths"]["cache"]
    temporary = cache_dir.with_name(cache_dir.name + ".incomplete")
    if cache_dir.exists():
        raise FileExistsError(f"refusing to overwrite activation cache: {cache_dir}")
    if temporary.exists():
        raise FileExistsError(
            f"incomplete cache exists at {temporary}; inspect it before explicitly removing it"
        )
    temporary.mkdir(parents=True)
    try:
        frame = load_assay(str(config["dataset"]), with_sequences=True)
        train, _ = get_split(
            str(config["dataset"]), str(config["source_split"]), fold=int(config["source_fold"])
        )
        training = frame.iloc[train].reset_index(drop=True)
        sequences = training["mutant_sequence"].astype(str).tolist()
        sequence_length = len(sequences[0])
        if len(training) != 895 or sequence_length != 290:
            raise AssertionError("approved IRED cache dimensions changed")
        np.save(temporary / "variant_rows.npy", training["variant_row"].to_numpy(np.int64))

        device = "cuda"
        base = _load_backbone(step3, device)
        base_tokens = base._tokenize(sequences[:1]).detach().cpu()
        evo_paths, evo_scalar, evo_realized = _extract_model(
            base, sequences, temporary / "evo", layer=int(config["analysis_layer"]),
            batch_size=int(config["extraction_batch_size"]),
            variants_per_shard=int(config["variants_per_shard"]),
            dimension=int(config["embedding_dimension"]),
        )
        evo_raw_mean = float(np.sqrt(int(config["embedding_dimension"])) / evo_scalar)
        del base
        gc.collect()
        torch.cuda.empty_cache()

        checkpoint = REPO_ROOT / config["paths"]["adapter_checkpoint"]
        verification = json.loads(
            (REPO_ROOT / config["paths"]["checkpoint_verification"]).read_text()
        )
        if sha256_file(checkpoint) != verification["real"]["sha256"]:
            raise AssertionError("real adapter checkpoint hash differs from Gate 1 verification")
        adapted_backbone = _load_backbone(step3, device)
        adapted = build_lora_adapter(
            adapted_backbone, rank=int(step3["adapter"]["rank"]),
            alpha=int(step3["adapter"]["alpha"]), dropout=float(step3["adapter"]["dropout"]),
        )
        wrapped = SequenceFitnessAdapter(adapted, int(config["embedding_dimension"])).to(device)
        load_trainable_state(wrapped, checkpoint)
        wrapped.eval()
        fit_tokens = wrapped.backbone._tokenize(sequences[:1]).detach().cpu()
        if not torch.equal(base_tokens, fit_tokens):
            raise AssertionError("base and adapted token alignment differs")
        fit_paths, fit_scalar, fit_realized = _extract_model(
            wrapped.backbone, sequences, temporary / "fit", layer=int(config["analysis_layer"]),
            batch_size=int(config["extraction_batch_size"]),
            variants_per_shard=int(config["variants_per_shard"]),
            dimension=int(config["embedding_dimension"]),
        )
        fit_raw_mean = float(np.sqrt(int(config["embedding_dimension"])) / fit_scalar)

        if [np.load(path, mmap_mode="r").shape for path in evo_paths] != [
            np.load(path, mmap_mode="r").shape for path in fit_paths
        ]:
            raise AssertionError("base and adapted cache shards are not aligned")
        manifest = {
            "status": "complete",
            "dataset": str(config["dataset"]),
            "split": f"{config['source_split']}_fold_{config['source_fold']}_training_only",
            "analysis_layer": int(config["analysis_layer"]),
            "training_variants": len(training),
            "sequence_length": sequence_length,
            "positions": len(training) * sequence_length,
            "dimension": int(config["embedding_dimension"]),
            "dtype": str(config["storage_dtype"]),
            "adapter_checkpoint_sha256": sha256_file(checkpoint),
            "token_alignment": "exact",
            "raw_adapted_to_base_mean_norm_ratio": fit_raw_mean / evo_raw_mean,
            "models": {
                "evo": _model_record(evo_paths, evo_scalar, evo_realized, evo_raw_mean),
                "fit": _model_record(fit_paths, fit_scalar, fit_realized, fit_raw_mean),
            },
            "payload_bytes": sum(path.stat().st_size for path in evo_paths + fit_paths),
            "gpu_wall_hours": (time.monotonic() - started) / 3600.0,
            "excluded": ["pg_kfold_test", "layer_freeze_reserve", "round_held_out_test", "plate"],
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, cache_dir)
        report = REPO_ROOT / config["paths"]["cache_manifest"]
        report.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache_dir / "manifest.json", report)
        print(json.dumps(manifest, indent=2), flush=True)
        return 0
    except BaseException:
        # Incomplete data remain visibly quarantined and are never consumed.
        raise


if __name__ == "__main__":
    raise SystemExit(main())
