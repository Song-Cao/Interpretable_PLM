#!/usr/bin/env python3
"""Cache normalized IRED depth-26 shuffled-adapter activations for null runs."""
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


def _extract(
    model: torch.nn.Module,
    sequences: list[str],
    output_dir: Path,
    *,
    layer: int,
    batch_size: int,
    variants_per_shard: int,
    dimension: int,
) -> tuple[list[Path], float, float, float]:
    output_dir.mkdir(parents=True)
    paths: list[Path] = []
    sum_norms = 0.0
    positions = 0
    for shard_index, start in enumerate(range(0, len(sequences), variants_per_shard)):
        values = extract_residue_layer(
            model, sequences[start : start + variants_per_shard],
            layer=layer, batch_size=batch_size,
        )
        if values.ndim != 3 or values.shape[-1] != dimension:
            raise AssertionError(f"unexpected shuffled activation shape {values.shape}")
        sum_norms += float(np.linalg.norm(values, axis=-1).sum())
        positions += int(np.prod(values.shape[:-1]))
        path = output_dir / f"shard_{shard_index:04d}.npy"
        np.save(path, values.astype(np.float16))
        paths.append(path)
    scalar = normalization_scalar(sum_norms, positions, dimension)
    realized = normalize_shards_in_place(paths, scalar)
    return paths, scalar, realized, sum_norms / positions


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("shuffled-adapter activation extraction requires the GPU pod")
    started = time.monotonic()
    config = load_config(REPO_ROOT / "Code/config/step6.yaml")
    step3 = load_config(REPO_ROOT / "Code/config/step3.yaml")
    step4 = load_config(REPO_ROOT / "Code/config/step4.yaml")
    cache_dir = REPO_ROOT / config["paths"]["shuffled_cache"]
    temporary = cache_dir.with_name(cache_dir.name + ".incomplete")
    if cache_dir.exists():
        raise FileExistsError(f"refusing to overwrite activation cache: {cache_dir}")
    if temporary.exists():
        raise FileExistsError(f"inspect the incomplete cache before removal: {temporary}")
    temporary.mkdir(parents=True)

    frame = load_assay(str(config["dataset"]), with_sequences=True)
    train, _ = get_split(
        str(config["dataset"]), str(step4["source_split"]), fold=int(step4["source_fold"])
    )
    training = frame.iloc[train].reset_index(drop=True)
    reference_rows = np.load(REPO_ROOT / config["paths"]["step4_cache"] / "variant_rows.npy")
    rows = training["variant_row"].to_numpy(np.int64)
    if not np.array_equal(rows, reference_rows):
        raise AssertionError("shuffled and real-pair cache rows are not exactly aligned")
    sequences = training["mutant_sequence"].astype(str).tolist()
    np.save(temporary / "variant_rows.npy", rows)

    checkpoint = REPO_ROOT / config["paths"]["shuffled_adapter_checkpoint"]
    verification = json.loads(
        (REPO_ROOT / config["paths"]["gate1_checkpoint_verification"]).read_text()
    )
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != verification["shuffled"]["sha256"]:
        raise AssertionError("shuffled checkpoint hash differs from Gate 1 verification")
    backbone = load_esmc(
        str(step3["backbone"]["name"]), "cuda",
        repository=str(step3["backbone"]["repository"]),
        expected_revision=str(step3["backbone"]["revision"]),
    )
    adapted = build_lora_adapter(
        backbone, rank=int(step3["adapter"]["rank"]),
        alpha=int(step3["adapter"]["alpha"]), dropout=float(step3["adapter"]["dropout"]),
    )
    wrapped = SequenceFitnessAdapter(adapted, int(config["dimension"])).to("cuda")
    load_trainable_state(wrapped, checkpoint)
    wrapped.eval()
    paths, scalar, realized, raw_mean = _extract(
        wrapped.backbone, sequences, temporary / "shuffled",
        layer=int(step4["analysis_layer"]), batch_size=int(step4["extraction_batch_size"]),
        variants_per_shard=int(step4["variants_per_shard"]), dimension=int(config["dimension"]),
    )
    del wrapped, adapted, backbone
    gc.collect()
    torch.cuda.empty_cache()

    manifest = {
        "status": "complete", "dataset": str(config["dataset"]),
        "split": f"{step4['source_split']}_fold_{step4['source_fold']}_training_only",
        "analysis_layer": int(step4["analysis_layer"]), "training_variants": len(training),
        "sequence_length": len(sequences[0]), "positions": len(training) * len(sequences[0]),
        "dimension": int(config["dimension"]), "dtype": "float16",
        "adapter": "shuffled_label", "adapter_checkpoint_sha256": checkpoint_hash,
        "models": {"shuffled": {
            "raw_mean_norm": raw_mean, "normalization_scalar": scalar,
            "normalized_mean_norm": realized,
            "shards": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": shard_sha256(path)}
                for path in paths
            ],
        }},
        "payload_bytes": sum(path.stat().st_size for path in paths),
        "gpu_wall_hours": (time.monotonic() - started) / 3600.0,
        "excluded": ["pg_kfold_test", "layer_freeze_reserve", "round_held_out_test", "plate"],
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, cache_dir)
    report = REPO_ROOT / config["paths"]["shuffled_cache_manifest"]
    report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_dir / "manifest.json", report)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
