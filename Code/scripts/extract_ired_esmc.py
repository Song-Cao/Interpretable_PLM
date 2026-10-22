#!/usr/bin/env python3
"""Extract IRED ESM-C representations, reserve layers, and mutated-position dPLL."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.datasets import load_assay, load_plate_dataset, wt_sequence
from crossfit.density import changed_positions
from crossfit.esmc import extract_residue_embeddings, load_esmc, masked_log_probabilities
from crossfit.utils import REPO_ROOT, load_config


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--masked-batch-size", type=int, default=32)
    return parser.parse_args()


def _safe_target(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite cache artifact: {path}")


def main() -> int:
    args = _arguments()
    config = load_config(REPO_ROOT / "Code" / "config" / "step2.yaml")
    step3 = load_config(REPO_ROOT / "Code" / "config" / "step3.yaml")
    cache = REPO_ROOT / config["paths"]["representations"]
    cache.mkdir(parents=True, exist_ok=True)
    targets = {
        "variant": cache / "variant_residue.npy",
        "rows": cache / "variant_rows.npy",
        "plate": cache / "plate_residue.npy",
        "wild_type": cache / "wild_type_residue.npy",
        "reserve": cache / "reserve_all_layers_base.npy",
        "reserve_rows": cache / "reserve_rows.npy",
        "metadata": cache / "metadata.json",
        "density": REPO_ROOT / config["paths"]["density_cache"],
    }
    for path in targets.values():
        _safe_target(path)
    targets["density"].parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    full = load_assay("IRED", with_sequences=True, include_reserve=True)
    plate = load_plate_dataset("IRED")
    wild_type = wt_sequence("IRED")
    model = load_esmc(
        str(step3["backbone"]["name"]), args.device,
        repository=str(step3["backbone"]["repository"]),
        expected_revision=str(step3["backbone"]["revision"]),
    )

    variant_batch = extract_residue_embeddings(
        model,
        full["mutant_sequence"].astype(str).tolist(),
        batch_size=args.embedding_batch_size,
    )
    plate_batch = extract_residue_embeddings(
        model, plate["aa_seq"].astype(str).tolist(), batch_size=args.embedding_batch_size
    )
    wild_type_batch = extract_residue_embeddings(
        model, [wild_type], batch_size=1
    )
    reserve = full["is_layer_freeze_reserve"].to_numpy(bool)
    reserve_batch = extract_residue_embeddings(
        model,
        full.loc[reserve, "mutant_sequence"].astype(str).tolist(),
        batch_size=args.embedding_batch_size,
        all_layers=True,
    )
    np.save(targets["variant"], variant_batch.final.astype(np.float16))
    np.save(targets["rows"], full["variant_row"].to_numpy(np.int64))
    np.save(targets["plate"], plate_batch.final.astype(np.float16))
    np.save(targets["wild_type"], wild_type_batch.final[0].astype(np.float16))
    np.save(targets["reserve"], reserve_batch.layers.astype(np.float16))
    np.save(targets["reserve_rows"], full.loc[reserve, "variant_row"].to_numpy(np.int64))

    density_sequences = full["mutant_sequence"].astype(str).tolist() + plate["aa_seq"].astype(str).tolist()
    density_sequences = list(dict.fromkeys([wild_type] + density_sequences))
    mutable_positions = sorted(
        set().union(*(set(changed_positions(wild_type, sequence)) for sequence in density_sequences))
    )
    wild_type_terms = masked_log_probabilities(
        model,
        [wild_type] * len(mutable_positions),
        mutable_positions,
        batch_size=args.masked_batch_size,
    )
    wild_type_by_position = dict(zip(mutable_positions, wild_type_terms))
    task_sequences: list[str] = []
    task_positions: list[int] = []
    task_slices: list[tuple[int, int]] = []
    for sequence in density_sequences:
        begin = len(task_sequences)
        positions = changed_positions(wild_type, sequence)
        task_sequences.extend([sequence] * len(positions))
        task_positions.extend(positions)
        task_slices.append((begin, len(task_sequences)))
    variant_terms = masked_log_probabilities(
        model, task_sequences, task_positions, batch_size=args.masked_batch_size
    )
    dpll = []
    for sequence, (begin, end) in zip(density_sequences, task_slices):
        positions = task_positions[begin:end]
        value = sum(
            float(variant_terms[index]) - float(wild_type_by_position[position])
            for index, position in zip(range(begin, end), positions)
        )
        dpll.append(value)
    pd.DataFrame({"aa_seq": density_sequences, "dpll": dpll}).to_csv(targets["density"], index=False)

    elapsed = time.perf_counter() - started
    cache_bytes = sum(path.stat().st_size for path in targets.values() if path.exists())
    metadata = {
        "dataset": "IRED",
        "model_name": step3["backbone"]["name"],
        "variant_count_including_reserve": len(full),
        "plate_count": len(plate),
        "reserve_count": int(reserve.sum()),
        "sequence_length": len(wild_type),
        "embedding_dimension": int(variant_batch.final.shape[-1]),
        "hidden_state_count": int(reserve_batch.layers.shape[1]),
        "variant_shape": list(variant_batch.final.shape),
        "reserve_all_layer_shape": list(reserve_batch.layers.shape),
        "dpll_sequence_count": len(density_sequences),
        "dpll_masked_variant_tasks": len(task_sequences),
        "dpll_wild_type_tasks": len(mutable_positions),
        "elapsed_seconds": elapsed,
        "cache_bytes_before_metadata": cache_bytes,
    }
    targets["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
