#!/usr/bin/env python3
"""Extract IRED campaign and plate activations at frozen analysis depth 26."""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.adaptation import SequenceFitnessAdapter, build_lora_adapter, load_trainable_state
from crossfit.cache_provenance import (
    manifest_provenance,
    sha256_file,
    write_reference_block,
)
from crossfit.datasets import load_assay, load_plate_dataset, wt_sequence
from crossfit.esmc import extract_residue_layer, load_esmc
from crossfit.utils import REPO_ROOT, load_config


def _backbone(step3: dict, device: str):
    return load_esmc(
        str(step3["backbone"]["name"]), device,
        repository=str(step3["backbone"]["repository"]),
        expected_revision=str(step3["backbone"]["revision"]),
    )


def _save(path: Path, values: np.ndarray) -> dict[str, object]:
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values, dtype=np.float16))
    os.replace(temporary, path)
    return {
        "file": path.name,
        "shape": list(values.shape),
        "dtype": "float16",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("analysis-layer extraction requires the GPU pod")
    started = time.monotonic()
    config = load_config(REPO_ROOT / "Code/config/step9.yaml")
    step3 = load_config(REPO_ROOT / "Code/config/step3.yaml")
    provenance = load_config(REPO_ROOT / config["paths"]["provenance_config"])
    destination = REPO_ROOT / config["paths"]["campaign_cache"]
    temporary = destination.with_name(destination.name + ".incomplete")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite analysis cache: {destination}")
    if temporary.exists():
        raise FileExistsError(f"inspect incomplete analysis cache before removal: {temporary}")
    temporary.mkdir(parents=True)

    campaign = load_assay("IRED", with_sequences=True, include_reserve=True)
    plate = load_plate_dataset("IRED")
    wild_type = wt_sequence("IRED")
    campaign_sequences = campaign["mutant_sequence"].astype(str).tolist()
    plate_sequences = plate["aa_seq"].astype(str).tolist()
    layer = int(config["analysis_layer"])
    batch_size = int(config["extraction_batch_size"])

    base = _backbone(step3, "cuda")
    compute_dtype = str(next(base.parameters()).dtype).removeprefix("torch.")
    base_tokens = base._tokenize([wild_type]).detach().cpu()
    base_reference = extract_residue_layer(base, [wild_type], layer=layer, batch_size=1)[0]
    campaign_base = extract_residue_layer(
        base, campaign_sequences, layer=layer, batch_size=batch_size
    )
    plate_base = extract_residue_layer(base, plate_sequences, layer=layer, batch_size=batch_size)
    payload = {
        "campaign_base": _save(temporary / "campaign_base.npy", campaign_base),
        "plate_base": _save(temporary / "plate_base.npy", plate_base),
    }
    base_record = write_reference_block(
        temporary, "base", base_reference, activation_units="raw"
    )
    del base, campaign_base, plate_base
    gc.collect()
    torch.cuda.empty_cache()

    checkpoint = REPO_ROOT / config["paths"]["step3_adapter"]
    checkpoint_verification = json.loads(
        (REPO_ROOT / config["paths"]["gate1_checkpoint_verification"]).read_text()
    )
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != checkpoint_verification["real"]["sha256"]:
        raise AssertionError("real adapter checkpoint hash differs from Gate 1 verification")
    adapted_backbone = _backbone(step3, "cuda")
    adapted = build_lora_adapter(
        adapted_backbone,
        rank=int(step3["adapter"]["rank"]),
        alpha=int(step3["adapter"]["alpha"]),
        dropout=float(step3["adapter"]["dropout"]),
    )
    wrapped = SequenceFitnessAdapter(adapted, int(config["dimension"])).to("cuda")
    load_trainable_state(wrapped, checkpoint)
    wrapped.eval()
    fit_tokens = wrapped.backbone._tokenize([wild_type]).detach().cpu()
    if not torch.equal(base_tokens, fit_tokens):
        raise AssertionError("base and adapted wild-type tokenization differs")
    fit_reference = extract_residue_layer(
        wrapped.backbone, [wild_type], layer=layer, batch_size=1
    )[0]
    campaign_fit = extract_residue_layer(
        wrapped.backbone, campaign_sequences, layer=layer, batch_size=batch_size
    )
    plate_fit = extract_residue_layer(
        wrapped.backbone, plate_sequences, layer=layer, batch_size=batch_size
    )
    payload.update({
        "campaign_fit": _save(temporary / "campaign_fit.npy", campaign_fit),
        "plate_fit": _save(temporary / "plate_fit.npy", plate_fit),
    })
    fit_record = write_reference_block(
        temporary, "fit", fit_reference, activation_units="raw"
    )
    np.save(temporary / "campaign_rows.npy", campaign["variant_row"].to_numpy(np.int64))
    np.save(temporary / "plate_rows.npy", np.arange(len(plate), dtype=np.int64))
    payload["campaign_rows"] = {
        "file": "campaign_rows.npy",
        "shape": [len(campaign)],
        "sha256": sha256_file(temporary / "campaign_rows.npy"),
    }
    payload["plate_rows"] = {
        "file": "plate_rows.npy",
        "shape": [len(plate)],
        "sha256": sha256_file(temporary / "plate_rows.npy"),
    }
    manifest = {
        "status": "complete",
        **manifest_provenance(
            contract_version=int(provenance["contract_version"]),
            dataset="IRED",
            split="campaign_all_including_reserve_plus_external_plate_evaluation",
            layer_index=layer,
            storage_dtype="float16",
            compute_dtype=compute_dtype,
            backbone_revision=str(step3["backbone"]["revision"]),
            activation_units="raw",
            reference_blocks={"base": base_record, "fit": fit_record},
            requested_attention=str(provenance["runtime"]["requested_attention_implementation"]),
        ),
        "dimension": int(config["dimension"]),
        "campaign_variants": int(len(campaign)),
        "plate_variants": int(len(plate)),
        "sequence_length": int(len(wild_type)),
        "adapter_checkpoint_sha256": checkpoint_hash,
        "payload": payload,
        "selection_use": {
            "reserve": False,
            "pg_kfold_test": False,
            "external_plate": False,
        },
        "gpu_wall_hours": (time.monotonic() - started) / 3600.0,
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, destination)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
