#!/usr/bin/env python3
"""Cache pooled IRED crosscoder features for campaign and plate variants."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.crosscoder import CrossCoder
from crossfit.feature_pooling import pool_codes
from crossfit.utils import REPO_ROOT, load_config


def _normalize(values: np.ndarray, scalar: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) * float(scalar)


@torch.inference_mode()
def _pool(
    model: CrossCoder,
    evo: np.ndarray,
    fit: np.ndarray,
    operators: list[str],
    *,
    device: str,
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    output = {name: [] for name in operators}
    model.to(device).eval()
    for start in range(0, len(evo), batch_size):
        left = torch.as_tensor(evo[start : start + batch_size], dtype=torch.float32, device=device)
        right = torch.as_tensor(fit[start : start + batch_size], dtype=torch.float32, device=device)
        flat = model.encode(left.flatten(0, 1), right.flatten(0, 1))
        codes = flat.reshape(left.shape[0], left.shape[1], -1).cpu().numpy()
        for name in operators:
            output[name].append(pool_codes(codes, name).astype(np.float32))
    return {name: np.concatenate(blocks) for name, blocks in output.items()}


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("phi caching requires the GPU pod")
    config = load_config(REPO_ROOT / "Code/config/step9.yaml")
    step4_manifest = json.loads((REPO_ROOT / "Cache/step4/IRED/manifest.json").read_text())
    evo_scalar = float(step4_manifest["models"]["evo"]["normalization_scalar"])
    fit_scalar = float(step4_manifest["models"]["fit"]["normalization_scalar"])
    campaign_evo = _normalize(np.load(REPO_ROOT / config["paths"]["campaign_base"], mmap_mode="r"), evo_scalar)
    campaign_fit = _normalize(np.load(REPO_ROOT / config["paths"]["campaign_fit"], mmap_mode="r"), fit_scalar)
    plate_dir = REPO_ROOT / config["paths"]["plate_cache"]
    plate_evo = _normalize(np.load(plate_dir / "base.npy", mmap_mode="r"), evo_scalar)
    plate_fit = _normalize(np.load(plate_dir / "fit.npy", mmap_mode="r"), fit_scalar)
    destination = REPO_ROOT / config["paths"]["phi_cache"]
    temporary = destination.with_name(destination.name + ".incomplete")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite phi cache: {destination}")
    if temporary.exists():
        raise FileExistsError(f"inspect incomplete phi cache before removal: {temporary}")
    temporary.mkdir(parents=True)
    rows = np.load(REPO_ROOT / config["paths"]["campaign_rows"])
    np.save(temporary / "campaign_rows.npy", rows)
    manifest = {"status": "complete", "dataset": str(config["dataset"]), "widths": {}}
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
        campaign = _pool(model, campaign_evo, campaign_fit, list(config["pooling_operators"]), device="cuda")
        plate = _pool(model, plate_evo, plate_fit, list(config["pooling_operators"]), device="cuda")
        width_dir = temporary / f"w{width}"
        width_dir.mkdir()
        for operator in config["pooling_operators"]:
            np.save(width_dir / f"campaign_{operator}.npy", campaign[operator])
            np.save(width_dir / f"plate_{operator}.npy", plate[operator])
        taxonomy = pd.read_csv(
            REPO_ROOT / config["paths"]["reports_step7"] / f"taxonomy_w{width}_seed{seed}.csv"
        )
        taxonomy[["feature", "provisional_class"]].to_csv(width_dir / "taxonomy.csv", index=False)
        manifest["widths"][str(width)] = {
            "features": model.features, "crosscoder_seed": seed,
            "campaign_variants": len(campaign_evo), "plate_variants": len(plate_evo),
            "operators": list(config["pooling_operators"]),
        }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, destination)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
