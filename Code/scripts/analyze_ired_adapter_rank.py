#!/usr/bin/env python3
"""Compare matched-reserve effective rank for base, real, and shuffled adapters."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.dictionary_diagnostics import covariance_spectrum, participation_ratio
from crossfit.utils import REPO_ROOT, load_config


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    config = load_config(REPO_ROOT / "Code/config/step7.yaml")
    rank = config["diagnostics"]["adapter_rank"]
    layer = int(rank["layer"])
    models = {
        "evo": rank["reserve_base"],
        "fit": rank["reserve_real"],
        "shuffled": rank["reserve_shuffled"],
    }
    result: dict[str, object] = {
        "dataset": config["dataset"],
        "layer": layer,
        "sample": "layer_freeze_reserve",
    }
    spectra = {}
    expected_shape = None
    for name, relative_path in models.items():
        values = np.load(REPO_ROOT / relative_path, mmap_mode="r")
        if values.ndim != 4 or layer >= values.shape[1]:
            raise ValueError(f"unexpected all-layer reserve tensor: {relative_path}")
        if expected_shape is None:
            expected_shape = values.shape
        elif values.shape != expected_shape:
            raise AssertionError("base, real, and shuffled reserve tensors are not aligned")
        flattened = np.asarray(values[:, layer]).reshape(-1, values.shape[-1])
        eigenvalues = covariance_spectrum(
            flattened,
            batch_size=int(rank["batch_size"]),
            device=args.device,
        )
        cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
        spectra[name] = {
            "participation_ratio": participation_ratio(eigenvalues),
            "observations": len(flattened),
            "components_90_percent": int(np.searchsorted(cumulative, 0.9) + 1),
            "components_99_percent": int(np.searchsorted(cumulative, 0.99) + 1),
        }
    evo_rank = spectra["evo"]["participation_ratio"]
    result["models"] = spectra
    result["real_change_percent"] = 100.0 * (
        spectra["fit"]["participation_ratio"] / evo_rank - 1.0
    )
    result["shuffled_change_percent"] = 100.0 * (
        spectra["shuffled"]["participation_ratio"] / evo_rank - 1.0
    )
    result["interpretation"] = "generic_finetuning_contraction"
    output = REPO_ROOT / config["paths"]["report_dir"] / "adapter_rank_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
