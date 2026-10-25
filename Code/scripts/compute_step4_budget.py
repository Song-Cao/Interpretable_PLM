#!/usr/bin/env python3
"""Print the IRED Step 4 realized training-only budget without caching."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.datasets import get_split, load_assay, wt_sequence
from crossfit.utils import REPO_ROOT, load_config


def _positions(values) -> set[int]:
    return set().union(*(set(map(int, str(value).split(";"))) for value in values))


def main() -> int:
    config = load_config(REPO_ROOT / "Code/config/step4.yaml")
    dataset = str(config["dataset"])
    frame = load_assay(dataset)
    train, _ = get_split(dataset, str(config["source_split"]), fold=int(config["source_fold"]))
    training = frame.iloc[train]
    dimension = int(config["embedding_dimension"])
    bytes_per_activation = int(config["bytes_per_value"])
    model_count = int(config["model_count"])
    sequence_length = len(wt_sequence(dataset))

    def row(name: str, positions: int) -> dict:
        byte_count = positions * dimension * bytes_per_activation * model_count
        return {
            "scenario": name,
            "positions": positions,
            "bytes": byte_count,
            "gb_decimal": byte_count / 1e9,
            "gib_binary": byte_count / (1024 ** 3),
        }

    if str(config["position_policy"]) != "all_training_residues":
        raise ValueError("the approved IRED budget policy is all_training_residues")
    mutable = _positions(training["mutated_positions"])
    realized_positions = len(training) * sequence_length
    per_model_bytes = realized_positions * dimension * bytes_per_activation
    width_multipliers = [int(value) for value in config["width_multipliers"]]
    output = {
        "dataset": dataset,
        "status": "approved_for_caching",
        "split": f"{config['source_split']}_fold_{config['source_fold']}_training_only",
        "analysis_layer": int(config["analysis_layer"]),
        "position_policy": str(config["position_policy"]),
        "training_variants": len(training),
        "sequence_length": sequence_length,
        "embedding_dimension": dimension,
        "training_mutable_positions": len(mutable),
        "realized_positions": realized_positions,
        "per_model_bytes": per_model_bytes,
        "per_model_gb_decimal": per_model_bytes / 1e9,
        "two_model_bytes": per_model_bytes * model_count,
        "two_model_gb_decimal": per_model_bytes * model_count / 1e9,
        "two_model_gib_binary": per_model_bytes * model_count / (1024 ** 3),
        "dictionary_width_diagnostic": [
            {
                "width_multiplier": width,
                "features": width * dimension,
                "position_samples_per_feature": realized_positions / (width * dimension),
            }
            for width in width_multipliers
        ],
        "excluded": ["pg_kfold_test", "layer_freeze_reserve", "round_held_out_test", "plate"],
        "rows": [row("realized_all_training_residues", realized_positions)],
    }
    json_path = REPO_ROOT / config["paths"]["report_json"]
    csv_path = REPO_ROOT / config["paths"]["report_csv"]
    markdown_path = REPO_ROOT / config["paths"]["report_markdown"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    pd.DataFrame(output["rows"]).to_csv(csv_path, index=False)
    markdown_path.write_text(
        "# IRED Step 4 realized caching budget\n\n"
        "Status: budget approved for activation caching.\n\n"
        "| training variants | residues per variant | positions | dimension | dtype | models | GB |\n"
        "|---:|---:|---:|---:|---|---:|---:|\n"
        f"| {len(training)} | {sequence_length} | {realized_positions} | {dimension} | "
        f"{config['storage_dtype']} | {model_count} | {per_model_bytes * model_count / 1e9:.6f} |\n\n"
        "The budget uses only the pg_kfold fold-0 training pool. Test, reserve, round-held-out "
        "test, and plate variants are excluded. The arithmetic is positions x dimension x "
        "2 bytes x 2 models. IRED is small enough to retain every residue position; this avoids "
        "introducing an unapproved window or distant-position sampling rule.\n"
        "\nThe registered dictionary width multipliers are 1, 2, 4, and 8. They provide "
        + ", ".join(
            f"{realized_positions / (width * dimension):.1f} samples/feature at w={width}"
            for width in width_multipliers
        )
        + ". Width is selected only after reporting dead-feature fraction and participation ratio.\n"
    )
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
