#!/usr/bin/env python3
"""Audit the Hamming constraint on the deterministic default training stream."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit import datasets
from crossfit.pair_analysis import audit_epoch_pair_distances
from crossfit.sampling import CappedSqrtSampler


def main() -> None:
    config_path = Path("Code/config/step2.yaml")
    config = yaml.safe_load(config_path.read_text())
    frames = {assay_id: datasets.load_assay(assay_id) for assay_id in datasets.list_assays()}
    sampler = CappedSqrtSampler(
        {assay_id: np.arange(len(frame)) for assay_id, frame in frames.items()},
        cap_median_multiplier=config["sampling"]["cap_median_multiplier"],
        batch_size=config["sampling"]["batch_size"],
        seed=config["sampling"]["seed"],
    )
    n_epochs = int(config["pair_distance_audit"]["epochs"])
    epochs = [sampler.sample_epoch(epoch) for epoch in range(n_epochs)]
    audit = audit_epoch_pair_distances(
        frames,
        epochs,
        margin_sd=config["pairs"]["margin_sd"],
    )
    output = audit.to_frame(config["pairs"]["max_hamming_distance"])
    output.insert(1, "audit_epochs", n_epochs)
    output.insert(2, "margin_sd", config["pairs"]["margin_sd"])
    output.insert(3, "max_hamming_distance", config["pairs"]["max_hamming_distance"])
    output_path = Path(config["pair_distance_audit"]["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    summary = output.drop_duplicates("assay_id")[
        ["assay_id", "pairs_without_constraint", "pairs_with_constraint", "removed_fraction"]
    ]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
