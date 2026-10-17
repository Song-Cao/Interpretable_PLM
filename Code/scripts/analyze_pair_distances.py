#!/usr/bin/env python3
"""Audit pair distances, margin gates, and the Gcn4 mutation-count mixture."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
    default_margin = config["pairs"]["margin_sd"]
    audit = audit_epoch_pair_distances(
        frames,
        epochs,
        margin_sd=default_margin,
    )
    output = audit.to_frame()
    output.insert(1, "audit_epochs", n_epochs)
    output.insert(2, "margin_sd", default_margin)
    output_path = Path(config["pair_distance_audit"]["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    summary = output.drop_duplicates("assay_id")[["assay_id", "total_gated_pairs"]]
    print(summary.to_string(index=False))

    margin_rows = []
    for margin in config["loss"]["sweeps"]["margin_sd"]:
        margin_audit = audit_epoch_pair_distances(frames, epochs, margin_sd=float(margin))
        for assay_id, distribution in sorted(margin_audit.counts.items()):
            margin_rows.append(
                {
                    "assay_id": assay_id,
                    "audit_epochs": n_epochs,
                    "margin_sd": float(margin),
                    "gated_pairs": sum(distribution.values()),
                }
            )
    pd.DataFrame(margin_rows).to_csv(
        config["pair_distance_audit"]["margin_output"], index=False
    )

    gcn4 = datasets.load_assay("GCN4_YEAST_Staller_2018", include_reserve=True)
    usable = gcn4.loc[~gcn4["is_layer_freeze_reserve"]]
    mutation_counts = (
        gcn4["n_mutations"]
        .value_counts()
        .sort_index()
        .rename("curated_count")
        .to_frame()
        .join(usable["n_mutations"].value_counts().rename("usable_count"), how="outer")
        .fillna(0)
        .astype(int)
        .reset_index(names="n_mutations")
    )
    mutation_counts["curated_fraction"] = mutation_counts["curated_count"] / len(gcn4)
    mutation_counts["usable_fraction"] = mutation_counts["usable_count"] / len(usable)
    mutation_counts.to_csv(config["pair_distance_audit"]["gcn4_output"], index=False)


if __name__ == "__main__":
    main()
