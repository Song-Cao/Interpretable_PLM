#!/usr/bin/env python3
"""Report IRED test hit rate and external-plate composition before training."""
from __future__ import annotations

import json
from pathlib import Path

from crossfit.datasets import get_split, load_assay, load_plate_dataset, wild_type_fitness
from crossfit.splits import mutated_positions
from crossfit.utils import REPO_ROOT, load_config


def main() -> int:
    config = load_config(REPO_ROOT / "Code" / "config" / "step2.yaml")
    dataset = config["ired_curation"]["dataset_id"]
    frame = load_assay(dataset)
    _, test = get_split(dataset, "pg_kfold", fold=config["evaluation"]["primary_test_fold"])
    cutoff = wild_type_fitness(dataset)
    test_labels = frame.iloc[test]["y"]

    plate = load_plate_dataset(dataset).copy()
    campaign_positions = {
        position for mutation in frame["mutations"] for position in mutated_positions(mutation)
    }
    plate["n_mutations"] = plate["mut"].map(lambda value: len(mutated_positions(value)))
    plate["has_unseen_position"] = plate["mut"].map(
        lambda value: any(position not in campaign_positions for position in mutated_positions(value))
    )
    unseen_positions = sorted(
        {
            position
            for mutation in plate.loc[plate["has_unseen_position"], "mut"]
            for position in mutated_positions(mutation)
        }
        - campaign_positions
    )
    distribution = plate["1/s"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    report = {
        "pg_kfold_test": {
            "fold": config["evaluation"]["primary_test_fold"],
            "n": len(test),
            "standardized_wild_type_cutoff": cutoff,
            "hits": int((test_labels > cutoff).sum()),
            "hit_rate": float((test_labels > cutoff).mean()),
        },
        "plate": {
            "n": len(plate),
            "one_over_s_distribution": {key: float(value) for key, value in distribution.items()},
            "mutation_count_distribution": {
                str(key): int(value)
                for key, value in plate["n_mutations"].value_counts().sort_index().items()
            },
            "variants_with_unseen_position": int(plate["has_unseen_position"].sum()),
            "unseen_positions": unseen_positions,
            "campaign_mutated_position_count": len(campaign_positions),
        },
    }
    output = REPO_ROOT / config["paths"]["preflight_report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
