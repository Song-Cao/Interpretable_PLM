#!/usr/bin/env python3
"""Print the IRED Step 4 activation-position storage bounds without caching."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.datasets import get_split, load_assay, wt_sequence
from crossfit.utils import REPO_ROOT


def _positions(values) -> set[int]:
    return set().union(*(set(map(int, str(value).split(";"))) for value in values))


def main() -> int:
    frame = load_assay("IRED")
    train, _ = get_split("IRED", "pg_kfold", fold=0)
    training = frame.iloc[train]
    dimension = 960
    bytes_per_activation = 2
    model_count = 2

    def row(name: str, positions: int) -> dict:
        byte_count = positions * dimension * bytes_per_activation * model_count
        return {
            "scenario": name,
            "positions": positions,
            "bytes": byte_count,
            "gb_decimal": byte_count / 1e9,
            "gib_binary": byte_count / (1024 ** 3),
        }

    mutable = _positions(training["mutated_positions"])
    output = {
        "dataset": "IRED",
        "training_variants": len(training),
        "sequence_length": len(wt_sequence("IRED")),
        "embedding_dimension": dimension,
        "training_mutable_positions": len(mutable),
        "rows": [
            row("all_residues_upper_bound", len(training) * len(wt_sequence("IRED"))),
            row("mutable_union_at_every_variant", len(training) * len(mutable)),
            row("mutated_positions_only_lower_bound", int(training["n_mutations"].sum())),
        ],
        "not_realized": (
            "v12 does not fix the mutation-window radius or distant-position draw count; "
            "a final realized budget cannot be named without those Step 4 choices"
        ),
    }
    path = REPO_ROOT / "Reports" / "step3" / "step4_budget_ired.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
