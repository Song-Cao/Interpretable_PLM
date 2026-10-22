#!/usr/bin/env python3
"""Assemble the IRED Step 2 arm reports without recomputing metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.utils import REPO_ROOT, load_config


EXPECTED = (
    "onehot_ridge_unaugmented",
    "onehot_ridge_augmented",
    "aaindex_flat_ridge_unaugmented",
    "aaindex_flat_ridge_augmented",
    "esm_mean_ridge",
    "esm_mean_mlp",
    "esm_mean_random_forest",
    "esm_mutpool_ridge",
    "esm_residue_conv",
    "esm_residue_delta_conv",
    "onehot_positional_conv",
)


def _table(results: list[dict], condition: str) -> pd.DataFrame:
    rows = []
    for result in results:
        row: dict[str, object] = {"dataset": "IRED", "arm": result["arm"]}
        for metric, interval in result[condition].items():
            row[metric] = interval["estimate"]
            row[f"{metric}_ci_lower"] = interval["lower"]
            row[f"{metric}_ci_upper"] = interval["upper"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("arm").reset_index(drop=True)


def main() -> int:
    config = load_config(REPO_ROOT / "Code" / "config" / "step2.yaml")
    report_dir = REPO_ROOT / config["paths"]["reports"]
    results = []
    for arm in EXPECTED:
        path = report_dir / f"{arm}_result.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
    if not results:
        raise FileNotFoundError("no Step 2 arm result files exist")
    _table(results, "pg_kfold_test").to_csv(report_dir / "ired_pg_kfold_test_results.csv", index=False)
    _table(results, "round_held_out").to_csv(report_dir / "ired_round_held_out_results.csv", index=False)
    _table(results, "plate").to_csv(report_dir / "ired_plate_results.csv", index=False)

    shortlist = pd.DataFrame([
        {
            "arm": result["arm"],
            "validation_spearman": result["validation_spearman_mean"],
            "selected_configuration": json.dumps(result["selected_configuration"], sort_keys=True),
            "boundary_hyperparameters": json.dumps(result.get("boundary_hyperparameters", {}), sort_keys=True),
        }
        for result in results
    ]).sort_values("validation_spearman", ascending=False)
    shortlist.to_csv(report_dir / "ired_architecture_shortlist.csv", index=False)

    by_arm = {result["arm"]: result for result in results}
    pooling = {"status": "incomplete"}
    if "esm_mean_ridge" in by_arm and "esm_mutpool_ridge" in by_arm:
        mean = by_arm["esm_mean_ridge"]["pg_kfold_test"]["spearman"]
        mutated = by_arm["esm_mutpool_ridge"]["pg_kfold_test"]["spearman"]
        mean_validation = by_arm["esm_mean_ridge"]["validation_spearman_mean"]
        mutated_validation = by_arm["esm_mutpool_ridge"]["validation_spearman_mean"]
        winner = "all_residue_mean" if mean_validation >= mutated_validation else "mutated_position_pool"
        pooling = {
            "status": "complete",
            "comparison_model": "ridge",
            "selection_protocol": "configurations selected only by inner-CV validation Spearman",
            "esm_mean_validation_spearman": mean_validation,
            "esm_mutpool_validation_spearman": mutated_validation,
            "esm_mean_test_spearman": mean,
            "esm_mutpool_test_spearman": mutated,
            "answer": winner,
        }
    (report_dir / "ired_pooling_domain.json").write_text(json.dumps(pooling, indent=2) + "\n")
    incomplete = [arm for arm in EXPECTED if arm not in by_arm]
    summary = {
        "completed_arms": [result["arm"] for result in results],
        "incomplete_arms": incomplete,
        "pooling_domain": pooling,
    }
    (report_dir / "ired_step2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
