#!/usr/bin/env python3
"""Analyze trajectory recovery, consensus, controls, and model disagreement."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from applications.ired_design.mutations import canonical
from applications.ired_design.run_search import ARMS
from crossfit.utils import REPO_ROOT, load_config


def main() -> int:
    config = load_config(REPO_ROOT / "Code/config/ired_design.yaml")
    report = REPO_ROOT / config["paths"]["reports"]
    lineage_root = report / "full_lineage"
    overlap_rows, passage_rows, final_rows = [], [], []
    start = canonical(config["controls"]["background"])
    targets = [
        "S68P-P203Q-T241A", "S68P-T241A-H244Y", "S68P-P203Q-T241A-H244Y"
    ]
    dead_variants = {
        mutation: canonical((*start.split("-"), mutation))
        for mutation in config["controls"]["dead_ends"]
    }
    for arm in ARMS:
        files = sorted((lineage_root / arm).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no completed lineage for {arm}")
        frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        selected = frame.loc[frame["selected_for_next_round"]]
        for (founder, round_number), group in selected.groupby(["founder_seed", "round"]):
            observed = group.loc[group["in_curated_dataset"]]
            overlap_rows.append({
                "model": arm, "founder_seed": founder, "round": int(round_number),
                "selected_count": len(group), "observed_in_campaign": len(observed),
                "observed_variants": ";".join(observed["mutation_set_canonical"].astype(str)),
                "measured_fitness": ";".join(observed["measured_fitness_if_known"].astype(str)),
            })
        start_selected = selected.loc[selected["founder_seed"] == start]
        for target in targets:
            rounds = start_selected.loc[start_selected["mutation_set_canonical"] == canonical(target), "round"]
            passage_rows.append({
                "model": arm, "founder_seed": start, "target": canonical(target),
                "passes_through": bool(len(rounds)),
                "first_selected_round": int(rounds.min()) if len(rounds) else np.nan,
            })
        for mutation, target in dead_variants.items():
            rounds = start_selected.loc[start_selected["mutation_set_canonical"] == target, "round"]
            passage_rows.append({
                "model": arm, "founder_seed": start, "target": target,
                "passes_through": bool(len(rounds)),
                "first_selected_round": int(rounds.min()) if len(rounds) else np.nan,
                "control_mutation": mutation,
            })
        final = selected.loc[
            (selected["round"] == int(config["search"]["rounds"]))
            & ~selected["in_curated_dataset"]
        ].copy()
        final["model_final_rank"] = final.groupby("founder_seed")["predicted_fitness"].rank(
            ascending=False, method="first"
        )
        final_rows.append(final)
    overlap = pd.DataFrame(overlap_rows)
    passage = pd.DataFrame(passage_rows)
    overlap.to_csv(report / "trajectory_overlap.csv", index=False)
    passage.to_csv(report / "trajectory_passage_controls.csv", index=False)

    finals = pd.concat(final_rows, ignore_index=True)
    membership = finals.groupby("mutation_set_canonical").agg(
        consensus_count=("model", "nunique"),
        mean_rank=("model_final_rank", "mean"),
        rank_sd=("model_final_rank", "std"),
        supporting_models=("model", lambda values: ";".join(sorted(set(values)))),
        founders=("founder_seed", lambda values: ";".join(sorted(set(values)))),
    ).reset_index()
    membership["rank_sd"] = membership["rank_sd"].fillna(0.0)
    membership = membership.sort_values(
        ["consensus_count", "mean_rank", "mutation_set_canonical"],
        ascending=[False, True, True],
    )
    count = min(
        int(config["output"]["consensus_max"]),
        max(int(config["output"]["consensus_min"]), int(config["search"]["final_candidate_count"])),
    )
    consensus = membership.head(count).copy()
    high_consensus = consensus["consensus_count"] >= consensus["consensus_count"].median()
    high_spread = consensus["rank_sd"] >= consensus["rank_sd"].quantile(0.75)
    consensus["high_consensus_high_disagreement"] = high_consensus & high_spread
    consensus.to_csv(report / "consensus_candidates.csv", index=False)
    membership.to_csv(report / "consensus_all_final_candidates.csv", index=False)
    manifest = {
        "status": "complete", "models": len(ARMS), "consensus_candidates": len(consensus),
        "known_winner_passages": int(
            passage.loc[passage["control_mutation"].isna(), "passes_through"].sum()
        ),
        "known_dead_end_passages": int(
            passage.loc[passage["control_mutation"].notna(), "passes_through"].sum()
        ),
        "known_campaign_recoveries": int(overlap["observed_in_campaign"].sum()),
        "ranking_interpretation": "proposal and known-trajectory audit, not a confident ranking",
    }
    (report / "trajectory_analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
