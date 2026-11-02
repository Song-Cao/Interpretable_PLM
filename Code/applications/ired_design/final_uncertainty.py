#!/usr/bin/env python3
"""Rescore the consensus pool across every model and preserve disagreement."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from applications.ired_design.gpu_engine import StreamingEngine
from applications.ired_design.run_search import ARMS, _score_maps
from applications.ired_design.training import campaign
from crossfit.utils import REPO_ROOT, load_config


def main() -> int:
    config = load_config(REPO_ROOT / "Code/config/ired_design.yaml")
    report = REPO_ROOT / config["paths"]["reports"]
    candidates = pd.read_csv(report / "consensus_candidates.csv")
    names = candidates["mutation_set_canonical"].astype(str).tolist()
    _, wild_type = campaign(config)
    engine = StreamingEngine(config, wild_type)
    maps = _score_maps({arm: names for arm in ARMS}, config, wild_type, engine)
    maps["adapter_linear"] = engine.score_adapter(names, all_members=True)
    rows = []
    rank_matrix = []
    for arm in ARMS:
        matrix = np.stack([maps[arm][name] for name in names], axis=1)
        mean = matrix.mean(axis=0)
        order = np.argsort(-mean, kind="stable")
        ranks = np.empty(len(mean), dtype=int); ranks[order] = np.arange(1, len(mean) + 1)
        rank_matrix.append(ranks)
        for index, name in enumerate(names):
            rows.append({
                "model": arm, "mutation_set_canonical": name,
                "predicted_fitness": float(mean[index]),
                "deep_ensemble_sd": float(matrix[:, index].std(ddof=1)),
                "rank_within_consensus_pool": int(ranks[index]),
                "conformal_lo": np.nan, "conformal_hi": np.nan,
                "conformal_status": "pending_existing_calibration_reconstruction",
            })
    detailed = pd.DataFrame(rows)
    detailed.to_csv(report / "final_candidate_model_scores.csv", index=False)
    rank_values = np.stack(rank_matrix)
    candidates["cross_model_rank_sd"] = rank_values.std(axis=0, ddof=1)
    candidates["cross_model_mean_rank"] = rank_values.mean(axis=0)
    candidates["high_consensus_high_disagreement"] = (
        (candidates["consensus_count"] >= candidates["consensus_count"].median())
        & (candidates["cross_model_rank_sd"] >= candidates["cross_model_rank_sd"].quantile(0.75))
    )
    candidates.to_csv(report / "consensus_candidates.csv", index=False)
    manifest = {
        "status": "complete", "candidates": len(candidates), "models": len(ARMS),
        "ensemble_members_per_model": 5,
        "conformal_status": "pending reconstruction of the existing evaluation calibrators",
    }
    (report / "uncertainty_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
