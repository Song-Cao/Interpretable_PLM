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
from applications.ired_design.mutations import parse_mutation, parse_set
from applications.ired_design.run_search import ARMS, _score_maps
from applications.ired_design.training import campaign
from crossfit.datasets import get_split, load_assay
from crossfit.utils import REPO_ROOT, load_config


NO_STEP2_CALIBRATOR = {"phi_w1_max_all_live", "adapter_linear"}


def _stratum_status(config: dict, names: list[str]) -> list[tuple[float, float, str]]:
    """Describe whether the registered Mondrian calibration supports each design."""
    frame = load_assay("IRED", with_sequences=True)
    step2 = load_config(REPO_ROOT / config["training"]["objective_config"])
    train_pool, _ = get_split(
        "IRED", "pg_kfold", fold=int(step2["evaluation"]["primary_test_fold"])
    )
    rng = np.random.default_rng(int(step2["evaluation"]["calibration_seed"]))
    shuffled = rng.permutation(train_pool)
    n_calibration = max(
        1, int(round(len(shuffled) * float(step2["evaluation"]["calibration_fraction"])))
    )
    fit_indices = np.sort(shuffled[n_calibration:])
    fit_positions = set()
    for value in frame.iloc[fit_indices]["mutated_positions"]:
        if not pd.isna(value) and str(value):
            fit_positions.update(map(int, str(value).split(";")))
    output = []
    for name in names:
        mutations = parse_set(name)
        positions = {parse_mutation(value)[1] for value in mutations}
        key = (len(mutations), bool(positions - fit_positions))
        if key == (6, False):
            # The existing calibration has one row here. At 90% coverage the
            # finite-sample conformal rank is two, so the registered radius is infinite.
            output.append((-np.inf, np.inf, "existing_stratum_unbounded_one_calibration_row"))
        else:
            output.append((np.nan, np.nan, f"no_existing_mondrian_stratum_{key}"))
    return output


def main() -> int:
    config = load_config(REPO_ROOT / "Code/config/ired_design.yaml")
    report = REPO_ROOT / config["paths"]["reports"]
    candidates = pd.read_csv(report / "consensus_candidates.csv")
    names = candidates["mutation_set_canonical"].astype(str).tolist()
    stratum_status = _stratum_status(config, names)
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
            if arm in NO_STEP2_CALIBRATOR:
                conformal_lo, conformal_hi = np.nan, np.nan
                conformal_status = "no_existing_step2_conformal_calibrator"
            else:
                conformal_lo, conformal_hi, conformal_status = stratum_status[index]
            rows.append({
                "model": arm, "mutation_set_canonical": name,
                "predicted_fitness": float(mean[index]),
                "deep_ensemble_sd": float(matrix[:, index].std(ddof=1)),
                "rank_within_consensus_pool": int(ranks[index]),
                "conformal_lo": conformal_lo, "conformal_hi": conformal_hi,
                "conformal_status": conformal_status,
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
        "conformal_status": "reported only where supported by the registered Step 2 Mondrian strata; one-row strata are explicitly unbounded",
    }
    (report / "uncertainty_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
