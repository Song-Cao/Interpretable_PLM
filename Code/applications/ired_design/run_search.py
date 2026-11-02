#!/usr/bin/env python3
"""Run the resumable multi-model in-silico directed-evolution application."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from applications.ired_design.gpu_engine import BASE_ARMS, StreamingEngine
from applications.ired_design.lineage import expand, score_round, selected_variants
from applications.ired_design.mutations import all_singles, apply, canonical
from applications.ired_design.scoring import GenerationScorer
from applications.ired_design.training import CLASSICAL_ARMS, campaign
from crossfit.representations import aaindex_flat, load_aaindex_table, onehot, onehot_positional
from crossfit.utils import REPO_ROOT, load_config


ARMS = (*CLASSICAL_ARMS, *BASE_ARMS, "onehot_positional_conv", "phi_w1_max_all_live", "adapter_linear")
DIRECT_ARMS = (*CLASSICAL_ARMS, "onehot_positional_conv")


class _Lookup:
    def __init__(self, predictions: dict[str, np.ndarray]) -> None:
        self.predictions = predictions

    def predict_members(self, sequences: list[str], mutation_sets: list[str]) -> np.ndarray:
        return np.stack([self.predictions[value] for value in mutation_sets], axis=1)


def _known(frame: pd.DataFrame) -> dict[str, float]:
    return {
        canonical(str(mutation).replace(":", "-")): float(fitness)
        for mutation, fitness in zip(frame["mutations"], frame["y_raw"])
    }


def _direct_maps(
    candidates: dict[str, list[str]], config: dict, wild_type: str,
    engine: StreamingEngine, *, chunk_size: int = 512,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    aaindex = load_aaindex_table(REPO_ROOT / config["paths"]["aaindex_table"])
    augmented_names = sorted(set().union(*(
        set(candidates.get(arm, []))
        for arm in CLASSICAL_ARMS
        if arm.endswith("augmented") and not arm.endswith("unaugmented")
    )))
    shared_density = engine.dpll(augmented_names) if augmented_names else {}
    for arm in DIRECT_ARMS:
        names = sorted(set(candidates.get(arm, [])))
        if not names:
            continue
        scorer = GenerationScorer(arm, config, device="cuda")
        augmented = arm.endswith("augmented") and not arm.endswith("unaugmented")
        arm_values = {}
        for start in range(0, len(names), chunk_size):
            batch = names[start : start + chunk_size]
            sequences = [apply(wild_type, value) for value in batch]
            if arm.startswith("onehot_ridge"):
                features = {"onehot": onehot(sequences)}
            elif arm.startswith("aaindex"):
                features = {"aaindex": aaindex_flat(sequences, aaindex)}
            else:
                features = {"onehot_positional": onehot_positional(sequences)}
            if augmented:
                features["density"] = np.asarray([shared_density[value] for value in batch])
            predictions = scorer.predict_prepared(features)
            arm_values.update({name: predictions[:, index] for index, name in enumerate(batch)})
        result[arm] = arm_values
    return result


def _score_maps(
    candidates: dict[str, list[str]], config: dict, wild_type: str, engine: StreamingEngine
) -> dict[str, dict[str, np.ndarray]]:
    output = _direct_maps(candidates, config, wild_type, engine)
    output.update(engine.score_base_arms(candidates))
    if candidates.get("phi_w1_max_all_live"):
        output["phi_w1_max_all_live"] = engine.score_phi(candidates["phi_w1_max_all_live"])
    if candidates.get("adapter_linear"):
        output["adapter_linear"] = engine.score_adapter(candidates["adapter_linear"])
    return output


def _budget(config: dict, benchmark: dict, *, pool: int, survivors: int) -> dict:
    search = config["search"]
    founders = len(config["founders"]["paper"]) + 1
    rounds = int(search["rounds"])
    scan = 290 * 19
    per_arm = scan + founders * (pool + (rounds - 1) * survivors * pool)
    base = per_arm * len(BASE_ARMS) * benchmark["seconds_per_sequence"]
    phi = per_arm * 2 * benchmark["seconds_per_sequence"]
    adapter = per_arm * benchmark["seconds_per_sequence"]
    direct_conv = per_arm * 0.0
    density = per_arm * 2 * 5.0 * benchmark["seconds_per_masked_task"]
    hours = (base + phi + adapter + direct_conv + density) / 3600.0
    return {
        "winner_pool_size": pool, "survivors_per_round": survivors,
        "upper_bound_candidates_per_arm": per_arm,
        "projected_gpu_hours_upper_bound": hours,
        "components_hours": {
            "base_esm_arms": base / 3600.0, "phi_two_backbones": phi / 3600.0,
            "adapter": adapter / 3600.0, "dpll_two_arms": density / 3600.0,
        },
    }


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".incomplete.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _valid_shard(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    required = {
        "model", "founder_seed", "round", "variant_id", "mutation_set_canonical",
        "parent_variant_id", "mutation_added", "predicted_fitness", "rank_within_round",
        "selected_for_next_round", "ensemble_sd", "conformal_lo", "conformal_hi",
        "in_curated_dataset", "measured_fitness_if_known",
    }
    return frame if required.issubset(frame.columns) and len(frame) else None


def main() -> int:
    started = time.monotonic()
    config = load_config(REPO_ROOT / "Code/config/ired_design.yaml")
    frame, wild_type = campaign(config)
    known = _known(frame)
    report = REPO_ROOT / config["paths"]["reports"]
    report.mkdir(parents=True, exist_ok=True)
    engine = StreamingEngine(config, wild_type)
    singles = all_singles(wild_type)
    representative = singles[: int(config["search"]["representative_embedding_sequences"])]
    benchmark = engine.benchmark(representative)
    budget = _budget(
        config, benchmark, pool=int(config["search"]["winner_pool_size"]),
        survivors=int(config["search"]["survivors_per_round"]),
    )
    if budget["projected_gpu_hours_upper_bound"] > float(config["search"]["estimated_gpu_hours"][1]):
        budget["reduced_from"] = {
            "winner_pool_size": int(config["search"]["winner_pool_size"]),
            "survivors_per_round": int(config["search"]["survivors_per_round"]),
        }
        budget = budget | _budget(
            config, benchmark,
            pool=int(config["search"]["budget_reduced_winner_pool_size"]),
            survivors=int(config["search"]["budget_reduced_survivors_per_round"]),
        )
        budget["reduction_reason"] = "representative timing projected above six GPU-hours"
    pool = int(budget["winner_pool_size"])
    survivors = int(budget["survivors_per_round"])
    budget["benchmark"] = benchmark
    (report / "embedding_budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    if budget["projected_gpu_hours_upper_bound"] > float(config["search"]["hard_gpu_hours"]):
        budget["hard_ceiling_warning"] = "upper bound remains above ceiling; shared-union realized time is enforced during execution"
        (report / "embedding_budget.json").write_text(json.dumps(budget, indent=2) + "\n")

    scan_candidates = {arm: singles for arm in ARMS}
    scan_maps = _score_maps(scan_candidates, config, wild_type, engine)
    pools = {}
    template = pd.DataFrame({
        "variant_id": singles, "mutation_set_canonical": singles,
        "parent_variant_id": "WT", "mutation_added": singles,
    })
    for arm in ARMS:
        scan = score_round(
            template, model=arm, founder="WT", round_number=0, wild_type=wild_type,
            scorer=_Lookup(scan_maps[arm]), survivors=pool, known_fitness=known,
        )
        scan.to_csv(report / f"{arm}_round1_single_scan.csv", index=False)
        winner = scan.loc[scan["selected_for_next_round"]]
        winner.to_csv(report / f"{arm}_winner_pool.csv", index=False)
        pools[arm] = selected_variants(scan)

    control_background = canonical(config["controls"]["background"])
    control_candidates = expand([control_background], singles)
    control_names = control_candidates["mutation_set_canonical"].astype(str).tolist()
    control_maps = _score_maps({arm: control_names for arm in ARMS}, config, wild_type, engine)
    control_targets = {
        mutation: canonical((*control_background.split("-"), mutation))
        for mutation in (*config["controls"]["dead_ends"], config["controls"]["positive"])
    }
    control_rows = []
    for arm in ARMS:
        values = np.asarray([control_maps[arm][name].mean() for name in control_names])
        order = np.argsort(-values, kind="stable")
        ranks = np.empty(len(values), dtype=int); ranks[order] = np.arange(1, len(values) + 1)
        lookup = {name: index for index, name in enumerate(control_names)}
        for mutation, variant in control_targets.items():
            index = lookup[variant]
            control_rows.append({
                "model": arm, "background": control_background, "control_mutation": mutation,
                "control_role": "positive" if mutation == config["controls"]["positive"] else "known_dead_end",
                "variant": variant, "rank_among_all_background_singles": int(ranks[index]),
                "predicted_fitness": float(values[index]), "candidate_count": len(values),
                "in_model_winner_pool": mutation in pools[arm],
            })
    pd.DataFrame(control_rows).to_csv(report / "controls_ranks.csv", index=False)

    founders = list(config["founders"]["paper"]) + [config["founders"]["campaign_peak"]]
    parent_map = {(arm, canonical(founder)): [canonical(founder)] for arm in ARMS for founder in founders}
    lineage_root = report / "full_lineage"
    realized_evaluations = 0
    for round_number in range(1, int(config["search"]["rounds"]) + 1):
        definitions: dict[tuple[str, str], pd.DataFrame] = {}
        requested: dict[str, list[str]] = {arm: [] for arm in ARMS}
        for arm in ARMS:
            for founder in map(canonical, founders):
                shard = lineage_root / arm / f"{founder}_round_{round_number}.parquet"
                complete = _valid_shard(shard)
                if complete is not None:
                    parent_map[(arm, founder)] = selected_variants(complete)
                    continue
                candidates = expand(parent_map[(arm, founder)], pools[arm])
                definitions[(arm, founder)] = candidates
                requested[arm].extend(candidates["mutation_set_canonical"].astype(str))
        if definitions:
            maps = _score_maps(requested, config, wild_type, engine)
            for (arm, founder), candidates in definitions.items():
                result = score_round(
                    candidates, model=arm, founder=founder, round_number=round_number,
                    wild_type=wild_type, scorer=_Lookup(maps[arm]), survivors=survivors,
                    known_fitness=known,
                )
                shard = lineage_root / arm / f"{founder}_round_{round_number}.parquet"
                _atomic_parquet(result, shard)
                top_dir = report / "round_top" / arm
                top_dir.mkdir(parents=True, exist_ok=True)
                result.loc[result["selected_for_next_round"]].to_csv(
                    top_dir / f"{founder}_round_{round_number}.csv", index=False
                )
                parent_map[(arm, founder)] = selected_variants(result)
                realized_evaluations += len(result)
        elapsed_hours = (time.monotonic() - started) / 3600.0
        if elapsed_hours >= float(config["search"]["hard_gpu_hours"]):
            raise RuntimeError("application search reached the six GPU-hour hard ceiling after a complete round")

    model_top = []
    for arm in ARMS:
        frames = [pd.read_parquet(path) for path in sorted((lineage_root / arm).glob("*.parquet"))]
        combined = pd.concat(frames, ignore_index=True)
        final = combined.loc[
            (combined["round"] == int(config["search"]["rounds"]))
            & ~combined["in_curated_dataset"]
        ].sort_values(["rank_within_round", "founder_seed"])
        top = final.drop_duplicates("mutation_set_canonical").head(int(config["output"]["commit_individual_top"]))
        top.to_csv(report / f"{arm}_top20.csv", index=False)
        model_top.append(top)
    manifest = {
        "status": "complete", "arms": list(ARMS), "founders": list(map(canonical, founders)),
        "winner_pool_size": pool, "survivors_per_round": survivors,
        "realized_evaluated_rows_new": realized_evaluations,
        "gpu_wall_hours": (time.monotonic() - started) / 3600.0,
    }
    (report / "search_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
