"""Beam-style directed evolution with complete, reconstructable lineage records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .mutations import add, apply, canonical, parse_set


class Scorer(Protocol):
    """Protocol for five-member generation ensembles."""

    def predict_members(self, sequences: list[str], mutation_sets: list[str]) -> np.ndarray:
        """Return member by candidate direct-fitness predictions."""


@dataclass(frozen=True)
class SearchSettings:
    """Fixed application search sizes."""

    rounds: int
    winner_pool_size: int
    survivors_per_round: int


def variant_id(mutation_set: str) -> str:
    """Return the stable human-readable variant identifier."""
    return canonical(mutation_set)


def expand(parents: list[str], mutation_pool: list[str]) -> pd.DataFrame:
    """Generate unique one-step children while retaining a deterministic parent."""
    records: dict[str, dict[str, str]] = {}
    for parent in parents:
        for mutation in mutation_pool:
            child = add(parent, mutation)
            if child is None:
                continue
            child_name = canonical(child)
            candidate = {
                "variant_id": variant_id(child_name),
                "mutation_set_canonical": child_name,
                "parent_variant_id": variant_id(parent),
                "mutation_added": mutation,
            }
            previous = records.get(child_name)
            if previous is None or (candidate["parent_variant_id"], mutation) < (
                previous["parent_variant_id"], previous["mutation_added"]
            ):
                records[child_name] = candidate
    return pd.DataFrame(records.values()).sort_values("mutation_set_canonical").reset_index(drop=True)


def score_round(
    candidates: pd.DataFrame,
    *,
    model: str,
    founder: str,
    round_number: int,
    wild_type: str,
    scorer: Scorer,
    survivors: int,
    known_fitness: dict[str, float],
) -> pd.DataFrame:
    """Score one complete candidate round and mark the retained beam."""
    if candidates.empty:
        raise ValueError("candidate round is empty")
    names = candidates["mutation_set_canonical"].astype(str).tolist()
    sequences = [apply(wild_type, name) for name in names]
    members = np.asarray(scorer.predict_members(sequences, names), dtype=float)
    if members.ndim != 2 or members.shape[1] != len(candidates) or not np.isfinite(members).all():
        raise ValueError("scorer must return a finite member-by-candidate matrix")
    output = candidates.copy()
    output["model"] = model
    output["founder_seed"] = canonical(founder)
    output["round"] = int(round_number)
    output["predicted_fitness"] = members.mean(axis=0)
    output["ensemble_sd"] = members.std(axis=0, ddof=0)
    order = np.argsort(-output["predicted_fitness"].to_numpy(), kind="stable")
    ranks = np.empty(len(output), dtype=int)
    ranks[order] = np.arange(1, len(output) + 1)
    output["rank_within_round"] = ranks
    output["selected_for_next_round"] = output["rank_within_round"] <= min(survivors, len(output))
    output["in_curated_dataset"] = output["mutation_set_canonical"].isin(known_fitness)
    output["measured_fitness_if_known"] = output["mutation_set_canonical"].map(known_fitness)
    output["conformal_lo"] = np.nan
    output["conformal_hi"] = np.nan
    columns = [
        "model", "founder_seed", "round", "variant_id", "mutation_set_canonical",
        "parent_variant_id", "mutation_added", "predicted_fitness", "rank_within_round",
        "selected_for_next_round", "ensemble_sd", "conformal_lo", "conformal_hi",
        "in_curated_dataset", "measured_fitness_if_known",
    ]
    return output[columns].sort_values("rank_within_round").reset_index(drop=True)


def selected_variants(frame: pd.DataFrame) -> list[str]:
    """Return selected variants in score order."""
    chosen = frame.loc[frame["selected_for_next_round"]].sort_values("rank_within_round")
    return chosen["mutation_set_canonical"].astype(str).tolist()

