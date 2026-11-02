#!/usr/bin/env python3
"""Assemble locally readable application tables from completed search artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from applications.ired_design.final_uncertainty import NO_STEP2_CALIBRATOR, _stratum_status
from applications.ired_design.mutations import parse_mutation, parse_set
from crossfit.utils import REPO_ROOT, load_config


REPORT = REPO_ROOT / "Reports/applications/ired_design"
DEAD_ENDS = {"A72T", "G37C"}


def _thermompnn_lookup() -> dict[str, float]:
    table = pd.read_csv(
        REPORT / "oracles/thermompnn/ThermoMPNN_inference_5ocm_deposited.csv"
    )
    substitutions = table.loc[table["wildtype"] != table["mutation"]].copy()
    if len(substitutions) != 290 * 19:
        raise RuntimeError(f"expected 5,510 non-identity ThermoMPNN values, found {len(substitutions)}")
    substitutions.to_csv(
        REPORT / "oracles/thermompnn/thermompnn_single_substitution_5510.csv", index=False
    )
    return {
        f"{row.wildtype}{int(row.position) + 1}{row.mutation}": float(row.ddG_pred)
        for row in substitutions.itertuples()
    }


def _conformal_reason(model: str, mutation_set: str, status: tuple[float, float, str]) -> tuple[object, object]:
    if model in NO_STEP2_CALIBRATOR:
        reason = "not_available: model has no registered Step 2 conformal calibrator"
        return reason, reason
    lo, hi, label = status
    if np.isneginf(lo) and np.isposinf(hi):
        reason = "unbounded: registered Mondrian stratum has one calibration row"
        return reason, reason
    n_mutations = len(parse_set(mutation_set))
    if "no_existing_mondrian_stratum" in label and n_mutations >= 6:
        reason = f"absent: no registered Mondrian stratum for this {n_mutations}-mutation design"
        return reason, reason
    reason = "not_available: generation refit did not persist the Step 2 calibrator quantile"
    return reason, reason


def assemble_round_detail() -> pd.DataFrame:
    paths = sorted((REPORT / "round_top").glob("*/*.csv"))
    if len(paths) != 208:
        raise RuntimeError(f"expected 208 round-top tables, found {len(paths)}")
    detail = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    names = detail["mutation_set_canonical"].astype(str).tolist()
    config = load_config(REPO_ROOT / "Code/config/ired_design.yaml")
    statuses = _stratum_status(config, names)
    lookup = _thermompnn_lookup()
    detail["n_mutations"] = [len(parse_set(name)) for name in names]
    detail["ddg_thermompnn_additive"] = [
        sum(lookup[mutation] for mutation in parse_set(name)) for name in names
    ]
    intervals = [
        _conformal_reason(str(model), name, status)
        for model, name, status in zip(detail["model"], names, statuses)
    ]
    detail["conformal_lo"] = [value[0] for value in intervals]
    detail["conformal_hi"] = [value[1] for value in intervals]
    detail["contains_known_dead_end"] = [
        bool(DEAD_ENDS.intersection(parse_set(name))) for name in names
    ]
    columns = [
        "model", "founder_seed", "round", "rank_within_round", "variant_id",
        "mutation_set_canonical", "parent_variant_id", "mutation_added",
        "n_mutations", "predicted_fitness", "ensemble_sd", "conformal_lo",
        "conformal_hi", "ddg_thermompnn_additive", "in_curated_dataset",
        "measured_fitness_if_known", "contains_known_dead_end",
    ]
    detail = detail[columns].sort_values(
        ["model", "founder_seed", "round", "rank_within_round"], kind="stable"
    )
    if len(detail) != 13 * 4 * 4 * 30:
        raise RuntimeError(f"unexpected round-detail row count: {len(detail)}")
    detail.to_csv(REPORT / "per_round_top_variants.csv", index=False)
    return detail


def assemble_biophysical_panel() -> pd.DataFrame:
    consensus = pd.read_csv(REPORT / "consensus_candidates.csv")
    stability = pd.read_csv(REPORT / "final_candidate_stability.csv")
    panel = consensus.merge(stability, on="mutation_set_canonical", how="left", validate="one_to_one")
    manifest = json.loads((REPORT / "biophysics_manifest.json").read_text())
    for key in (
        "active_site_geometry_status", "solubility_status", "foldx_status", "docking_status"
    ):
        panel[key] = manifest[key]
    panel.to_csv(REPORT / "final_consensus_biophysical_panel.csv", index=False)
    return panel


def main() -> int:
    detail = assemble_round_detail()
    panel = assemble_biophysical_panel()
    print(json.dumps({"round_detail_rows": len(detail), "final_panel_rows": len(panel)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
