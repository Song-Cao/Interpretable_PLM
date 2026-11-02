#!/usr/bin/env python3
"""Validate and apply the stability oracle to the final design pool."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from applications.ired_design.mutations import parse_mutation, parse_set
from applications.ired_design.training import campaign
from crossfit.utils import REPO_ROOT, load_config


ANCHORS = {
    "T241A": "stabilising",
    "S68P": "destabilising",
    "P203Q": "destabilising",
    "H244Y": "destabilising",
    "A72T": "destabilising",
    "G37C": "destabilising",
}


def oracle_lookup(ssm: pd.DataFrame) -> dict[str, float]:
    """Map zero-indexed ThermoMPNN positions to one-indexed substitutions."""
    required = {"position", "wildtype", "mutation", "ddG_pred"}
    if not required.issubset(ssm.columns):
        raise ValueError(f"ThermoMPNN table lacks {sorted(required - set(ssm.columns))}")
    return {
        f"{row.wildtype}{int(row.position) + 1}{row.mutation}": float(row.ddG_pred)
        for row in ssm.itertuples(index=False)
    }


def validate_thermompnn(ssm: pd.DataFrame, wild_type: str) -> pd.DataFrame:
    """Check sequence alignment and all experimentally required anchor signs."""
    sequence = "".join(
        ssm.sort_values("position").groupby("position", sort=True)["wildtype"].first().astype(str)
    )
    if sequence != wild_type:
        raise AssertionError("ThermoMPNN structure sequence does not exactly match campaign wild type")
    lookup = oracle_lookup(ssm)
    rows = []
    for mutation, expected in ANCHORS.items():
        value = lookup[mutation]
        observed = "stabilising" if value < 0 else "destabilising"
        rows.append({
            "mutation": mutation,
            "experimental_anchor": expected,
            "thermompnn_ddg_kcal_mol": value,
            "predicted_sign": observed,
            "sign_matches": observed == expected,
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssm", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(REPO_ROOT / "Code/config/ired_design.yaml")
    report = REPO_ROOT / config["paths"]["reports"]
    report.mkdir(parents=True, exist_ok=True)
    _, wild_type = campaign(config)
    ssm = pd.read_csv(args.ssm)
    anchors = validate_thermompnn(ssm, wild_type)
    anchors.to_csv(report / "thermompnn_anchor_validation.csv", index=False)
    validated = bool(anchors["sign_matches"].all())

    candidates_path = report / "consensus_candidates.csv"
    score_rows = []
    if validated and candidates_path.exists():
        lookup = oracle_lookup(ssm)
        candidates = pd.read_csv(candidates_path)
        shells = pd.read_csv(REPO_ROOT / "Reports/step8/ired_functional_shells.csv")
        shell_lookup = shells.set_index("campaign_position")["shell"].to_dict()
        for name in candidates["mutation_set_canonical"].astype(str):
            mutations = parse_set(name)
            missing = [mutation for mutation in mutations if mutation not in lookup]
            values = [lookup[mutation] for mutation in mutations if mutation in lookup]
            positions = [parse_mutation(mutation)[1] for mutation in mutations]
            score_rows.append({
                "mutation_set_canonical": name,
                "thermompnn_additive_ddg_kcal_mol": float(np.sum(values)) if not missing else np.nan,
                "thermompnn_component_ddg": ";".join(
                    f"{mutation}:{lookup[mutation]:.6g}" for mutation in mutations if mutation in lookup
                ),
                "missing_mutations": ";".join(missing),
                "functional_shells_touched": ";".join(
                    sorted({str(shell_lookup.get(position, "unknown")) for position in positions})
                ),
                "stability_consistency": (
                    "consistent_with_improved_stability" if values and np.sum(values) < 0
                    else "predicted_net_destabilising"
                ),
                "oracle_disagreement": "not_assessable_one_validated_oracle",
                "interpretation": "additive composition of single-substitution estimates; not a direct multi-mutant prediction",
            })
        pd.DataFrame(score_rows).to_csv(report / "final_candidate_stability.csv", index=False)

    manifest = {
        "oracle": "ThermoMPNN",
        "validated": validated,
        "anchor_count": len(anchors),
        "anchor_signs_matched": int(anchors["sign_matches"].sum()),
        "structure_sequence_exact_match": True,
        "final_candidates_scored": len(score_rows),
        "multi_mutant_method": "sum of validated single-substitution ddG estimates",
        "multi_mutant_limitation": "additivity is an approximation and does not model epistasis",
        "active_site_geometry_status": "not_scored_without_a_validated_mutant-structure oracle",
        "solubility_status": "not_scored_without_an IRED anchor",
        "foldx_status": "not_available; cut before trajectory deliverables under the registered priority order",
        "docking_status": "not_run; first item cut under the registered priority order",
    }
    (report / "biophysics_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
