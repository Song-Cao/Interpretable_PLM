#!/usr/bin/env python3
"""Assemble IRED Step 8 numbering, provenance, contact, and shell tracks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.datasets import wt_sequence
from crossfit.persistence import sha256_file
from crossfit.structural_annotations import (
    align_campaign_sequence,
    assign_functional_shells,
    biological_assembly_polymer_chains,
    canonical_polymer_sequence,
    catalytic_distances,
    ligand_contacts,
    parse_residue_label,
    protein_residues,
    validate_residue_tracks,
)
from crossfit.utils import REPO_ROOT, load_config


def main() -> int:
    config = load_config(REPO_ROOT / "Code/config/step8.yaml")
    structure_config = config["structure"]
    structure_path = REPO_ROOT / structure_config["deposited_mmcif"]
    assembly_path = REPO_ROOT / structure_config["assembly_1_mmcif"]
    if sha256_file(structure_path) != structure_config["deposited_mmcif_sha256"]:
        raise AssertionError("5OCM deposited mmCIF hash differs from the registered file")
    if sha256_file(assembly_path) != structure_config["assembly_1_mmcif_sha256"]:
        raise AssertionError("5OCM assembly-1 mmCIF hash differs from the registered file")

    sequence = wt_sequence(str(config["dataset"]))
    if len(sequence) != int(structure_config["campaign_length"]):
        raise AssertionError("campaign sequence length differs from Step 8 config")
    pdb_sequence = canonical_polymer_sequence(structure_path)
    alignment = align_campaign_sequence(sequence, pdb_sequence)
    if alignment.identity < 0.95 or alignment.campaign_coverage < 0.95:
        raise RuntimeError("campaign-to-5OCM alignment is too poor for structural annotation")

    observed_assemblies = biological_assembly_polymer_chains(structure_path)
    expected_assemblies = {
        str(key): tuple(map(str, value))
        for key, value in structure_config["biological_assembly_chain_sets"].items()
    }
    if observed_assemblies != expected_assemblies:
        raise AssertionError(
            f"5OCM biological assemblies changed: {observed_assemblies} != {expected_assemblies}"
        )

    supplied = validate_residue_tracks(sequence, config["annotation_tracks"])
    supplied["structure_position"] = supplied["campaign_position"].map(alignment.mapping)
    supplied["structure_residue"] = supplied["structure_position"].map(
        lambda position: pdb_sequence[int(position) - 1]
    )
    supplied["provenance_kind"] = "experimentally_observed"
    supplied["source"] = "Hollfelder_lab_preprints_user_supplied"
    supplied["role"] = supplied["track"].map(config["track_roles"])
    supplied["claim_eligible"] = supplied["role"].ne("positive_control_only")

    catalytic = [parse_residue_label(value)[1] for value in config["annotation_tracks"]["catalytic_proposed"]]
    functional = []
    for track in (
        "substrate_direct_contact", "NADPH_direct_contact",
        "NADPH_proximal_or_functionally_coupled",
    ):
        functional.extend(parse_residue_label(value)[1] for value in config["annotation_tracks"][track])
    residues = protein_residues(structure_path, len(sequence))
    distances = catalytic_distances(residues, expected_assemblies, len(sequence), catalytic)
    shells = assign_functional_shells(
        sequence, distances, catalytic, functional,
        first_shell_cutoff=float(structure_config["first_shell_cutoff_angstrom"]),
        second_shell_cutoff=float(structure_config["second_shell_cutoff_angstrom"]),
    )

    contact_frames = []
    for resname, track, proxy_note in (
        (
            structure_config["ketone_analogue_resname"],
            "crystallographic_ketone_analogue_contact",
            "5OCM 9RH ketone analogue; proxy for the campaign imine substrate",
        ),
        (
            structure_config["cofactor_resname"],
            "crystallographic_NADP_contact",
            "5OCM NAP crystallographic cofactor contact",
        ),
    ):
        contacts = ligand_contacts(
            structure_path, expected_assemblies, len(sequence), str(resname),
            cutoff=float(structure_config["contact_cutoff_angstrom"]),
        )
        contacts.insert(0, "track", track)
        contacts["provenance_kind"] = "crystallographic_contact"
        contacts["source"] = "PDB_5OCM"
        contacts["proxy_note"] = proxy_note
        contact_frames.append(contacts)
    contacts = pd.concat(contact_frames, ignore_index=True)

    report = REPO_ROOT / config["paths"]["reports"]
    report.mkdir(parents=True, exist_ok=True)
    supplied.to_csv(report / "ired_annotation_tracks.csv", index=False)
    contacts.to_csv(report / "ired_crystallographic_contacts.csv", index=False)
    shells.to_csv(report / "ired_functional_shells.csv", index=False)
    shell_counts = shells.groupby("shell", as_index=False).size().rename(columns={"size": "residue_count"})
    shell_counts.to_csv(report / "ired_functional_shell_counts.csv", index=False)
    pd.DataFrame([
        {
            "track": "docking_derived_substrate_contact", "provenance_kind": "docking_derived",
            "status": "not_provided_not_inferred", "residue_count": 0,
            "claim_eligible": False,
        },
        {
            "track": "evolutionary_peak_residue", "provenance_kind": "experimentally_observed",
            "status": "positive_control_only", "residue_count": int(
                (supplied["track"] == "evolutionary_peak_residue").sum()
            ), "claim_eligible": False,
        },
    ]).to_csv(report / "ired_contact_provenance_status.csv", index=False)

    supplied_context = {
        int(position): sequence[int(position) - 1]
        for position in structure_config["pdb_reported_context_positions"]
    }
    manifest = {
        "status": "complete",
        "dataset": str(config["dataset"]),
        "pdb_id": str(structure_config["pdb_id"]),
        "campaign_length": len(sequence),
        "structure_sequence_length": len(pdb_sequence),
        "alignment_identity": alignment.identity,
        "campaign_coverage": alignment.campaign_coverage,
        "mismatches": list(alignment.mismatches),
        "campaign_gaps": list(alignment.campaign_gaps),
        "structure_terminal_extension": pdb_sequence[len(sequence):],
        "biological_assemblies": {key: list(value) for key, value in expected_assemblies.items()},
        "assembly_interpretation": (
            "The deposited six-chain coordinate entry contains three RCSB-defined biological "
            "dimers (A/F, B/C, D/E). Contacts use all three copies but never cross between dimers."
        ),
        "supplied_residues_all_map": bool(
            (supplied["campaign_residue"] == supplied["structure_residue"]).all()
        ),
        "reported_5ocm_context_positions": supplied_context,
        "numbering_note": (
            "W178 and H244 map exactly to 5OCM W178 and H244. They are distinct from the "
            "entry context residues W208 and H238 and were not substituted."
        ),
        "shell_counts": dict(zip(shell_counts["shell"], shell_counts["residue_count"])),
        "crystallographic_contact_rows": int(len(contacts)),
        "enrichment_tests_run": False,
        "patching_run": False,
        "oracle_validation_run": False,
    }
    (report / "ired_annotation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
