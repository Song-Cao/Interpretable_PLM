#!/usr/bin/env python3
"""Build IRED structural-domain, dimer-interface, and core/surface tracks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.datasets import wt_sequence
from crossfit.structural_domains import (
    aggregate_structure_axes,
    assembly_surface_and_interface,
    assign_structural_domains,
    overlap_counts,
    secondary_structure_by_chain,
)
from crossfit.utils import REPO_ROOT, load_config


def main() -> int:
    config = load_config(REPO_ROOT / "Code/config/step8.yaml")
    structure = config["structure"]
    partition = config["domain_partition"]
    sequence = wt_sequence(str(config["dataset"]))
    campaign_length = int(structure["campaign_length"])
    if len(sequence) != campaign_length:
        raise AssertionError("IRED wild-type length changed")
    assemblies = {
        str(key): tuple(map(str, value))
        for key, value in structure["biological_assembly_chain_sets"].items()
    }
    chains = tuple(chain for members in assemblies.values() for chain in members)
    structure_path = REPO_ROOT / structure["deposited_mmcif"]

    domains = assign_structural_domains(
        campaign_length, partition["assignments"], partition["unassigned_campaign_positions"]
    )
    chain_values = assembly_surface_and_interface(
        structure_path, assemblies, campaign_length,
        interface_cutoff=float(partition["interface_contact_cutoff_angstrom"]),
        rsa_scale=str(partition["rsa_maximum_scale"]),
    )
    secondary = secondary_structure_by_chain(structure_path, chains, campaign_length)
    chain_values = chain_values.merge(
        secondary, on=["chain_origin", "campaign_position"], how="left", validate="one_to_one"
    )
    secondary_consensus = secondary.groupby("campaign_position")["secondary_structure"].agg(
        lambda values: ";".join(sorted(set(map(str, values))))
    ).rename("secondary_structure")
    aggregate = aggregate_structure_axes(
        chain_values,
        interface_min_fraction=float(partition["interface_min_chain_copy_fraction"]),
        core_surface_rsa_cutoff=float(partition["core_surface_rsa_cutoff"]),
    )
    shells = pd.read_csv(REPO_ROOT / "Reports/step8/ired_functional_shells.csv")
    contacts = pd.read_csv(REPO_ROOT / "Reports/step8/ired_crystallographic_contacts.csv")
    contact_sets = {
        track: set(group["campaign_position"].astype(int))
        for track, group in contacts.groupby("track")
    }
    domain_table = domains.merge(aggregate, on="campaign_position", validate="one_to_one")
    domain_table.insert(1, "campaign_residue", list(sequence))
    domain_table = domain_table.merge(
        secondary_consensus, on="campaign_position", validate="one_to_one"
    )
    domain_table = domain_table.merge(
        shells[["campaign_position", "shell"]], on="campaign_position", validate="one_to_one"
    )
    for track, positions in contact_sets.items():
        domain_table[track] = domain_table["campaign_position"].isin(positions)

    report = REPO_ROOT / config["paths"]["reports"]
    chain_values.to_csv(report / "ired_domain_partition_chain_provenance.csv", index=False)
    domain_table.to_csv(report / "ired_domain_partition.csv", index=False)
    counts = []
    for axis in ("structural_domain", "dimer_interface", "core_surface"):
        for value, count in domain_table[axis].value_counts(dropna=False).items():
            counts.append({"axis": axis, "category": value, "residue_count": int(count)})
        overlap_counts(domain_table, axis).to_csv(
            report / f"ired_{axis}_by_shell_overlap.csv"
        )
    pd.DataFrame(counts).to_csv(report / "ired_domain_partition_counts.csv", index=False)

    topology = chain_values.groupby(
        ["chain_origin", "secondary_structure"], as_index=False
    ).size().rename(columns={"size": "residue_count"})
    topology.to_csv(report / "ired_secondary_structure_counts_by_chain.csv", index=False)
    overlap_counts(
        domain_table, "structural_domain", shell_column="secondary_structure"
    ).to_csv(report / "ired_structural_domain_by_secondary_structure.csv")
    contact_summary = domain_table.groupby("structural_domain", as_index=False).agg(
        residue_count=("campaign_position", "size"),
        NADP_contact_residues=("crystallographic_NADP_contact", "sum"),
        ketone_analogue_contact_residues=("crystallographic_ketone_analogue_contact", "sum"),
    )
    contact_summary.to_csv(report / "ired_domain_contact_summary.csv", index=False)
    manifest = {
        "status": "complete",
        "dataset": "IRED",
        "pdb_id": structure["pdb_id"],
        "source": partition["source"],
        "source_url": partition["source_url"],
        "retrieved_date": partition["retrieved_date"],
        "biological_assembly_chain_sets": {key: list(value) for key, value in assemblies.items()},
        "interface_contact_cutoff_angstrom": partition["interface_contact_cutoff_angstrom"],
        "interface_min_chain_copy_fraction": partition["interface_min_chain_copy_fraction"],
        "rsa_maximum_scale": partition["rsa_maximum_scale"],
        "core_surface_rsa_cutoff": partition["core_surface_rsa_cutoff"],
        "domain_counts": domain_table["structural_domain"].value_counts().to_dict(),
        "interface_residues": int(domain_table["dimer_interface"].sum()),
        "core_surface_counts": domain_table["core_surface"].value_counts().to_dict(),
        "chain_provenance_rows": int(len(chain_values)),
        "unassigned_fraction": float(
            (domain_table["structural_domain"] == "unassigned").mean()
        ),
    }
    (report / "ired_domain_partition_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
