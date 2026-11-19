"""Structure-derived domain, interface, secondary-structure, and RSA annotations."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, NeighborSearch, ShrakeRupley
from Bio.PDB.DSSP import residue_max_acc
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure


def assign_structural_domains(
    campaign_length: int,
    assignments: Mapping[str, Mapping[str, object]],
    unassigned_positions: Sequence[int],
) -> pd.DataFrame:
    """Assign each campaign position to exactly one registered structural domain."""
    labels: dict[int, str] = {}
    for domain, specification in assignments.items():
        for start, stop in specification["campaign_ranges"]:
            for position in range(int(start), int(stop) + 1):
                if position in labels:
                    raise ValueError(f"position {position} appears in multiple structural domains")
                labels[position] = str(domain)
    unassigned = set(map(int, unassigned_positions))
    if set(labels).intersection(unassigned):
        raise ValueError("assigned and unassigned structural positions overlap")
    positions = range(1, int(campaign_length) + 1)
    expected = set(positions)
    if set(labels).union(unassigned) != expected:
        missing = sorted(expected - set(labels) - unassigned)
        raise ValueError(f"structural domains do not cover campaign positions: {missing}")
    return pd.DataFrame({
        "campaign_position": positions,
        "structural_domain": [labels.get(position, "unassigned") for position in positions],
    })


def secondary_structure_by_chain(
    mmcif_path: str | Path, chains: Sequence[str], campaign_length: int
) -> pd.DataFrame:
    """Read deposited helix and sheet ranges for each requested chain."""
    values = MMCIF2Dict(str(mmcif_path))
    rows = [
        {"chain_origin": str(chain), "campaign_position": position, "secondary_structure": "coil"}
        for chain in chains for position in range(1, int(campaign_length) + 1)
    ]
    frame = pd.DataFrame(rows).set_index(["chain_origin", "campaign_position"])
    for prefix, state in (("_struct_sheet_range", "sheet"), ("_struct_conf", "helix")):
        chain_key = f"{prefix}.beg_auth_asym_id"
        start_key = f"{prefix}.beg_auth_seq_id"
        stop_key = f"{prefix}.end_auth_seq_id"
        for chain, start, stop in zip(values[chain_key], values[start_key], values[stop_key]):
            if str(chain) not in chains:
                continue
            for position in range(max(1, int(start)), min(int(campaign_length), int(stop)) + 1):
                frame.loc[(str(chain), position), "secondary_structure"] = state
    return frame.reset_index()


def _coordinate_model(mmcif_path: str | Path) -> Model:
    structure = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=False).get_structure(
        "5ocm", str(mmcif_path)
    )
    return next(structure.get_models())


def _assembly_model(source: Model, chains: Sequence[str]) -> Model:
    structure = Structure("biological_assembly")
    model = Model(0)
    structure.add(model)
    for chain_id in chains:
        model.add(deepcopy(source[str(chain_id)]))
    return model


def assembly_surface_and_interface(
    mmcif_path: str | Path,
    assembly_chains: Mapping[str, Sequence[str]],
    campaign_length: int,
    *,
    interface_cutoff: float,
    rsa_scale: str,
) -> pd.DataFrame:
    """Compute chain-level RSA and cross-subunit contacts in each biological dimer."""
    if interface_cutoff <= 0:
        raise ValueError("interface cutoff must be positive")
    if rsa_scale not in residue_max_acc:
        raise ValueError(f"unknown RSA scale: {rsa_scale}")
    source = _coordinate_model(mmcif_path)
    maxima = residue_max_acc[rsa_scale]
    rows = []
    for assembly, chain_ids in assembly_chains.items():
        if len(chain_ids) != 2:
            raise ValueError("IRED interface calculation expects biological dimers")
        model = _assembly_model(source, chain_ids)
        ShrakeRupley().compute(model, level="R")
        for chain_id, partner_id in ((chain_ids[0], chain_ids[1]), (chain_ids[1], chain_ids[0])):
            chain = model[str(chain_id)]
            partner = model[str(partner_id)]
            partner_atoms = [
                atom for residue in partner for atom in residue
                if residue.id[0] == " " and str(atom.element).upper() != "H"
            ]
            search = NeighborSearch(partner_atoms)
            for position in range(1, int(campaign_length) + 1):
                residue = chain.child_dict.get((" ", position, " "))
                if residue is None:
                    rows.append({
                        "biological_assembly": str(assembly), "chain_origin": str(chain_id),
                        "partner_chain": str(partner_id), "campaign_position": position,
                        "relative_sasa": np.nan, "interface_contact": False,
                        "nearest_partner_position": np.nan,
                        "minimum_partner_distance_angstrom": np.nan,
                    })
                    continue
                maximum = maxima.get(str(residue.resname))
                relative_sasa = float(residue.sasa / maximum) if maximum else np.nan
                best_distance, best_position = float("inf"), np.nan
                for atom in residue:
                    if str(atom.element).upper() == "H":
                        continue
                    for other in search.search(atom.coord, interface_cutoff, level="A"):
                        distance = float(atom - other)
                        other_position = int(other.get_parent().id[1])
                        if (distance, other_position) < (best_distance, best_position):
                            best_distance, best_position = distance, other_position
                rows.append({
                    "biological_assembly": str(assembly), "chain_origin": str(chain_id),
                    "partner_chain": str(partner_id), "campaign_position": position,
                    "relative_sasa": relative_sasa,
                    "interface_contact": bool(np.isfinite(best_distance)),
                    "nearest_partner_position": best_position,
                    "minimum_partner_distance_angstrom": (
                        best_distance if np.isfinite(best_distance) else np.nan
                    ),
                })
    return pd.DataFrame(rows)


def aggregate_structure_axes(
    chain_values: pd.DataFrame,
    *,
    interface_min_fraction: float,
    core_surface_rsa_cutoff: float,
) -> pd.DataFrame:
    """Aggregate biological-dimer copies without discarding chain-level provenance."""
    if not 0 < interface_min_fraction <= 1:
        raise ValueError("interface fraction threshold must lie in (0, 1]")
    if not 0 < core_surface_rsa_cutoff < 1:
        raise ValueError("RSA cutoff must lie in (0, 1)")
    grouped = chain_values.groupby("campaign_position", sort=True)
    result = grouped.agg(
        coordinate_chain_copies=("relative_sasa", "count"),
        mean_relative_sasa=("relative_sasa", "mean"),
        min_relative_sasa=("relative_sasa", "min"),
        max_relative_sasa=("relative_sasa", "max"),
        interface_chain_copy_fraction=("interface_contact", "mean"),
    ).reset_index()
    result["dimer_interface"] = (
        result["interface_chain_copy_fraction"] >= float(interface_min_fraction)
    )
    result["core_surface"] = np.where(
        result["mean_relative_sasa"].isna(),
        "unassigned",
        np.where(
            result["mean_relative_sasa"] < float(core_surface_rsa_cutoff), "core", "surface"
        ),
    )
    return result


def overlap_counts(frame: pd.DataFrame, axis: str, shell_column: str = "shell") -> pd.DataFrame:
    """Count residues in an annotation axis by functional shell."""
    return frame.groupby([axis, shell_column], dropna=False).size().unstack(fill_value=0)
