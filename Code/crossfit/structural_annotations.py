"""Sequence mapping and structure-derived annotation tracks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


@dataclass(frozen=True)
class SequenceAlignment:
    """Mapping from one-based campaign positions to structure sequence positions."""

    mapping: dict[int, int]
    identity: float
    campaign_coverage: float
    mismatches: tuple[tuple[int, str, int, str], ...]
    campaign_gaps: tuple[int, ...]


def canonical_polymer_sequence(mmcif_path: str | Path) -> str:
    """Read the canonical polymer sequence from a single-entity mmCIF entry."""
    values = MMCIF2Dict(str(mmcif_path)).get("_entity_poly.pdbx_seq_one_letter_code_can")
    if not values or len(values) != 1:
        raise ValueError("expected one canonical polymer entity")
    return "".join(str(values[0]).split()).upper()


def align_campaign_sequence(campaign: str, structure: str) -> SequenceAlignment:
    """Globally align a campaign sequence and return an auditable position mapping."""
    if not campaign or not structure:
        raise ValueError("alignment sequences must be non-empty")
    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(campaign, structure)[0]
    mapping: dict[int, int] = {}
    mismatches: list[tuple[int, str, int, str]] = []
    for (campaign_start, campaign_stop), (structure_start, structure_stop) in zip(
        alignment.aligned[0], alignment.aligned[1]
    ):
        if campaign_stop - campaign_start != structure_stop - structure_start:
            raise AssertionError("aligned blocks must have equal ungapped length")
        for left, right in zip(
            range(int(campaign_start), int(campaign_stop)),
            range(int(structure_start), int(structure_stop)),
        ):
            mapping[left + 1] = right + 1
            if campaign[left] != structure[right]:
                mismatches.append((left + 1, campaign[left], right + 1, structure[right]))
    gaps = tuple(position for position in range(1, len(campaign) + 1) if position not in mapping)
    matches = len(mapping) - len(mismatches)
    return SequenceAlignment(
        mapping=mapping,
        identity=matches / len(mapping) if mapping else 0.0,
        campaign_coverage=len(mapping) / len(campaign),
        mismatches=tuple(mismatches),
        campaign_gaps=gaps,
    )


def biological_assembly_polymer_chains(mmcif_path: str | Path) -> dict[str, tuple[str, ...]]:
    """Read polymer-chain membership for each biological assembly from mmCIF metadata."""
    values = MMCIF2Dict(str(mmcif_path))
    entity_ids = values["_entity_poly.entity_id"]
    polymer_entities = set(entity_ids if isinstance(entity_ids, list) else [entity_ids])
    asym_ids = values["_struct_asym.id"]
    asym_entities = values["_struct_asym.entity_id"]
    polymer_asym = {
        str(asym) for asym, entity in zip(asym_ids, asym_entities) if str(entity) in polymer_entities
    }
    assembly_ids = values["_pdbx_struct_assembly_gen.assembly_id"]
    asym_lists = values["_pdbx_struct_assembly_gen.asym_id_list"]
    output: dict[str, tuple[str, ...]] = {}
    for assembly, members in zip(assembly_ids, asym_lists):
        chains = tuple(part.strip() for part in str(members).split(",") if part.strip() in polymer_asym)
        output[str(assembly)] = chains
    return output


def parse_residue_label(label: str) -> tuple[str, int]:
    """Parse a residue label such as S94 into residue letter and one-based position."""
    value = str(label).strip().upper()
    if len(value) < 2 or not value[1:].isdigit():
        raise ValueError(f"invalid residue label: {label}")
    return value[0], int(value[1:])


def validate_residue_tracks(sequence: str, tracks: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Validate supplied residue labels against the campaign sequence."""
    rows = []
    for track, labels in tracks.items():
        for label in labels:
            expected, position = parse_residue_label(label)
            if position < 1 or position > len(sequence):
                raise ValueError(f"{track} residue {label} is outside the campaign sequence")
            observed = sequence[position - 1]
            if observed != expected:
                raise AssertionError(
                    f"{track} residue {label} disagrees with campaign sequence ({observed}{position})"
                )
            rows.append({
                "track": str(track), "residue_label": str(label),
                "campaign_position": position, "campaign_residue": observed,
            })
    return pd.DataFrame(rows)


def _structure(mmcif_path: str | Path, *, auth_residues: bool = False):
    return MMCIFParser(QUIET=True, auth_chains=True, auth_residues=auth_residues).get_structure(
        "structure", str(mmcif_path)
    )


def protein_residues(mmcif_path: str | Path, campaign_length: int) -> dict[tuple[str, int], object]:
    """Return coordinate-bearing polymer residues keyed by auth chain and label position."""
    model = next(_structure(mmcif_path).get_models())
    output = {}
    for chain in model:
        for residue in chain:
            if residue.id[0] == " " and 1 <= int(residue.id[1]) <= campaign_length:
                output[(str(chain.id), int(residue.id[1]))] = residue
    return output


def _heavy_atoms(residue: object) -> list[object]:
    return [atom for atom in residue.get_atoms() if str(atom.element).upper() != "H"]


def residue_distance(left: object, right: object) -> float:
    """Return the minimum heavy-atom distance between two residues."""
    left_atoms, right_atoms = _heavy_atoms(left), _heavy_atoms(right)
    if not left_atoms or not right_atoms:
        return float("inf")
    return min(float(atom_left - atom_right) for atom_left in left_atoms for atom_right in right_atoms)


def catalytic_distances(
    residues: Mapping[tuple[str, int], object],
    assembly_chains: Mapping[str, Sequence[str]],
    campaign_length: int,
    catalytic_positions: Iterable[int],
) -> pd.DataFrame:
    """Find each position's nearest catalytic copy within a biological assembly."""
    catalytic = tuple(sorted(set(map(int, catalytic_positions))))
    rows = []
    for position in range(1, campaign_length + 1):
        best = (float("inf"), "", "", -1, "")
        for assembly, chains in assembly_chains.items():
            for residue_chain in chains:
                residue = residues.get((str(residue_chain), position))
                if residue is None:
                    continue
                for catalytic_chain in chains:
                    for catalytic_position in catalytic:
                        target = residues.get((str(catalytic_chain), catalytic_position))
                        if target is None:
                            continue
                        distance = residue_distance(residue, target)
                        candidate = (
                            distance, str(residue_chain), str(catalytic_chain),
                            catalytic_position, str(assembly),
                        )
                        if candidate < best:
                            best = candidate
        rows.append({
            "campaign_position": position, "minimum_catalytic_distance_angstrom": best[0],
            "residue_chain": best[1], "catalytic_chain": best[2],
            "nearest_catalytic_position": best[3], "biological_assembly": best[4],
        })
    return pd.DataFrame(rows)


def assign_functional_shells(
    sequence: str,
    distances: pd.DataFrame,
    catalytic_positions: Iterable[int],
    functional_network_positions: Iterable[int],
    *,
    first_shell_cutoff: float,
    second_shell_cutoff: float,
) -> pd.DataFrame:
    """Assign mutually exclusive functional shells in the registered precedence order."""
    if first_shell_cutoff <= 0 or second_shell_cutoff <= first_shell_cutoff:
        raise ValueError("shell cutoffs must satisfy 0 < first < second")
    catalytic = set(map(int, catalytic_positions))
    functional = set(map(int, functional_network_positions)) - catalytic
    frame = distances.copy().sort_values("campaign_position").reset_index(drop=True)
    if frame["campaign_position"].tolist() != list(range(1, len(sequence) + 1)):
        raise ValueError("distance table must contain each campaign position exactly once")
    shells = []
    for row in frame.itertuples(index=False):
        position = int(row.campaign_position)
        distance = float(row.minimum_catalytic_distance_angstrom)
        if position in catalytic:
            shell = "catalytic"
        elif distance <= first_shell_cutoff:
            shell = "first_shell"
        elif distance <= second_shell_cutoff:
            shell = "second_shell"
        elif position in functional:
            shell = "distal_functional_network"
        else:
            shell = "generic_distal"
        shells.append(shell)
    frame.insert(1, "campaign_residue", list(sequence))
    frame["shell"] = shells
    return frame


def ligand_contacts(
    mmcif_path: str | Path,
    assembly_chains: Mapping[str, Sequence[str]],
    campaign_length: int,
    ligand_resname: str,
    *,
    cutoff: float,
) -> pd.DataFrame:
    """Find heavy-atom ligand contacts within each biological assembly copy."""
    # Non-polymers have no label_seq_id and are omitted by Biopython when
    # auth_residues=False. Use author residue ids here; 5OCM's polymer author
    # numbering is verified against its canonical sequence before this call.
    model = next(_structure(mmcif_path, auth_residues=True).get_models())
    residues = {
        (str(chain.id), int(residue.id[1])): residue
        for chain in model for residue in chain
        if residue.id[0] == " " and 1 <= int(residue.id[1]) <= campaign_length
    }
    chains = {str(chain.id): chain for chain in model}
    rows = []
    for assembly, members in assembly_chains.items():
        for ligand_chain in members:
            for ligand in chains[str(ligand_chain)]:
                if ligand.resname != ligand_resname:
                    continue
                for protein_chain in members:
                    for position in range(1, campaign_length + 1):
                        residue = residues.get((str(protein_chain), position))
                        if residue is None:
                            continue
                        distance = residue_distance(residue, ligand)
                        if distance <= cutoff:
                            rows.append({
                                "biological_assembly": str(assembly),
                                "protein_chain": str(protein_chain),
                                "campaign_position": position,
                                "ligand_chain": str(ligand_chain),
                                "ligand_resname": str(ligand.resname),
                                "ligand_auth_seq_id": int(ligand.id[1]),
                                "minimum_distance_angstrom": distance,
                            })
    columns = [
        "biological_assembly", "protein_chain", "campaign_position", "ligand_chain",
        "ligand_resname", "ligand_auth_seq_id", "minimum_distance_angstrom",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["biological_assembly", "ligand_chain", "protein_chain", "campaign_position"]
    ).reset_index(drop=True)


def mutation_responsive_codes(mutant: np.ndarray, wild_type: np.ndarray) -> np.ndarray:
    """Return residue-level mutant-minus-wild-type latent responses."""
    mutant_values = np.asarray(mutant)
    wild_type_values = np.asarray(wild_type)
    if mutant_values.ndim != 2 or mutant_values.shape != wild_type_values.shape:
        raise ValueError("mutant and wild-type codes must be aligned L x m matrices")
    return mutant_values - wild_type_values
