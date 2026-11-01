# Step 8 preparation handoff — IRED annotation tracks

## Status and boundary

Annotation inputs, sequence numbering, biological-assembly contacts, functional shells, and
mutation-responsive latent differences are assembled. No enrichment test, feature annotation
test, patch, oracle fit, or oracle validation was run. This is the requested stopping boundary.

IRED remains exploratory because Gate 1 failed. Nothing here revises that verdict or enters a
confirmatory central-claim table.

Under v16, recalibration of every taxonomy threshold against the unconditional drift-matched
null leaves zero emergent candidates at `w=1`. The user-specified condition for beginning Part C
is therefore not met. The prepared tracks and Delta-z cache remain intact, but no enrichment,
patching-budget calculation, ablation, or oracle validation has been run.

## Numbering verification

The 290-residue campaign wild type globally aligns to canonical 5OCM residues 1-290 with 100%
identity and 100% campaign coverage. There are no mismatches or campaign gaps. The 5OCM entity
has one additional terminal Ala at position 291.

All 13 supplied residue annotations map exactly to the campaign and structure sequences. In
particular, campaign W178 and H244 are 5OCM W178 and H244. They are distinct positions from the
entry's contextual W208 and H238; none was substituted or renumbered. This mapping is asserted
by `Code/tests/test_structural_annotations.py` against the frozen mmCIF.

## Assembly and contact provenance

The deposited 5OCM coordinate entry contains six polymer chains, but official mmCIF metadata
defines three biological dimers: A/F, B/C, and D/E. Calculations use all three biological copies
and allow contacts across the two subunits of each dimer; contacts between different dimers are
excluded as crystal-neighbour contacts. Every crystallographic contact records the biological
assembly, protein chain, ligand chain, position, and minimum heavy-atom distance.

At the registered 5 A cutoff, 5OCM yields:

| crystallographic track | contact rows | unique sequence positions | cross-subunit rows |
|---|---:|---:|---:|
| NAP cofactor contact | 174 | 29 | 24 |
| 9RH ketone-analogue contact | 82 | 14 | 42 |

The 9RH ligand is explicitly labelled a ketone-analogue proxy for the campaign's imine
substrate. Supplied experimental contacts, these crystallographic contacts, and docking-derived
contacts remain separate. No docking input was supplied, so the docking track is present only as
`not_provided_not_inferred` with zero residues.

The supplied tracks are also separate: `catalytic_proposed`, `substrate_direct_contact`,
`NADPH_direct_contact`, `NADPH_proximal_or_functionally_coupled`, and
`evolutionary_peak_residue`. The last is marked `positive_control_only` and is ineligible for a
mechanistic claim.

## Functional shells

Shells are mutually exclusive and assigned in this order: catalytic; first shell at <=5 A;
second shell at >5 to <=8 A; supplied distal functional network; generic distal. Distances are
minimum heavy-atom distances to any proposed catalytic residue copy within the same biological
dimer. The evolutionary-peak positive control is excluded from the functional network.

| shell | residues |
|---|---:|
| catalytic | 3 |
| first shell | 37 |
| second shell | 36 |
| distal functional network | 3 |
| generic distal | 211 |

The exact position, nearest catalytic position, source and target chains, assembly, and distance
are in `ired_functional_shells.csv`.

## Mutation-responsive codes

For both working dictionaries, the pod cache stores the exact residue-level arithmetic
`Delta z_j(x) = z_j(x_mutant) - z_j(x_WT)` for all 1,182 curated campaign variants and 290
positions. Each sequence is encoded independently, so BatchTopK competition is fixed within one
290-position sequence rather than depending on an arbitrary multi-sequence inference batch.

| w | sparse shape | nonzeros | density | compressed cache |
|---:|---:|---:|---:|---:|
| 1 | 342,780 x 960 | 17,961,806 | 5.46% | 80 MB |
| 2 | 342,780 x 1,920 | 18,518,511 | 2.81% | 85 MB |

The full pod-only cache is 177 MB including wild-type codes, per-position mean codes, row index,
and manifest. It is under `Cache/step8/IRED/delta_z/` and is excluded from git. The committed
manifest records checkpoint hashes and the 0.04321 GPU-hour cost.

## Artifacts and schemas

- `ired_annotation_tracks.csv`: one row per supplied annotation with campaign/PDB position,
  residue, provenance, role, and claim eligibility.
- `ired_crystallographic_contacts.csv`: one row per chain-specific <=5 A ligand contact with
  assembly and chain provenance.
- `ired_functional_shells.csv`: one row per campaign position with shell and nearest-catalytic
  geometry.
- `ired_functional_shell_counts.csv`: realized shell counts.
- `ired_contact_provenance_status.csv`: unavailable docking track and positive-control status.
- `ired_annotation_manifest.json`: sequence/assembly validation and safety-stop flags.
- `ired_delta_z_manifest.json`: sparse Delta-z shapes, density, hashes, and cost.

Frozen inputs are `Data/raw/structures/5ocm.cif` and `5ocm-assembly1.cif`; their SHA-256 hashes
are registered in `Code/config/step8.yaml`.

## Incomplete or uncertain

The proposed catalytic and experimental contact assignments are accepted as supplied Hollfelder
preprint annotations; this preparation does not independently adjudicate their biological role.
The full deposited entry is not a biological hexamer, so the implementation follows the official
three-dimer assembly metadata. 9RH contacts are proxy geometry, not observed imine-substrate
contacts. No access-tunnel, conservation, DCA, energetic, docking, or oracle track was supplied
or inferred. Final `emergent` labels remain unavailable until causal validation.
