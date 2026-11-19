# IRED structural-domain partition

Date: 2026-11-19
Status: complete from local CPU inputs

## Boundary evidence

The partition uses the deposited 5OCM structure and its author/PISA biological dimers A/F,
B/C, and D/E. CATH v4.3.0 assigns residues 2-160 in every chain to the NAD(P)-binding
Rossmann-like domain (3.40.50.720) and residues 161-291 to the mainly alpha-helical second
domain (1.10.1040.10). The campaign sequence covers residues 1-290, so residue 1 is retained as
structurally unassigned rather than forced into a domain.

The deposited secondary-structure topology supports the split: the cofactor-binding domain
contains the eight-strand sheet (37-43 sheet residues per crystal copy, with small copy-specific
boundary differences), whereas the substrate/catalytic region contains 107 helical and 23 coil
positions and no sheet position. Ligand contacts independently support the labels: 25 of 29
NADP-contact positions fall in the cofactor-binding domain, while 10 of 14 ketone-analogue
contact positions fall in the substrate/catalytic region. The active site therefore spans both
domains, as expected for the dimeric assembly; these labels describe structural organization,
not mutually isolated chemistry.

## Residue counts

| axis | category | residues |
|---|---|---:|
| structural domain | cofactor-binding domain | 159 |
| structural domain | substrate/catalytic region | 130 |
| structural domain | unassigned | 1 |
| dimer interface | interface | 105 |
| dimer interface | non-interface | 185 |
| relative solvent accessibility | core | 184 |
| relative solvent accessibility | surface | 106 |

Dimer-interface status is based on a 5.0-A heavy-atom contact to the partner chain and must be
present in at least half of the six chain copies. Ninety-two positions are interface contacts
in all six copies; the remaining 13 consensus-interface positions capture crystallographic
copy variation. Core is mean assembly-level relative solvent accessibility below 0.25 on the
Wilke maximum-ASA scale. Every chain-level RSA and nearest partner contact is retained in
`ired_domain_partition_chain_provenance.csv`.

## Structural domains by distance shell

| structural domain | catalytic | first shell | second shell | distal functional | generic distal |
|---|---:|---:|---:|---:|---:|
| cofactor-binding | 1 | 13 | 11 | 3 | 131 |
| substrate/catalytic | 2 | 24 | 25 | 0 | 79 |
| unassigned | 0 | 0 | 0 | 0 | 1 |

## Orthogonal structural axes by distance shell

| dimer-interface status | catalytic | first shell | second shell | distal functional | generic distal |
|---|---:|---:|---:|---:|---:|
| non-interface | 1 | 8 | 18 | 3 | 155 |
| interface | 2 | 29 | 18 | 0 | 56 |

| RSA class | catalytic | first shell | second shell | distal functional | generic distal |
|---|---:|---:|---:|---:|---:|
| core | 2 | 35 | 33 | 1 | 113 |
| surface | 1 | 2 | 3 | 2 | 98 |

These matrices show why shells and domains are not substitutes. The substrate/catalytic region
contains residues from every distance shell, while the first shell is predominantly both core
and dimer-interface. The domain partition is the primary spatial axis; shells remain a
secondary distance axis.

## Files

- `ired_domain_partition.csv`: one row per campaign position with all structural axes.
- `ired_domain_partition_chain_provenance.csv`: one row per chain copy and position, including
  biological assembly, chain of origin, partner chain, RSA, nearest cross-chain contact, and
  deposited secondary structure.
- `ired_domain_partition_counts.csv`: category sizes.
- `ired_*_by_shell_overlap.csv`: overlap matrices reproduced above.
- `ired_structural_domain_by_secondary_structure.csv`: deposited topology by domain.
- `ired_domain_contact_summary.csv`: crystallographic ligand contacts by domain.
- `ired_domain_partition_manifest.json`: registered thresholds and provenance.
