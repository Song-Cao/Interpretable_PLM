# v20 Step 8 local-input blocker

Date: 2026-11-19

The structural domain partition and per-run audit are complete. The remaining v20 analyses do
not require GPU arithmetic, but their registered inputs are not present on the local machine:
`Cache/` is empty.

The minimum files that must be copied from the persistent network volume are:

- `Cache/step8/IRED/delta_z/variant_rows.npy`
- `Cache/step8/IRED/delta_z/w1/delta_z.npz`
- `Cache/step8/IRED/delta_z/w1/wild_type_codes.npy`
- `Cache/step8/IRED/delta_z/v18_semantic_profiles/w1_seed20261311_residue_mean.npy`
- `Cache/step8/IRED/delta_z/v18_semantic_profiles/w1_seed20261312_residue_mean.npy`
- `Cache/step8/IRED/delta_z/v18_semantic_profiles/w1_seed20261313_residue_mean.npy`
- `Cache/step8/IRED/delta_z/v18_semantic_profiles/w1_seed20261314_residue_mean.npy`
- `Cache/step8/IRED/delta_z/v18_semantic_profiles/w1_seed20261315_residue_mean.npy`
- `Cache/step8/IRED/delta_z/manifest.json` and its reference blocks/checksums

These files support the domain enrichment, domain-fraction causal linkage, long-range response,
per-feature top positions, annotation profiles, seed replication, sub-typing, and causal-effect
comparisons. Copying them requires no recomputation and no GPU work, but accessing the network
volume requires a pod session; the instruction for this session explicitly prohibits starting
one.

ThermoMPNN and SoluProt are also not installed locally. ThermoMPNN supplies an official CPU
environment, so inference is not inherently GPU-bound, but the all-variant aggregation rule for
multi-mutants must be explicit because the registered ThermoMPNN model predicts point-mutation
ddG. SoluProt is an exploratory whole-sequence predictor and its own documentation warns that it
is not intended to predict point-mutation effects. Neither oracle was substituted or silently
approximated in this session.
