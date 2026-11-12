# IRED representation-layer convention

Step 2's `esm_mean`, `esm_mutpool`, `esm_residue`, and `esm_residue_delta` representations were
all extracted from ESM-C's final output (`output.embeddings`). This is the conventional embedding
baseline and remains valid.

Analysis-layer objects use the frozen selected hidden-state depth, explicitly depth 26 for IRED.
The authoritative Step 4 crosscoder cache follows this convention. The withdrawn Step 5 dense
probes and Step 9 phi cache did not: their configuration said depth 26, but the generic Step 3
files they loaded were the superseded depth-29 extraction. Those benchmark numbers are not
comparable to Step 2 and remain withdrawn. The v18 recomputation reads only
`Cache/step5/IRED/analysis_layer_26/`.

This distinction is now a standing convention rather than an artifact of execution order:
embedding baselines use the final output, while dense probes, crosscoder inputs, sparse pooled
features, semantic tests, and patching use the selected analysis depth. Every result names its
depth.
