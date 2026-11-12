# IRED Step 5 cache-depth notice

The committed dense-probe results are superseded pending recomputation. Their campaign inputs
came from `Cache/step3/IRED/full_selected_layer_{base,adapted}.npy`, which stores the original
depth-29 extraction rather than accepted depth 26. Consequently these files must not be used as
the Gate 2 dense reference. They are retained for audit only.

The diagnosis and numerical evidence are in
`Reports/step8/v17_activation_cache_diagnosis.{md,json}`.
