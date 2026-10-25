# Step 6 handoff — IRED self-pair null

## Registered architecture

The crosscoder has a shared encoder over concatenated activations, separate bias-free decoder
matrices, and BatchTopK reconstruction training with no L1 or auxiliary term. `k=32` was fixed
before results because it is the published BatchTopK operating point and is 3.33% of the
960-channel input dimension. It is held fixed across every width.

## Self-pair result

The cheapest registered null used `(h_evo, h_evo)` at `width_multiplier=1` over all 259,550
cached positions. Across all 960 features, Delta had minimum 0.5, maximum 0.5, and maximum
absolute deviation from 0.5 of **0.0**, passing the `1e-7` numerical tolerance. Realized L0 was
32.0 against nominal 32. The run consumed 0.03354 A100 wall-hours.

The pod-only checkpoint is
`Models/step6/IRED/self_pair_w1_seed20261301_nobias.pt`, SHA-256
`737f390da8a2e0ea421e4ed05d95c4322d80422c2cd010cb1c3d192c0d89720e`.
The committed result is `Reports/step6/self_pair_null.json`.

## Defect found and disposition

The first implementation used affine decoder layers, although v13 specifies the matrix maps
`D_evo z` and `D_fit z`. That self-pair also produced 0.5 because symmetric biases cannot be
detected by this null. Its report and the associated width curve are retained under
`Reports/archive/step6_decoder_bias_invalid/` and
`Reports/archive/step7_decoder_bias_invalid/`; they are invalid and must never be loaded.
Config and tests now assert `decoder_bias: false`, and the authoritative null restarted from
the beginning under a new checkpoint name.

## Incomplete or uncertain

Only the self-pair component was requested in this pass. The full Step 6 shuffled-label
crosscoder, drift-matching check, Beta fit, tau calibration, QQ plot, and null uncertainty have
not run. The self-pair result authorizes real-pair alignment work but does not calibrate a
taxonomy threshold.
