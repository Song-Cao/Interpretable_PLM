# IRED Step 4 realized caching budget

Status: budget approved for activation caching.

| training variants | residues per variant | positions | dimension | dtype | models | GB |
|---:|---:|---:|---:|---|---:|---:|
| 895 | 290 | 259550 | 960 | float16 | 2 | 0.996672 |

The budget uses only the pg_kfold fold-0 training pool. Test, reserve, round-held-out test, and plate variants are excluded. The arithmetic is positions x dimension x 2 bytes x 2 models. IRED is small enough to retain every residue position; this avoids introducing an unapproved window or distant-position sampling rule.

The registered dictionary width multipliers are 1, 2, 4, and 8. They provide 270.4 samples/feature at w=1, 135.2 samples/feature at w=2, 67.6 samples/feature at w=4, 33.8 samples/feature at w=8. Width is selected only after reporting dead-feature fraction and participation ratio.
