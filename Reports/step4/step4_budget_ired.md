# IRED Step 4 realized caching budget

Status: budget computed; activation caching has not started.

| training variants | residues per variant | positions | dimension | dtype | models | GB |
|---:|---:|---:|---:|---|---:|---:|
| 895 | 290 | 259550 | 960 | float16 | 2 | 0.996672 |

The budget uses only the pg_kfold fold-0 training pool. Test, reserve, round-held-out test, and plate variants are excluded. The arithmetic is positions x dimension x 2 bytes x 2 models. IRED is small enough to retain every residue position; this avoids introducing an unapproved window or distant-position sampling rule.

At the reference alpha=8, the dictionary would contain 7,680 features and only 33.8 position-samples per feature. Alpha is not selected at the budget stage; the exploratory shakedown must choose a substantially smaller width using participation ratio and dead-feature fraction as evidence.
