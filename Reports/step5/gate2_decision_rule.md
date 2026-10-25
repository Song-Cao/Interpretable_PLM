# Gate 2 decision rule (pre-committed)

This rule was recorded before any Step 7 sparse-feature result was produced.

The decision metric is test Spearman on the dataset's pg_kfold split. Dense probes are fitted in
Step 5, but Gate 2 is evaluated only in Step 7 after pooled crosscoder features exist. Dense
**dominates** only when it beats the best sparse-feature arm and a paired-bootstrap confidence
interval for that per-variant performance difference excludes zero in the dense-favouring
direction. Sparse is **competitive** when the interval contains zero or favours sparse.

Sample-efficiency curves are reported alongside the comparison but never determine Gate 2. No
model selection, calibration, or threshold fitting uses pg_kfold test predictions.
