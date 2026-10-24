# IRED conclusion

IRED produced a complete Step 2 comparison of 11 predictor arms under three reporting
conditions: the pg_kfold test split, the round-5 held-out split, and the external plate set.
Selection used only five-fold validation within the pg_kfold training pool. The plate set was
never used for fitting, tuning, calibration, or model selection.

The plate set exposes a representation-transfer split. The one-hot and AAIndex arms have
Spearman 0.5300-0.5742, while every ESM arm has 0.2317-0.4443. The ranges do not overlap.
This holds across the ESM linear/ridge, MLP, random-forest, and residual-convolution
architecture families, so it is not attributable to a single ESM readout architecture.

Inner-validation performance does not predict plate transfer. ESM-residue convolution has
the highest validation Spearman of all arms, 0.6327, but plate Spearman is 0.3336, nearly the
lowest result. ESM-residue-delta convolution improves over absolute ESM-residue convolution
on both plate (0.3900 versus 0.3336) and round-held-out data (0.5408 versus 0.4985).

Gate 1 failed on IRED because task gain was not established. The registered adapter's mean
five-fold gain is 0.01906 with per-fold SD 0.04371, `t(4)=0.975`, `p=0.385`, and Cohen's
d = 0.436. Mean gain is positive in all four tested configurations and increases monotonically
across the tested capacity settings, but none is significant. At d = 0.44, approximately
42.5 folds are required for 80% power; a hypothetical ten-fold comparison gives only
`t(9)=1.379`, `p=0.201`. The task-gain criterion is therefore underpowered at 895 training
variants, not established as absent.

The independent representational criteria were met cleanly at depth 26. Real relative drift
is 0.6338 against shuffled-label drift 0.3417, with gap interval [0.2832, 0.3012], and real
CKA is 0.9189. Thus label-driven representational change is measurable, but the complete
Gate 1 verdict remains failed because its task-gain criterion did not pass.
