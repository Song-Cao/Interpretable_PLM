# IRED emergent-gate resolution audit

## Result

The `w=1` emergent-candidate count remains zero after making the empirical null fine enough to
resolve the first Benjamini-Hochberg cutoff. This is now a supported negative result rather than
an artifact of the original three-null resolution.

The registered eligibility filter retained 19,609 feature observations across 34 independent
drift-matched null crosscoders. The smallest attainable empirical p-value is `5.0994e-5`, below
the first BH cutoff of `5.2083e-5` for 960 tests at `q=0.05`. Twenty null runs would not have
been adequate; even 32 realized only 18,458 eligible observations, below the exact requirement
of 19,199.

Across the five real dictionary seeds, the registered pooled-empirical analysis gives mean (SD)
class proportions: dead 32.94% (1.41%), emergent candidate 0%, evolution-only 0.06% (0.06%),
rotated 25.06% (1.63%), re-weighted 0.04% (0.06%), and conserved 41.90% (1.10%). The expansion
of the null changes the accepted rotated fraction only from 25.23% to 25.06%; rotation remains
the populated shifted class.

## Tail-fit check

Reconstruction ratios are not confined to `[0,1]`: the most extreme real values are negative,
and null values can exceed one. A Beta fit is therefore invalid for this statistic. The headline
test retains the registered pooled empirical p-values. As a diagnostic, a generalized Pareto was
fit only below the pooled null's fifth percentile (`0.51553`). Its shape is `0.73126` and scale is
`0.05464`; the lower-tail QQ plot is
`reconstruction_ratio_gpd_lower_tail_qq.png`.

For the primary real seed, the ten most extreme reconstruction ratios have empirical p-values
from `0.001785` to `0.004335`; the fitted-tail values range from `0.001132` to `0.003488`.
Across all five seeds, the largest empirical-versus-fitted discrepancy is 0.259 log10 units,
less than a factor of two. The fitted-tail sensitivity analysis also returns zero emergent
candidates in every seed. Thus neither empirical resolution nor an overly fat parametric tail is
the binding explanation for the zero.

The bounded Delta statistic retains its pooled-Beta treatment. Its lower-tail fit restricted to
the empirical lower 5% is shown in `delta_beta_lower_tail_qq.png`. Exact per-feature empirical
ranks and both p-values are in `most_extreme_reconstruction_ratios.csv`.

## Artifacts

- `class_proportions_by_seed.csv` and `class_proportions_summary.csv`: registered and fitted-tail
  sensitivity classifications.
- `taxonomy_resolution_w1_seed*.csv`: full five-seed taxonomies and gate flags.
- `most_extreme_reconstruction_ratios.csv`: ten most extreme eligible features per real seed.
- `thresholds.json`: pooled calibrations and leave-one-null-run-out uncertainty.
- `null_crosscoder_runs.csv` and `null_crosscoder_manifest.json`: all 34 null checkpoints and
  hashes.
- `resolution_manifest.json`: empirical resolution, fitted-tail parameters, hashes, and cost.

All crosscoder checkpoints and per-seed statistic caches are on the `/workspace` network volume.
The report artifacts above are committed. Causal patching was not started before the pod was
terminated.
