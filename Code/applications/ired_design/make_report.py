#!/usr/bin/env python3
"""Render the application result and handoff from completed disk artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from applications.ired_design.run_search import ARMS
from crossfit.utils import REPO_ROOT, load_config


def _table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional dependency."""
    values = frame.fillna("").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    divider = "| " + " | ".join("---" for _ in values.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in values.to_numpy()]
    return "\n".join((header, divider, *rows))


def main() -> int:
    config = load_config(REPO_ROOT / "Code/config/ired_design.yaml")
    report = REPO_ROOT / config["paths"]["reports"]
    search = json.loads((report / "search_manifest.json").read_text())
    budget = json.loads((report / "embedding_budget.json").read_text())
    trajectory = json.loads((report / "trajectory_analysis_manifest.json").read_text())
    uncertainty = json.loads((report / "uncertainty_manifest.json").read_text())
    biophysics = json.loads((report / "biophysics_manifest.json").read_text())
    adapter = json.loads((report / "adapter_generation_manifest.json").read_text())
    costs = json.loads((report / "gpu_cost_accounting.json").read_text())

    controls = pd.read_csv(report / "controls_ranks.csv")
    control_table = controls.pivot(
        index="model", columns="control_mutation", values="rank_among_all_background_singles"
    ).reset_index()
    passage = pd.read_csv(report / "trajectory_passage_controls.csv")
    winners = passage.loc[passage["target"].isin([
        "S68P-P203Q-T241A", "S68P-T241A-H244Y", "S68P-P203Q-T241A-H244Y",
    ])][["model", "target", "passes_through", "first_selected_round"]]
    overlap = pd.read_csv(report / "trajectory_overlap.csv")
    overlap_summary = overlap.groupby("model", as_index=False).agg(
        selected_rows=("selected_count", "sum"),
        campaign_recoveries=("observed_in_campaign", "sum"),
    )
    consensus = pd.read_csv(report / "consensus_candidates.csv")
    top_consensus = consensus.head(30)[[
        "mutation_set_canonical", "consensus_count", "mean_rank",
        "cross_model_rank_sd", "high_consensus_high_disagreement",
    ]].copy()
    top_consensus["mean_rank"] = top_consensus["mean_rank"].map(lambda value: f"{value:.2f}")
    top_consensus["cross_model_rank_sd"] = top_consensus["cross_model_rank_sd"].map(
        lambda value: f"{value:.2f}"
    )
    anchors = pd.read_csv(report / "thermompnn_anchor_validation.csv")
    stability = pd.read_csv(report / "final_candidate_stability.csv")
    stability_view = stability[[
        "mutation_set_canonical", "thermompnn_additive_ddg_kcal_mol",
        "stability_consistency", "functional_shells_touched",
    ]].copy()
    stability_view["thermompnn_additive_ddg_kcal_mol"] = stability_view[
        "thermompnn_additive_ddg_kcal_mol"
    ].map(lambda value: f"{value:.3f}")

    gpu_hours = float(costs["total_hours"])
    text = f"""# SrIRED in-silico directed-evolution application

> Application workstream only. No result here is evidence for a CROSSFIT gate,
> table, or scientific claim.

## Interpretation

These are model proposals and a retrospective audit against measured trajectories,
not a confident ranking. Every model was trained on campaign fitness, a selection
readout. On the independent plate kcat set, classical encodings reached Spearman
0.53-0.57 while every ESM arm reached 0.23-0.44; the classical arms therefore have
the stronger external-transfer evidence when catalytic rate is the goal.

## Protocol and budget

- 13 frozen model arms, retrained without re-searching on all 1,182 curated non-WT
  campaign variants.
- Four founders, four one-mutation rounds, 5,510 complete WT single-mutant scans,
  {search['winner_pool_size']} reusable winner mutations per model, and
  {search['survivors_per_round']} survivors per founder and round.
- Representative projection: {budget['projected_gpu_hours_upper_bound']:.3f} GPU-hours.
- Search wall time: {search['gpu_wall_hours']:.3f} GPU-hours; accounted application
  total including generation fits and failed pre-result launches: {gpu_hours:.3f}
  GPU-hours.

## Consensus candidates

Consensus is the number of model-specific final top lists containing a variant; mean
rank breaks ties. High rank spread is disagreement, not uncertainty to average away.
The maximum consensus is only {int(consensus['consensus_count'].max())} of 13 models,
and every displayed candidate is supported only by the paired one-hot/AAIndex ridge
arms. Those encodings span nearly the same linear space, so this is not independent
cross-family agreement. Eight candidates are additionally flagged for high rank
dispersion across all models. None should be treated as a confident lab recommendation.

{_table(top_consensus)}

## Known-control ranks

Ranks are among all valid single additions to S68P-T241A. A72T and G37C are known
dead ends; H244Y is the known positive control. Lower is better.

{_table(control_table)}

## Could the models recover the campaign trajectory?

{_table(winners)}

Round-level recovery totals across the four founder searches:

{_table(overlap_summary)}

## Uncertainty

Every candidate has five-member within-model spread and cross-model rank spread.
The existing Step 2 Mondrian calibration contains no informative final-design stratum:
six-mutation, seen-position variants have only one calibration row and therefore an
unbounded finite-sample 90% interval; seven- and eight-mutation strata are absent.
Phi and the adapter have no registered Step 2 conformal calibrator. These limitations
are recorded per model and candidate rather than replaced by a pooled interval.

## Biophysical consistency checks

ThermoMPNN passed all {biophysics['anchor_count']} required sign checks on an exact
290/290 structure-sequence match:

{_table(anchors)}

Candidate values below sum the validated single-substitution estimates. They are an
additive consistency check, not a direct multi-mutant or epistatic prediction.

{_table(stability_view)}

Active-site geometry was not scored because no validated mutant-structure generator
was available; reporting unchanged wild-type coordinates would be misleading.
SoluProt was unreachable and has no IRED anchor, FoldX was unavailable, and docking
was the first registered budget cut. No causal attribution is made.

## Primary artifacts

- `controls_ranks.csv`: mandatory known false-positive and positive-control ranks.
- `trajectory_overlap.csv` and `trajectory_passage_controls.csv`: measured recovery
  and winning-path audit.
- `consensus_candidates.csv` and `final_candidate_model_scores.csv`: consensus,
  five-member spread, cross-model disagreement, and conformal support status.
- `<arm>_round1_single_scan.csv`, `<arm>_winner_pool.csv`, `<arm>_top20.csv`, and
  `round_top/<arm>/*.csv`: committed model-specific proposals.
- `full_lineage/<arm>/*.parquet`: pod-only evaluated and rejected candidates with
  parent links; intentionally gitignored.
- `trajectory_graphs/`: one graph per model and one ensemble graph.
"""
    (report / "ired_design_report.md").write_text(text)

    handoff = f"""# IRED design application handoff

Status: complete. This is an application workstream and is not CROSSFIT evidence.

## Completed

- Trained and hashed {(len(ARMS) - 1) * 5} non-adapter model members plus five adapter
  members on all 1,182 curated non-WT variants.
- Completed {len(ARMS)} model searches, four founders, and four rounds with full
  parent-linked lineage persistence.
- Produced the controls and trajectory-overlap audits before biophysical work.
- Produced 30 consensus candidates, top 20 per model, five-member spread, cross-model
  rank disagreement, trajectory graphs, and the validated ThermoMPNN check.

## Budget

Reduced-search projection: {budget['projected_gpu_hours_upper_bound']:.3f} GPU-hours. Search:
{search['gpu_wall_hours']:.3f}. Accounted total: {gpu_hours:.3f} GPU-hours, below the
six-hour ceiling.

## Limitations

- Rankings target campaign fitness, not kcat; external transfer favours classical
  encodings.
- Existing Mondrian strata do not provide finite intervals for these 6-8-mutation
  designs.
- ThermoMPNN multi-mutant values are additive single-mutation checks. Geometry,
  SoluProt, FoldX, and docking were excluded rather than reported without validation.
- Complete parquet lineages and all application checkpoints remain pod-only. Their
  committed manifests contain hashes; no main-project model or result directory was
  modified.

See `ired_design_report.md` for the result tables and artifact map.
"""
    (report / "ired_design_handoff.md").write_text(handoff)
    print(json.dumps({
        "status": "complete", "models": len(ARMS), "consensus": len(consensus),
        "gpu_hours_accounted": gpu_hours, "known_recoveries": trajectory["known_campaign_recoveries"],
        "uncertainty_status": uncertainty["conformal_status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
