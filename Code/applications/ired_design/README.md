# IRED design application

This directory contains an application workstream that proposes and audits SrIRED
directed-evolution trajectories using model families previously selected in CROSSFIT.
It is operationally and inferentially separate from CROSSFIT: no result produced here
is evidence for a CROSSFIT gate, result table, or scientific claim.

Reusable application logic lives in this package. `run.py` is the resumable entry
point; detailed artifacts go only to `Reports/applications/ired_design/`, and fitted
generation checkpoints go only to `Models/applications/ired_design/`. Hyperparameters,
seeds, paths, budgets, founders, and controls are fixed in
`Code/config/ired_design.yaml`.

Run from the repository root. Every stage writes a completion manifest only after all
expected rows and files validate; interrupted stages resume from complete shards and
otherwise restart the incomplete shard.

