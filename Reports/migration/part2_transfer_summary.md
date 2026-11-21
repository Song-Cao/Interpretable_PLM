# RunPod to Modal migration: Part 2 transfer summary

Date: 2026-11-21

## Result

The approved CROSSFIT set was copied from the RunPod network volume to:

`/Users/songcao/workdir/Interpretable_PLM_RunPod_Phase2_2026-11-21`

The transfer contains 7,621 payload files and 9,221,227,791 bytes. Every local file was re-hashed with SHA-256 and compared with the refreshed pod manifest: **7,621 matches, 0 mismatches, 0 missing**. No retry was required. The per-file evidence is `transfer_verification.csv`; the selected source set is `approved_transfer_manifest.csv`.

| Transfer component | Files | Bytes |
|---|---:|---:|
| all CROSSFIT `Models/` | 3,797 | 8,813,137,553 |
| active repository `.git/` | 1,802 | 80,237,189 |
| irreplaceable active `Reports/` files | 314 | 52,775,287 |
| active `Cache/step8/` | 15 | 125,012,286 |
| `Interpretable_PLM_v17/.git/` | 67 | 33,547,930 |
| `Interpretable_PLM_v17_patching/.git/` | 367 | 40,522,460 |
| `Interpretable_PLM_v17_resolution/.git/` | 1,181 | 42,744,733 |
| `Interpretable_PLM_v17_semantics/.git/` | 78 | 33,250,353 |

Free space immediately before transfer was 45,220,708,352 bytes. The approved set required 11,065,473,349 bytes with 20% headroom, so the check passed. Free space after transfer and the auxiliary verified bundles was 32,490,938,368 bytes.

## Six report copies

The six pod report copies initially labelled irreplaceable were not uncommitted work. Each exactly matched a historical commit already contained in current remote `main`; the pod checkout was stale and divergent. Overwriting the canonical copies would have restored withdrawn numbers and removed the cache provenance contract. The current canonical files were therefore retained. The complete unified diff and commit provenance are in `pod_vs_current_reports.patch` and `pretransfer_report_audit.md`.

## Repository state

All five working copies were audited with all refs, stashes, porcelain state, branch tracking, and `fsck --no-reflogs --lost-found`. Exact working-tree snapshot commits were created with alternate indexes, leaving checked-out branches and files unchanged. Original heads, two stashes, and all dangling commits were preserved in 16 non-overwriting `migration/...` branches on GitHub. Five complete local bundles also pass `git bundle verify`. See `repository_audit_summary.md` and `repository_audit/`.

## Quarantine and exclusions

`Models/archive/` contains 1,830 files / 3,292,588,495 bytes of quarantined depth-29 material. It was transferred for the audit trail and every row is marked `modal_eligible=false`; it must not be pushed to Modal or resolved by active loaders.

The following unrelated material was not transferred:

| Excluded material | Bytes |
|---|---:|
| `/workspace/FM_project/` | 10,168,093,732 |
| FMTS archives and bundle | 8,301,407,357 |
| `/workspace/meta-flow-matching/` | 52,227,916 |
| `/workspace/fmts-workshop-venv/` | 5,447,461,249 |

The FMTS environment is classified regenerable at zero GPU-hours. Its pinned package list is `fmts_workshop_requirements.txt`. The other excluded materials remain a separate user decision.

## Source safety and cost

No source payload on the pod or network volume was deleted, moved, renamed, or overwritten. The requested lost-found audit and preservation operation only added Git lost-found records, objects, and migration refs; those additions were re-inventoried before transfer. The pod remains reachable and running.

This session used approximately 0.52 hours of pod wall time and 0 GPU-hours.

