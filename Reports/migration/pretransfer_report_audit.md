# Pre-transfer report audit

Date: 2026-11-21

The six pod copies flagged during Phase 1 are not uncommitted reports. Each exactly matches a historical commit that is already an ancestor of the current local and remote `main`. The pod checkout is divergent and stale (`50eeb3e`, ahead 12 and behind 15 relative to its stale `origin/main`); replacing the current files with the pod copies would reverse later corrections.

| Path | Pod content commit | Status in current history | Disposition |
|---|---|---|---|
| `Reports/step2/step2_handoff.md` | `c4b408a` | ancestor of current `main` | retain current canonical version |
| `Reports/step7/step7_handoff.md` | `3301f77` | ancestor of current `main` | retain current canonical version |
| `Reports/step8/step8_preparation_handoff.md` | `f395e2d` | ancestor of current `main` | retain current canonical version |
| `Reports/step9/step9_handoff.md` | `5d14f11` | ancestor of current `main` | retain current canonical version |
| `Reports/step4/cache_manifest.json` | `e67472b` | ancestor of current `main` | retain current provenance contract |
| `Reports/step7/gate2_result.json` | `3301f77` | ancestor of current `main` | retain current corrected Gate 2 result |

The complete unified diff is in `pod_vs_current_reports.patch`. In particular, the pod copy would restore the withdrawn pre-v18 Gate 2 margin, remove the cache provenance contract, and replace the corrected depth-26 Step 9 result with its superseded predecessor. No canonical report was overwritten. The exact pod states remain recoverable from the listed commits and from the transferred active `.git` directory.

