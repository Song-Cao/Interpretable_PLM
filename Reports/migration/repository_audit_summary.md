# RunPod repository-state audit

Date: 2026-11-21

This audit supersedes the earlier `origin/main..HEAD` check. The exact requested commands were run in all five volume checkouts: `git log --all --not --remotes --oneline`, `git stash list`, `git status --porcelain`, `git branch -vv`, and `git fsck --no-reflogs --lost-found`. Raw output is committed under `repository_audit/<repository>/`.

| Repository | Commits on local refs only | Stashes | Porcelain entries | Branch state | Dangling commits |
|---|---:|---:|---:|---|---:|
| `Interpretable_PLM` | 15 | 2 | 33 | `main` ahead 12, behind 15; local `bundle-results` | 1 |
| `Interpretable_PLM_v17` | 1 | 0 | 2 | `main` ahead 1 | 0 |
| `Interpretable_PLM_v17_patching` | 1 | 0 | 13 | `main` ahead 1 | 0 |
| `Interpretable_PLM_v17_resolution` | 0 | 0 | 3 | `main` matches its recorded upstream | 3 |
| `Interpretable_PLM_v17_semantics` | 0 | 0 | 12 | `main` behind 2 | 0 |

The active checkout's two stash commits are `9512cd4` and `b60af85`; the latter was also the active checkout's dangling commit. The resolution checkout exposed `9512cd4`, `b60af85`, and `50eeb3e` as dangling commits. The patching checkout contained two modified tracked files, two untracked code files, and seven untracked Step 8 result files. Resolution and semantics contained untracked result directories/files. `Cache` and `Models` in the four historical worktrees are symlinks to the active checkout and were captured as symlinks, not duplicated payloads.

## Preservation action

No checked-out branch, index, or working file was altered. For each checkout, an alternate Git index was populated from the exact working tree and committed to a new `migration/*-worktree-20261121` ref. Original HEADs, both stashes, and all dangling commits received separate migration refs. Sixteen unique, non-overwriting `migration/...` branches were pushed to GitHub; their hashes are in `repository_audit/pushed_remote_refs.txt`.

A complete `git bundle --all` was also streamed to the laptop for every repository and passed `git bundle verify`. Bundle paths, sizes, and SHA-256 values are in `repository_audit/bundle_verification.csv`. The bundles are stored under `Interpretable_PLM_RunPod_Phase2_2026-11-21/preserved_working_copies/` and are included in the local migration set. They provide a second recovery route independent of GitHub.

The requested `--lost-found` scan and creation of preservation refs added files under the five `.git` directories after Phase 1. Those directory entries were re-inventoried and re-hashed before transfer; Part 2 verification therefore uses the refreshed hashes.

