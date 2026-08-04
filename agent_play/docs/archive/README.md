---
title: Archive
type: moc
updated: 2026-07-30
tags: [archive, records, agent-env]
---

# Archive

Nothing in this directory is written to be read start to finish. It exists so that a number quoted in the documentation can be traced back to the run that produced it, and so that a decision can be traced back to the day it was taken.

Every file here is dated, machine-pinned, or commit-pinned, which means every file here goes stale by design. Quote from it only with its date attached, and never treat a measurement in this directory as current.

| Path | What it records | Staleness warning |
|---|---|---|
| `log.md` | The dated project history, oldest first | Narrative only; current state lives in [[../implementation/README|implementation]] |
| `benchmarks/2026-07-26-source-audit-apple-m3.md` | The Phase 0 engine audit and assumption table | Measured on an Apple M3, not the target machine; its own header says the numbers must be re-run before being quoted |
| `benchmarks/2026-07-27-apple-m2.md` | Target-hardware throughput, Mode A only | Pinned to commit `b16e6f698`, three milestones ago |
| `research-runs/2026-07-27-rl-approaches.md` | Claim-by-claim verification transcript with vote tallies | Superseded as reading by [[../research/findings|findings]]; kept for the per-claim votes and arXiv identifiers |
| `research-runs/2026-07-29-spatial-observations.md` | The same, for the spatial-observation question | Same |
| `experiments/2026-08-03-training-runs.md` | Every training and measurement run from the day the learning side was built, with configurations and numbers | Conclusions live in [[../rl/training-design|training-design]] and [[../rl/scenario-distribution|scenario-distribution]]; this is the raw record and includes runs later superseded |
| `sources/` | 43 fetched third-party PDFs, HTML snapshots, and vendored READMEs, plus `manifest.tsv` and `fetch_references.sh` | Third-party material, unmodified |

The reading path starts at [[../README|the tree README]].
