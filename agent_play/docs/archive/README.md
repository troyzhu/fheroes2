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
| `experiments/2026-08-04-flip-and-collapse.md` | The resolution of the trust-region flip and the collapse-condition hunt, with registered predictions and the runs that decided them | Same; conclusions live in [[../rl/training-design|training-design]] |
| `experiments/2026-08-05-diversity-and-encoding.md` | The diversity push: diverse demonstrations, the clone cross-evaluation, the leaky split, and the count-extrapolation evidence behind ADR 0006 | Conclusions live in [[../decisions/0006-encoding-count-scaling|ADR 0006]] and [[../implementation/observation-design|observation-design]] |
| `experiments/2026-08-05-real-engine-replay.md` | Rendering recorded episodes through the game's own battle interface, with the digest verification and the two defects it surfaced | Mechanism lives in [[../implementation/replay-rendering|replay-rendering]] |
| `experiments/files/` | Raw run artifacts a log depends on: replay recordings, dated run-report and recording-manifest snapshots, and the anchor checkpoints pools and recordings are calibrated against | Meaningful only through the logs that cite them; datasets re-record from their manifests |
| `sources/` | Fetched third-party PDFs, HTML snapshots, and vendored READMEs; `manifest.tsv` is the authoritative list and `fetch_references.sh` reproduces it | Third-party material, unmodified |

The reading path starts at [[../README|the tree README]].
