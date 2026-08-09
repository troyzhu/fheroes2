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
| `log.md` | The dated project history through Phase 0 and the milestones, oldest first; the training era continues in the per-day experiment logs below | Narrative only; current state lives in [[../implementation/README|implementation]] |
| `benchmarks/2026-07-26-source-audit-apple-m3.md` | The Phase 0 engine audit and assumption table | Measured on an Apple M3, not the target machine; its own header says the numbers must be re-run before being quoted |
| `benchmarks/2026-07-27-apple-m2.md` | Target-hardware throughput, Mode A only | Pinned to commit `b16e6f698`, three milestones ago |
| `research-runs/2026-07-27-rl-approaches.md` | Claim-by-claim verification transcript with vote tallies | Superseded as reading by [[../research/findings|findings]]; kept for the per-claim votes and arXiv identifiers |
| `research-runs/2026-07-29-spatial-observations.md` | The same, for the spatial-observation question | Same |
| `experiments/2026-08-03-training-runs.md` | Every training and measurement run from the day the learning side was built, with configurations and numbers | Conclusions live in [[../rl/training-design|training-design]] and [[../rl/scenario-distribution|scenario-distribution]]; this is the raw record and includes runs later superseded |
| `experiments/2026-08-04-flip-and-collapse.md` | The resolution of the trust-region flip and the collapse-condition hunt, with registered predictions and the runs that decided them | Same; conclusions live in [[../rl/training-design|training-design]] |
| `experiments/2026-08-05-diversity-and-encoding.md` | The diversity push: diverse demonstrations, the clone cross-evaluation, the leaky split, and the count-extrapolation evidence behind ADR 0006 | Conclusions live in [[../decisions/0006-encoding-count-scaling|ADR 0006]] and [[../implementation/observation-design|observation-design]] |
| `experiments/2026-08-05-real-engine-replay.md` | Rendering recorded episodes through the game's own battle interface, with the digest verification and the two defects it surfaced | Mechanism lives in [[../implementation/replay-rendering|replay-rendering]] |
| `experiments/2026-08-05-planner-reward-battlefields.md` | The planner-query resolution that unblocks DAgger, the difficulty-weighted reward measurement, and the battlefield-spread finding with the dead-seeds discovery | Conclusions live in [[../rl/training-design|training-design]], [[../rl/reward-design|reward-design]], and [[../rl/scenario-distribution|scenario-distribution]] |
| `experiments/2026-08-06-night-block-search-generations.md` | The overnight search-teaching generations over the fresh distribution, battery-gated per generation | Conclusions promote to [[../rl/training-design|training-design]] once the block closes |
| `experiments/2026-08-05-dagger-and-battlefield-transfer.md` | DAgger's first round with its matched teacher-data control, and pool transfer re-measured over battlefields | Conclusions live in [[../rl/training-design|training-design]] |
| `experiments/2026-08-06-offline-arms-and-planes.md` | The offline-improvement arms and the planes ablation, capacity-controlled | Conclusions live in [[../rl/off-support-and-offline-improvement|off-support-and-offline-improvement]] and [[../rl/the-policy-network|the-policy-network]] |
| `experiments/2026-08-07-overnight-champion-mixture.md` | The champion-mixture day: owner-objective labels, sharpness arms, self-play opening, the full-apparatus correction, convergence | Conclusions promote to [[../rl/training-design|training-design]] and the conventions of `../../experiments/README.md` |
| `experiments/2026-08-08-selfplay-round2-and-trust-region.md` | Self-play rounds two to four judged by the full apparatus: the trust-region rematch, the wide round, the KL leash, the coverage-forced corpus and its scale test, the rounds double-check, and the corrected reward column | Conclusions live in [[../rl/reward-design|reward-design]], [[../decisions/0007-anchored-ppo|ADR 0007]] and [[../rl/program-review|program-review]] |
| `experiments/2026-08-08-audit-and-the-deviation-finding.md` | The fifteen-agent audit, the three measurement defects it exposed, and the deviation probe showing the action-level signal sits in the matchups the policy loses | Conclusions live in [[../rl/program-review|program-review]] |
| `experiments/files/` | Raw run artifacts a log depends on: replay recordings, dated run-report and recording-manifest snapshots, and the anchor checkpoints pools and recordings are calibrated against | Meaningful only through the logs that cite them; datasets re-record from their manifests |
| `sources/` | Fetched third-party PDFs, HTML snapshots, and vendored READMEs; `manifest.tsv` is the authoritative list and `fetch_references.sh` reproduces it | Third-party material, unmodified |

The reading path starts at [[../README|the tree README]].
