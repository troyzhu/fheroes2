---
title: Reference vault — index
type: moc
updated: 2026-07-30
related_concepts: ["[[../implementation/legal-actions-and-masking]]", "[[../implementation/observation-design]]"]
tags: [reference, index, moc, agent-env]
---

> **What this note is.** The scannable catalogue of every work behind the two verified literature runs. Read [[findings]] first for what the corpus establishes; come here to find a specific source, its quality, and where it is used. Provenance and vault mechanics are at the bottom.

For what each codebase actually contains and where to look inside it, read [[prior-art|the repository orientation]] first. Reading order for someone new to the corpus: [[works/vcmi-gym]] for the closest prior art, then [[works/gym-microrts]] for the masking evidence and single-machine feasibility, then [[works/sc2le]] for observation design.

## The corpus

| Source | Type | Year | Quality | What it establishes | Feeds |
|---|---|---|---|---|---|
| [[works/vcmi-gym]] | project | 2023–26 | primary | Heroes III battle RL shipped as an in-game AI; padded-entity plus per-hex encoding; flat masked action space; CleanRL-style stack | ADR 0002, ADR 0003 |
| [[works/gym-microrts]] | paper | 2021 | primary | Masking ablations (0% unmasked to 82–91% fully masked); factorized heads; state of the art in ~60 h on one 16 GB machine | ADR 0002, hardware plan |
| [[works/invalid-action-masking]] | paper | 2022 | primary | Masking is a valid policy gradient; the $-10^8$ implementation; penalties collapse as the illegal space grows | ADR 0002 |
| [[works/sc2le]] | paper | 2017 | primary | Feature layers as synthetic semantic rasterizations, never RGB; structured tensors alongside planes | ADR 0004 |
| [[works/alphastar]] | paper | 2019 | primary | Entity transformer, semantic minimap, scatter connections; supervised stage reaching 87% before any RL | ADR 0004, BC staging |
| [[works/griddly]] | project | 2020–23 | primary | Multiple observers over one state; semantic planes match pixel observers at roughly 14× the throughput | ADR 0004 |
| [[works/microrts-py]] | codebase | 2021–25 | primary | `CategoricalMasked` reference implementation; `partial_obs` flag; TrueSkill league evaluation | ADR 0001, evaluation |
| [[works/nle]] | project | 2020 | primary | Symbolic multi-component observations; CNN over embedded per-cell glyphs at small board scale | ADR 0004 |
| [[works/stratega]] | project | 2020 | primary | Forward-model-centric agent API for planning methods | planning option |
| [[works/arlinbfw]] | codebase | 2019 | primary | Headless C++ game engine driven out-of-process over a text channel | worker architecture |
| [[works/entity-based-rl]] | project | 2022–23 | mixed | Entity-list APIs and ragged-batch transformers | upgrade path |
| [[works/openai-five]] | paper | 2019 | primary | Structured arrays over pixels at scale | pixel-cost rationale |
| [[works/asymmetric-actor-critic]] | paper-group | 2017–22 | primary | Privileged-critic and recurrent POMDP baselines | ADR 0001 (option only) |
| [[works/misc-pipeline-sources]] | collection | mixed | mixed | Sample Factory, Lux AI winner, board-game scaling laws, NetHack follow-up, FootsiesGym | background |

Quality reads as the source class, not as our endorsement. Two entries carry caveats that matter at the point of use: [[works/entity-based-rl]] and [[works/openai-five]] contributed claims that never reached a verified claim set, so cite those sources directly rather than citing us.

## How the corpus maps to our decisions

| Decision | Anchored by |
|---|---|
| ADR 0001, observability profiles | [[works/microrts-py]], [[works/asymmetric-actor-critic]], plus the engine's own `WAR_INFO` behavior |
| ADR 0002, fixed action space with mask | [[works/invalid-action-masking]], [[works/gym-microrts]], [[works/vcmi-gym]], [[works/alphastar]] |
| ADR 0003, versioned configuration | [[works/vcmi-gym]] |
| ADR 0004, semantic planes and no pixels | [[works/sc2le]], [[works/griddly]], [[works/alphastar]], [[works/nle]] |

## Provenance

Two runs of the same pipeline produced this corpus: literature on environment and agent design (2026-07-27, 23 sources) and coarse-spatial observation design (2026-07-29, 20 sources). Counting the overlap, that is roughly 35 distinct works, 43 fetched source files, and 15 per-work notes.

Local copies live in `files/`, with `manifest.tsv` recording the URL, fetch status, byte size, and title of every file. `fetch_references.sh` re-fetches them reproducibly. One repository ships no README upstream, which the manifest records as a failed fetch rather than hiding.

Opening `agent_play/docs/` or the repository root as an Obsidian vault resolves every wikilink here.

## Related

- [[findings]], what the corpus establishes, with confidence markers.
- [[../archive/research-runs/2026-07-27-rl-approaches]] and [[../archive/research-runs/2026-07-29-spatial-observations]], the claim-by-claim reports with verification votes.
- [[prior-art]], an orientation to the open-source codebases behind it.
- [[../implementation/README|Concept primers]], the teaching layer these findings feed.
- [[../overview]], the system as it stands.
