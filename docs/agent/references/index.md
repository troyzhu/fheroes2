---
title: "Reference vault — fheroes2 agent environment"
tags: [reference, index, moc]
updated: 2026-07-29
---

# Reference vault — index

Consolidated references behind the two adversarially verified research runs
([[../research_rl_approaches|RL approaches]], 2026-07-27, 23 sources · 
[[../research_minimap_observations|minimap/hybrid observations]], 2026-07-29, 20 sources) and
the four ADRs in `../decisions/`. One note per work in `notes/`; local copies in `files/`
(43 files, ~59 MB — see `manifest.tsv` for URL, status, size, and scraped title of every file;
re-fetch with `./fetch_references.sh`).

> Obsidian: open `docs/agent/` (or the repo root) as a vault; wikilinks below resolve to
> `notes/`. Each note's frontmatter carries `urls`, `local` file paths, `tags`, and which
> research run(s) cited it.

## Direct prior art (study first)

- [[ref-vcmi-gym]] — HoMM3 battle RL → shipped MMAI combat AI in VCMI 1.7.0. *The* template.
- [[ref-arlinbfw]] — Battle for Wesnoth: headless C++ engine + text protocol, existence proof.
- [[ref-stratega]] — forward-model-centric strategy-games framework (planning door).

## Observation design

- [[ref-sc2le]] — feature layers, the canonical semantic minimap; why not RGB.
- [[ref-pysc2]] — one game state, multiple toggleable observers (feature/RGB/raw).
- [[ref-griddly]] — VECTOR vs pixel observers: performance parity at ~14× throughput.
- [[ref-nle]] — symbolic multi-component observations; 9×9-crop CNN practice at small scale.
- [[ref-entity-based-rl]] — entity-list APIs and transformers (upgrade path; partly unverified).
- [[ref-openai-five]] — structured-arrays-over-pixels at scale (unverified rationale).

## Action space and training

- [[ref-invalid-action-masking]] — masking is a valid policy gradient; implementation canon.
- [[ref-gym-microrts]] — masking ablations (0.0 → 0.91), factorized heads, 60 h/16 GB SOTA.
- [[ref-microrts-py]] — CategoricalMasked code, partial_obs flag, TrueSkill league.
- [[ref-alphastar]] — entity transformer + semantic minimap + scatter connections; BC → 87 %.

## Partial observability

- [[ref-asymmetric-actor-critic]] — privileged-critic and POMDP baselines (verified gap: kept
  as option, no surviving performance claims).

## Peripheral / fetched-only

- [[ref-misc-pipeline-sources]] — Sample Factory, Lux AI winner, board-game scaling laws,
  NetHack-is-hard-to-hack, FootsiesGym, cyber-defence entity RL.

## How these map to our decisions

| Decision | Anchored by |
|---|---|
| [[../decisions/0001-observation-profiles\|ADR 0001]] full/observable profiles | [[ref-microrts-py]], [[ref-asymmetric-actor-critic]], engine `WAR_INFO` fact |
| [[../decisions/0002-action-space\|ADR 0002]] fixed space + mask | [[ref-invalid-action-masking]], [[ref-gym-microrts]], [[ref-vcmi-gym]], [[ref-alphastar]] |
| [[../decisions/0003-config-management\|ADR 0003]] YAML configs | [[ref-vcmi-gym]] (PBT/W&B practice) |
| [[../decisions/0004-spatial-observation-modality\|ADR 0004]] planes, no pixels | [[ref-sc2le]], [[ref-pysc2]], [[ref-griddly]], [[ref-alphastar]], [[ref-nle]] |

See [[summary]] for the consolidated analysis across the whole corpus.
