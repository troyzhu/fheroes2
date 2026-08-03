---
title: "ADR 0004, semantic spatial planes, pixels rejected"
type: adr
status: accepted
updated: 2026-08-03
related_concepts: ["[[../implementation/observation-design]]", "[[0001-observation-profiles]]", "[[../rl/training-design]]"]
tags: [adr, observation, modality, agent-env]
---

# ADR 0004 — Semantic spatial-plane observation modality; true pixel rendering rejected

- Status: accepted 2026-07-29, plane emitter lands with Milestone 4
- Implementation: not built. No occurrence of `planes_v1` or `observation_modalities` exists in `src/fheroes2/agent/` or `python/` as of 2026-07-31.
- Evidence: user proposal 2026-07-29 (a coarse minimap view for the agent); [[../archive/research-runs/2026-07-29-spatial-observations]], 24 of 25 claims confirmed; [[../research/works/pysc2]], [[../research/works/griddly]], [[../research/works/alphastar]]
- Extends: [[0001-observation-profiles]], orthogonally. A profile says what may be seen, a modality says how it is shaped.
- Mechanism detail: [[../implementation/observation-design]]; encoder consequences in [[../rl/training-design]]

## Table of contents
- [[#Context]]
- [[#The sub-problem]]
- [[#Options considered]]
- [[#Why this one, and what it cost]]
- [[#Decision]]
- [[#Consequences]]

## Context

The owner proposed giving the agent a small, coarse, pixel-style minimap plus auxiliary structured information, as a complementary observation mode. The verified literature answer (24/25 claims confirmed): the instinct is right, the pixels are not —

- SC2LE/PySC2's "minimap" is a synthetic semantic rasterization of game state (typed categorical/scalar planes at configurable resolution), never captured render output; DeepMind's stated rationale is that agents shouldn't burn capacity reading numbers out of pixels.
- Griddly ships a non-pixel VECTOR observer next to three pixel renderers over one game state: final RL performance is consistent across representations in its 150-experiment baseline, and the vector observer is ~14× faster than rendering.
- AlphaStar's spatial encoder consumes semantic planes, and its scatter connections write each unit's learned embedding into the grid cell that unit occupies, so the two modalities coexist and fuse rather than competing. See [[../research/works/alphastar]].
- Our headless core loads zero game assets (Phase 0's headline result). A true rendered minimap would reintroduce the display/AGG dependencies the whole architecture exists to avoid.

## The sub-problem

Should the agent receive a spatial, grid-shaped view of the board in addition to per-unit records, and if so, made of what?

The owner's proposal was a coarse minimap. Two separable questions hide inside it. Whether spatial structure should be exposed at all, since a list of ten unit records does encode position but does not make adjacency, blocking, or threat geometrically apparent. And whether such a view should be rendered pixels or synthesized semantic channels.

## Options considered

| Option | What it is | For | Against |
|---|---|---|---|
| Entity records only | Ten padded per-unit slots, no grid | Already specified, small, fast | Spatial relations must be learned from coordinates rather than read off structure |
| Rendered pixels | Actually draw the board and downscale | Matches what a human sees | Reintroduces the display and asset dependencies the headless core exists to avoid, and Griddly measures a vector observer at roughly 14 times the speed of rendering with consistent task performance |
| Semantic planes (chosen) | An 11 by 9 grid of typed channels synthesized from engine state | Spatial structure explicit, no assets, one source of truth with the entity list | Another representation to specify and maintain, and its value at this board size is unmeasured |
| Planes instead of entities | Replace the entity list | One representation | Discards exact per-unit values that a grid quantizes, and forecloses the hybrid fusion the evidence supports |

## Why this one, and what it cost

Pixels are rejected permanently rather than deferred, and the evidence is unusually clean. What SC2LE calls a minimap was never captured render output but a synthetic semantic rasterization, on DeepMind's stated rationale that agents should not spend capacity reading numbers off a screen. Griddly ships a non-pixel vector observer beside three pixel renderers over one game state and reports consistent final performance across representations in a 150-experiment baseline. Against that, Phase 0's headline result is that the headless core loads no game assets at all, so rendering would undo the property the architecture is built on.

Planes are added alongside entities rather than replacing them because AlphaStar's spatial encoder consumes semantic planes while scatter connections inject entity embeddings into those planes, so the two modalities coexist and fuse. Making them independently toggleable follows PySC2 and Griddly practice and costs nothing when unused.

Two costs are recorded honestly. Whether planes help at 11 by 9 is unmeasured anywhere, so this is an experiment rather than an improvement, and [[../rl/training-design]] keeps the entity encoder as the baseline for that reason. And the board's adjacency is hexagonal with six neighbours in row-offset geometry, so a square convolution kernel covers a receptive field that does not match the game's adjacency. No published evidence favours any hex rasterization convention, which the sweep recorded as a genuine gap, so the engine's own `Battle::Board` indexing is standardized on and written into the schema rather than assumed.

## Decision

1. No rendered pixels in the training environment, ever. Anything pixel-real stays on the separate `play-harness` branch. (Rejecting this permanently for the env; revisit only with evidence the literature currently does not contain.)
2. The observation schema gains a second modality axis, orthogonal to ADR 0001's profiles:

   ```json
   "observation_profile":   "full_v1" | "observable_v1",
   "observation_modalities": ["entities", "planes"]        // default: ["entities"]
   ```

   Modalities are independently toggleable (PySC2/Griddly precedent);

requesting `planes` adds a `planes` section to each observation, filtered by the active profile like every other field.
3. `planes_v1` layout (fixed 11×9 board-index grid; exact channel list frozen at Milestone 3/4 implementation): per-side occupancy, unit-class identity, count fraction, HP fraction, speed, shooter flag/shots fraction, passability/obstacles; reachability and threat channels join once the Milestone 3 resolvers exist. Channels are typed categorical or scalar-normalized, SC2 feature-layer style.

   The tensor is derived from the same engine state that feeds the entity list, one source of truth, one digest (unchanged, still over full state).
4. Hex convention documented, not assumed: planes use the engine's own 11×9 `Battle::Board` cell indexing (row-offset hex). No published evidence favors any hex rasterization convention (verified gap), so we standardize on the engine's and record it in the schema.
5. Policy-head choice is deferred to the training milestone, CNN over planes vs entity-transformer vs MLP at this board size is an open question the literature does not answer; the plane emitter costs nothing when unused, and having both modalities enables AlphaStar-style hybrid/scatter fusion as well as the in-house ablation.

## Consequences

- Spec §12 observation schema gains the modality field and `planes` section; the JSONL protocol (Milestone 4) carries it from day one, no schema break later.
- Trajectory headers record requested modalities (dataset-compatibility axis, spec §15.4).
- `configs/env/` (ADR 0003) exposes profile and modalities as config keys.
- The state digest and all existing verification gates are untouched.
