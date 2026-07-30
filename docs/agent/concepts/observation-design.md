---
title: Observation design — a primer
aliases:
  - observations
  - observation-profiles
  - feature-planes
tags:
  - agent-env
  - primer
concept: structured observation representation and observability profiles
domain: RL environment design
grounded_in: "docs/agent/references/ (SC2LE, Griddly, AlphaStar, NLE); ADR 0001 and ADR 0004"
depth: quick
updated: 2026-07-30
---

# Observation design — a primer

What the policy sees is set by two independent choices: the representation, meaning entity records
or spatial planes, and the observability, meaning true state or player-obtainable state. This
primer explains both axes, the rule that keeps the environment an MDP, and why rendered pixels are
excluded permanently.

## Motivation

A battle state is a set of typed objects on a small board. Serializing it badly costs more than
throughput. Handing the policy an image forces it to spend capacity recovering numbers the engine
already knows, and hiding a field that still affects the dynamics silently converts the problem
into a partially observed one that nobody chose.

The naive approach, screenshotting the game, fails on all three counts here. It reintroduces the
display and asset stack that the headless core was built to avoid, it measures roughly an order of
magnitude slower, and it discards exact values in favor of rendered glyphs.

## The idea in one sentence

Emit structured engine state, as padded entity records plus an optional semantic plane tensor,
filtered by an observability profile, and never emit an image.

## Intuition

The representation choice is one an ML engineer makes routinely: a set of tokens for a transformer
against a channel-stacked grid for a convolutional network. Units are naturally tokens, since
there are few of them and each carries typed attributes. The board is naturally an image, since
adjacency and reach are spatial. Both views describe the same state, and production systems feed
both to one network rather than picking a winner.

The observability choice is the asymmetric actor-critic split, stated in the environment rather
than in the training code. The critic may read privileged state during training; the actor reads
only what a player could obtain, because that is what it will have at deployment.

## How it works

**Axis 1, representation.** An entity list carries one record per stack: identity, position, count,
hit points, speed, shots, status flags. It is variable-length in principle, so it ships as fixed
slots with an explicit NULL category for empty ones, the standard answer at this scale. A plane
tensor rasterizes the same state onto the board as `11 × 9 × C`, where C is the channel count
fixed by ADR 0004 and finalized with Milestone 4: occupancy per side, unit class, count and
hit-point fractions, passability, and later reachability and threat.

The two are complementary. AlphaStar feeds an entity transformer, a plane convolutional network,
and a scalar vector into one core, and additionally scatters entity embeddings into the plane
stack. Our schema therefore treats them as independently toggleable modalities, `entities` and
`planes`, following the one-state-many-observers pattern that PySC2 and Griddly both ship.

**Axis 2, observability.** In MDP terms, `full_v1` hands the network the true state $s$, while
`observable_v1` hands it an observation $o = O(s)$ and thereby a POMDP. Keeping both is the
standard asymmetric split, critic on `full_v1` and actor on `observable_v1`, which is why the
profile is a per-consumer setting rather than a global switch.

Today the two differ only by fields tagged `oracle`, chiefly `engine_strength`, the built-in AI's
own evaluator output. Creature-only battles are informationally symmetric, because the battle
interface shows any unit's full stat sheet with no ownership gating (`Cursor::WAR_INFO` reaching
`Dialog::ArmyInfo`). Genuine hidden information arrives with enemy hero mana and adventure-map fog,
and `observable_v1` extends to cover it without forking the schema. Omitted fields are absent
rather than zero-filled, so the two profiles stay distinguishable on the wire, and the state digest
always covers full state regardless of profile.

**Axis 3, why not pixels.** Three reasons, in increasing order of finality. Griddly measured
semantic planes against three real pixel renderers at roughly 72,800 against 5,000 frames per
second with consistent task performance across representations. SC2's minimap was never RGB, since
DeepMind shipped synthetic feature layers on the stated rationale that agents should not learn to
read numbers off a screen. And our headless core loads no game assets at all, so rendering would
undo the finding the environment is built on. A minimap here therefore means a semantic plane
tensor. Anything pixel-real belongs to the separate `play-harness` branch.

## The rule underneath both axes: keep it an MDP

An attribute that influences the dynamics must be either in the observation or out of the
dynamics. Stochastic transitions keep the problem an MDP; unobserved state demotes it to a POMDP
nobody chose, and a policy trained on such a state is learning from something that cannot explain
its own transitions.

The reference project met this and chose removal, deleting morale, luck, and terrain effects from
the game. We chose exposure instead, so those mechanics stay live and their fields appear in both
profiles.

## Comparison with alternatives

| Representation | Shape | Strength | Cost | When preferred |
|---|---|---|---|---|
| Padded entity slots (ours, default) | fixed slots with NULL category | exact values, small, order-stable | wastes slots when armies are small | Few typed objects, as here with at most ten stacks |
| Semantic planes (ours, optional) | `11 × 9 × C` tensor | spatial structure is explicit | redundant with entities | Convolutional policies, spatial reasoning |
| Entity transformer over a ragged set | variable-length set | scales to hundreds of units | more machinery, ragged batching | Large or highly variable unit counts |
| Graph over cells | typed nodes and edges | encodes reach and threat directly | heaviest to build and debug | After the simpler encodings plateau |
| Rendered pixels | image | none here | roughly 14× throughput, reintroduces assets | Only when no structured state exists |

Padded slots plus optional planes is the combination the shipped comparable system used, and it
keeps the upgrade path to transformers and graphs open, since both consume the same underlying
records.

## When to use it

Use `observable_v1` for anything that will be deployed, and `full_v1` for critics, teachers, and
debugging. Request `planes` only when a policy actually consumes them, since the emitter costs
nothing when unused.

## Key terms

- Observability profile: `full_v1` or `observable_v1`, controlling how much true state is revealed.
- Modality: `entities` or `planes`, controlling representation; orthogonal to profile.
- Oracle field: a value no player could obtain, present only in `full_v1`.
- Feature layer: a coarse semantic plane rendered from state, not captured from a renderer.

## Why it came up here

Both axes are committed in ADR 0001 and ADR 0004 and bind the Milestone 4 serializer. Settling
them before the protocol ships avoids a schema break later.

## What this does not say

It does not settle whether a convolutional network over planes beats an entity transformer or a
plain multilayer perceptron at this board size. No published ablation exists at `11 × 9`, which is
exactly why both modalities exist and why the comparison is a planned in-house experiment.

## Go deeper

- [[legal-actions-and-masking]] — the action side of the same interface.
- [[determinism-seeds-and-digests]] — why the digest ignores profiles.
- `docs/agent/research_minimap_observations.md` — the verified evidence behind the pixel decision.
