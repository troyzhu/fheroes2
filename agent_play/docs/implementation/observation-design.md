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
grounded_in: "agent_play/docs/references/ (SC2LE, Griddly, AlphaStar, NLE); ADR 0001 and ADR 0004"
depth: quick
updated: 2026-07-30
---

# Observation design — a primer

What the policy sees is set by two independent choices: the representation, meaning entity records or spatial planes, and the observability, meaning true state or player-obtainable state. This primer explains both axes, the rule that keeps the environment an MDP, and why rendered pixels are excluded permanently.

## Motivation

A battle state is a set of typed objects on a small board. Serializing it badly costs more than throughput. Handing the policy an image forces it to spend capacity recovering numbers the engine already knows, and hiding a field that still affects the dynamics silently converts the problem into a partially observed one that nobody chose.

The naive approach, screenshotting the game, fails on all three counts here. It reintroduces the display and asset stack that the headless core was built to avoid, it measures roughly an order of magnitude slower, and it discards exact values in favor of rendered glyphs.

## The idea in one sentence

Emit structured engine state, as padded entity records plus an optional semantic plane tensor, filtered by an observability profile, and never emit an image.

## Intuition

The representation choice is one an ML engineer makes routinely: a set of tokens for a transformer against a channel-stacked grid for a convolutional network. Units are naturally tokens, since there are few of them and each carries typed attributes. The board is naturally an image, since adjacency and reach are spatial. Both views describe the same state, and production systems feed both to one network rather than picking a winner.

The observability choice is the asymmetric actor-critic split, stated in the environment rather than in the training code. The critic may read privileged state during training; the actor reads only what a player could obtain, because that is what it will have at deployment.

## How it works

**Axis 1, representation.** An entity list carries one record per stack: identity, position, count, hit points, speed, shots, status flags. It is variable-length in principle, so it ships as fixed slots with an explicit NULL category for empty ones, the standard answer at this scale. A plane tensor rasterizes the same state onto the board as `11 × 9 × C`, where C is the channel count fixed by ADR 0004 and finalized with Milestone 4: occupancy per side, unit class, count and hit-point fractions, passability, and later reachability and threat.

The two are complementary. [[../research/works/alphastar|AlphaStar]] feeds an entity transformer, a plane convolutional network, and a scalar vector into one core, and additionally scatters entity embeddings into the plane stack. Our schema therefore treats them as independently toggleable modalities, `entities` and `planes`, following the one-state-many-observers pattern that [[../research/works/pysc2|PySC2]] and [[../research/works/griddly|Griddly]] both ship.

**Axis 2, observability.** The two profiles are two amounts of information about the same battle. `observable_v1` is what a player at the screen could obtain; `full_v1` adds a handful of fields tagged `oracle` that no player could read off, chiefly the engine's own strength estimate. That is the entire difference: a short list of extra fields, not a different kind of problem.

One caveat keeps the vocabulary honest. Neither profile is the true state $s$ in the MDP sense, because some state that drives the dynamics, such as the combat random generator's position, is serialized in neither. Formally both are observation functions over $s$, and the problem is partially observed under either. In practice the gap only matters where a hidden quantity changes what happens next, and the section below on keeping it an MDP says how the design keeps that gap small.

Why keep two profiles at all: the tempting training setup gives the critic the richer view and the actor the deployable one, critic on `full_v1`, actor on `observable_v1`, which is why the profile is chosen per consumer rather than globally. The temptation has a known cost. When the actor cannot see everything, judging its actions with a critic that can produces a subtly wrong training signal: the critic's advantages answer "was that good given the full state" while the actor needed "was that good given what I could see", and the two differ exactly where the hidden information matters. The theoretically clean repair conditions the critic on the actor's history rather than on privileged state (Baisero and Amato, 2022). Our literature sweep found no verified case where the naive split simply worked, so this project treats the asymmetric setup as an available experiment, never a default.

In the shipped serializer only `full_v1` exists, and it carries no oracle-tagged field yet: `observable_v1` and the `engine_strength` field are ADR 0001's designed schema, not code, so everything in this paragraph describes the committed design rather than the current wire.

That field deserves care. It is not privileged state but the output of a different agent's hand-written evaluator, so feeding it to a critic is value distillation from the scripted AI rather than an asymmetric observation, and it imports that AI's tactical priors and its blind spots. Whether a critic regresses on it is a decision to take deliberately. Creature-only battles are informationally symmetric, because the battle interface shows any unit's full stat sheet with no ownership gating (`Cursor::WAR_INFO` reaching `Dialog::ArmyInfo`). Genuine hidden information arrives with enemy hero mana and adventure-map fog, and `observable_v1` extends to cover it without forking the schema. Omitted fields are absent rather than zero-filled, so the two profiles stay distinguishable on the wire, and the state digest always covers full state regardless of profile.

**Axis 3, why not pixels.** Three reasons, in increasing order of finality. Griddly measured semantic planes against three real pixel renderers at roughly 72,800 against 5,000 frames per second with consistent task performance across representations. [[../research/works/sc2le|SC2]]'s minimap was never RGB, since DeepMind shipped synthetic feature layers on the stated rationale that agents should not learn to read numbers off a screen. And our headless core loads no game assets at all, so rendering would undo the finding the environment is built on. A minimap here therefore means a semantic plane tensor. Anything pixel-real belongs to the separate `play-harness` branch.

## The rule underneath both axes: keep it an MDP

An attribute that influences the transition distribution and is not observed makes the observation process non-Markov. That is not fatal, since partially observed problems are well posed and routinely trained, but it carries costs worth choosing rather than inheriting. The best memoryless policy may have to be stochastic and may still be strictly suboptimal, observation-conditioned value estimates are biased as estimates of $V^\pi(s)$, and temporal-difference targets inherit that bias. Prefer exposure so the costs are opt-in.

The reference project met this and chose removal, deleting morale, luck, and terrain effects from the game. We chose exposure instead, so those mechanics stay live and their fields appear in both profiles.

## Comparison with alternatives

| Representation | Shape | Strength | Cost | When preferred |
|---|---|---|---|---|
| Padded entity slots (ours, default) | fixed slots with NULL category | exact values, small, order-stable | wastes slots when armies are small | Few typed objects, as here with at most ten stacks |
| Semantic planes (ours, optional) | `11 × 9 × C` tensor | spatial structure is explicit | redundant with entities | Convolutional policies, spatial reasoning |
| Entity transformer over a ragged set | variable-length set | scales to hundreds of units | more machinery, ragged batching | Large or highly variable unit counts |
| Graph over cells | typed nodes and edges | encodes reach and threat directly | heaviest to build and debug | After the simpler encodings plateau |
| Rendered pixels | image | none here | roughly 14× throughput, reintroduces assets | Only when no structured state exists |

Padded slots plus optional planes is the combination the shipped comparable system used, and it keeps the upgrade path to transformers and graphs open, since both consume the same underlying records.

## When to use it

Use `observable_v1` for anything that will be deployed, and `full_v1` for critics, teachers, and debugging. Request `planes` only when a policy actually consumes them, since the emitter costs nothing when unused.

## Key terms

- Observability profile: `full_v1` or `observable_v1`, controlling how much true state is revealed.
- Modality: `entities` or `planes`, controlling representation; orthogonal to profile.
- Oracle field: a value no player could obtain, present only in `full_v1`.
- Feature layer: a coarse semantic plane rendered from state, not captured from a renderer.

## Why it came up here

Both axes are committed in ADR 0001 and ADR 0004 and bind the Milestone 4 serializer. Settling them before the protocol ships avoids a schema break later.

## What this does not say

It does not settle whether a convolutional network over planes beats an entity transformer or a plain multilayer perceptron at this board size. No published ablation exists at `11 × 9`, which is exactly why both modalities exist and why the comparison is a planned in-house experiment.

## Go deeper

- [[legal-actions-and-masking]] — the action side of the same interface.
- [[determinism-seeds-and-digests]] — why the digest ignores profiles.
- `agent_play/docs/references/report-spatial-observations.md` — the verified evidence behind the pixel decision.

## The v3 revision, measured in

`obs_encoding_v3` log-scales counts and hit points and changes nothing else. [[../decisions/0006-encoding-count-scaling]] is the record.

The concrete tensor as of v3 is 634 wide, ten slots of 63 named per-stack features (presence and side flags, counts and hit points, the six stat fields, position, the four ability flags, and a 41-way creature one-hot) plus four globals. `FEATURE_NAMES` in `python/fheroes2_agent/encoding.py` is deliberately the single authoritative layout, named so an encoded row reads back by a human; the widths here are quoted from it rather than owned, and [[../rl/training-design#As built, 2026-08-05|training-design]] carries the network that consumes it.

## What the board does not say

The entity list carries every stack's position, so the policy knows where units are; it carries nothing about the terrain between them. There are no passability, obstacle, reachability or threat features, and the board's shape reaches the policy only through the legality mask, which reports which moves the acting stack may make this turn and says nothing about anyone else's options. Positional plans that constrain an opponent's path, the kind that force a slow stack to walk around an obstacle, are therefore not expressible in what the policy sees, however well it plays with what it has. That is the strongest argument for the `planes_v1` modality of [[../decisions/0004-spatial-observation-modality]], and it was sharpened by the owner on 2026-08-06 against a looser claim that the agent simply cannot see the board.

## The planes, built across 2026-08-06 and 07

The engine half of `planes_v1` exists: the worker's `--planes` flag appends a 99-cell `obstacles` array to every serialized observation, engine-read from `Cell::GetObject`, off by default with transcripts proven byte-identical when off. The tensor builder `encode_planes` (in `encoding.py`) rasterizes the committed channel list from the units plus that layer, seven channels of shape 9 by 11 in the engine's own row-offset indexing: per-side occupancy, count fraction, log-scaled hit points, speed, shooter, obstacle. The design insight that shrank the wire format is that every channel except obstacles is derivable from the entity list, so the engine emits only what entities cannot carry. The convolutional fusion arm consuming the tensor exists as of 2026-08-06 (`BattlePolicy(planes=True)`, inferred from the state dict by `load_policy`) and was measured by the capacity-controlled three-seed ablation; corpora recorded without `--planes` have an all-zero obstacle channel that consumers must treat as unknown rather than open ground.

## Ability records, three layers, one built

The four ability flags above compress the bestiary's rule diversity to almost nothing, and the owner-supplied guide ([[../research/works/generalized-battle-agent-guide]], its sections 3 through 5) lays out the repair as three layers. Layer 1, raw engine records: every ability and weakness exported as categorical type id plus typed payload (`percentage`, `value`), never as text, with the engine staying the authority on what the rule does. Layer 2, a semantic adapter mapping each raw record to a typed schema of trigger, target, and effect primitives, needed because `value` is type-dependent, a spell id for one type and a magnitude for another. Layer 3, action-conditioned effect summaries, engine-computed answers to "what would this ability do for this candidate in this state", which is what static encoding cannot carry.

Layer 1 is built as of 2026-08-05: the capability audit (`agent_capabilities.cpp`, regenerated into `python/fheroes2_agent/data/monster_capabilities_v1.json`) now carries `abilities` and `weaknesses` arrays per monster, additively, every earlier field unchanged. Nothing consumes them yet; the consumer is a Deep-Sets-style pooled embedding per the guide, an encoding change that ADR discipline says needs its own ablation before any version bump. Layers 2 and 3 are designed in the guide's digest and deliberately deferred, layer 3 because it needs a per-candidate engine query seam of the same shape the teacher probe used, which the probe's digest methodology would also verify.
