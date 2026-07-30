---
title: fheroes2 battles as an RL problem, and how they differ from other games
aliases:
  - fheroes2-battles
  - homm2-battles
  - game-comparison
tags:
  - agent-env
  - primer
concept: the fheroes2 battle domain, stated in RL terms and placed among comparable environments
domain: reinforcement learning for games
grounded_in: "the fheroes2 engine; docs/agent/references/ for the comparison targets"
depth: standard
updated: 2026-07-30
---

# fheroes2 battles as an RL problem, and how they differ from other games

A Heroes of Might and Magic II battle is a small, turn-based, nearly fully observed tactical game with dice. This primer explains the game itself for someone who has never played it, restates it in the vocabulary of [[rl-for-games]], and places it against the environments this project borrows from. The comparison is what justifies most of our design: the domain is unusually easy along the axes that normally cost the most, and unusually awkward along two that rarely come up.

## Motivation

Every decision here was argued from another environment's evidence, so the transfer needs checking. StarCraft is real-time with thousands of units, microRTS is real-time on a grid, NetHack is a single-agent dungeon crawl, and Heroes III battles are the closest neighbor but still differ in scale. Borrowing a design without stating those differences is how a project inherits machinery it does not need, or misses a constraint that does not exist elsewhere.

## The game, for someone who has never played it

Two armies meet on a fixed battlefield. Each army holds up to five stacks, where a stack is a number of identical creatures acting as one unit, so "fifty peasants" is one stack that moves and attacks together and loses members as it takes damage.

The field is a grid of 11 by 9 cells, drawn as hexes, so each cell has six neighbors rather than eight. Attackers start on the left, defenders on the right.

Play proceeds in rounds. Within a round every stack that can act does so once, in order of speed, and its options are to move, to attack an adjacent enemy, to shoot a distant one if it has ammunition and nothing adjacent is blocking it, or to skip. A melee attack usually draws a retaliation from the defender. Damage depends on the attacker's attack rating against the defender's defense rating, multiplied by the number of creatures in the stack, and rolled within a range, so outcomes are stochastic. Morale and luck can grant an extra action or a damage bonus at random. The battle ends when one side has no living stacks.

Our Phase 1a scope removes the parts that complicate this: no heroes, so no spells and no leadership bonuses; no castle, so no walls, moat, or towers; and only creatures whose action space is ordinary, meaning single-cell, walking, without special targeting.

## The same thing in RL terms

| Object | In a Heroes battle |
|---|---|
| State $s$ | Position, count, hit points, remaining shots, and status of every stack, plus whose turn it is and the combat generator's position. |
| Action $a$ | What the active stack does: move to a cell, attack a specific enemy from a specific direction, shoot, or skip. |
| Legal set $\mathcal{A}(s)$ | Typically 5 to 30 actions, out of a fixed space of 793. |
| Transition $P$ | The engine, stochastic through damage rolls, morale, and luck, but exactly reproducible under a fixed combat seed. |
| Reward $R$ | Deliberately undefined in Phase 1a. The terminal outcome and surviving force are recorded so an objective can be chosen later. |
| Observation $o$ | Structured records, either the true state or the player-obtainable subset, never pixels. |
| Episode | One battle: roughly 5 to 40 decisions in the fixtures measured so far. |
| Players | Two, zero-sum, alternating within a speed-ordered turn queue. |

Two properties of this mapping deserve emphasis because they shape everything downstream. The decision boundary is a unit's turn, not a round, so a five-stack battle produces about ten decisions per round rather than one. And the environment is *nearly* fully observed: the interface reveals any unit's full statistics regardless of ownership, so the only genuinely hidden quantity is the future of the random generator.

## How it compares

| Environment | Timing | Board | Units per side | Action space | Observability | Stochastic | Steps/s |
|---|---|---|---|---|---|---|---|
| fheroes2 battle (this project) | turn-based, alternating | 99 hex cells | 1 to 5 stacks | 793, masked | effectively full | yes, damage and morale | ~4,600 episodes/s |
| Heroes III battle (vcmi-gym) | turn-based, alternating | 165 hex cells | up to 7 stacks | 2,312, masked | effectively full | yes | not published |
| microRTS | real-time, simultaneous | 16×16 grid | dozens | factorized per cell | full or fogged by flag | mostly deterministic | very high |
| StarCraft II (PySC2) | real-time, simultaneous | large minimap | hundreds | huge, structured | fogged | yes | low |
| NetHack (NLE) | turn-based, single agent | 21×79 glyphs | one hero | ~100 discrete | fogged, partial | yes | high |
| Battle for Wesnoth (ARLinBfW) | turn-based, alternating | hex map | several | small discrete | full on the map | yes | low |
| Chess and Go | turn-based, alternating | 64 or 361 cells | fixed | hundreds to thousands | full | no | very high |
| Dota 2 (OpenAI Five) | real-time, simultaneous | large map | five heroes | large, parameterized | fogged | yes | low |

## What is unusually easy here

The scale is small in every dimension that normally forces machinery. With at most ten stacks and 99 cells, a padded entity list fits comfortably and an entity transformer is an optimization rather than a necessity. The action space is about 793 entries, so a flat masked softmax works and factorization buys nothing.

Observability is effectively complete, so recurrence, belief states, and frame stacking are unnecessary at this stage. That removes an entire class of architecture decisions that dominate work on fogged games.

Turn-based alternating play means one decision at a time and no action-delay modeling, which is a large simplification against every real-time environment in the table.

The engine is fast and deterministic under a seed, at roughly 4,600 episodes per second on the target machine, so the learner will be the bottleneck rather than the simulator, and planning methods stay viable later.

A competent scripted opponent already exists inside the game. It plays both sides for free, which supplies both a demonstration source and an evaluation baseline without writing either.

## What is unusually awkward here

Two constraints show up rarely in published environments, and both come from embedding in a real game engine rather than a purpose-built simulator.

The engine exposes no legal-action API. Its validation logic lived inside the functions that execute commands, so the environment either extracts that logic or re-derives battle legality and risks disagreeing with the engine. This was the project's largest risk and it is why the validators were lifted into a shared module rather than reimplemented. See [[legal-actions-and-masking]].

Control is inverted and the arena is a singleton. The engine advances a whole round per call and owns the call stack, so the environment blocks inside a hook rather than exposing a callable step, and only one battle can exist per process, which makes parallelism process-level. See [[battle-turn-dispatch]].

A third, milder awkwardness: the stochasticity is coarse. Morale and luck can grant an entire extra action, so a single lucky roll changes a turn's structure rather than nudging a number, which makes variance across seeds larger than damage rolls alone would suggest.

## What this implies for the design

The comparison explains why this project's decisions diverge from the environments it borrows from. We use a flat masked action space rather than microRTS's factorized one because 793 entries do not need factoring. We keep both an entity list and an optional plane tensor rather than committing, because at this scale neither is expensive and no published ablation settles which wins on an 11 by 9 board. We ship an observability profile despite full observability today, because the seam is free now and expensive to retrofit when hero mana and fog arrive. And we invest in seed and digest discipline more heavily than comparable projects do, because a fast deterministic engine makes that discipline cheap and it is the only affordable proof that engine edits changed nothing.

## Key terms

- Stack: a group of identical creatures acting as a single unit, the atom of a Heroes battle.
- Round: one pass in which every eligible stack acts once, ordered by speed.
- Retaliation: the defender's automatic counter-attack after a melee strike.
- Shooter: a stack with ranged ammunition, which reverts to melee when an enemy is adjacent.
- Wide unit: a creature occupying two cells, excluded from the current profile.
- Morale and luck: random effects granting an extra action or bonus damage.

## Why it came up here

The absence of this comparison was a real gap in the documentation. Design records cited evidence from other environments without stating whether the evidence transferred, which made choices such as the flat action space or the dual observation profiles look like preferences rather than conclusions.

## What this does not say

The comparison covers the battle domain only. The fheroes2 adventure map, with fog of war, resource management, and a much larger action space, is a different problem that would sit far closer to StarCraft in this table, and nothing here should be read as applying to it.

## Go deeper

- [[rl-for-games]] — the vocabulary this primer instantiates.
- [[legal-actions-and-masking]] and [[battle-turn-dispatch]] — the two awkward constraints in detail.
- `../references/repos.md` — the codebases behind the comparison column.
