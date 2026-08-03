---
title: Open-source repositories — an orientation
type: overview
updated: 2026-07-30
related_concepts: ["[[../rl/rl-and-the-battle-domain]]", "[[../implementation/legal-actions-and-masking]]"]
tags: [reference, repos, agent-env]
---

> **What this note is.** A practical orientation to the codebases behind this project's evidence: what each one is, how it is built, what is worth reading in it, and what we took or deliberately left. The per-work notes in `notes/` carry the verified claims; this note is for someone deciding which repository to open and where to look inside it.

Local snapshots of the READMEs and papers live in `files/`, listed in `manifest.tsv`. Nothing here is vendored into our build; the influence is design-level.

## The one closest to us

vcmi-gym ([github.com/smanolloff/vcmi-gym](https://github.com/smanolloff/vcmi-gym)) is a Gymnasium-compatible RL environment for Heroes of Might and Magic III battles, built on VCMI, the open-source reimplementation of that game. Its own description is exact: a gym-compatible environment "along with implementations of RL algorithms and other supplementary code (orchestration, hyperparameter tuning, observability)". Python wrapping a C++ engine through a connector layer, with the learners as single-file scripts under `rl/algos/` in the CleanRL style, hyperparameter search through Ray population-based training in `rl/raytune/`, and the observation and action encodings documented in `doc/env_info.md`.

Read it for three things: the observation layout, which is a flat float vector combining padded stack slots with per-hex features; the action space, a flat discrete space with a boolean mask exposed by the environment; and the honest engineering retrospective on the author's site, which walks through twelve observation iterations and says plainly what did not work.

The companion vcmi-MMAI and vcmi-mods/mmai hold the exported models that shipped inside VCMI 1.7.0 as a selectable battle AI, which is what makes this project an existence proof rather than a prototype.

What we took: the padded-slot-plus-per-cell encoding shape, the flat masked action space, the CleanRL-style single-file learner as the target training stack, and the Markov discipline of never half-observing an attribute. What we left: the specific constants, since the published documentation lags the code by several environment versions, and the decision to delete morale and luck from the game rather than expose them.

## The masking evidence base

MicroRTS-Py ([github.com/Farama-Foundation/MicroRTS-Py](https://github.com/Farama-Foundation/MicroRTS-Py)) is the Python wrapper around the microRTS Java engine, and the reference implementation of nearly everything this project does with masks. Deprecated by Farama in August 2025, so treat it as a frozen canonical reference rather than a dependency.

Worth reading: `CategoricalMasked` in the PPO training scripts, which is the twenty-line implementation of masked sampling that every later project reproduces; the observation encoder, showing per-cell one-hot feature planes with a `partial_obs` flag that appends visibility planes to the same tensor; and `league.py`, a TrueSkill evaluation loop that schedules matches until rating uncertainty converges instead of running a fixed count.

What we took: the masking implementation pattern, the one-schema-with-a-flag approach to partial observability, and the evaluation protocol. What we left: the factorized per-cell action space, which our 793-entry space does not need.

The companion paper repository, gym-microrts-paper, holds the ablation runs behind the numbers we cite, including the unmasked baseline that scores zero.

## The observation-design canon

PySC2 ([github.com/google-deepmind/pysc2](https://github.com/google-deepmind/pysc2)) is DeepMind's StarCraft II environment. Two files repay reading even if you never touch StarCraft: `pysc2/lib/features.py`, which defines every feature layer with an explicit scalar or categorical type, and `docs/environment.md`, which states the design rationale for shipping semantic layers rather than pixels. The interface toggles are the part we copied conceptually: feature layers, RGB, and raw observations are three independently enableable views over one game state.

Griddly ([github.com/Bam4d/Griddly](https://github.com/Bam4d/Griddly)) is a grid-world engine built for AI research, describing itself as "heavily optimized and flexible", with games defined in a YAML dialect and environments running "up to 70k FPS on a single thread". Its value here is the controlled comparison nobody else provides: the same games run under a semantic VECTOR observer and three pixel renderers, so the throughput and performance difference is measured rather than argued. Look at `src/Griddly/Core/Observers/VectorObserver.cpp` for what a semantic observer actually emits.

NLE, the NetHack Learning Environment ([github.com/heiner/nle](https://github.com/heiner/nle)), wraps NetHack with a symbolic observation of glyph, character, and color planes plus a statistics vector. Read its baseline agent for a compact example of the hybrid pattern: embed per-cell symbols, run a convolutional network over the embeddings, run a second one over a 9 by 9 crop around the hero, encode the statistics vector with a multilayer perceptron, then concatenate. MiniHack builds configurable smaller tasks on the same base.

## Architecture precedents

ARLinBfW ([github.com/DStelter94/ARLinBfW](https://github.com/DStelter94/ARLinBfW)) is the closest precedent for our integration shape and the smallest repository here, at two commits from 2019. It wraps Battle for Wesnoth in a Gym-like environment by running the game headless and bridging to Python out of process: a Lua add-on inside the game prints observations to standard output, which Python reads, while chosen actions are written to a file the Lua code polls. The README is worth ten minutes because it shows the pattern working against a real C++ game engine, including the source patch and the exact headless invocation. Our bidirectional JSONL over standard input and output is the same idea with a cleaner channel.

Stratega ([github.com/GAIGResearch/Stratega](https://github.com/GAIGResearch/Stratega)) is a general strategy-games framework whose agent interface is built around a forward model rather than a step function: an agent receives a copyable game state and a simulator, and is expected to plan. It is the reminder that a fast deterministic engine is an asset for search methods, not only for gradient methods, and it is why we intend to keep a copyable-state path open.

## Interfaces and learners

entity-gym ([github.com/entity-neural-network/entity-gym](https://github.com/entity-neural-network/entity-gym)) defines an entity-based environment API, extending "the standard paradigm of fixed-size observation spaces by allowing observations to contain dynamically-sized lists of entities". Its companion rogue-net implements the ragged-batch transformer that consumes those observations. This is the upgrade path from our padded slots, worth adopting only if slot padding becomes the constraint.

invalid-action-masking ([github.com/vwxyzjn/invalid-action-masking](https://github.com/vwxyzjn/invalid-action-masking)) holds the experiments behind the masking paper, including the penalty-versus-mask scaling comparison. The repository ships no README, which the fetch manifest records honestly as a failure rather than hiding.

sb3-contrib provides `MaskablePPO`, the off-the-shelf masked learner and our fallback if the single-file approach proves inconvenient. Sample Factory is the high-throughput single-machine alternative if the learner ever needs more than a straightforward implementation gives.

## How to use this list

For the closest analogue to what we are building, read vcmi-gym first, including the author's retrospective. For the masking implementation, read MicroRTS-Py's `CategoricalMasked`. For observation design, read PySC2's `features.py` and Griddly's observer comparison. For the process architecture, read ARLinBfW's README. Everything else is background, and the per-work notes in `notes/` say which claims each source actually supports.

## Related

- [[README]] — the scannable source catalogue with quality grades.
- [[findings]] — what the corpus collectively establishes.
- [[../rl/rl-and-the-battle-domain]] — how these environments differ from ours.
