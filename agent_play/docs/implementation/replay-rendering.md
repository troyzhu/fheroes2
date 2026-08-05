---
title: Replay rendering — a primer
aliases:
  - replay-rendering
  - real-engine-replay
  - battle-videos
tags:
  - agent-env
  - primer
concept: exact re-execution of a recorded episode through the game's own battle interface
domain: RL environment tooling
grounded_in: "determinism machinery of Milestone 1; the DecisionController seam of Milestone 2; verified in [[../archive/experiments/2026-08-05-real-engine-replay]]"
depth: quick
updated: 2026-08-05
---

# Replay rendering — a primer

The environment is headless by design, so nothing about a trained policy is watchable. This primer explains the mechanism that renders a recorded episode through the game's real battle interface, why the result is provably the same battle the policy played rather than a re-enactment, and where the engine had to be touched to allow it.

## Motivation

Win rates say a policy improved; they do not show what it does. The owner asked to see the battles, and a schematic board is a poor substitute for the engine's own sprites, animations, and combat log. The obvious approaches both fail: screenshotting a live game session captures whatever battle happens then, not the recorded one, and re-running the policy inside a rendering process would need the network in C++ and would still sample different actions. What is wanted is the recorded episode itself, drawn by the engine.

## The idea in one sentence

Feed the recorded canonical action indices back through the same `ExternalDecisionController` the worker used, under the same scenario and world seed, with the arena constructed to show its interface, and prove exactness by comparing the terminal state digest against a headless run of the same recording.

## Intuition

Because every battle is a deterministic function of the scenario, the world seed, and the decision sequence, a recording is not a video but a program: replaying its inputs reproduces its states. Rendering is then a pure observer attached to that re-execution. The claim rests on the determinism machinery in [[determinism-seeds-and-digests]], and it is checkable per replay rather than trusted: the SHA-256 over the canonical terminal state must come out identical with the interface on and off.

## How it works

`capture_replay.py` records an episode as JSON: the scenario (armies, commanders, fixture, side), each decision's pre-decision unit list, and the canonical action index the policy chose. It encodes observations per the checkpoint's stamped encoding version, so checkpoints trained before [[../decisions/0006-encoding-count-scaling|ADR 0006]] replay faithfully.

`fheroes2_agent_replay` (`src/agent_replay/`, the same relink build as the worker) rebuilds the scenario, wraps the recorded list in a `DecideFn`, and calls the runner. The runner gained one defaulted parameter, `showInterface`, forwarded to the arena constructor, which is where the engine has always decided whether a battle owns an interface. Every existing caller is byte-identical because the default is off.

Frame capture stays out of the engine: the display offers a generic render observer, a null-by-default callback invoked once per composed frame (`Display::setRenderObserver`, `src/engine/screen.h`), and the replay tool's `--frames-dir` installs a dumper on it that saves every frame as a numbered BMP with a capture-time manifest line. `render_replay.py` orchestrates the whole pipeline: headless run, rendered run, digest comparison, then ffmpeg assembly at the manifest's real timing.

The one rendering repair: scenario commanders carry no faction, and the battle interface chooses hero art by race, so `Race::NONE` now draws the Knight captain instead of asserting. No real game hero reaches that branch.

## The exactness invariant

A replay is accepted only if every recorded action was legal at its decision point (the controller rejects otherwise), every recorded action was consumed, the live battle asked for no extra decisions, the termination matches the recording, and the headless and rendered terminal digests are equal. The last check is the strong one: equal digests mean the interface changed nothing, same rolls, same kills, same survivors. All three episodes rendered on 2026-08-05 passed all five checks; the run log carries the digests.

## Comparison with alternatives

Screenshotting a live session was rejected because it shows a different battle and needs the macOS capture permissions this project deliberately avoids. Driving the policy live inside the rendering process was rejected because sampling would not reproduce the recorded trajectory and the network runs in Python. The `play-harness` branch's frame dump solves an adjacent problem, one overwritten frame throttled for a concurrent reader, and stays deliberately unmerged; this capture keeps every frame because its reader is an encoder running afterwards.

## Key terms

**Replay JSON.** The recording: scenario, per-decision frames, chosen action indices. Not reproducible from the checkpoint alone, since capture-time sampling is stochastic; the file is the provenance.

**Exactness.** The five-part acceptance above, digest equality being the decisive part.

**Frame manifest.** `manifest.tsv` beside the BMPs, one capture timestamp per frame, so the video reproduces the engine's animation cadence rather than a guessed frame rate.

## Why it came up here

The first visualization was a schematic SVG viewer, delivered the same day. The owner asked for the real engine, and the scout found the pieces were already lying in place: the arena's interface flag, the controller seam, and seed determinism. The engine surface it added is enumerated in [[inventory#Engine-source surface (what could possibly affect the game)|the inventory]].

## What this does not say

Rendering is an output channel for audit and presentation; the policy never sees pixels, and the exclusion argued in [[observation-design]] stands untouched. The video's wall-clock timing is the engine's animation cadence at the chosen battle speed, which says nothing about decision compute. And a rendered replay needs a display session and the game assets, so it runs on a desktop, not in the training loop.

## Go deeper

- [[determinism-seeds-and-digests]], the machinery the exactness claim rests on.
- [[battle-turn-dispatch]], where the controller seam sits in the turn loop.
- [[../archive/experiments/2026-08-05-real-engine-replay]], the run log with digests, frame counts, and the two defects found.
- `agent_play/experiments/capture_replay.py` and `render_replay.py`, the two ends of the pipeline.
