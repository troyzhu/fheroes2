---
title: "Real-engine replay rendering, 2026-08-05"
type: experiment-log
updated: 2026-08-05
tags: [agent-env, archive, experiment, replay, rendering]
---

# Real-engine replay rendering, 2026-08-05

The owner asked to see the agent's progress as rendered battles, and after a schematic SVG viewer shipped the same hour, asked whether the actual game engine could do the rendering instead. This log records the scout, the build, the two defects it surfaced, and the verification that makes the videos evidence rather than illustration. The mechanism's documentation lives in [[../../implementation/replay-rendering]]; this is the raw record.

## The scout, and why the answer was yes

Three pieces were already in place. The arena constructor has always taken `isShowInterface`, and the headless runner simply passed false. The `ExternalDecisionController` takes any decide callback, so replaying a recorded action list is a callback that pops a vector. And Milestone 1's seed discipline makes a battle a deterministic function of scenario, world seed, and decision sequence, so a recording is sufficient to reproduce the battle exactly. The missing pieces were process initialization (display, AGG assets, palette, animation delays, mirrored from the game's own `main()` minus config files and audio), a frame capture point, and a tool to drive it.

The captures came from `capture_replay.py` driving the usual protocol worker: the checkpoint's stamped encoding version selects the encoder, so the pre-ADR-0006 checkpoints replay under the linear scaling they were trained with. Each JSON stores the scenario and the canonical action per decision. These recordings are not reproducible from the checkpoints, capture-time sampling is stochastic, so the three JSONs are vendored beside this log under `files/`.

## What was added, and where

`src/agent_replay/` holds the tool, built by the same relink script pattern as the worker. `runEpisode` gained a defaulted `showInterface` parameter forwarded to the arena constructor; every existing caller is byte-identical. The display gained a generic render observer, a null-by-default callback invoked once per composed frame, after the owner asked for a cleaner separation than the first cut's env-gated dump inside `screen.cpp`; the dump logic, every frame as a numbered BMP plus a capture-time manifest line, now lives in the replay tool behind `--frames-dir`, deliberately unlike the play-harness branch's throttled single-file dump because this reader is an encoder, not a concurrent agent. `render_replay.py` orchestrates: headless verification, rendered capture, digest comparison, ffmpeg assembly at manifest timing.

## The two defects the rendered path surfaced

The raceless captain. Scenario commanders are captains with `Race::NONE`, and `OpponentSprite` chooses battlefield art by race with an assert on the default branch, so the first rendered run aborted at `battle_interface.cpp:1015`. Fixed by drawing `Race::NONE` with the Knight captain art; no real game hero reaches the branch, and the alternative, giving scenario commanders a real race, was rejected because commander identity feeds the combat seed path and would have invalidated every recorded battle.

The teardown order. The tool initially crashed in static destruction after printing its terminal record, because the display was never released while the SDL core still lived. The game's own `DisplayInitializer` destructor does `unregisterRenderers` then `Display::release` in exactly that order; mirroring it fixed the exit.

One environment lesson, not a defect: `resurrection.h2d` ships with fheroes2 rather than the original game data, and the CLI data root at `~/.fheroes2` lacked it, so the wrapper adds the repository root through `FHEROES2_DATA`, where `files/data/resurrection.h2d` lives.

## Verification

Each replay must pass five checks: every recorded action legal at its decision point, all actions consumed, no extra decisions requested, termination equal to the recording, and the SHA-256 terminal state digest equal between a headless and a rendered run of the same recording. Digest equality is the decisive one, since it proves the interface perturbed nothing.

| Recording | Decisions | Result | Headless digest = rendered digest | Frames | Video |
|---|---|---|---|---|---|
| `files/2026-08-05-replay-clone.json`, clone on the Thunk fight | 21 | defeat, 641 Peasants standing | `3a16f6283ebf…`, yes | 1,050 | 34 s |
| `files/2026-08-05-replay-trained.json`, curriculum policy, same fight | 24 | victory, 2 stacks alive | `db31aba901de…`, yes | 1,177 | 48 s |
| `files/2026-08-05-replay-diverse.json`, clone v4 on a budget-sampled matchup | 23 | victory, 4 creatures alive | `98879ae4400d…`, yes | 1,536 | 65 s |

The full gate suite stayed green after the engine touches: Phase 0, M1 5/5, M2 8/8, M3 9/9, agent 11/11, and the 145 Python checks. The golden digests are the proof that the headless path is unchanged, which is exactly what they exist to prove.

## What the videos show

The clone loses the Thunk fight the way the numbers said it would, feeding stacks into the Peasant mass piecemeal. The trained policy wins it with two stacks standing, which is the 0.000 to 0.891 curriculum result made visible. The diverse battle shows clone v4 fighting a cross-faction budget-sampled matchup, Steel Golem and Mummies among Orcs and Goblins, with the engine's combat log narrating. Battle speed 10, 640x480 upscaled 2x nearest-neighbor.
