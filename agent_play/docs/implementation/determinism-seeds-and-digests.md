---
title: Determinism, seeds, and digests — a primer
aliases:
  - determinism
  - seeds
  - state-digest
tags:
  - agent-env
  - primer
concept: deterministic episode reproduction
domain: RL environment engineering
grounded-in: "this repo's Phase 0 audit and Milestone 1; agent_play/docs/references/"
depth: quick
updated: 2026-07-30
---

# Determinism, seeds, and digests — a primer

An fheroes2 battle is a pure function of two inputs, a world seed and an army composition, so fixing both makes the whole episode reproducible. This primer explains which seed controls what, how we pin them without patching the engine, and why every outcome is hashed. Digest equality is the only evidence we accept that an engine change changed nothing.

## Motivation

Reproducibility carries three separate jobs here.

Recorded demonstrations are worthless if replaying them produces a different battle, so recorded demonstrations would not correspond to reproducible battles. Regression safety has no cheaper instrument. We are editing engine internals, where re-reading the diff proves nothing and a matching outcome hash proves a great deal. Debugging requires repeated runs until the failure recurs.

The naive approach fails for a specific reason. The engine picks its map seed from a global random device at construction time and exposes no setter, so calling the battle entry point twice gives two different battlefields. Grepping the repository for `SetMapSeed` returns nothing.

## The idea in one sentence

Pin the world seed before the map is generated, derive the combat seed through one shared helper, then hash the terminal state so any divergence is loud rather than silent.

## Intuition

Reseeding after `generateBattleOnlyMap` is too late, because the map seed has already been drawn from the device. The same discipline governs any seeded pipeline. In PyTorch, `torch.manual_seed` has to run before the `DataLoader` iterator is built rather than after.

The digest plays the role a checksum plays on a dataset shard. Nobody inspects the bytes; they compare the hash and move on.

## How it works

The engine draws battle randomness from two independent places, and conflating them is the classic error.

**World seed to map seed.** `World::Defaults()` sets its map seed from `Rand::Get()`, which draws from a `thread_local PCG32`. The map seed then drives battlefield obstacle placement, where `battle_arena.cpp:430` seeds a separate generator with `world.GetMapSeed() + tileIndex`. Phase 0 found that `Rand::CurrentThreadRandomDevice()` (`rand.cpp:85`) returns a mutable reference, so the device can simply be replaced:

```cpp
Rand::CurrentThreadRandomDevice() = Rand::PCG32( worldSeed );
world.generateBattleOnlyMap( groundType );
```

Two properties follow. The call is positional, so the two lines stay adjacent and nothing that consumes global randomness may sit between them. It is also blunt, pinning every global draw in the process, which is correct for a dedicated worker and wrong inside the interactive game.

**Combat seed.** Computed from the tile index, the map seed, and every army slot, then used to construct the generator the arena consumes for damage, morale, and luck. The arena does not own it. `Arena` holds `Rand::PCG32 & _randomGenerator` (`battle_arena.h:385`) and the object lives in the caller (`battle_main.cpp:360`), so a worker must construct and reconstruct it itself. Slot position participates, so the same troop in slot 0 and slot 1 produces different battles, which is why scenarios always carry all five slots explicitly. The formula lives in `Battle::computeBattleSeed` (`battle_seed.{h,cpp}`), extracted in Milestone 1 so the engine's battle loader and our runner derive it through one implementation.

**Digests.** A digest is a hash over a canonical serialization of state. Three are in use.

| Digest | Covers | Purpose |
|---|---|---|
| Spike FNV fold (`2cfd42cb104aa5e7`) | Phase 0 terminal state | Historical baseline, still checked |
| `agent_terminal_v1` (SHA-256) | Seeds, rounds, result bits, termination, every unit record | Canonical episode outcome |
| `agent_decisions_v0` (SHA-256) | The recorded decision stream | Proves the teacher's choices replayed |

The canonicalization rules carry the weight, not the choice of hash function. Integers are encoded little-endian at fixed width, strings are length-prefixed so that `"a" + "bc"` cannot collide with `"ab" + "c"`, and no wall-clock field appears anywhere. That last rule is why two independent worker processes emit byte-identical trajectory files.

> [!note]- Why the combat-seed formula is frozen The fold order and contents are a compatibility contract. Changing them means the same scenario now produces a different battle, so every previously recorded trajectory becomes unreplayable. `agent_play/tests/test_battle_seed.cpp` pins four engine-derived golden values against exactly that risk, and a failure there is a compatibility break rather than a test to update.

## Comparison with alternatives

| Approach | Engine change | Scope of control | When preferred |
|---|---|---|---|
| Reseed the thread-local device (ours) | none | all global randomness in the process | A dedicated worker process, where blunt pinning is harmless |
| Add a `World` seed-API overload | small, invasive | the map seed alone | Production use inside the interactive game, where global pinning is unacceptable |
| Record and replay the command log | none | exact command sequence | Strict replay debugging, where you need the identical command stream rather than the identical state |
| Accept nondeterminism, average over runs | none | none | Never here, because it destroys both replay and regression testing |

The overload remains the better long-term shape and is a deferred cleanup, not a prerequisite. It was demoted once Phase 0 proved the reseed sufficient, which removed an engine patch from the Milestone 1 critical path.

## When to use it

Reseed before every episode, always adjacent to map generation. Compare digests, never prose descriptions of behavior, when judging whether an engine edit was safe.

## Key terms

- World seed: our input, pinned by reseeding; determines the map seed.
- Map seed: derived by the engine; drives obstacle placement.
- Combat seed: `computeBattleSeed(tileIndex, mapSeed, attacker, defender)`; seeds the arena.
- State digest: SHA-256 over canonical terminal state, `agent_terminal_v1`.
- Canonicalization: the fixed encoding rules that make a hash comparable across machines.

## Why it came up here

Every gate under `agent_play/verify_m*.sh` reduces to a digest comparison, and the same five golden fixture digests have now survived a seed-helper extraction, a decision-hook insertion, a legality-rule lift, and candidate enumeration running at every decision. Five fixture digests on one machine are the extent of that evidence.

### The single-arena rule is only enforced where assertions live

The arena is a pointer in an anonymous namespace (`battle_arena.cpp:74`) and the constructor asserts it is null (`battle_arena.cpp:356`). Under `NDEBUG` that assertion disappears, so a second arena silently overwrites the global and every static accessor built on it reads the wrong battle. The `src/dist` Makefile never defines `NDEBUG`, but a CMake `Release` build does, which is the configuration Milestone 4's `ENABLE_AGENT` target will introduce. Enforcement has to be added there rather than assumed.

## What this does not say

Determinism is not stability across engine versions. An upstream change to damage rules will legitimately move every digest. Digests detect change; judging whether a change is acceptable stays human work.

## Go deeper

- [[battle-turn-dispatch]] — where in the turn loop randomness is consumed.
- [[command-encoding-and-snapshots]] — the command stream behind the decision digest.
- `agent_play/docs/local_source_audit.md` — the Phase 0 evidence, with file and line citations.
