---
title: "Determinism: seeds, digests, and why we hash everything"
type: concept-primer
depth: quick
grounded-in: "fheroes2 agent-env branch (Phase 0, Milestone 1)"
related_concepts: ["[[battle-turn-dispatch]]", "[[command-encoding-and-snapshots]]"]
tags: [concept, determinism, rng, hashing, agent-env]
---

> **What this is.** A ground-up primer on the two seeds that control an fheroes2 battle, the
> digests that prove a run reproduced, and why an RL environment collapses without both. Read it
> before touching anything that calls `Rand::` or `computeBattleSeed`.

## The one-sentence version

A battle is a pure function of *(world seed, army composition)*, so if we control both and hash
the outcome, any divergence — across runs, machines, compilers, or refactors — becomes visible
immediately instead of silently poisoning training data.

## Why determinism is non-negotiable for RL

Three separate jobs depend on it:

1. **Replay.** A trajectory is only worth storing if replaying it reproduces the same states. A
   non-deterministic env turns recorded demonstrations into fiction.
2. **Regression safety.** We are modifying a 20-year-old game engine. The only affordable proof
   that a refactor changed nothing is *the outcome hash did not move*. Every engine change on
   this branch was accepted on exactly that evidence.
3. **Debuggability.** "Reproduce the bug" must mean re-running one command, not re-running until
   luck strikes.

## The two seeds

fheroes2 draws battle randomness from two independent places, and conflating them is the classic
mistake.

**The world seed → map seed.** `World::Defaults()` picks a map seed by calling `Rand::Get()`,
which draws from a `thread_local PCG32` random device. The map seed then drives *battlefield
obstacle placement* (`battle_arena.cpp:427` seeds a separate `Rand::PCG32` with
`world.GetMapSeed() + tileIndex`). No public setter for the map seed exists.

The trick that unlocked this project (Phase 0 finding #2): `Rand::CurrentThreadRandomDevice()`
(`rand.cpp:85`) returns a **mutable reference**, so we simply assign a fresh generator before
generating the map:

```cpp
Rand::CurrentThreadRandomDevice() = Rand::PCG32( worldSeed );
world.generateBattleOnlyMap( groundType );
```

Two properties matter. It is **positional** — anything that consumes global randomness between
those two lines shifts the result, so they stay adjacent. And it is **blunt**: it pins *all*
global randomness in the process. That is correct for a dedicated worker and would be wrong
inside the interactive game.

**The combat seed.** Computed from the tile index, the map seed, and every army slot, then used
to seed the `Rand::PCG32` the arena consumes for damage rolls, morale, luck. The formula lives in
`Battle::computeBattleSeed` (`battle_seed.{h,cpp}`) — extracted in Milestone 1 so the engine's
own battle loader and our runner derive it through *one* implementation.

Slot **position** is part of the seed: the same troop in slot 0 and in slot 1 produces different
battles. That is why scenarios always carry all five slots explicitly.

> [!note]- Why the combat seed must never be "improved"
> The fold order and contents are a compatibility contract. Change them and every previously
> recorded trajectory becomes unreplayable, because the same scenario now produces a different
> battle. `agent_play/tests/test_battle_seed.cpp` pins four engine-derived golden values for
> exactly this reason; a failure there is a compatibility break, not a test to update.

## Digests: what we hash and why

A **digest** is a hash over a canonical serialization of state. We use three:

| Digest | Covers | Purpose |
|---|---|---|
| Spike FNV fold (`2cfd42cb104aa5e7`) | Phase 0 terminal state | Cheap historical baseline; still checked to prove old behavior holds |
| `agent_terminal_v1` (SHA-256) | Seeds, rounds, result bits, termination, every unit's terminal record | The canonical episode outcome |
| `agent_decisions_v0` (SHA-256) | Every recorded decision's command stream | Proves the *teacher's choices* replayed, not just the outcome |

The canonicalization rules are the load-bearing part, not the hash function: fixed-width
little-endian integers, length-prefixed strings (so `"a"+"bc"` can never collide with `"ab"+"c"`),
and **no wall-clock fields anywhere**. That last rule is why two independent worker processes
produce byte-identical trajectory files.

## Why it matters here

Every gate in `agent_play/verify_m*.sh` is ultimately a digest comparison, and the same five
golden fixture digests have now survived: a seed-helper extraction, a decision-hook insertion, a
legality-rule lift, and candidate enumeration running at every decision. That is the entire basis
on which we claim the engine still behaves identically.

## What this does *not* say

Determinism is not *stability across engine versions*. An upstream change to damage rules will
legitimately move every digest. The digests detect change; deciding whether a change is
acceptable is still human work.

## See also
- [[battle-turn-dispatch]] — where in the turn loop randomness is consumed.
- [[command-encoding-and-snapshots]] — the command stream that feeds the decision digest.
