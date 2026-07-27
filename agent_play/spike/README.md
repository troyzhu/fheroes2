# Phase 0 spike — headless battle smoke test

> New here? Read **`docs/agent/START_HERE.md`** first.

Throwaway diagnostic that answers the Phase 0 questions in
`agent_play/fheroes2_agent_system_spec_v0.3.md` §2.4. **Not** the eventual agent worker:
there is no decision hook, no candidate generation, no protocol, and no Python here.

Findings are written up in `docs/agent/local_source_audit.md`.

## Build

```bash
make -C src/dist -j10          # once, from the repo root — produces the game objects
./agent_play/spike/build_spike.sh
```

For a sanitizer run, build the engine and the spike with the same switch (the script mirrors the
`src/dist` Makefile convention and adds the matching `-fsanitize` flags to compile and link):

```bash
FHEROES2_WITH_ASAN=1 make -C src/dist -j10
FHEROES2_WITH_ASAN=1 ./agent_play/spike/build_spike.sh
```

`build_spike.sh` compiles only `smoke_battle.cpp` and relinks it against the existing
`src/dist/fheroes2/*.o`, **excluding `fheroes2.o`** — the translation unit holding the game's real
`main()`. That is deliberate: it is the cheapest possible test of the spec's constraint that the
agent entry point must not collide with the game entry point, and it proves the non-entry object
set is already usable as a library.

## Run

```bash
cd "$(mktemp -d)"                                    # keeps stray config out of $HOME
SPIKE=/path/to/fheroes2/agent_play/spike/smoke_battle

$SPIKE --episodes 10                                  # determinism check
$SPIKE --episodes 10 --no-global-seed                 # control: should diverge
$SPIKE --episodes 2000 --quiet                        # sequential-reuse stress
$SPIKE --monster-a 2 --count-a 20 --monster-b 1 --count-b 60   # Archer vs Peasant
```

| Flag | Default | Meaning |
|---|---|---|
| `--episodes N` | 1 | fresh arenas to run sequentially in this process |
| `--world-seed S` | 20260726 | seed for the thread-local RNG, which fixes the map seed |
| `--monster-a/-b ID` | 1 (PEASANT) | monster id per side; 2 = ARCHER, 3 = RANGER |
| `--count-a/-b N` | 50 | stack size per side |
| `--max-rounds N` | 200 | truncation cap |
| `--no-global-seed` | off | skip the RNG reseed — used to demonstrate the engine's default nondeterminism |
| `--quiet` | off | summary line only |

Protocol-ish output goes to stdout; the config banner goes to stderr.

## What it demonstrates

- A `Battle::Arena` runs headless with **no display, audio, AGG, or game data**.
- Reseeding `Rand::CurrentThreadRandomDevice()` makes the world seed reproducible **without any
  engine change**, contradicting the spec's assumption that a new `World` API was required.
- 2000 sequential arenas in one process, one distinct digest, 12.9 MB peak RSS.
- Input `Army` objects are *not* synchronized after battle (`army_synced=no`), so terminal state
  must be read from the `Force` objects before the arena is destroyed.

## Known limitations

- ~~Release-only.~~ Resolved 2026-07-27: the `FHEROES2_WITH_DEBUG` run passed 7/7 with digests
  identical to Release, and an ASan+UBSan build ran 1 900 episodes across five compositions with
  zero reports. (Note: the Makefile's `-O3` build never defines `-DNDEBUG`, so asserts were live
  in every "Release" run all along.)
- `computeBattleSeed` is duplicated verbatim from `battle_main.cpp` because the engine does not
  export it. That duplication is the argument for the shared helper in spec §7.3.
- The digest is an FNV-style fold, not SHA-256. Sufficient to detect divergence between runs; the
  real environment needs the canonical digest defined in spec §12.5.
- ~~Measured on an Apple M3, not the target Mac mini M2.~~ Resolved 2026-07-27: reproduced on the
  M2 with identical seed and digest; see `docs/agent/benchmark_m2.md`.
