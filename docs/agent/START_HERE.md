# START HERE — fheroes2 agent environment

**Single entry point.** If you are a new session picking this work up, read this file first and in
full. It is the index, the status report, and the runbook. Everything else is detail behind it.

---

## 1. What this project is

Build a deterministic, headless, structured environment for fheroes2 **battles**, so a policy can be
trained on them. The environment reads **true engine state** and selects from **engine-generated
legal actions** — never pixels, never synthetic input.

The first deliverable is deliberately narrow: creature-only field battles, no heroes, no spells, no
castles, no adventure map. It must contain no LLM, no RL loop, no screenshot parsing, and no GUI
automation. Build a trustworthy substrate first.

**Not to be confused with** the `play-harness` branch, which is a *different* project on this same
repo: frame-dumping + an input FIFO so Claude can play the game through the real UI. That one is
pixels-and-GUI by design. Keep them separate. Engine modifications on this branch are minimal and
deliberate: the shared `Battle::computeBattleSeed` helper (Milestone 1, behavior-preserving —
digest-verified) and the entry-point-free `src/fheroes2/agent/` library that compiles into the
normal executable without changing it.

---

## 2. Status as of 2026-07-27

**Phase 0 and Milestone 1 are complete and verified on the target Mac mini M2. Milestone 2 is
next.**

- ✅ Spec written, source-cross-checked, then corrected against runtime evidence (v0.3)
- ✅ Phase 0 validated: 7/7 automated experiments pass, reproduced from a clean clone
- ✅ Reproduced on the target **Mac mini M2** (2026-07-27): 7/7 in both the `-O3` and the
  `FHEROES2_WITH_DEBUG` (`-O0 -g`) builds; map seed and digest identical to the M3 baseline,
  and identical between the two build types
- ✅ M2 benchmark measured and written up: `docs/agent/benchmark_m2.md` (Mode A only —
  ~4.6k eps/s tiny-melee, 12 MB RSS, multi-process sweet spot at 4 workers)
- ✅ CMake normal-game regression builds and launches on the mini; ASan+UBSan pass clean over
  1 900 episodes across five compositions (2026-07-27)
- ✅ **Milestone 1 complete** (2026-07-27): shared `Battle::computeBattleSeed` helper (extraction
  digest-verified, golden unit tests), agent library under `src/fheroes2/agent/` (scenario
  fixtures + validation, SHA-256 canonical terminal digest, headless AI-vs-AI runner), worker
  entry point in `src/agent_worker/`, and `agent_play/verify_m1.sh` — ten-run determinism holds
  for all five fixtures, cross-process output is byte-identical, both build systems still build
  the normal game
- ⚠️ One runbook item needs a human: a real Battle Only battle played through the UI (§6.3)
- ❌ Milestone 2 onward (decision hook, passive logging): **not started**

Branch: **`agent-env`**, branched from `master`, pushed to `origin` (the `troyzhu` fork).
Engine-source changes are limited to the Milestone 1 seed-helper extraction plus the additive
`src/fheroes2/agent/` library — enumerate with `git diff master --stat -- src/`.

---

## 3. The four findings that changed the plan

These overturned assumptions in the original spec. Do not re-derive them; do not undo them.

1. **Headless battles need no game assets at all.** No display, no audio, no AGG, no h2d, no HoMM2
   data. Monster stats are hardcoded (`monster_info.cpp:384`) and obstacle setup uses ICN ids as
   plain enum tags (`battle_board.cpp:573`), so battle resolution never touches an asset. This was
   the spec's #1 risk; it is closed.

2. **The world seed needs no engine change.** `Rand::CurrentThreadRandomDevice()` (`rand.cpp:85`)
   returns a *mutable reference* to a `thread_local PCG32`. Reseeding it before
   `world.generateBattleOnlyMap()` makes the map seed reproducible. The spec originally called for a
   new `World` API overload; that is now a deferred cleanup, not a prerequisite.
   Caveat: it pins *all* global randomness in the process. Fine for a dedicated worker, wrong for
   in-game use.

3. **A second entry point needs no CMake refactor.** Linking a new `main` against the game objects
   minus `fheroes2.o` works first try, no undefined symbols. The object-library refactor in spec §6.3
   is an optimization, not a requirement.

4. **This repo has two build systems.** The spec assumes CMake; there is also a plain Makefile under
   `src/dist`. CMake was absent on the dev machine, so all validation used the Makefile path.
   **Open decision:** whether the eventual agent target supports both. Settle it before Milestone 4.

Full evidence and a 25-row assumption table: `local_source_audit.md`.

---

## 4. Where everything is

| Path | What it is | Read when |
|---|---|---|
| `docs/agent/START_HERE.md` | this file | first |
| `agent_play/fheroes2_agent_system_spec_v0.3.md` §0.1 | validation results table — what changed and why | second |
| `docs/agent/local_source_audit.md` | Phase 0 report; assumption table; exact file:line evidence | before touching battle code |
| `agent_play/spike/README.md` | how to drive the spike, its limitations | before running it |
| `agent_play/fheroes2_agent_system_spec_v0.3.md` (rest) | the full 2600-line design: protocol, schemas, milestones | when implementing a milestone |

The spec is large. §0.1 (validation), §4 (scope), §9 (decision hook), §10 (legal actions) and §22
(definition of done) are the sections that matter for Milestone 1–3.

---

## 5. Get running (about 5 minutes)

```bash
# Toolchain — one time
xcode-select --install
brew bundle --file script/macos/Brewfile     # gettext, sdl2, sdl2_mixer, sdl2_image
brew install cmake                           # NOT in the Brewfile; only needed for the CMake path

# Source
git clone git@github.com:troyzhu/fheroes2.git && cd fheroes2
git switch agent-env

# Build and verify
make -C src/dist -j"$(sysctl -n hw.ncpu)"
./agent_play/spike/build_spike.sh
./agent_play/spike/verify_phase0.sh
./agent_play/verify_m1.sh                    # Milestone 1: unit tests + worker + determinism
```

**Expected: `7 passed, 0 failed`**, with map seed `2227197244` and melee digest `2cfd42cb104aa5e7`,
then `4 passed, 0 failed` from the Milestone 1 verification.

Those two values are machine-independent and have reproduced across three separate working trees.
**A mismatch on the mini is a real finding** — stop and investigate before building anything on top,
because it would mean determinism does not hold on the target hardware.

---

## 6. What to do next, in order

1. ✅ **Debug-build assertion run** — done 2026-07-27 on the M2: 7/7, no assertion fired, digest
   identical to Release. Correction discovered on the way: the `src/dist` Makefile never defines
   `-DNDEBUG`, so its `-O3` build already had `assert()` live — every Phase 0 run on both machines
   exercised the `ApplyAction` asserts. (The original rationale here — "Release compiles out the
   asserts" — is true only for CMake Release builds.) The `FHEROES2_WITH_DEBUG` run still added
   `-O0` codegen and `-DWITH_DEBUG` coverage.
2. ✅ **Re-measure on the mini** — done 2026-07-27: `docs/agent/benchmark_m2.md` (Mode A via
   `agent_play/spike/bench_m2.sh`). Headline: ~4.6k eps/s tiny-melee (M2 ≈ 9 % slower than M3),
   12 MB RSS, startup 10 ms, scaling linear to 4 workers then flat. Modes B/C blocked on M4–M5.
3. ◐ **CMake normal-game regression** — done 2026-07-27: `cmake -S . -B build/cmake-regression
   -DCMAKE_BUILD_TYPE=Release` configures and builds to 100 % on the mini, and both the CMake and
   the Makefile binaries launch to the main menu against the local `devdata/` root and survive
   12 s with clean logs. **Still open (needs a human at the screen): one real Battle Only battle
   played through the UI.** Launch: `FHEROES2_DATA="$PWD/devdata" ./src/dist/fheroes2/fheroes2`.
4. ✅ **Sanitizers** — done 2026-07-27: full `FHEROES2_WITH_ASAN=1` engine build (implies UBSan),
   spike relinked with matching flags (`build_spike.sh` now honors the same env vars). 1 900
   episodes across five compositions (mirror melee 50/1000, archer-vs-peasant, ranger duel,
   archer-vs-ranger): **zero ASan/UBSan reports**, all runs deterministic. ~700 eps/s, 207 MB RSS
   under instrumentation.
5. ✅ **Milestone 1** — done 2026-07-27, exit criterion met (`./agent_play/verify_m1.sh`: ten
   identical runs per fixture, cross-process byte-identical). The world-seed reseed (spec §7.2
   option 1) is used as planned; no `World` API overload was needed. `m1_tiny_melee` reproduces
   the historical map/combat seeds (2227197244 / 1356111745); note its terminal digest is the new
   canonical SHA-256 (`agent_terminal_v1`), intentionally not comparable to the spike's FNV fold.
6. **Next: Milestone 2** per spec §20 — optional `Battle::DecisionController` seam in
   `Arena::UnitTurn` (§9), typed `CommandSnapshot` (§10.5), passive built-in-AI trajectory
   logging. Exit: built-in behavior unchanged with a null controller (digest-verified) and
   passive logs replay deterministically. The §9 dispatch sketch and §9.3 invariants are the
   contract; read §3.6–§3.8 (turn dispatch, RNG stream semantics) before touching
   `battle_arena.cpp`.

---

## 7. Decisions already made — don't relitigate

- Agent work lives on `agent-env`, branched from `master`, **not** on top of `play-harness`. Both
  trees produce the identical battle digest, proving the harness patch is inert, but the baseline
  stays clean anyway.
- ~~The spike duplicates `computeBattleSeed` from `battle_main.cpp` verbatim.~~ Resolved in
  Milestone 1: the helper lives in `src/fheroes2/battle/battle_seed.{h,cpp}`, used by the engine,
  the spike and the agent runner, with golden-value tests in `agent_play/tests/`.
- The spike's digest is an FNV fold, not SHA-256. Enough to detect divergence; the real environment
  needs the canonical digest in spec §12.5.
- Baseline is the current `master` lineage, **not** the `1.1.17` tag the spec originally pinned. The
  tag is 42 commits behind and every spec-critical battle file is byte-identical, so pinning would
  fork the work off the live branch for nothing.

---

## 8. Gotchas that will bite

- **A `FHEROES2_DATA` root needs the repo's own h2d bundle, not just the GOG extraction.**
  Non-bundle builds resolve `files/data/resurrection.h2d` against each data root
  (`h2d.cpp:110`) and throw at startup if it is nowhere. A root made only of extracted HoMM2
  assets fails ~9 s in, after the intro. Fix: `mkdir -p "$FHEROES2_DATA/files/data" && cp
  files/data/resurrection.h2d "$FHEROES2_DATA/files/data/"`. (Running the repo-root `./fheroes2`
  copy masks this, because the executable-directory search root then contains `files/data/`.)
- **Repo paths with spaces.** The Mac mini's clone lives at `/Volumes/External Drive/…`.
  `build_spike.sh` originally kept its `-I` flags in a whitespace-joined string and broke there;
  it now uses bash arrays. Any future build script must pass flag lists as arrays
  (`"${FLAGS[@]}"`), never as unquoted strings.
- **The Makefile build has no `-DNDEBUG`.** `assert()` is live even in the `-O3` "Release"
  `src/dist` build. Do not reason from CMake habits: a CMake `Release` build *does* define
  `NDEBUG` and strips asserts. Benchmark numbers from the two systems are not interchangeable.
- **`make -C src/dist clean` after every upstream sync.** The `-MD` depfiles hard-code header paths,
  so an upstream header rename breaks incremental builds with `No rule to make target '.../x.h'`.
  This is a stale-depfile artifact, never a code fault.
- **`xgettext` must be the real one.** On the dev machine it resolved to a pyenv shim and happened to
  work. If the Makefile build fails in the `.pot` step, put Homebrew's `gettext` ahead of pyenv on
  `PATH`.
- **One arena per process.** `battle_arena.cpp:73` holds a file-static pointer and the constructor
  asserts it is null. Every episode must destroy its arena before the next starts. Parallelism means
  multiple *processes*.
- **Input `Army` objects are not synced after battle.** Terminal state must be read from the `Force`
  objects *before* the arena is destroyed, or you get the pre-battle numbers back.

---

## 9. The real remaining risk

With the asset and seed risks closed, the top risk is now **legal-action generation** (spec §3.9 and
§10). The engine has no public API returning the complete legal action set for a unit, and the
validation helpers live in anonymous namespaces in `battle_action.cpp`. Enumerating legal actions
without either extracting those helpers or duplicating battle rules incorrectly is the hard part of
this project. Budget accordingly, and do not test legality by applying candidates to the live arena.
