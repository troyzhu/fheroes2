# fheroes2 Agent Local Source Audit

Fills the template in Appendix D of `agent_play/fheroes2_agent_system_spec_v0.3.md`.
This is the Phase 0 report the spec makes mandatory before broader implementation.

> **Scope caveat, stated up front:** this audit was produced on a **MacBook-class M3, not the
> target Mac mini M2**. Everything below that is machine-independent (source facts, determinism,
> lifecycle, API shapes) transfers. Everything that is a *measurement* (throughput, memory,
> worker count) must be re-run on the Mac mini before it is quoted as a baseline.

---

## Environment

| Field | Value |
|---|---|
| Date | 2026-07-26 |
| Machine | `Mac15,12` (Apple M3, 8 cores) |
| RAM | 16 GB |
| macOS | 26.5.2 |
| Compiler | Apple clang 21.0.0 (clang-2100.1.1.101) |
| CMake | **not installed** — see "Build-target decision" |
| Make | GNU Make 3.81 |
| SDL | 2.32.10 (Homebrew), SDL2_mixer present |
| fheroes2 commit | `336dde2f6` (branch `agent-env`, branched clean from `master`) |
| Dirty tree | only the new `agent_play/` and `docs/agent/` trees — **no engine source is modified** |

The agent work is deliberately kept on a branch taken straight from `master`, *not* on top of the
unrelated `play-harness` engine patch (screenshot dumping + input FIFO). Two reasons: the spec's §4.5
non-goals exclude GUI automation from this deliverable, and a determinism baseline should not carry
an unrelated engine modification.

Phase 0 was first executed on the `play-harness` tree and then re-run in full on this clean branch.
Both produce the **identical** terminal digest `2cfd42cb104aa5e7`, which independently confirms the
harness patch is inert with respect to battle resolution.

### Baseline commit deviation (important)

The spec pins `2685c2188b541660f1ce261b554c3e92f79b1775` (tag `1.1.17`). That commit is real and the
tag resolves to it. **The working tree is 42 commits past it.** Rather than checking out the tag,
the drift was measured directly, because the audit only matters for spec-critical files:

| File | Tag → HEAD |
|---|---|
| `CMakeLists.txt` | unchanged |
| `src/fheroes2/CMakeLists.txt` | unchanged |
| `src/fheroes2/battle/battle_arena.cpp` | unchanged |
| `src/fheroes2/battle/battle_main.cpp` | unchanged |
| `src/fheroes2/battle/battle_action.cpp` | unchanged |
| `src/fheroes2/battle/battle_command.cpp` | unchanged |
| `src/fheroes2/world/world.h` | unchanged |
| `src/fheroes2/world/world.cpp` | 26+/20− — **none of it in `Defaults()`, `generateBattleOnlyMap()`, or `generateUninitializedMap()`** |
| `src/fheroes2/battle/battle_only.cpp` | 9+/9− (UI-level) |

**Conclusion:** the spec's line-level audit is still valid at HEAD. Pinning to the `1.1.17` tag is
unnecessary and would fork the agent work off the branch the repo actually uses. Recommend
re-pinning the baseline to the current master lineage and recording the drift table above instead.

---

## Resource setup

**No game resources were needed at all.** This is the headline finding.

- Source of game/demo resources: **none used**
- Paths: none
- Environment variables: none (`HOME` was redirected only to keep config files out of the real home)
- Display/audio requirements: **none** — no `SDL_VIDEODRIVER=dummy`, no audio device, no window

The headless smoke binary never calls `fheroes2::Display::instance()`, `AGG::AGGInitializer`,
`h2d::H2DInitializer`, `Assets::getImage`, or `Audio::`. It runs on a machine with no game data
present.

Two source facts explain why this works, and both were verified by reading the code:

1. **Monster stats are hardcoded, not asset-derived.** `monsterData` in
   `src/fheroes2/monster/monster_info.cpp:384` is built from in-source `monsterBattleStats[]` /
   `monsterGeneralStats[]` arrays. The `icnId` and `binFileName` members are only rendering/animation
   handles.
2. **Battlefield obstacle setup uses ICN ids as enum tags, not images.**
   `Battle::Board::SetCovrObjects` (`battle_board.cpp:573`) is a pure `switch` over `ICN::COVR*`
   constants that calls `at(n).SetObject(...)`. No image is ever loaded.

---

## Baseline build

### CMake path — blocked on this machine

CMake is not installed. The spec's §2.2 commands could not run. The repo, however, ships a
**second, complete build system** the spec does not mention: a plain recursive Makefile under
`src/dist`, driven by the root `Makefile`. That is what was used:

```bash
make -C src/dist clean
make -C src/dist -j10        # exit 0
```

One real gotcha, recorded because it will recur on the Mac mini: the `-MD` generated `.d` files
hard-code header paths, so any upstream header rename breaks an *incremental* build with
`No rule to make target '.../agg_image.h'`. `make -C src/dist clean` first is mandatory after
syncing upstream. This is a stale-depfile artifact, not a code problem.

- Normal game run: not re-verified in this session (the same tree built and ran previously).
- Battle Only run: **not performed** — see "Go/no-go".

---

## Headless smoke

Artifacts:

- `agent_play/spike/smoke_battle.cpp` — the runner
- `agent_play/spike/build_spike.sh` — build script

### Initialization sequence that actually works

Far shorter than the spec's 17-step §8.1 list. This is the whole of it:

```cpp
Logging::InitLog();

Rand::CurrentThreadRandomDevice() = Rand::PCG32( worldSeed );   // determinism, see below
world.generateBattleOnlyMap( Maps::Ground::GRASS );

Settings & conf = Settings::Get();
conf.GetPlayers().Init( BLUE | RED );
world.InitKingdoms();
Players::SetPlayerRace( BLUE, Race::KNGT );   Players::SetPlayerControl( BLUE, CONTROL_AI );
Players::SetPlayerRace( RED,  Race::KNGT );   Players::SetPlayerControl( RED,  CONTROL_AI );

Army attacker; attacker.Reset( false ); attacker.SetColor( BLUE );
attacker.GetTroop( 0 )->Set( Monster( id ), count );
// ... same for defender ...

const uint32_t combatSeed = computeBattleSeed( 1, world.GetMapSeed(), attacker, defender );
Rand::PCG32 gen( combatSeed );
{
    Battle::Arena arena( attacker, defender, 1, /*isShowInterface=*/false, gen );
    while ( arena.BattleValid() && rounds < cap ) { arena.Turns(); ++rounds; }
    // read terminal state HERE, before the arena is destroyed
}
```

No `Game::Init()`, no display, no palette, no assets.

### Results

| Scenario | Rounds | Winner | Terminal state |
|---|---|---|---|
| Peasant 50 vs Peasant 50 | 3 | attacker | a: 1 stack / 50 creatures / 50 HP; d: wiped |
| Archer 20 vs Peasant 60 | 1 | attacker | a: 1 stack / 20 creatures / 200 HP; d: wiped |
| Peasant 10 vs Archer 30 | 1 | defender | a: wiped; d: 1 stack / 30 creatures / 300 HP |

- Effective world seed: `20260726` (input) → map seed `2227197244` (derived, stable)
- Effective combat seed: varies correctly with army composition (`1356111745`, `1381489788`, `1164091793`)
- UI opened: **no**
- Errors/warnings: none

---

## Determinism

| Test | Runs | Distinct digests | Verdict |
|---|---|---|---|
| Same process, global RNG reseeded | 10 | 1 | **deterministic** |
| Same process, global RNG *not* reseeded | 10 | 10 | non-deterministic (as predicted) |
| Same process, 100 episodes | 100 | 1 | deterministic |
| Same process, 2000 episodes | 2000 | 1 | deterministic |
| Two fresh processes | 2 | 1 | **cross-process deterministic** |

The digest folds map seed, combat seed, round count, both result words, per-side live stacks /
creatures / HP, and an order-sensitive fold over every unit's `(UID, monster id, count, hit points,
head cell, validity)`.

### The significant finding: no engine change is required for a deterministic world seed

The spec is correct that `World::Defaults()` randomizes the map seed
(`world.cpp:291`, `_seed = Rand::Get( ... )`) and that no public setter exists — `grep` for
`SetMapSeed` returns nothing repo-wide.

But the spec's conclusion — that a new `generateBattleOnlyMap(groundType, mapSeed)` overload plus
`Defaults(std::optional<uint32_t>)` is *required* — does not hold. `Rand::Get()` draws from
`Rand::CurrentThreadRandomDevice()` (`rand.cpp:85`), which returns a **non-const reference** to a
`thread_local PCG32`. Assigning to it makes the map seed fully reproducible:

```cpp
Rand::CurrentThreadRandomDevice() = Rand::PCG32( worldSeed );
```

Verified: 10/10 identical with it, 10/10 divergent without it, and reproducible across separate
processes. **Milestone 1 can therefore ship with zero engine modifications.**

Tradeoff to record honestly: this is a blunter instrument than the proposed overload. It pins
*all* global randomness in the process, not just the map seed. For a dedicated single-purpose
worker process that is arguably a feature. It would be wrong for in-game use. Recommendation:
adopt the reseed for Phase 0/Milestone 1, and treat the narrow engine overload as a later
cleanup rather than a blocking prerequisite.

---

## Sequential lifecycle

| Metric | Value |
|---|---|
| Episodes in one process | 2000, no crash / assertion / degradation |
| Peak RSS | **12.9 MB** at 500 episodes |
| Wall clock | 2000 episodes in 0.40 s ≈ **5000 episodes/s** |
| Arena singleton | respected — each episode scopes its arena so `~Arena()` clears the file-static pointer |

The spec's 100-sequential-arena requirement passes by a factor of 20.

**Do not quote the throughput number as a baseline.** It is a 1–3 round, single-stack, no-protocol,
no-JSON battle on an M3. It establishes only that the engine is not the bottleneck and that per-episode
teardown does not leak. Real numbers must come from the Mac mini with the protocol layer attached.

---

## Source assumptions

| Assumption (spec §) | Confirmed | Evidence / change |
|---|---|---|
| CMake 3.24, C++17, `GLOB_RECURSE` game sources (§3.1) | ✅ | `CMakeLists.txt:21,28`; `src/fheroes2/CMakeLists.txt:21` |
| Worker `main` must sit outside the glob (§3.1) | ✅ | Confirmed, and validated in practice by excluding `fheroes2.o` at link time |
| `StartBattleOnly` → `generateBattleOnlyMap`, tile index 1 (§3.2) | ✅ | `game_startgame.cpp:215` |
| `generateUninitializedMap(2)` → `Defaults()` → random `_seed` (§3.3) | ✅ | `world.cpp:340,284,291` |
| No public map-seed setter (§3.3) | ✅ | repo-wide grep empty |
| A new engine seed API is **required** (§3.3, §7.2) | ❌ **refuted** | `Rand::CurrentThreadRandomDevice()` is a mutable ref (`rand.cpp:85`); verified deterministic |
| Combat seed = tile + mapSeed, folding all slots (§3.4) | ✅ | `battle_main.cpp:134-161`; duplicated in the spike and it reproduces engine behaviour |
| Seed loop folds "five" slots (§3.4) | ⚠️ nuance | loop is over `Army::Size()`, not a literal 5; `Army::maximumTroopCount = 5` (`army.h:164`) makes it 5 in practice |
| Arena ctor signature (§3.5) | ✅ | `battle_arena.h:97` |
| One arena per process, file-static + assert (§3.5) | ✅ | `battle_arena.cpp:73,353,456` |
| Four branches in `UnitTurn`; AI call in the 4th (§3.6) | ✅ | `battle_arena.cpp:480,488,496,499`; AI at `:507-508` |
| Observer window before stream update / apply (§9.2) | ✅ | `BattleTurn` `:508` → `updatePCG32Stream` `:517` → `ApplyAction` `:522` |
| `Turns()` advances a whole round (§3.7) | ✅ | `battle_arena.cpp:552`, returns `void` |
| Commands store params reversed, popped from back (§3.8) | ✅ | `operator<<` push_back; `operator>>` `val = back(); pop_back();` (`battle_command.cpp:85-93`) |
| Decode a copy via `GetNextValue()` (§10.5) | ✅ viable | `GetNextValue()` is **public**; `operator>>` is private; `Command` derives from `std::vector<int>` so it copies |
| ATTACK stream ignores target cell + direction (§3.8) | ✅ | `battle_command.cpp:42-51`, uses only `at(2),at(3),at(4)` — source comment says exactly why |
| `arena.getAllAvailableMoves()` exists and is public (§10.3) | ✅ | `battle_arena.h:192` (public region) |
| Board is 11×9 = 99 (§12.2) | ✅ | `battle_board.h:73-77` — names are `Board::widthInCells` / `heightInCells` / `sizeInCells` |
| Both-AI commander-free armies take the AI branch (§8.2) | ✅ | `Army::GetControl()` (`army.cpp:1389`) → `Players::GetPlayerControl(color)`; battles resolve, so the branch is live |
| Input armies are **not** synced post-battle (§8.4) | ✅ | spike reports `army_synced=no`; terminal state must be read from `Force` before destruction |
| Headless needs no display/audio/AGG (§3.10 LOCAL-VERIFY) | ✅ **resolved** | ran with zero game resources |
| Obstacles derive from world seed, separate RNG (§7.1) | ✅ + detail | `battle_arena.cpp:427` — `Rand::PCG32( world.GetMapSeed() + tileIndex )`, wholly independent of the combat RNG |
| `AI::BattlePlanner` reset at arena construction (§8.3) | ✅ | `battle_arena.cpp:438`, `battleBegins()` |

---

## simple_v1 capability decision

Not yet decided — this is Phase 0 only. Established so far:

- Monster ids are dense from `Monster::UNKNOWN = 0`: `PEASANT = 1`, `ARCHER = 2`, `RANGER = 3`, …
  (`monster.h:54-62`).
- Peasant (pure melee) and Archer (shooter) both run clean end-to-end headless, which makes them
  usable as the §4.3 smoke fixtures.
- `RANGER` already carries `MonsterAbilityType::DOUBLE_SHOOTING` (`monster_info.cpp:391`), a concrete
  example of the ability data the §4.2 capability audit must be generated from rather than hand-listed.

Rejected abilities, provisional supported-id list, and teacher-coverage fixtures: **still open.**

---

## Build-target decision

**Chosen for the spike:** compile only the spike TU and relink it against the existing
`src/dist/fheroes2/*.o`, excluding `fheroes2.o` (the TU holding the game's `main`).

```
[1/2] compiling smoke_battle.cpp
[2/2] linking 220 game objects (fheroes2.o excluded) -> smoke_battle
```

**Why it matters beyond convenience:** it linked on the first attempt with no undefined symbols.
That empirically proves the game's non-entry object set is already usable as a library for an
external `main`, which is the exact premise of the spec's §6.3 object-library refactor — without
having to perform that refactor to find out. The spec's §6.4 "compile everything twice" fallback is
not needed either; a relink is enough.

Normal-game regression: not re-run under CMake (unavailable). The Makefile build of the normal
executable succeeded and is unaffected — the spike adds no files under `src/`.

---

## Deviations from v0.2

1. Baseline is current master lineage, not the `1.1.17` tag; drift table above shows why that is safe.
2. Build performed with the `src/dist` Makefile, not CMake.
3. Deterministic world seed obtained by reseeding the thread-local RNG instead of adding an engine API.
4. `computeBattleSeed` duplicated verbatim in the spike rather than extracted — deliberate, so the
   spike could stay a pure add-on. Extraction (§7.3) remains the right production move.

No ADRs written yet; these belong in `docs/agent/decisions/` once the direction is accepted.

---

## Go/no-go

**GO** for Milestone 1, with the two highest risks in the register materially reduced.

Passed (spec §2.4):

- ✅ Headless startup — no UI
- ✅ Asset dependence — none required
- ✅ Display/audio dependence — none required
- ✅ Deterministic duplicate — 10/10, and 2000/2000
- ✅ Sequential reuse — 2000 arenas, one process, 12.9 MB peak
- ✅ Seed visibility — map and combat seeds logged and stable

Not yet done, carried into Milestone 1:

- ❌ **Debug-build assertion run.** The whole spike ran Release. `UnitTurn`'s non-AI path asserts
  `_interface != nullptr`, and `ApplyAction` validates in asserts — so a Debug run is the real test
  that no invalid command is being produced. Deliberately skipped here because rebuilding with
  `FHEROES2_WITH_DEBUG` would overwrite the Release objects that the user's play-harness binary
  and this spike are linked from.
- ❌ Normal-game regression under CMake (CMake absent on this machine).
- ❌ Battle Only run through the real UI.
- ❌ Sanitizers (ASan/UBSan).
- ❌ Anything measured on the actual Mac mini M2.

Blocking issues: none.
