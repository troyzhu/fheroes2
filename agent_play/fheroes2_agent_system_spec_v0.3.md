# fheroes2 Agent System — Source-Cross-Checked Implementation Specification

> **Status:** v0.3 — source-cross-checked **and Phase 0 runtime-validated**  
> **Original draft:** 2026-07-25  
> **Source audit revision:** 2026-07-25  
> **Runtime validation revision:** 2026-07-26 — see §0.1 and `agent_play/docs/archive/benchmarks/2026-07-26-source-audit-apple-m3.md`  
> **Primary target:** Apple Silicon Mac mini M2; assume 16 GB unified memory until the local machine is measured  
> **Validation machine:** Apple M3, 8 cores, 16 GB, macOS 26.5.2 — **not the target Mac mini.** Source-level and determinism findings transfer; all *measurements* must be re-taken on the mini.  
> **Pinned engine baseline:** see §0.1 — the `1.1.17` pin is superseded by the current master lineage  
> **Audience:** a local coding agent or engineer with an IDE, terminal, compiler, game resources, and permission to modify a fork  
> **First deliverable:** a deterministic, headless, structured environment for a deliberately restricted class of creature-only field battles

---

## 0. Document status and how to use it

This is an implementation handoff, but it is not a claim that the proposed patch has already been built or executed.

The source-level conclusions below were cross-checked against files at the pinned commit. The environment used to prepare this revision could inspect exact-commit source files but could not complete a local checkout and executable run. The local implementation agent must therefore start with the mandatory **Phase 0 runtime spike** in Section 2.

Throughout this document:

- **CONFIRMED-SOURCE** means the behavior or interface is directly visible in the pinned source.
- **PROPOSED** means the document recommends a design that is not presently in fheroes2.
- **LOCAL-VERIFY** means source inspection is insufficient; the local agent must build or run the engine.
- **DEFERRED** means deliberately outside the first mergeable implementation.

The local agent may refine this specification when runtime evidence contradicts it. Any meaningful change should be recorded in `agent_play/docs/decisions/` rather than silently broadening or changing scope.

The first mergeable result must not contain an LLM, RL training loop, whole-map agent, screenshot parser, or GUI automation. It must first create a trustworthy environment substrate.

---

## 0.1 Validation results (v0.3)

Phase 0 was executed. A headless creature-only battle was built and run against this repository.
Full report: **`agent_play/docs/archive/benchmarks/2026-07-26-source-audit-apple-m3.md`**. Artifacts: `agent_play/spike/`.

**Phase 0 verdict: GO.** The two highest-ranked risks in §23 are materially reduced.

### What was confirmed

Every `CONFIRMED-SOURCE` claim in §3 held, with the small corrections listed below. Specifically
confirmed by build-and-run, not just by reading: the arena constructs and runs headless; `UnitTurn`'s
four branches and the AI dispatch site; the combat-seed formula (a duplicated implementation
reproduces engine behaviour); the single-arena file-static invariant; and that input `Army` objects
are not synchronized post-battle.

### What was refuted or changed

| # | v0.2 claim | v0.3 status |
|---|---|---|
| 1 | A new engine seed API is **required** for a deterministic world (§3.3, §7.2) | **Refuted.** `Rand::CurrentThreadRandomDevice()` (`rand.cpp:85`) returns a mutable reference to a `thread_local PCG32`. Reseeding it makes the map seed reproducible with **zero engine changes**. Verified 10/10 identical, 10/10 divergent without, and reproducible across processes. |
| 2 | Headless init needs unknown display/audio/AGG setup (§3.10, `LOCAL-VERIFY`) | **Resolved: none needed.** The smoke ran with no display, no audio, no AGG, no h2d, and **no game resources of any kind**. Monster stats are hardcoded (`monster_info.cpp:384`); obstacle setup uses ICN ids as enum tags only (`battle_board.cpp:573`). |
| 3 | Pin to tag `1.1.17` = `2685c2188…` (§2.1) | **Superseded.** The tag is real, but the repo is 42 commits past it and every spec-critical battle file is byte-identical between the two. Pinning would fork the work off the live branch for no benefit. Re-pin to master lineage; keep the drift table in the audit. |
| 4 | Build via CMake (§2.2) | **Incomplete.** CMake is not installed on the validation machine. The repo ships a second, complete build system the spec never mentions — a plain Makefile under `src/dist`. §6 must define the agent target for **both**, or declare the Makefile path unsupported. |
| 5 | Object-library refactor or double-compile needed for a second entry point (§6.3, §6.4) | **Cheaper path proven.** Linking a new `main` against the existing game objects minus `fheroes2.o` succeeded on the first attempt, no undefined symbols, 220 objects. The non-entry object set is already library-shaped; the refactor is an optimization, not a prerequisite. |
| 6 | Seed folds "the five" army slots (§3.4) | **Nuance.** The loop is over `Army::Size()`, not a literal 5. `Army::maximumTroopCount = 5` makes it 5 in practice, but the shared helper (§7.3) must iterate `Size()`. |
| 7 | Board constants (§12.2) | **Naming.** 11×9=99 is correct; the real identifiers are `Board::widthInCells` / `heightInCells` / `sizeInCells` (`battle_board.h:73-77`). |
| 8 | `Battle::Command` decoding (§10.5) | **Viable, with a caveat.** `operator>>` is **private**; `GetNextValue()` is public and `Command` derives from `std::vector<int>`, so the copy-then-decode plan works. Do not plan on `operator>>`. |

### Measured on the M3 (indicative only — re-measure on the Mac mini)

| Metric | Value |
|---|---|
| Sequential arenas in one process | 2000, zero failures |
| Determinism | 2000/2000 identical digests; identical across fresh processes |
| Throughput | ~5000 episodes/s (1–3 round single-stack battle, no protocol layer) |
| Peak RSS | 12.9 MB |

The throughput figure is **not** a baseline. It has no JSON, no protocol, no Python, and a trivial
battle. Its only real content is that the engine is not the bottleneck and teardown does not leak —
which is what §19 asks Phase 0 to establish before setting any target.

### Still open after Phase 0

Debug-build assertion run (the spike was Release-only); sanitizers; normal-game regression under
CMake; a real Battle Only UI run; and every measurement on the actual Mac mini M2.

---

## 1. Executive decision

Build a hierarchical agent eventually, but begin with the smallest engine-native environment that exercises the real turn loop:

```text
Future pretrained strategic model
        │
        │ ranks engine-generated macro-actions
        ▼
Python policy and training runtime
        │
        │ selects an ephemeral legal-action ID
        ▼
Out-of-process fheroes2 agent worker
        │
        ├── synchronous decision hook
        ├── observation serializer
        ├── legal-action generator
        ├── deterministic episode runner
        └── unmodified battle mechanics
```

The first implementation slice is:

> A deterministic, headless, creature-only field-battle worker in which fheroes2 continues to schedule turns and apply commands, while an external policy may replace the built-in AI at selected full-fledged unit decisions.

The main design choices are:

1. **Use true engine state, not pixels.**
2. **Keep the worker out of process.** One worker owns one engine process and runs one arena at a time.
3. **Let the engine generate and retain legal commands.** Python selects an action ID; it does not synthesize `Battle::Command`.
4. **Use a synchronous blocking hook in `Battle::Arena::UnitTurn`.** No worker-side threads are required for the first version.
5. **Reuse the engine’s Battle Only world path and battle-seed formula.**
6. **Restrict the first action profile.** Phase 1a supports only audited simple creatures and canonical `SKIP`, `MOVE`, simple ranged `ATTACK`, and simple melee `ATTACK`.
7. **Use the built-in tactical AI as the first teacher and baseline.**
8. **Treat determinism as an explicit contract with state-digest replay tests.**

This is an achievable systems milestone on the Mac mini. Model training should begin only after the environment passes the definition of done in Section 22.

---

## 2. Mandatory Phase 0: local source and runtime spike

**LOCAL-VERIFY — this phase must precede the larger implementation.**

The local agent must first prove that the exact pinned engine can be built and that a headless creature-only battle can be constructed repeatedly in the intended environment.

### 2.1 Checkout and provenance

**v0.3 — revised.** Phase 0 has been executed; this subsection is retained for the Mac mini re-run.

Do **not** check out the `1.1.17` tag. The tag is genuine (`2685c2188b541660f1ce261b554c3e92f79b1775`),
but the working repository is 42 commits past it, and every file this document audits at line level
is byte-identical between the two:

```bash
# Verify the audit still applies at whatever commit you are on:
for f in CMakeLists.txt src/fheroes2/CMakeLists.txt \
         src/fheroes2/battle/battle_arena.cpp src/fheroes2/battle/battle_main.cpp \
         src/fheroes2/battle/battle_action.cpp src/fheroes2/battle/battle_command.cpp \
         src/fheroes2/world/world.h src/fheroes2/world/world.cpp; do
  echo "$(git diff --numstat 2685c2188b541660f1ce261b554c3e92f79b1775..HEAD -- "$f" | awk '{print $1"+/"$2"-"}')  $f"
done
```

At the time of validation only `world.cpp` differed (26+/20−), and none of it touched `Defaults()`,
`generateBattleOnlyMap()`, or `generateUninitializedMap()`. Record the table; if a battle file starts
showing changes, re-audit that file before trusting §3.

Create:

```text
agent_play/docs/local_source_audit.md
```

Record:

- operating system and version;
- Mac model, CPU, RAM;
- compiler and version;
- CMake version;
- SDL package versions;
- checked-out commit;
- build commands;
- location and provenance of required game/demo resources;
- whether a display or audio device was required;
- every source assumption that failed.

### 2.2 Clean baseline build

**v0.3 — revised.** This repository has **two** build systems. The spec originally assumed only CMake.

**Path A — CMake** (preferred where available; requires CMake ≥ 3.24):

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DMACOS_APP_BUNDLE=OFF

cmake --build build-release -j
```

**Path B — plain Makefile** (`src/dist`; used for the Phase 0 validation because CMake was absent):

```bash
make -C src/dist clean      # see the gotcha below — do not skip
make -C src/dist -j10       # -> src/dist/fheroes2/fheroes2
```

> **Gotcha, will recur:** the Makefile build emits `-MD` dependency files that hard-code header
> paths. After syncing upstream, any header rename breaks the *incremental* build with
> `No rule to make target '.../<old_header>.h'`. Always `clean` first after a sync. This is a stale
> depfile artifact, never a code fault.

§6.2's `ENABLE_AGENT` option covers Path A only. Before Milestone 4, decide explicitly whether the
worker target is also wired into Path B or whether Path B is declared unsupported for agent builds.
Leaving this undecided will strand whichever machine lacks CMake.

Run the ordinary executable and enter Battle Only once. This confirms that the local resources and standard initialization path work before agent changes are introduced.

### 2.3 Minimal headless battle smoke target

Before implementing JSON, Python, candidate enumeration, or a reusable worker, add a temporary or narrowly scoped smoke executable that:

1. performs the minimum engine/resource initialization established from the normal entry point;
2. calls `world.generateBattleOnlyMap(explicitTerrain)`;
3. forces a known map seed through the narrow seed API proposed in Section 7;
4. initializes BLUE and RED players/kingdoms with AI control;
5. constructs two fresh commander-free `Army` objects;
6. populates one simple stack on each side;
7. computes the engine-compatible combat seed;
8. constructs `Battle::Arena(attacker, defender, 1, false, randomGenerator)`;
9. runs `arena.Turns()` until terminal or a hard round cap;
10. records winner, round count, final stacks, effective map seed, and effective combat seed;
11. destroys the arena cleanly.

Use an explicit terrain such as GRASS. Do not pass `Maps::Ground::UNKNOWN`, because Battle Only resolves UNKNOWN through global randomness.

### 2.4 Required spike experiments

The local report must answer these questions:

**v0.3 — results filled in.** Six of eight pass; two remain open.

| Experiment | Pass condition | Result (M3, 2026-07-26) |
|---|---|---|
| Headless startup | The smoke battle runs without opening the battle UI. | ✅ **PASS** — no window, no UI |
| Asset dependence | Exact required AGG/demo/full-game resources are documented. | ✅ **PASS — none required.** Ran with zero game data |
| Display/audio dependence | Document whether SDL dummy video/audio drivers are required. | ✅ **PASS — neither required.** No `SDL_VIDEODRIVER=dummy`, no audio device |
| Deterministic duplicate | Ten identical runs produce the same result and canonical terminal digest. | ✅ **PASS** — 10/10, and 2000/2000 |
| Sequential reuse | One process runs at least 100 fresh arenas sequentially without assertion, crash, or growing live arena state. | ✅ **PASS** — 2000 arenas, 12.9 MB peak RSS |
| Debug behavior | Debug build completes representative battles without invalid-command assertions. | ❌ **NOT RUN** — spike was Release-only; rebuilding Debug would have clobbered the Release objects in use. Carry to Mac mini |
| Normal game regression | The standard `fheroes2` executable still starts and Battle Only still works. | ⚠️ **PARTIAL** — Makefile build of the normal executable succeeds and the spike adds nothing under `src/`; CMake regression and a real Battle Only run not performed |
| Seed visibility | Logged map and combat seeds match the values used by obstacle and combat randomization. | ✅ **PASS** — map seed stable under a fixed world seed; combat seed varies correctly with army composition |

### 2.5 Phase 0 decision gate

Proceed only when the local report identifies a viable initialization sequence.

If direct `Battle::Arena` construction proves impractical because of hidden resource or singleton requirements, the fallback is not GUI automation. The fallback is to refactor the existing `Battle::Loader`/Battle Only path into a reusable runner while preserving its behavior.

---

## 3. Cross-checked source baseline

### 3.1 Build topology

**CONFIRMED-SOURCE**

The root project requires CMake 3.24, selects C++17, and exposes options including `ENABLE_TOOLS`, `MACOS_APP_BUNDLE`, and `GET_HOMM2_DEMO`.

The important build constraint is in `src/fheroes2/CMakeLists.txt`:

```cmake
file(GLOB_RECURSE FHEROES2_SOURCES CONFIGURE_DEPENDS *.cpp)
add_executable(fheroes2 ... ${FHEROES2_SOURCES} ...)
```

Consequences:

- A second `main.cpp` placed anywhere below `src/fheroes2/` would be globbed into the ordinary executable.
- The worker entry point must live outside that glob, or the source list must be made explicit.
- The current game source is not already packaged as a reusable library.
- Existing `src/tools` targets link primarily against `engine`; an agent worker needs much of the `src/fheroes2` game code and is not equivalent to the current lightweight tools.

### 3.2 Battle Only world construction

**CONFIRMED-SOURCE**

`Game::StartBattleOnly()` calls:

```cpp
world.generateBattleOnlyMap( battleOnlySetup.terrainType() );
```

and uses tile index `1` when starting the battle.

`World::generateBattleOnlyMap(groundType)`:

1. calls `generateUninitializedMap(2)`;
2. assigns indices to the four map tiles;
3. applies a uniform terrain.

This is the correct synthetic-world path to reuse. The worker should not invent a fake `Maps::Tile` or bypass the global `World`.

### 3.3 Hidden map randomness

**CONFIRMED-SOURCE**

`generateUninitializedMap(2)` calls `World::Defaults()`. `Defaults()` sets:

```cpp
_seed = Rand::Get( std::numeric_limits<uint32_t>::max() );
```

The public header exposes `GetMapSeed()` but no setter — a repo-wide grep for `SetMapSeed` returns nothing. Therefore, the v0.1 assumption that a scenario could simply supply an independent `map_seed` was incorrect.

**v0.3 — CONFIRMED-RUNTIME, but the prescribed fix is no longer required.**

`Rand::Get()` draws from `Rand::CurrentThreadRandomDevice()` (`src/engine/rand.cpp:85`), which returns a **non-const reference** to a `thread_local PCG32`:

```cpp
Rand::PCG32 & Rand::CurrentThreadRandomDevice()
{
    thread_local std::random_device rd;
    thread_local PCG32 gen( rd );

    return gen;
}
```

So the map seed can be pinned from outside the engine, before calling into it:

```cpp
Rand::CurrentThreadRandomDevice() = Rand::PCG32( worldSeed );
world.generateBattleOnlyMap( Maps::Ground::GRASS );
// world.GetMapSeed() is now a pure function of worldSeed
```

Measured: 10/10 runs identical with the reseed, 10/10 divergent without it, and identical across
separate processes. **Milestone 1 no longer depends on an engine patch.**

The tradeoff must be recorded rather than hidden: this pins *all* global randomness in the process,
not just the map seed. In a dedicated single-purpose worker that is acceptable and arguably
desirable; it would be wrong inside the real game. See the revised §7.2.

### 3.4 Battle seed derivation

**CONFIRMED-SOURCE**

`battle_main.cpp` computes a combat seed by:

1. starting from `uint32_t(tileIndex) + mapSeed`;
2. folding each of the five attacking army slots in order;
3. folding each of the five defending army slots in order;
4. folding monster ID and count for valid slots, and zero for invalid slots.

`Battle::Loader` then creates `Rand::PCG32(battleSeed)` and constructs the arena.

Therefore:

- map/obstacle randomness and combat randomness are related but distinct;
- the scenario should normally provide a deterministic **world seed**, not a free-standing combat seed;
- the worker should call the same extracted seed helper as `Battle::Loader`;
- empty slot positions are part of the seed contract.

### 3.5 Arena construction and process-level singleton

**CONFIRMED-SOURCE**

`Battle::Arena` accepts:

```cpp
Arena(
    Army & attackingArmy,
    Army & defendingArmy,
    int32_t tileIndex,
    bool isShowInterface,
    Rand::PCG32 & randomGenerator );
```

When `isShowInterface == false`, no battle interface is created. The constructor still relies on the global `world`, reads the map tile/castle context, constructs battle `Force` objects, and initializes the board.

`battle_arena.cpp` also contains a file-static `Battle::Arena * arena`. Construction asserts that no arena already exists; destruction clears the pointer.

Consequences:

- only one live arena is supported per process;
- worker parallelism must use multiple processes, not multiple arenas or threads in one process;
- every episode must destroy its arena before starting the next;
- a crash/assertion is isolated by the out-of-process worker design.

### 3.6 Turn dispatch seam

**CONFIRMED-SOURCE**

`Battle::Arena::UnitTurn` handles several engine-controlled cases before asking a controller for a decision:

1. pending UI actions;
2. already-finished, dead, standing, or immovable units;
3. bad-morale automatic action;
4. the full-fledged action branch.

Only in the fourth branch does it call:

```cpp
AI::BattlePlanner::Get().BattleTurn( *this, *_currentUnit, actions );
```

or the human interface.

After the chosen `Actions` exist, the method:

1. updates the PCG stream from the command sequence;
2. applies each command;
3. removes dead units;
4. may append an automatic good-morale action;
5. may give the same unit another full decision.

The agent hook belongs in this full-fledged branch. It must not intercept automatic morale, tower, catapult, or pending UI actions.

### 3.7 Round boundary

**CONFIRMED-SOURCE**

`Arena::Turns()` advances one complete battle round, not one policy step. It initializes both forces for the new round, repeatedly chooses the next unit, and calls `UnitTurn` until no eligible unit remains or the battle ends.

The environment’s external `step` boundary must therefore be implemented by the blocking decision hook inside `UnitTurn`; repeatedly calling `Turns()` from Python cannot provide one-decision stepping.

### 3.8 Commands and RNG stream semantics

**CONFIRMED-SOURCE**

`Battle::Command` stores integer parameters in reverse vector order and consumes them with `GetNextValue()`, which pops from the back.

The command stream update is not a byte-for-byte hash of every semantic field. In particular, `ATTACK` intentionally combines only:

- command type;
- movement cell;
- defender UID;
- attacker UID.

The target cell and attack direction are intentionally excluded because AI and human control may encode equivalent attacks differently.

Consequences:

- never serialize a command by assuming normal vector iteration is semantic argument order;
- decode a copy with `GetNextValue()` into a typed snapshot;
- distinguish semantic replay from strict raw-command replay;
- state-digest equivalence is the authoritative determinism test.

### 3.9 Action validation is embedded in execution code

**CONFIRMED-SOURCE**

`battle_action.cpp` validates move and attack parameters inside the action-application path. Several useful helpers, such as move validation and attack target/direction calculation, are in an anonymous namespace. The tactical AI also has private attack-vector logic in `ai_battle.cpp`.

There is no single existing public method that returns the complete legal semantic action set for arbitrary creatures.

This is the largest Phase 1 engineering risk. The v0.1 document overstated how directly the legal-action list could be obtained.

### 3.10 Runtime initialization and assets

**v0.3 — RESOLVED. Previously `LOCAL-VERIFY`; now CONFIRMED-RUNTIME.**

A standalone worker can omit **all** display, audio, AGG, h2d, translation, and game-resource
initialization. The Phase 0 smoke ran on a machine with no Heroes II or demo data present, opened no
window, and touched no audio device.

Two source facts explain it:

1. **Monster stats are hardcoded, not asset-derived.** `monsterData` (`monster_info.cpp:384`) is built
   from in-source `monsterBattleStats[]` / `monsterGeneralStats[]`. The `icnId` and `binFileName`
   members are rendering/animation handles that battle resolution never dereferences.
2. **Obstacle setup uses ICN ids as enum tags, not images.** `Battle::Board::SetCovrObjects`
   (`battle_board.cpp:573`) is a `switch` over `ICN::COVR*` constants calling `at(n).SetObject(...)`.
   `GetCovr` (`battle_arena.cpp:120`) likewise only *selects* an id. No image is loaded.

The empirically minimal startup path is far shorter than the 17 steps in §8.1:

```cpp
Logging::InitLog();
Rand::CurrentThreadRandomDevice() = Rand::PCG32( worldSeed );
world.generateBattleOnlyMap( Maps::Ground::GRASS );

Settings::Get().GetPlayers().Init( BLUE | RED );
world.InitKingdoms();
Players::SetPlayerRace( BLUE, Race::KNGT );  Players::SetPlayerControl( BLUE, CONTROL_AI );
Players::SetPlayerRace( RED,  Race::KNGT );  Players::SetPlayerControl( RED,  CONTROL_AI );
// fresh armies, computeBattleSeed, PCG32, Arena(..., isShowInterface=false, ...)
```

No `Game::Init()`, no `Display::instance()`, no palette, no assets. §8.1 should be trimmed to this.

Remaining caveat: this was validated in **Release**. A Debug build re-enables the asserts inside
`ApplyAction` and the `assert( _interface != nullptr )` on the non-AI path, so the Debug run in §2.4
is still required before trusting the action pipeline.

---

## 4. First implementation scope

### 4.1 Supported battle profile

The first accepted profile is:

```text
battle_profile = "creature_field_v1"
action_profile = "simple_v1"
```

It means:

- commander-free armies;
- BLUE attacker and RED defender;
- one to five stack slots per side;
- open field on the 2 × 2 Battle Only world;
- fixed tile index `1`;
- explicit non-UNKNOWN terrain;
- no castle;
- no heroes or captains;
- no spell casting;
- no retreat or surrender exposed to the external policy;
- no UI;
- only audited creatures accepted by the `simple_v1` capability allowlist.

### 4.2 Why an allowlist is required

The game includes wide units, flyers, shooters, blocked shooters, double-cell attacks, area effects, multi-target damage, special retaliation behavior, and other creature-specific mechanics. Some of these affect the number or semantics of legal attack vectors.

The first candidate generator must not claim completeness for all creatures before those cases are tested.

**PROPOSED:** generate and check in a machine-readable capability audit:

```text
python/fheroes2_agent/data/monster_capabilities_v1.json
```

For every monster ID available at the pinned commit, record at least:

```json
{
  "monster_id": 1,
  "name": "example",
  "is_valid": true,
  "is_wide": false,
  "is_flying": false,
  "is_archer": false,
  "is_double_cell_attack": false,
  "has_area_or_multi_target_attack": false,
  "simple_v1_supported": true,
  "reason": "single-cell ordinary melee"
}
```

The exact capability accessors and special-ability inventory must be confirmed by the local agent. Do not hand-maintain an unexplained list.

### 4.3 Phase 1a creature coverage

Phase 1a should begin with fixtures representing:

- an ordinary single-cell melee creature;
- an ordinary single-cell shooter;
- a shooter that becomes hand-fighting when blocked.

Peasant and Archer are reasonable provisional smoke fixtures, but the local capability audit and runtime tests are authoritative.

Phase 1a rejects scenarios containing unsupported creatures with a structured validation error.

### 4.4 Phase 1b expansion

Only after Phase 1a passes should the agent add:

- wide units;
- flying movement;
- double-cell attacks;
- special multi-target attacks;
- unusual ranged attacks;
- other audited creature abilities.

Phase 1b may remain a follow-on PR. The Phase 1a environment is useful for validating the model/data pipeline without pretending to cover every battle mechanic.

### 4.5 Non-goals

The first mergeable implementation excludes:

- adventure-map control;
- castles, sieges, walls, towers, catapults, bridges, and moats;
- heroes, captains, artifacts, primary/secondary skills, and spells;
- retreat, surrender, auto-combat toggling, and quick combat as external actions;
- screenshots, OCR, mouse, or keyboard control;
- an in-process Python extension;
- network service exposure;
- LLM inference or model downloads;
- RL algorithms;
- self-play leagues;
- proprietary game assets in source control.

---

## 5. Architecture

### 5.1 Process topology

```text
Python training/evaluation process
    │
    ├── WorkerClient #0 ── stdin/stdout JSONL ── fheroes2_agent_worker #0
    ├── WorkerClient #1 ── stdin/stdout JSONL ── fheroes2_agent_worker #1
    └── WorkerClient #N ── stdin/stdout JSONL ── fheroes2_agent_worker #N
```

Each worker:

- owns the global fheroes2 world and settings;
- has at most one live arena;
- runs episodes sequentially;
- emits protocol messages only on stdout;
- emits diagnostics only on stderr;
- can be killed and restarted independently.

### 5.2 Synchronous control flow

No worker-side concurrency is needed in v1.

The worker command loop receives a `reset` request and enters `runEpisode()`. When the engine reaches an externally controlled full decision, the hook:

1. snapshots the state;
2. enumerates legal candidates;
3. writes a `decision` message to stdout;
4. blocks reading stdin;
5. accepts only a matching `act` or `close`;
6. copies the selected engine-owned `Battle::Actions` into the turn’s output;
7. returns to `Arena::UnitTurn`.

This design preserves the engine’s natural call stack and avoids pausing/resuming an arena through a second thread.

### 5.3 Ownership

The recommended ownership graph is:

```text
AgentWorker
  ├── ProtocolChannel
  ├── AgentBattleRunner
  │     ├── fresh attacker Army
  │     ├── fresh defender Army
  │     ├── Rand::PCG32
  │     ├── optional live Battle::Arena
  │     └── BlockingDecisionController
  └── Episode/trajectory metadata
```

The candidate table is owned by the controller for exactly one outstanding decision. It contains actual `Battle::Actions` or typed command objects already validated against the current arena. Python receives metadata and an ephemeral candidate ID only.

### 5.4 Failure isolation

Protocol errors should not result in an invalid `Battle::Command` being applied.

For a stale or unknown candidate ID:

- emit a recoverable `error`;
- keep the same decision outstanding;
- continue waiting for a valid `act`.

On stdin EOF or explicit close while blocked:

- mark the episode aborted;
- supply a safe `SKIP` for the current unit so the stack can unwind;
- do not begin another round;
- destroy the arena;
- exit the worker.

Fatal engine assertions still terminate the worker and are detected by the Python client.

---

## 6. Repository and build plan

### 6.1 Proposed layout

```text
fheroes2/
├── CMakeLists.txt
├── src/
│   ├── fheroes2/
│   │   ├── agent/
│   │   │   ├── agent_battle_runner.cpp
│   │   │   ├── agent_battle_runner.h
│   │   │   ├── agent_decision_controller.cpp
│   │   │   ├── agent_decision_controller.h
│   │   │   ├── agent_action_generator.cpp
│   │   │   ├── agent_action_generator.h
│   │   │   ├── agent_observation.cpp
│   │   │   ├── agent_observation.h
│   │   │   ├── agent_scenario.cpp
│   │   │   ├── agent_scenario.h
│   │   │   ├── agent_command_snapshot.cpp
│   │   │   └── agent_command_snapshot.h
│   │   └── ... existing source ...
│   ├── agent_worker/
│   │   ├── CMakeLists.txt
│   │   ├── main.cpp
│   │   ├── protocol.cpp
│   │   └── protocol.h
│   └── thirdparty/
│       └── agent_json/             # only if a vendored JSON dependency is approved
├── python/
│   ├── pyproject.toml
│   ├── fheroes2_agent/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── env.py
│   │   ├── protocol.py
│   │   ├── replay.py
│   │   ├── policies.py
│   │   └── data/
│   │       └── monster_capabilities_v1.json
│   └── tests/
├── scenarios/
│   ├── smoke_melee_v1.json
│   ├── smoke_ranged_v1.json
│   └── ...
└── docs/
    └── agent/
        ├── local_source_audit.md
        ├── implementation_report.md
        ├── benchmark_m2.md
        └── decisions/
```

Shared agent implementation files may live under `src/fheroes2/agent/` because they contain no entry point and can compile into the normal executable without changing behavior. The worker `main.cpp` must remain outside the recursive `src/fheroes2/*.cpp` glob.

### 6.2 Build option

Add:

```cmake
option(ENABLE_AGENT "Build the fheroes2 structured agent worker" OFF)
option(ENABLE_AGENT_TESTS "Build agent-specific C++ tests" OFF)
```

With both off:

- the ordinary build remains functionally unchanged;
- no JSON dependency is required;
- no agent executable is installed;
- the standard game entry point and behavior remain the same.

### 6.3 Preferred source-target refactor

**PROPOSED**

Refactor the CMake source collection so the ordinary entry point is separated from reusable game sources:

```cmake
file(GLOB_RECURSE FHEROES2_ALL_SOURCES CONFIGURE_DEPENDS *.cpp)

set(FHEROES2_MAIN_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/game/fheroes2.cpp")

set(FHEROES2_GAME_SOURCES ${FHEROES2_ALL_SOURCES})
list(REMOVE_ITEM FHEROES2_GAME_SOURCES ${FHEROES2_MAIN_SOURCE})
```

Then use either:

- an object library shared by `fheroes2` and `fheroes2_agent_worker`; or
- the same explicit source list compiled into both targets.

An object library is preferred only if compile definitions, include directories, and platform linkage remain clear. Do not force a large architectural library split merely to satisfy elegance.

The normal executable still receives its current manifest/resources, `FHEROES2_DATA` definition, platform flags, and SDL main linkage.

The worker has a standard `main`. Whether it links `${USE_SDL_VERSION}main` must be determined locally; do not assume the normal executable’s exact entry-point linkage is appropriate.

### 6.4 Acceptable first fallback

If the object-library refactor destabilizes the normal build, the first worker may compile the non-entry fheroes2 sources a second time. Record the compile-time cost as technical debt and leave a clean source-list boundary for later optimization.

### 6.5 JSON dependency

No general-purpose JSON library was apparent in the audited game source.

Recommended approach:

- vendor a pinned, permissively licensed, single-header or small JSON library only for `ENABLE_AGENT`;
- record its version and license;
- do not use network `FetchContent` in the default build;
- parse strictly and return path-specific validation errors;
- enforce maximum input line and collection sizes.

A custom ad hoc JSON parser is not recommended.

---

## 7. Deterministic world and battle seeds

### 7.1 Correct seed model

The environment records two effective seeds:

```text
effective_world_seed
effective_combat_seed
```

The world seed controls map-derived battle setup such as field obstacle generation. The combat seed initializes the `Rand::PCG32` used by the arena and is computed from the world seed, tile index, and all army slots.

### 7.2 World seed control

**v0.3 — rewritten. The engine change is now optional, not a prerequisite.**

#### Option 1 (adopted for Milestone 1): reseed the thread-local RNG — zero engine changes

```cpp
Rand::CurrentThreadRandomDevice() = Rand::PCG32( worldSeed );
world.generateBattleOnlyMap( terrain );
```

Validated in Phase 0: deterministic within a process, across 2000 sequential episodes, and across
fresh processes. It requires no patch, so Milestone 1 no longer carries an engine-modification
dependency and the normal game is untouched by construction.

Limits, stated plainly:

- It pins **all** global randomness in the process, not only the map seed. Correct for a dedicated
  worker; unacceptable inside the real game.
- It is positional — it must run before `generateBattleOnlyMap`, and anything else that consumes
  global randomness in between will shift the result. Keep the two lines adjacent.
- It fixes the *world* seed only. The combat seed is still derived per §3.4 and stays independent.

#### Option 2 (deferred cleanup): the narrow overload

```cpp
void generateBattleOnlyMap( int32_t groundType );                       // unchanged, random
void generateBattleOnlyMap( int32_t groundType, uint32_t mapSeed );     // agent/test overload
void Defaults( std::optional<uint32_t> mapSeed );                       // seed before dependents
```

Still the better long-term shape: surgical, self-documenting, and safe for in-game use. Schedule it
when the environment is otherwise working, not as a blocker. Avoid a broad public `SetMapSeed()` that
would permit reseeding a loaded world at arbitrary times, and avoid setting `_seed` after `Defaults()`
unless it is proven no dependent state matters.

**Whichever option is used, the scenario contract in §7.4 is unchanged:** the environment records an
`effective_world_seed` and an `effective_combat_seed`, and both go into the trajectory header.

### 7.3 Shared combat-seed helper

Move the current anonymous `computeBattleSeed` implementation into a small shared battle module, for example:

```cpp
uint32_t Battle::computeBattleSeed(
    int32_t mapIndex,
    uint32_t mapSeed,
    const Army & attackingArmy,
    const Army & defendingArmy );
```

Use the helper from both:

- existing `Battle::Loader`;
- the new agent runner.

Add a unit test with fixed slot layouts, including empty slots. The helper’s behavior is part of the trajectory compatibility contract.

### 7.4 Seed modes

Scenario schema v1 supports:

```json
"seeding": {
  "mode": "engine_compatible",
  "world_seed": 123456789
}
```

`engine_compatible` is required for standard tests, demonstrations, and benchmarks.

An optional debug-only mode may be added later:

```json
"seeding": {
  "mode": "explicit_combat_seed",
  "world_seed": 123456789,
  "combat_seed": 987654321
}
```

Trajectories using an explicit combat override must be labeled non-stock-compatible. This mode is not required for the first merge.

### 7.5 Terrain and tile index

For schema v1:

- `terrain` is mandatory and cannot be UNKNOWN;
- `tile_index` is fixed to `1`;
- the worker rejects any other tile index.

This matches the existing Battle Only call and avoids expanding the contract before needed.

---

## 8. Engine initialization and episode lifecycle

### 8.1 Reference initialization sequence

**PROPOSED, LOCAL-VERIFY**

The runner should begin from the sequence used by Battle Only:

1. validate scenario;
2. create the 2 × 2 world with explicit terrain and world seed;
3. initialize the Settings player collection for BLUE and RED;
4. initialize kingdoms;
5. set both player races to valid values;
6. set both player controls to `CONTROL_AI`;
7. set current color as required by the engine;
8. construct fresh commander-free armies;
9. set attacker color BLUE and defender color RED;
10. populate exactly five deterministic slots, leaving unspecified slots empty;
11. compute the shared combat seed;
12. create `Rand::PCG32`;
13. create headless arena with the optional decision controller;
14. run complete rounds until terminal or truncation;
15. collect final `Force`/result data before arena destruction;
16. destroy arena;
17. reset current color and episode-owned state.

The local spike must determine the minimum required subset without relying on accidental global state from a previously launched game UI.

### 8.2 Why both armies remain engine-AI controlled

The existing headless constructor forces auto-combat for human-controlled armies. To avoid that alternative path and to preserve normal scheduling:

- configure both armies as engine-level AI;
- let the new decision controller intercept selected sides before the built-in AI call.

This is cleaner than marking an army human without an interface.

### 8.3 Fresh objects per episode

Never reuse:

- `Battle::Arena`;
- battle `Force` objects;
- scenario armies;
- the combat RNG;
- the per-decision candidate table.

The process may be reused, but episode objects are fresh.

`AI::BattlePlanner` is a singleton with battle-lifetime state. Arena construction currently calls its battle reset path; the local agent must verify this remains true after any constructor refactor and add a regression test.

### 8.4 Terminal result and army synchronization

`Battle::Loader` synchronizes battle forces back to original armies after normal battles. The worker does not need persistent post-battle adventure-map armies.

For v1:

- serialize terminal force state and result before arena destruction;
- discard the input armies after the episode;
- do not add synchronization solely for the agent unless a test demonstrates it is needed.

### 8.5 Truncation

Scenario limits:

```json
"limits": {
  "max_rounds": 100,
  "max_external_decisions": 2000
}
```

Round truncation is checked before starting another `Arena::Turns()`.

If the external-decision limit is reached while inside a round:

- mark `truncation_pending`;
- automatically choose engine-valid `SKIP` at subsequent intercepted decisions;
- allow the current `Turns()` call to return;
- stop before the next round.

This avoids unsafe long jumps out of the engine stack.

Terminal metadata distinguishes:

```text
victory
defeat
mutual_elimination_or_engine_draw
round_limit
decision_limit
client_closed
engine_error
```

---

## 9. Minimal battle decision hook

### 9.1 Proposed interface

Add a small optional interface in the battle layer:

```cpp
namespace Battle
{
    class DecisionController
    {
    public:
        virtual ~DecisionController() = default;

        // Called only for the full-fledged unit-decision branch.
        virtual bool handlesDecision(
            const Arena & arena,
            const Unit & currentUnit ) const = 0;

        // Must append at least one valid action when handlesDecision() is true.
        virtual void chooseActions(
            Arena & arena,
            const Unit & currentUnit,
            Actions & output ) = 0;

        // Called after external, built-in AI, or human selection, but before
        // stream update and before ApplyAction mutates commands.
        virtual void observeChosenActions(
            const Arena & arena,
            const Unit & currentUnit,
            const Actions & actions )
        {}
    };
}
```

Add an optional pointer/reference to the Arena constructor with a default that preserves all existing callers:

```cpp
Arena(
    Army & attackingArmy,
    Army & defendingArmy,
    int32_t tileIndex,
    bool isShowInterface,
    Rand::PCG32 & randomGenerator,
    DecisionController * controller = nullptr );
```

Alternative injection patterns are acceptable if they preserve the same semantics and avoid a new global hook.

### 9.2 Dispatch logic

In the full-fledged action branch:

```cpp
bool handledExternally = false;

if ( _decisionController != nullptr
     && _decisionController->handlesDecision( *this, *_currentUnit ) ) {
    _decisionController->chooseActions( *this, *_currentUnit, actions );
    handledExternally = true;
}

if ( !handledExternally ) {
    if ( ( _currentUnit->GetCurrentControl() & CONTROL_AI )
         || ( _autoCombatColors & _currentUnit->GetCurrentColor() ) ) {
        AI::BattlePlanner::Get().BattleTurn( *this, *_currentUnit, actions );
    }
    else {
        // Existing human path unchanged.
    }
}

if ( _decisionController != nullptr ) {
    _decisionController->observeChosenActions(
        *this, *_currentUnit, actions );
}
```

Call the observer before `updatePCG32Stream` and before `ApplyAction`.

### 9.3 Invariants

The hook must not:

- alter turn ordering;
- call `BattlePlanner::BattleTurn` twice;
- apply commands itself;
- consume combat RNG;
- mutate the arena while serializing;
- intercept automatic morale;
- intercept towers/catapult;
- expose mutable engine pointers to Python;
- leave `actions` empty when it claims to handle a decision.

### 9.4 Decision identity

Every full-fledged decision receives a monotonic `engine_decision_index`, including built-in AI decisions. Every externally surfaced decision also receives a monotonic `external_decision_index` and a random or monotonic `decision_id` scoped to the episode.

Good morale may produce a second decision for the same unit. It is a distinct decision and must receive distinct indices.

---

## 10. Legal-action generation

### 10.1 Principle

Python must never send arbitrary command fields.

For each external decision, the worker constructs:

```cpp
struct ActionCandidate
{
    uint32_t actionId;            // ephemeral within this decision
    CanonicalAction semantic;
    Battle::Actions engineActions;
};
```

The protocol exposes only the semantic metadata and `action_id`. Selecting an ID copies the prevalidated `engineActions` into `UnitTurn`.

### 10.2 Extract shared validation/canonicalization

The existing action application code contains authoritative legality checks but keeps important helpers private.

**PROPOSED:** extract non-mutating helpers used by both candidate generation and execution, rather than duplicating battle rules.

A possible interface:

```cpp
struct ResolvedMove
{
    uint32_t unitUid;
    int32_t destinationHeadCell;
};

struct ResolvedAttack
{
    uint32_t attackerUid;
    uint32_t defenderUid;
    int32_t movementHeadCell;   // -1 means current position
    int32_t targetCell;
    CellDirection direction;
};

std::optional<ResolvedMove> resolveMove(
    const Arena & arena,
    const Unit & unit,
    int32_t destinationHeadCell );

std::optional<ResolvedAttack> resolveAttack(
    const Arena & arena,
    const Unit & attacker,
    const Unit & defender,
    int32_t movementHeadCell,
    std::optional<int32_t> targetCell = std::nullopt,
    std::optional<CellDirection> direction = std::nullopt );
```

The exact placement may be an `Arena` method, a battle action-validation module, or an agent adapter with friend access. The critical property is that `ApplyActionMove` and `ApplyActionAttack` use the same logic, or are regression-tested against it.

Do not test legality by applying a candidate to the live arena.

### 10.3 `simple_v1` candidates

Always include:

```text
SKIP(active_unit_uid)
```

Moves:

1. call `arena.getAllAvailableMoves(activeUnit)`;
2. treat returned values as candidate head cells;
3. resolve and validate each move;
4. exclude the current head position;
5. deduplicate.

Simple non-blocked ranged attacks:

1. for each living enemy stack;
2. resolve an attack from current position;
3. emit one canonical ranged attack per enemy.

Simple melee attacks:

1. consider current position plus every reachable movement head cell;
2. for each living enemy;
3. call the shared resolver with target/direction unspecified;
4. emit the resolved canonical attack if legal;
5. deduplicate by all resolved semantic fields.

For Phase 1a’s single-cell creatures, one movement position and defender should normally resolve to a single attack vector. If the resolver finds multiple meaningful vectors, the creature or state is outside `simple_v1` until Phase 1b defines how to expose them.

Blocked shooters follow melee legality and must have explicit tests.

### 10.4 Deterministic ordering

Sort candidates by a stable tuple:

```text
action_type_rank,
attacker_uid,
defender_uid_or_0,
movement_head_cell_or_-1,
target_cell_or_-1,
direction_or_0
```

Suggested type rank:

```text
SKIP = 0
MOVE = 1
RANGED_ATTACK = 2
MELEE_ATTACK = 3
```

Assign contiguous `action_id` values only after sorting.

Do not treat `action_id` as stable across states, versions, or builds. Store the canonical semantic key in trajectories.

### 10.5 Candidate/action command snapshots

Because `Battle::Command` stores parameters in reverse order, add a typed decoder:

```cpp
CommandSnapshot snapshotCommand( const Battle::Command & command );
```

Implementation must copy the command and call `GetNextValue()` in documented semantic order. It must never consume the original command.

**v0.3 — verified feasible, with one constraint.** `Command::operator>>` is **private**
(`battle_command.h:141`); only `GetNextValue()` is public (`:80`). Since `Command` derives from
`std::vector<int>` (`:53`), the copy is trivial — but the copy must be **non-const**, because
`GetNextValue()` mutates. The signature above therefore has to take its parameter by const reference
and copy internally:

```cpp
CommandSnapshot snapshotCommand( const Battle::Command & command )
{
    Battle::Command scratch = command;   // non-const copy; GetNextValue() pops from the back
    ...
}
```

Confirmed pop semantics (`battle_command.cpp:85-93`): `val = back(); pop_back();`.

Note also that ATTACK's stream contribution reads `at(2)`, `at(3)`, `at(4)` **positionally**
(`battle_command.cpp:44-51`) rather than through `GetNextValue()`. A snapshot decoder and the RNG
stream therefore index the same command in two different ways; keep them in separate, individually
tested functions.

Example snapshot:

```json
{
  "type": "ATTACK",
  "attacker_uid": 1,
  "defender_uid": 6,
  "movement_head_cell": 34,
  "target_cell": 45,
  "direction": 3
}
```

### 10.6 Built-in AI matching

For demonstration collection:

1. allow the built-in AI to choose once;
2. snapshot and canonicalize its action sequence;
3. match it to the candidate table;
4. record `teacher_action_id` when matched;
5. record a structured coverage failure otherwise.

Do not coerce an unsupported teacher command into a superficially similar candidate.

Acceptance for the supported `simple_v1` fixture matrix is **100% teacher-action coverage**. Coverage on out-of-profile creatures is reported separately and is not a Phase 1a acceptance criterion.

---

## 11. Scenario schema v1

Example:

```json
{
  "schema_version": 1,
  "scenario_id": "simple_melee_001",
  "battle_profile": "creature_field_v1",
  "action_profile": "simple_v1",
  "terrain": "GRASS",
  "tile_index": 1,
  "seeding": {
    "mode": "engine_compatible",
    "world_seed": 123456789
  },
  "external_control": "attacker",
  "limits": {
    "max_rounds": 100,
    "max_external_decisions": 2000
  },
  "attacker": {
    "stacks": [
      { "slot": 0, "monster_id": 1, "count": 50 }
    ]
  },
  "defender": {
    "stacks": [
      { "slot": 0, "monster_id": 1, "count": 50 }
    ]
  },
  "metadata": {
    "purpose": "smoke"
  }
}
```

### 11.1 Validation

Reject before constructing an arena when:

- `schema_version != 1`;
- profile is unknown;
- terrain is absent, UNKNOWN, or unsupported;
- tile index is not `1`;
- seed is missing or outside `uint32_t`;
- external control is not `attacker`, `defender`, `both`, or `none`;
- a side has zero valid stacks or more than five slots;
- slot index is outside `[0, 4]`;
- a slot is duplicated;
- monster ID is invalid;
- count is zero or exceeds an explicit safety maximum;
- monster is not supported by the requested action profile;
- limits are zero or exceed configured safety maxima;
- unknown fields are present in strict mode.

Validation errors include a JSON path and stable error code.

### 11.2 Army slot semantics

Army slot order affects the engine-compatible combat seed and likely initial placement. Preserve all five positions exactly.

The canonical scenario representation fills missing slots with empty entries before hashing and seed computation.

### 11.3 Scenario digest

Compute:

```text
scenario_sha256
```

from canonical UTF-8 JSON with:

- sorted object keys;
- normalized integer formatting;
- all five slots represented;
- no insignificant whitespace;
- metadata excluded unless explicitly declared semantically relevant.

Store both the original scenario and digest in the trajectory header.

---

## 12. Observation schema v1

### 12.1 Top-level shape

```json
{
  "schema_version": 1,
  "episode_id": "e-000001",
  "engine_decision_index": 17,
  "external_decision_index": 9,
  "round_number": 3,
  "active_unit_uid": 2,
  "active_side": "attacker",
  "active_army_color": "BLUE",
  "effective_world_seed": 123456789,
  "effective_combat_seed": 2345678901,
  "board": {},
  "units": [],
  "side_summary": {},
  "state_digest": "sha256:..."
}
```

Do not expose candidate IDs inside the canonical state digest.

### 12.2 Board

The battle board is 11 × 9 with 99 indexed cells.

**v0.3 — CONFIRMED-SOURCE, with the real identifiers.** Do not invent constants; use
`battle_board.h:73-77`:

```cpp
static constexpr int widthInCells{ 11 };
static constexpr int heightInCells{ 9 };
static constexpr int sizeInCells{ widthInCells * heightInCells };   // 99
```

`Board::isValidIndex()` (`battle_board.h:115`) is the engine's own bounds check and should be
mirrored rather than reimplemented.

Suggested representation:

```json
{
  "width": 11,
  "height": 9,
  "cells": [
    {
      "index": 0,
      "x": 0,
      "y": 0,
      "obstacle_id": null,
      "occupant_uid": null
    }
  ]
}
```

The local implementation should define `x` and `y` as display-grid coordinates only; policies should not assume square-grid adjacency. Either:

- include a static six-neighbor table in the worker `ready.capabilities`; or
- provide a documented Python helper matching `Battle::Board` direction rules.

Do not duplicate the neighbor table in every observation.

### 12.3 Unit record

Serialize living battle units only:

```json
{
  "uid": 2,
  "side": "attacker",
  "army_color": "BLUE",
  "current_color": "BLUE",
  "monster_id": 1,
  "count": 42,
  "initial_count": 50,
  "dead_count": 8,
  "total_hit_points": 42,
  "top_creature_hit_points": 1,
  "attack": 1,
  "defense": 1,
  "base_speed": 3,
  "effective_speed": 3,
  "shots_left": 0,
  "engine_strength": 42.0,
  "head_cell": 34,
  "tail_cell": null,
  "is_wide": false,
  "is_flying": false,
  "is_archer": false,
  "is_hand_fighting": true,
  "is_immovable": false,
  "has_retaliation": true,
  "moved": false,
  "skipped": false,
  "morale_state": "none",
  "luck_state": "none",
  "spell_effect_ids": []
}
```

Accessor semantics to preserve:

- `total_hit_points` is total HP remaining in the stack;
- `top_creature_hit_points` is HP remaining in the top creature;
- `engine_strength` must document the exact engine method used and must not be mislabeled as a learned value;
- UID is stable only within an episode;
- `current_color` may differ from army color in broader spell-enabled profiles, even though it should not in `simple_v1`.

The local agent should add a field-to-accessor table in code comments or generated documentation.

### 12.4 Side summary

```json
{
  "attacker": {
    "living_stacks": 2,
    "living_creatures": 61,
    "remaining_total_hit_points": 95,
    "engine_strength_sum": 123.4
  },
  "defender": {
    "living_stacks": 1,
    "living_creatures": 20,
    "remaining_total_hit_points": 40,
    "engine_strength_sum": 52.1
  }
}
```

These are diagnostics, not necessarily the training reward.

### 12.5 Canonical state digest

Build a `state_core` object containing only deterministic game state, excluding:

- episode UUID;
- wall-clock time;
- process ID;
- candidate IDs;
- log messages;
- request IDs.

Canonicalize and hash it with SHA-256.

The digest is used for replay comparison. Prefer computing it in one shared language implementation or cross-testing C++ and Python canonicalization with golden fixtures.

---

## 13. Protocol v1

### 13.1 Transport

- UTF-8 JSON Lines;
- exactly one object per line;
- stdout reserved for protocol;
- stderr reserved for human-readable logs;
- line length capped, for example at 4 MiB;
- worker flushes after every outbound message;
- protocol version negotiated by the initial `ready`.

### 13.2 Common envelope

```json
{
  "protocol_version": 1,
  "type": "decision",
  "request_id": "r-17",
  "episode_id": "e-1"
}
```

Fields not meaningful to a message may be omitted.

### 13.3 `ready`

Worker emits once:

```json
{
  "protocol_version": 1,
  "type": "ready",
  "engine": {
    "name": "fheroes2",
    "version": "1.1.17",
    "commit": "2685c2188b541660f1ce261b554c3e92f79b1775",
    "build_type": "Release"
  },
  "capabilities": {
    "scenario_schema_versions": [1],
    "action_profiles": ["simple_v1"],
    "external_control": ["attacker", "defender", "both", "none"],
    "board_width": 11,
    "board_height": 9,
    "max_stacks_per_side": 5
  }
}
```

### 13.4 `reset`

Client sends:

```json
{
  "protocol_version": 1,
  "type": "reset",
  "request_id": "r-1",
  "scenario": { "...": "..." }
}
```

Worker validates before constructing the arena. It then emits either `error`, a first `decision`, or `episode_end` if no external side is controlled.

An optional `reset_ok` may be emitted, but the client must not require it unless it is included in the final protocol schema.

### 13.5 `decision`

```json
{
  "protocol_version": 1,
  "type": "decision",
  "request_id": "r-1",
  "episode_id": "e-1",
  "decision_id": "d-9",
  "observation": { "...": "..." },
  "legal_actions": [
    {
      "action_id": 0,
      "type": "SKIP",
      "canonical_key": "skip:2",
      "actor_uid": 2
    },
    {
      "action_id": 1,
      "type": "MOVE",
      "canonical_key": "move:2:34",
      "actor_uid": 2,
      "destination_head_cell": 34
    }
  ]
}
```

### 13.6 `act`

```json
{
  "protocol_version": 1,
  "type": "act",
  "request_id": "r-1",
  "episode_id": "e-1",
  "decision_id": "d-9",
  "action_id": 1
}
```

The worker rejects:

- wrong episode;
- wrong decision;
- unknown action ID;
- duplicate action after the decision has advanced.

A rejected action does not mutate the arena.

### 13.7 `episode_end`

```json
{
  "protocol_version": 1,
  "type": "episode_end",
  "request_id": "r-1",
  "episode_id": "e-1",
  "terminated": true,
  "truncated": false,
  "reason": "victory",
  "winner": "attacker",
  "result": {
    "attacker": "wins",
    "defender": "loss"
  },
  "rounds": 8,
  "engine_decisions": 41,
  "external_decisions": 21,
  "effective_world_seed": 123456789,
  "effective_combat_seed": 2345678901,
  "terminal_state_digest": "sha256:...",
  "final_summary": {}
}
```

### 13.8 `error`

```json
{
  "protocol_version": 1,
  "type": "error",
  "request_id": "r-1",
  "episode_id": "e-1",
  "recoverable": true,
  "code": "UNKNOWN_ACTION_ID",
  "message": "action_id 99 is not legal for decision d-9",
  "path": "$.action_id"
}
```

Stable codes should cover parse, validation, lifecycle, and action-selection errors.

### 13.9 `close`

Client may send `close` while idle or while a decision is outstanding. Worker acknowledges when practical and exits cleanly. The Python client must also support process termination when the worker is unresponsive.

---

## 14. Python package

### 14.1 Core API

```python
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

@dataclass(frozen=True)
class StepResult:
    observation: Mapping[str, Any] | None
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]

class FHeroes2BattleEnv:
    def reset(
        self,
        *,
        scenario: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any] | None, Mapping[str, Any]]:
        ...

    def legal_actions(self) -> Sequence[Mapping[str, Any]]:
        ...

    def step(self, action_id: int) -> StepResult:
        ...

    def close(self) -> None:
        ...
```

`reset()` may return `observation=None` when `external_control="none"` and the battle immediately runs to completion.

Gymnasium compatibility may be added as an adapter, not as a dependency of the transport/client core.

### 14.2 Client requirements

The client must:

- spawn the worker with explicit resource/config paths;
- read stderr separately to avoid pipe deadlock;
- enforce startup and decision timeouts;
- validate protocol version;
- detect premature process exit;
- include the tail of stderr in raised exceptions;
- reject out-of-order messages;
- close or kill child processes on interpreter exit;
- support deterministic restart after worker failure.

### 14.3 Policies

Ship:

```python
class Policy(Protocol):
    def select_action(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> int:
        ...
```

Baselines:

- `RandomPolicy(seed)`;
- `SkipPolicy`;
- `GreedyDamageHeuristicPolicy` after fields are sufficient;
- `ReplayPolicy`;
- built-in AI mode through scenario `external_control="none"` or one-sided control.

Random policy randomness belongs to Python and is logged separately from engine seeds.

### 14.4 Worker pool

A simple pool launches one process per environment. Start with process counts `1`, `2`, and `4` on the M2 and measure before increasing.

Do not share one worker among simultaneous episodes.

---

## 15. Trajectories and demonstration data

### 15.1 Header record

Each trajectory file begins with:

```json
{
  "record_type": "episode_header",
  "trajectory_schema_version": 1,
  "engine_commit": "2685c2188b541660f1ce261b554c3e92f79b1775",
  "worker_build": {},
  "scenario": {},
  "scenario_sha256": "sha256:...",
  "effective_world_seed": 123456789,
  "effective_combat_seed": 2345678901,
  "action_profile": "simple_v1",
  "policy": {},
  "started_at_utc": "..."
}
```

Wall-clock fields are metadata and excluded from determinism checks.

### 15.2 Decision record

```json
{
  "record_type": "decision",
  "engine_decision_index": 17,
  "external_decision_index": 9,
  "state_digest_before": "sha256:...",
  "observation": {},
  "legal_actions": [],
  "selected_action_id": 3,
  "selected_canonical_key": "attack:2:7:34:45:3",
  "selected_command_snapshots": [],
  "teacher": {
    "available": true,
    "matched": true,
    "action_id": 3,
    "canonical_key": "attack:2:7:34:45:3"
  },
  "state_digest_after_next_decision_or_terminal": "sha256:..."
}
```

For built-in-only demonstrations, record all full decisions through `observeChosenActions`, not only externally surfaced ones.

### 15.3 Terminal record

Includes:

- terminal/truncation reason;
- winner/result bits;
- final state digest;
- round and decision counts;
- remaining stack summaries;
- teacher coverage statistics;
- worker exit health.

### 15.4 Data compatibility

A dataset is identified by at least:

```text
engine commit
scenario schema version
observation schema version
action profile version
trajectory schema version
command canonicalization version
```

Do not silently combine trajectories across incompatible versions.

---

## 16. Determinism and replay contract

### 16.1 Level A — episode determinism

Given:

- same engine commit and build-relevant flags;
- same scenario;
- same effective seeds;
- same policy RNG seed;
- same selected canonical actions;

the sequence of state digests and terminal outcome must match.

Acceptance: ten identical runs for every golden scenario.

### 16.2 Level B — semantic replay

A replay file selects actions by canonical key, not transient ID. At every decision:

1. regenerate candidates;
2. find exactly one candidate with the recorded key;
3. verify `state_digest_before`;
4. apply;
5. verify next digest.

Failure reports the first divergent decision and a structured state diff.

### 16.3 Level C — strict command replay

Optional debugging mode also compares typed command snapshots.

Do not require raw `std::vector<int>` equality as the principal contract. ATTACK RNG normalization intentionally ignores target cell and direction, and equivalent encodings may exist.

### 16.4 Metadata

Log:

- commit;
- dirty-tree indicator;
- compiler and version;
- build type;
- relevant CMake flags;
- protocol and schema versions;
- action profile;
- world and combat seeds;
- scenario digest;
- Python policy seed;
- worker count for parallel runs.

---

## 17. Reward and evaluation

### 17.1 Default reward

For a one-sided external environment:

```text
+1.0  external side wins
-1.0  external side loses
 0.0  truncation or unresolved/draw outcome
```

Intermediate reward is `0.0`.

This keeps the environment neutral and avoids embedding an unvalidated shaping objective in v1.

### 17.2 Diagnostic metrics

Expose but do not automatically add to reward:

- remaining total HP by side;
- creature-count fraction by side;
- engine strength sum by side;
- stack deaths;
- rounds;
- decisions;
- action-type counts;
- teacher agreement;
- illegal/stale protocol attempts;
- wall-clock duration.

A later training config may compute shaping externally.

### 17.3 Baseline evaluation suite

Maintain fixed scenarios partitioned into:

- smoke;
- deterministic regression;
- simple melee;
- simple ranged;
- blocked shooter;
- asymmetric strength;
- movement/obstacle;
- truncation/stalemate.

Report win rate with fixed paired seeds against:

- built-in AI;
- random;
- skip;
- simple heuristic.

---

## 18. Testing strategy

### 18.1 Build regression

CI/local commands must verify:

- normal game with `ENABLE_AGENT=OFF`;
- normal game with `ENABLE_AGENT=ON`;
- worker target;
- Debug and Release where feasible;
- no second-main collision;
- no agent JSON dependency when disabled.

### 18.2 C++ unit tests

At minimum:

1. engine combat-seed helper, including empty slot positions;
2. scenario validation and canonicalization;
3. command snapshot decoding for MOVE, ATTACK, and SKIP;
4. canonical action key and deterministic ordering;
5. candidate ID assignment;
6. stale/unknown action rejection;
7. state-core canonicalization;
8. capability allowlist loading/validation.

If the project has no suitable existing test framework, a small agent-specific executable and Python-driven assertions are acceptable. Do not force a repository-wide testing-framework migration.

### 18.3 Engine integration tests

Required fixtures:

- simple melee vs simple melee;
- simple shooter vs melee;
- blocked shooter acts as melee;
- unreachable move absent;
- attack only appears when resolver accepts it;
- external attacker vs built-in defender;
- built-in attacker vs external defender;
- both external;
- both built-in;
- good/bad morale handling does not produce a protocol decision for automatic morale;
- round limit;
- decision limit;
- invalid action ID leaves state unchanged;
- client close while blocked;
- 100 sequential episodes in one worker;
- two workers run independently.

### 18.4 Candidate completeness tests

For every supported fixture state:

- every emitted candidate passes shared validation;
- no two candidates have the same canonical key;
- built-in teacher command matches exactly one candidate;
- selected candidate applies without Debug assertion;
- `SKIP` always exists for a valid full decision.

For Phase 1a supported scenarios, teacher coverage must be 100%. Do not dilute this with unsupported-monster states.

### 18.5 Determinism tests

- ten identical runs per golden scenario;
- semantic trajectory replay;
- same scenario after 99 unrelated prior episodes;
- one-worker and fresh-worker comparison;
- Debug and Release outcome comparison where practical;
- canonical state digest independent of JSON key order.

### 18.6 Sanitizers

On a supported compiler/platform, run representative integration tests with:

- AddressSanitizer;
- UndefinedBehaviorSanitizer.

LeakSanitizer availability on macOS varies; record what actually ran rather than claiming unsupported coverage.

### 18.7 Fuzz and robustness tests

Fuzz or property-test:

- malformed JSON;
- oversized lines;
- unknown fields;
- invalid monster/count/slot;
- stale decision IDs;
- duplicate messages;
- EOF at every protocol phase.

No malformed client input should reach `Arena::ApplyAction`.

---

## 19. Benchmark plan for the M2 Mac mini

Do not set an arbitrary battles-per-second gate before measuring the engine.

### 19.1 Benchmark modes

Measure separately:

**A. Pure engine baseline**

- built-in AI vs built-in AI;
- no per-decision JSON;
- one worker.

**B. Worker serialization overhead**

- external loopback policy in the same process/protocol path;
- observations and actions serialized;
- one worker.

**C. Python end-to-end**

- Python random or replay policy;
- worker subprocess;
- `1`, `2`, and `4` concurrent workers.

### 19.2 Workloads

Use fixed scenario suites:

- tiny one-stack;
- medium three-stack;
- five-stack;
- ranged-heavy simple profile;
- longer balanced battle.

Run enough warm-up episodes to exclude first-load effects, then at least 100 measured episodes per configuration; scale higher if runtime permits.

### 19.3 Metrics

Record:

- battles per second;
- decisions per second;
- median and p95 episode time;
- median and p95 external-decision round-trip latency;
- CPU utilization;
- resident memory per worker;
- peak memory;
- worker startup time;
- stdout bytes per decision;
- determinism failures;
- crashes/restarts.

### 19.4 Deliverable

Create:

```text
agent_play/docs/benchmark_m2.md
```

Include exact hardware/software, commands, scenario digests, raw summary tables, and a recommendation for default worker count.

---

## 20. Milestones and commit sequence

### Milestone 0 — local audit and smoke

Deliver:

- exact checkout verification;
- standard build/run;
- Battle Only run;
- deterministic headless AI-v-AI smoke;
- 100 sequential arena stress;
- `agent_play/docs/archive/benchmarks/2026-07-26-source-audit-apple-m3.md`.

Exit only when the initialization and asset path are known.

### Milestone 1 — deterministic runner foundation

Changes:

- optional deterministic Battle Only world-seed path;
- extracted shared combat-seed helper;
- fixed scenario construction in C++;
- headless AI-v-AI runner;
- terminal digest;
- normal executable regression.

Exit when ten identical runs match.

### Milestone 2 — decision hook and passive logging

Changes:

- optional `DecisionController`;
- observer before command mutation;
- typed `CommandSnapshot`;
- passive built-in-AI trajectory logging.

Exit when built-in behavior is unchanged with a null controller and passive logs replay deterministically.

### Milestone 3 — `simple_v1` legal actions

Changes:

- shared non-mutating move/attack resolution;
- monster capability audit;
- scenario allowlist;
- candidate generation/sorting;
- built-in teacher matching.

Exit when all supported fixtures have valid candidates and 100% teacher coverage.

### Milestone 4 — JSONL worker

Changes:

- dedicated worker target outside the fheroes2 source glob;
- protocol v1;
- scenario parser;
- blocking external control;
- lifecycle/error handling.

Exit when scripted stdin/stdout tests control both sides without invalid commands.

### Milestone 5 — Python environment and replay

Changes:

- subprocess client;
- environment wrapper;
- random/skip/replay policies;
- trajectory writer;
- semantic replay and divergence report;
- worker pool.

Exit when golden trajectories reproduce across fresh and reused workers.

### Milestone 6 — hardening and benchmark

Changes:

- sanitizers/robustness;
- 100+ episode stress;
- M2 benchmark;
- implementation report;
- unresolved-risk list.

Exit when the definition of done in Section 22 passes.

### Suggested commit granularity

1. `docs(agent): add local source audit and ADR scaffolding`
2. `refactor(battle): share engine-compatible battle seed helper`
3. `feat(world): add deterministic battle-only map generation`
4. `feat(agent): add headless deterministic battle smoke runner`
5. `feat(battle): add optional decision controller`
6. `feat(agent): add command snapshots and passive trace logging`
7. `feat(agent): add simple_v1 capability audit and action generator`
8. `feat(agent): add worker target and protocol v1`
9. `feat(python): add client, environment, and baseline policies`
10. `test(agent): add replay, stress, and protocol robustness`
11. `docs(agent): add implementation and M2 benchmark reports`

The local agent may reorder commits to keep each buildable.

---

## 21. Training roadmap after the environment is accepted

### 21.1 Demonstration collection

Run built-in AI on supported scenario distributions and store:

```text
observation
legal candidate set
teacher candidate
terminal outcome
```

Split by scenario family and seed, not by individual decisions, to prevent trajectory leakage.

### 21.2 Behavior cloning

Start with candidate scoring:

\[
s_\theta(o, a_i)
\]

and cross-entropy over the legal candidate set:

\[
\mathcal{L}_{BC}
= -\log
\frac{\exp s_\theta(o,a^\*)}
     {\sum_{a_i \in \mathcal{A}(o)} \exp s_\theta(o,a_i)}.
\]

A small MLP/transformer/graph model is enough for the first tactical policy. An LLM is not required for battle cells.

### 21.3 DAgger-style correction

Let the learned policy visit states, query the built-in AI at those exact states through passive teacher matching, aggregate, and retrain.

Track:

- teacher coverage;
- learner-teacher agreement;
- win rate against frozen teacher;
- out-of-distribution state rate.

### 21.4 Battle RL

Only after behavior cloning is reliable:

- use short battle episodes;
- terminal win/loss reward;
- optional externally computed HP shaping;
- frozen opponent mixtures;
- checkpoint evaluation on hidden scenarios.

Whole-game GRPO is not the starting point. If a group-relative method is tested, use bounded battle or macro-decision rollouts rather than full game trajectories.

### 21.5 Expand mechanics deliberately

Add one capability group at a time:

1. wide movement;
2. flying movement;
3. special melee vectors;
4. special ranged/multi-target attacks;
5. morale/luck diversity;
6. heroes and spells;
7. siege mechanics.

Each expansion increments the action/observation profile version and adds teacher-coverage tests.

---

## 22. Definition of done for Phase 1a

### Build and isolation

- [ ] Exact baseline commit is recorded.
- [ ] Normal `fheroes2` builds with agent disabled.
- [ ] Normal `fheroes2` builds and runs with agent enabled.
- [ ] Worker entry point is outside the recursive game-source glob.
- [ ] No proprietary assets are committed.
- [ ] Agent-only dependencies are disabled by default.

### Runtime correctness

- [ ] Headless simple battle completes.
- [ ] Explicit world seed is effective before dependent initialization.
- [ ] Shared combat seed matches `Battle::Loader`.
- [ ] One process runs at least 100 sequential arenas.
- [ ] Multiple workers operate independently.
- [ ] Null decision controller preserves existing behavior.
- [ ] No emitted candidate triggers a Debug assertion.

### Environment contract

- [ ] Strict scenario schema v1.
- [ ] Structured observation schema v1.
- [ ] Stable state digest.
- [ ] Deterministically sorted legal candidates.
- [ ] Python selects only action IDs.
- [ ] Stale/invalid selections do not mutate state.
- [ ] `simple_v1` rejects unsupported creatures.
- [ ] Built-in teacher coverage is 100% on supported fixture states.

### Reproducibility and data

- [ ] Ten-run deterministic golden tests pass.
- [ ] Semantic trajectory replay passes.
- [ ] Engine and schema metadata are in every trajectory.
- [ ] Passive built-in AI traces are collectable.
- [ ] First divergence is diagnosable with state diff.

### Operational quality

- [ ] Stdout contains protocol only.
- [ ] Client handles worker crash/EOF/timeouts.
- [ ] Representative sanitizer tests pass where supported.
- [ ] M2 benchmark report exists.
- [ ] Local source audit and implementation report exist.
- [ ] Remaining Phase 1b mechanics are explicitly listed.

---

## 23. Risk register

**v0.3:** the first, second, fifth and sixth rows are **closed by Phase 0**. Retained with their
outcomes so the reasoning is not lost.

| Risk | Impact | Mitigation |
|---|---|---|
| ~~Standalone worker needs more game/SDL initialization than expected~~ | ~~High~~ → **CLOSED** | Phase 0 ran headless with **no** display, audio, AGG, h2d, or game resources. See §3.10. |
| ~~World seed cannot be injected cleanly~~ | ~~High~~ → **CLOSED** | Thread-local RNG reseed gives determinism with zero engine changes. See §7.2. |
| Legal action rules are duplicated incorrectly | **High — now the top risk** | Extract shared resolver used by execution and enumeration; restrict `simple_v1`. Untouched by Phase 0; §3.9 stands. |
| Special creature behavior slips into the allowlist | High | Generated capability inventory plus runtime teacher-coverage fixtures. `RANGER`'s `DOUBLE_SHOOTING` (`monster_info.cpp:391`) is a live example. |
| ~~Global Arena/World state leaks across episodes~~ | ~~High~~ → **CLOSED** | 2000 sequential arenas in one process, 1 distinct digest, 12.9 MB peak RSS. |
| ~~Worker `main` collides with recursive source glob~~ | ~~High~~ → **CLOSED** | Linking a new `main` against the 220 game objects minus `fheroes2.o` worked first try, no undefined symbols. |
| Debug-only assertions reject actions that Release silently accepts | **Medium — new** | Phase 0 was Release-only. Run the §2.4 Debug experiment before trusting the action pipeline. |
| Agent target defined for CMake but the machine only has the `src/dist` Makefile | **Medium — new** | Decide the Path A / Path B question in §2.2 before Milestone 4. |
| Command decoding is reversed or mutates commands | High | Decode a copy via `GetNextValue`; unit tests for every supported type. |
| Equivalent ATTACK encodings appear different | Medium | Canonical resolver and state-digest replay; distinguish strict from semantic replay. |
| Invalid protocol action reaches engine assertion | High | Engine-owned candidate table; no raw command input; retry same decision. |
| stdout logs corrupt JSONL | Medium | Protocol-only stdout; diagnostics redirected to stderr; parser tests. |
| Runtime resource loading dominates startup | Medium | Reuse process for sequential episodes; benchmark startup separately. |
| M2 process count causes memory pressure | Medium | Measure 1/2/4 workers; choose empirical default. |
| Engine update invalidates integration seams | Medium | Pin commit; version all schemas; re-audit before rebase. |
| Scope expands to all creatures before substrate works | High | Phase 1a allowlist and explicit Phase 1b gate. |

---

## 24. Local implementation-agent assignment

The following can be pasted into the local coding agent.

### Assignment

You are implementing the first structured agent environment for the fheroes2 repository.

Use release `1.1.17` and verify that `git rev-parse HEAD` equals:

```text
2685c2188b541660f1ce261b554c3e92f79b1775
```

Read this specification fully before editing. Do not implement it blindly: begin with **Phase 0**, build the normal game, run Battle Only, and produce `agent_play/docs/archive/benchmarks/2026-07-26-source-audit-apple-m3.md`.

The first deliverable is a deterministic, headless, creature-only field-battle environment. It must use true engine state and legal engine commands. It must not contain an LLM, RL algorithm, adventure-map control, screenshot parsing, or GUI automation.

Key constraints:

1. Preserve normal fheroes2 behavior when the agent feature is disabled.
2. Keep the worker `main.cpp` outside `src/fheroes2`, because that directory’s CMake recursively globs all `.cpp` files.
3. Reuse `World::generateBattleOnlyMap`, but add a narrow deterministic seed path that sets the map seed before dependent initialization.
4. Extract and share the existing engine-compatible battle-seed computation.
5. Add an optional synchronous decision controller only in the full-fledged branch of `Battle::Arena::UnitTurn`.
6. Do not intercept automatic morale, tower, catapult, or pending UI actions.
7. Never accept raw `Battle::Command` fields from Python.
8. Generate and retain engine-valid candidates; Python chooses an ephemeral `action_id`.
9. Begin with the audited `simple_v1` creature/action profile. Reject unsupported creatures.
10. Extract or share non-mutating move/attack validation with the execution path rather than reimplementing battle legality independently.
11. Decode commands by copying them and consuming semantic values with `GetNextValue()`.
12. Use state-digest replay as the primary determinism test.
13. Run one arena at a time per process and use multiple worker processes for parallelism.
14. Reserve stdout for JSONL protocol and stderr for logs.

Before broad implementation, answer in the local audit:

- What exact initialization and resource loading are required by a headless arena?
- Can ten identical battles reproduce exactly?
- Can one process run 100 fresh arenas?
- Does setting both commander-free BLUE/RED armies to AI control follow the expected `UnitTurn` branch?
- Which creature abilities must be excluded from `simple_v1`?
- What is the smallest safe CMake refactor that leaves the normal executable unchanged?

Deliverables:

```text
fheroes2_agent_worker
python/fheroes2_agent package
strict scenario/observation/protocol schema v1
simple_v1 candidate generator
built-in AI passive demonstrations and candidate matching
deterministic semantic replay
tests and stress runs
agent_play/docs/local_source_audit.md
agent_play/docs/implementation_report.md
agent_play/docs/benchmark_m2.md
ADRs for deviations
```

Stop Phase 1a at the definition of done in Section 22. Record unsupported mechanics for Phase 1b rather than silently approximating them.

### Expected implementation report

The final report should contain:

- changed files and architecture;
- exact build/run commands;
- initialization findings;
- seed verification;
- supported creature capability rule;
- test matrix and outcomes;
- sanitizer results;
- benchmark results;
- normal-game regression result;
- known gaps;
- recommended next implementation step.

---

## Appendix A. Exact-commit source references used in this audit

All links below are pinned to commit `2685c2188b541660f1ce261b554c3e92f79b1775`.

- Root build and options:  
  `https://raw.githubusercontent.com/ihhub/fheroes2/2685c2188b541660f1ce261b554c3e92f79b1775/CMakeLists.txt`
- Game executable source glob and target:  
  `https://raw.githubusercontent.com/ihhub/fheroes2/2685c2188b541660f1ce261b554c3e92f79b1775/src/fheroes2/CMakeLists.txt`
- Battle Only entry path:  
  `https://raw.githubusercontent.com/ihhub/fheroes2/2685c2188b541660f1ce261b554c3e92f79b1775/src/fheroes2/game/game_startgame.cpp`
- Battle Only setup and tile index:  
  `https://raw.githubusercontent.com/ihhub/fheroes2/2685c2188b541660f1ce261b554c3e92f79b1775/src/fheroes2/battle/battle_only.cpp`
- World generation and random map seed:  
  `https://raw.githubusercontent.com/ihhub/fheroes2/2685c2188b541660f1ce261b554c3e92f79b1775/src/fheroes2/world/world.cpp`
- World public API:  
  `https://raw.githubusercontent.com/ihhub/fheroes2/2685c2188b541660f1ce261b554c3e92f79b1775/src/fheroes2/world/world.h`
- Battle seed and loader loop:  
  `https://raw.githubusercontent.com/ihhub/fheroes2/2685c2188b541660f1ce261b554c3e92f79b1775/src/fheroes2/battle/battle_main.cpp`
- Arena constructor, singleton, `UnitTurn`, and `Turns`:  
  `https://raw.githubusercontent.com/ihhub/fheroes2/2685c2188b541660f1ce261b554c3e92f79b1775/src/fheroes2/battle/battle_arena.cpp`
- Arena public API:  
  `https://raw.githubusercontent.com/ihhub/fheroes2/2685c2188b541660f1ce261b554c3e92f79b1775/src/fheroes2/battle/battle_arena.h`
- Action validation and application:  
  `https://raw.githubusercontent.com/ihhub/fheroes2/2685c2188b541660f1ce261b554c3e92f79b1775/src/fheroes2/battle/battle_action.cpp`
- Command storage and stream update:  
  `https://raw.githubusercontent.com/ihhub/fheroes2/2685c2188b541660f1ce261b554c3e92f79b1775/src/fheroes2/battle/battle_command.cpp`
- Command types and constructor semantics:  
  `https://raw.githubusercontent.com/ihhub/fheroes2/2685c2188b541660f1ce261b554c3e92f79b1775/src/fheroes2/battle/battle_command.h`
- Tactical AI interface and state:  
  `https://raw.githubusercontent.com/ihhub/fheroes2/2685c2188b541660f1ce261b554c3e92f79b1775/src/fheroes2/ai/ai_battle.h`
- Tactical AI command generation:  
  `https://raw.githubusercontent.com/ihhub/fheroes2/2685c2188b541660f1ce261b554c3e92f79b1775/src/fheroes2/ai/ai_battle.cpp`
- Army control and strength behavior:  
  `https://raw.githubusercontent.com/ihhub/fheroes2/2685c2188b541660f1ce261b554c3e92f79b1775/src/fheroes2/army/army.cpp`
- Army slot count and interface:  
  `https://raw.githubusercontent.com/ihhub/fheroes2/2685c2188b541660f1ce261b554c3e92f79b1775/src/fheroes2/army/army.h`

---

## Appendix B. Example semantic replay failure

```text
Replay divergence
  episode: e-000017
  scenario_sha256: sha256:...
  engine_commit: 2685c218...
  decision: 23
  recorded_state: sha256:abc...
  actual_state:   sha256:def...
  selected_key: attack:2:7:34:45:3

First structured differences:
  units[uid=7].count: recorded 12, actual 13
  units[uid=7].total_hit_points: recorded 28, actual 31
  side_summary.defender.remaining_total_hit_points: recorded 54, actual 57

Recent command snapshots:
  20 move:...
  21 attack:...
  22 skip:...
```

The report should also include effective seeds and worker stderr tail.

---

## Appendix C. Principal corrections from v0.1

1. Reclassified the document as source-cross-checked rather than implementation-verified.
2. Added mandatory local build/runtime Phase 0.
3. Confirmed that all `src/fheroes2/*.cpp` files are recursively globbed, making worker-entry placement a hard constraint.
4. Replaced hypothetical world initialization with the actual `generateBattleOnlyMap` path.
5. Corrected seed semantics: `World::Defaults()` randomizes the map seed, and combat seed is derived from map seed, tile index, and all army slots.
6. Added a narrow deterministic world-seed API requirement.
7. Located the decision hook precisely in the full-fledged branch of `UnitTurn`.
8. Clarified that `Turns()` advances a complete round and that stepping must block inside the hook.
9. Corrected command serialization: parameters are stored in reverse order and must be decoded from a copy.
10. Corrected ATTACK RNG expectations: target cell and direction are intentionally excluded from stream updates.
11. Removed the unsupported claim that the engine already exposes one complete legal-action API.
12. Narrowed the initial action/creature scope to `simple_v1`.
13. Required shared non-mutating legality/canonicalization instead of ad hoc command construction.
14. Changed teacher coverage from a vague percentage to 100% on explicitly supported fixtures.
15. Added three levels of determinism and semantic replay.
16. Removed arbitrary throughput acceptance targets until the M2 baseline is measured.
17. Added concrete local-agent deliverables and a paste-ready assignment.

---

## Appendix D. Local source-audit report template

```markdown
# fheroes2 Agent Local Source Audit

## Environment
- Date:
- Machine:
- RAM:
- macOS:
- Compiler:
- CMake:
- SDL:
- fheroes2 commit:
- Dirty tree:

## Resource setup
- Source of game/demo resources:
- Paths:
- Environment variables:
- Display/audio requirements:

## Baseline build
- Commands:
- Result:
- Normal game run:
- Battle Only run:

## Headless smoke
- Initialization sequence:
- Scenario:
- Effective world seed:
- Effective combat seed:
- Result:
- UI opened:
- Errors/warnings:

## Determinism
- Number of repeated runs:
- State/terminal digest result:
- Any divergence:

## Sequential lifecycle
- Episodes in one process:
- Peak RSS:
- Crash/assertion/leak observations:

## Source assumptions
| Assumption | Confirmed | Evidence/change |
|---|---:|---|

## simple_v1 capability decision
- Supported predicates:
- Rejected abilities:
- Provisional supported monster IDs:
- Teacher coverage fixtures:

## Build-target decision
- Chosen CMake structure:
- Why:
- Normal-game regression:

## Deviations from v0.2
- ADR links:

## Go/no-go
- Decision:
- Blocking issues:
```
