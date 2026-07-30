---
title: "fheroes2 agent environment — start here"
status: active
goal: "A deterministic, headless, structured battle environment for fheroes2 that a policy can be trained on"
branch: agent-env
date_started: 2026-07-26
last_updated: 2026-07-30
related_concepts: ["[[determinism-seeds-and-digests]]", "[[battle-turn-dispatch]]", "[[legal-actions-and-masking]]", "[[observation-design]]", "[[command-encoding-and-snapshots]]", "[[teacher-coverage-and-behavior-cloning]]"]
tags: [agent-env, rl-environment, fheroes2, entry-point]
---

> **What this note is.** The single entry point for the agent environment: what it is, the terms
> it uses, how the pieces fit, what state it is in, and how to build and verify it. It explains
> the system **as it stands** — the dated history of how it got here lives in [[log]], and the
> concepts it leans on are one click away in `concepts/`. If you are picking this work up cold,
> read this file in full first; it is written to be learnable, not just referenced.

## Table of contents
- [[#What this project is]]
- [[#Notation and key terms]]
- [[#Architecture at a glance]]
- [[#The five ideas the design rests on]]
- [[#What we learned that changed the plan]]
- [[#Where the project stands]]
- [[#Build and verify]]
- [[#Where everything is]]
- [[#Decisions not to relitigate]]
- [[#Gotchas that will bite]]
- [[#Remaining risks, in order]]

## What this project is

Build a deterministic, headless, structured environment for fheroes2 **battles**, so a policy can
be trained on them. The environment reads **true engine state** and selects from
**engine-generated legal actions** — never pixels, never synthetic input.

The first deliverable is deliberately narrow: creature-only field battles. No heroes, no spells,
no castles, no adventure map, and — inside the environment itself — no LLM, no RL loop, no
screenshot parsing, no GUI automation. The goal of Phase 1a is a *trustworthy substrate*;
learning comes after it, on top of it.

**Scope boundaries**

| In scope now (`creature_field_v1` / `simple_v1`) | Deferred (Phase 1b+) | Never (this branch) |
|---|---|---|
| Commander-free armies, 1–5 stacks/side | Wide units, flyers | Adventure-map control |
| Open field, fixed tile index 1 | Two-cell and all-adjacent attacks | Castles, sieges, towers |
| Single-cell walking creatures, shooters incl. blocked | Area shots, unusual ranged attacks | Heroes, spells, artifacts |
| MOVE / ATTACK / SKIP | Spellcasting | Screenshots, OCR, mouse/keyboard |
| Both sides engine-AI, hook-interceptable | Retreat/surrender as actions | Rendered pixels *(→ `play-harness`)* |

**Not to be confused with** the `play-harness` branch — a *different* project in this repo where
Claude plays through the real UI via frame dumps and an input FIFO. That one is pixels-and-GUI by
design. Keep them separate.

## Notation and key terms

Terms recur throughout the docs and the code; these are their exact meanings here.

| Term | Meaning |
|---|---|
| **Episode** | One complete battle, from arena construction to terminal state or round truncation. |
| **Round** | One pass in which every eligible unit acts once — what `Arena::Turns()` advances. **Not** an RL step. |
| **Decision** | One *full-fledged* unit choice — the branch of `UnitTurn` where the engine asks the AI or a human. The RL step boundary. Indexed by `engine_decision_index`. |
| **Arena** | `Battle::Arena`, the engine's live battle object. **One per process** (file-static singleton). |
| **Force** | The engine's battle-side unit container. Terminal state must be read from it *before* the arena is destroyed. |
| **World seed** | Our input seed; pinned by reseeding the thread-local RNG. Determines the **map seed**. |
| **Map seed** | Derived by the engine; drives battlefield obstacle placement. |
| **Combat seed** | `computeBattleSeed(tileIndex, mapSeed, attacker, defender)`; seeds the arena's damage/morale/luck RNG. |
| **Scenario** | A fixed battle definition: terrain, tile index, world seed, five explicit slots per side, limits. |
| **Fixture** | One of the five committed Milestone-1 scenarios used as regression anchors. |
| **State digest** | SHA-256 over canonical terminal state (`agent_terminal_v1`). The determinism test of record. |
| **Decision digest** | SHA-256 over the recorded decision stream (`agent_decisions_v0`). |
| **Candidate** | A legal action: canonical index + semantic metadata + engine-ready command parameters. |
| **Legal mask** | `uint8[793]`; `mask[i]=1` ⟺ a candidate with index `i` exists. Same enumeration as the candidates. |
| **Canonical action index** | Position in the fixed `Discrete(793)` space. Stable across states and machines. |
| **Profile** | *Observability* setting: `full_v1` (everything) or `observable_v1` (player-obtainable only). |
| **Modality** | *Representation* setting: `entities` (padded slot records) and/or `planes` (11×9×C tensor). Orthogonal to profile. |
| **Teacher** | The built-in `AI::BattlePlanner`. Source of demonstrations; never a human. |
| **Teacher coverage** | Fraction of teacher decisions that map onto a legal canonical action. Milestone 3's exit criterion. |
| **Gate** | A `verify_m*.sh` script; the milestone's pass/fail evidence. |
| **Worker** | `fheroes2_agent_worker` — the entry point outside both build systems' source globs. |

Board constants: **11 × 9 = 99 cells** (`Board::widthInCells/heightInCells/sizeInCells`), six hex
directions, action space **793**.

## Architecture at a glance

```
        ┌── engine (unmodified behavior, digest-proven) ──────────────────┐
        │  Arena::Turns ─► UnitTurn ─┬─ pending UI / standing / bad morale │
        │                            └─ FULL DECISION ──► DecisionController hook
        │                                    │                  ▲          │
        │                    battle_action_validation ◄──────────┘          │
        │                    (one legality implementation)                  │
        └───────────────────────────────┬───────────────────────────────────┘
                                        │
        ┌── src/fheroes2/agent/ (compiled in, entry-point-free) ───────────┐
        │  scenario ─► runner ─► recorder ─► trajectory (JSONL)            │
        │  capabilities (simple_v1 allowlist)                              │
        │  action_space: ONE enumeration ─► legal mask + candidates        │
        │  digest (SHA-256 + canonical writer)                             │
        └───────────────────────────────┬───────────────────────────────────┘
                                        │
        src/agent_worker/  ──►  [Milestone 4: JSONL protocol]  ──►  Python
```

Three properties hold this together:

1. **The engine's behavior is unchanged.** Every engine edit is either a verbatim lift or an
   optional hook that is inert when unused — and each is accepted only on unchanged digests.
2. **Legality has one implementation.** Execution and enumeration call the same functions, so a
   mask can never disagree with what the engine accepts.
3. **Everything is hashed.** Outcomes, decision streams, and (later) configs, so any drift is
   loud.

## The five ideas the design rests on

Each has a primer; skim these five paragraphs, then read the primer for whatever you are about to
touch.

1. **[[determinism-seeds-and-digests|Determinism]].** A battle is a pure function of *(world
   seed, army composition)*. We pin the world seed by reseeding the thread-local RNG, derive the
   combat seed through one shared helper, and hash the outcome. Digest equality is how every
   engine change on this branch was proven safe.

2. **[[battle-turn-dispatch|Turn dispatch]].** `Turns()` advances a whole round, so an RL step
   cannot be a call into the engine — the engine calls *us*, at exactly one branch of `UnitTurn`,
   and the observer must run before the command stream perturbs the RNG.

3. **[[legal-actions-and-masking|Legal actions and masking]].** A fixed `Discrete(793)` space
   plus a per-state boolean mask: provably still a correct policy gradient, and empirically the
   difference between 0.0 and ~0.85 win rate in comparable work. One enumeration yields both the
   mask and the candidate list.

4. **[[observation-design|Observation design]].** Structured state only: padded entity records
   plus an optional semantic 11×9×C plane tensor, filtered by an observability profile. Rendered
   pixels are rejected — ~14× cost, no measured benefit, and they would undo the asset-free core.

5. **[[teacher-coverage-and-behavior-cloning|Teacher coverage]].** The engine's own AI plays and
   we record it. The fraction of its decisions our action space can express is the sharpest
   completeness test we have — and the same recordings are the behavior-cloning dataset. No human
   play is involved anywhere.

## What we learned that changed the plan

Four Phase 0 findings overturned the original spec. Do not re-derive them; do not undo them.

1. **Headless battles need no game assets at all.** No display, audio, AGG, h2d, or HoMM2 data.
   Monster stats are hardcoded (`monster_info.cpp:384`) and obstacle setup uses ICN ids as plain
   enum tags (`battle_board.cpp:573`), so battle resolution never touches an asset. This was the
   spec's #1 risk; it is closed, and it is why pixels stay out (see
   [[observation-design]]).

2. **The world seed needs no engine change.** `Rand::CurrentThreadRandomDevice()` (`rand.cpp:85`)
   returns a *mutable reference* to a `thread_local PCG32`. The proposed `World` API overload
   became a deferred cleanup rather than a prerequisite.

3. **A second entry point needs no CMake refactor.** Linking a new `main` against the game objects
   minus `fheroes2.o` worked first try, no undefined symbols — the non-entry object set is already
   library-shaped.

4. **This repo has two build systems** — CMake and a plain Makefile under `src/dist`. The Makefile
   path is the one in regular use. *Open decision:* whether the agent target supports both.
   Settle it in Milestone 4.

Full evidence and the 25-row assumption table: `local_source_audit.md`.

## Where the project stands

| Phase | State | Evidence |
|---|---|---|
| Phase 0 — audit and headless spike | ✅ complete, reproduced on target hardware | `verify_phase0.sh` 7/7 |
| Milestone 1 — deterministic runner | ✅ complete | `verify_m1.sh` 4/4 |
| Milestone 2 — decision hook, passive logging | ✅ complete | `verify_m2.sh` 8/8 |
| Milestone 3 — `simple_v1` legal actions | ✅ complete — **top risk closed** | `verify_m3.sh` 8/8, 100 % teacher coverage (116/116) |
| Milestone 4 — JSONL worker + protocol v1 | ❌ next | — |
| Milestones 5–6 — Python env, hardening, benchmark | ❌ not started | — |
| Optional QA — one Battle Only battle via the UI | ◻ accepted risk, owner's call | normal-game regression only, not a training prerequisite |

**Branch:** `agent-env`, from `master`, pushed to `origin` (the `troyzhu` fork). Engine-source
changes are limited to two verbatim lifts (`battle_seed`, `battle_action_validation`), one
optional hook (`DecisionController`), and the additive `src/fheroes2/agent/` library — enumerate
with `git diff master --stat -- src/`.

**Milestone 4, concretely:** protocol v1 (spec §13), strict scenario JSON parsing (§11, vendored
JSON lib per §6.5), blocking external control through the existing hook, observation
serialization implementing ADR 0001 profiles **and** ADR 0004 `planes_v1`, lifecycle/error
handling (§5.4), and the `ENABLE_AGENT` CMake target (§6.2). Exit: scripted stdin/stdout tests
control both sides without a single invalid command.

Dated history of everything above: [[log]].

## Build and verify

```bash
# Toolchain — one time
xcode-select --install
brew bundle --file script/macos/Brewfile     # gettext, sdl2, sdl2_mixer, sdl2_image
brew install cmake                           # NOT in the Brewfile; only for the CMake path

# Source
git clone git@github.com:troyzhu/fheroes2.git && cd fheroes2
git switch agent-env

# Build and verify (all four gates)
make -C src/dist -j"$(sysctl -n hw.ncpu)"
./agent_play/spike/build_spike.sh
./agent_play/spike/verify_phase0.sh    # 7 passed — Phase 0 invariants
./agent_play/verify_m1.sh              # 4 passed — deterministic runner
./agent_play/verify_m2.sh              # 8 passed — hook inertness + passive logs
./agent_play/verify_m3.sh              # 8 passed — legal actions + 100% teacher coverage
```

**The two numbers that matter:** map seed `2227197244` and spike digest `2cfd42cb104aa5e7`. They
are machine-independent and have reproduced across three working trees, two machines, and both
optimization levels. **A mismatch is a real finding** — stop and investigate before building
anything on top, because it means determinism does not hold on your hardware.

Useful extras:

```bash
./src/agent_worker/fheroes2_agent_worker --list                        # fixtures
./src/agent_worker/fheroes2_agent_worker --runs 1 --audit-coverage     # coverage report
./src/agent_worker/fheroes2_agent_worker --runs 1 --trajectory-dir OUT # teacher trajectories
./src/agent_worker/fheroes2_agent_worker --capability-audit caps.json  # regenerate the audit
FHEROES2_WITH_ASAN=1 make -C src/dist -j8 && FHEROES2_WITH_ASAN=1 ./agent_play/spike/build_spike.sh
```

## Where everything is

| Path | What it is | Read when |
|---|---|---|
| `docs/agent/START_HERE.md` | this file — the system as it stands | first |
| `docs/agent/concepts/` | six concept primers (the teaching layer) | when a term or mechanism is unfamiliar |
| `docs/agent/log.md` | dated project history and evidence | when you need "why" or "when" |
| `docs/agent/decisions/` | accepted ADRs amending the spec (0001 profiles, 0002 action space, 0003 configs, 0004 planes) | before implementing the area they touch |
| `docs/agent/implementation_report.md` | review inventory: commits, engine surface, verification matrices | to review what exists |
| `docs/agent/references/` | Obsidian reference vault: `index`, `summary`, per-work notes, 43 local sources | to consult or extend the evidence base |
| `docs/agent/research_rl_approaches.md` | verified RL literature consolidation | before design work |
| `docs/agent/research_minimap_observations.md` | verified research on spatial/hybrid observations | before observation serialization |
| `docs/agent/local_source_audit.md` | Phase 0 report; assumption table; file:line evidence | before touching battle code |
| `docs/agent/benchmark_m2.md` | target-hardware performance (Mode A) | before sizing workers or models |
| `agent_play/fheroes2_agent_system_spec_v0.3.md` | the full 2,600-line design | when implementing a milestone |
| `agent_play/spike/README.md` | Phase 0 spike usage and limits | before running it |

The spec is large; §0.1 (validation), §4 (scope), §9 (hook), §10 (actions), §13 (protocol) and
§22 (definition of done) are the load-bearing sections. Where an ADR and the spec disagree, **the
ADR wins** — it was written later and with verified evidence.

## Decisions not to relitigate

- Agent work lives on `agent-env`, branched from `master`, **not** on top of `play-harness`. Both
  trees produce the identical battle digest, proving the harness patch inert, but the baseline
  stays clean anyway.
- Baseline is the current `master` lineage, **not** the `1.1.17` tag the spec pinned. The tag is
  42 commits behind and every spec-critical battle file is byte-identical.
- The world seed is pinned by RNG reseed (spec §7.2 option 1). The narrow `World` overload remains
  a deferred cleanup.
- The spike's FNV digest stays as a historical baseline; the environment uses SHA-256
  (`agent_terminal_v1`). They are intentionally not comparable.
- Legality is **extracted, never re-derived** — the tactical AI and the human interface already
  carry their own near-duplicates of those rules; a third copy is forbidden.
- Rendered pixels are out of the training environment permanently (ADR 0004).

## Gotchas that will bite

- **One arena per process.** `battle_arena.cpp:73` holds a file-static pointer and the constructor
  asserts it is null. Destroy each arena before the next; parallelism means multiple *processes*.
- **Input `Army` objects are not synced after battle.** Read terminal state from the `Force`
  objects *before* the arena is destroyed, or you get pre-battle numbers.
- **`Battle::Command` stores parameters reversed** and `GetNextValue()` pops (and mutates).
  Decode a *copy*, never the live command — see [[command-encoding-and-snapshots]].
- **The Makefile build has no `-DNDEBUG`.** `assert()` is live even at `-O3`. CMake `Release`
  *does* define `NDEBUG`; benchmark numbers from the two are not interchangeable.
- **`make -C src/dist clean` after every upstream sync.** `-MD` depfiles hard-code header paths,
  so a header rename breaks incremental builds with `No rule to make target`.
- **A `FHEROES2_DATA` root needs the repo's own `files/data/resurrection.h2d`**, not just the GOG
  extraction, or startup throws ~9 s in (`h2d.cpp:110`).
- **Repo paths with spaces** (this clone lives under `/Volumes/External Drive/`). Build scripts
  must pass flag lists as bash arrays, never unquoted strings. macOS ships bash 3.2, where
  `"${arr[@]}"` on an empty array trips `set -u` — use `${arr[@]+"${arr[@]}"}`.
- **`xgettext` must be the real one.** If the Makefile build fails in the `.pot` step, put
  Homebrew's `gettext` ahead of pyenv on `PATH`.

## Remaining risks, in order

The historical #1 risk — legal-action generation — **is closed** (Milestone 3: validators
extracted, 100 % teacher coverage, no live-arena probing). What is left:

1. **Protocol/JSON surface (Milestone 4).** A strict parser boundary and a vendored JSON
   dependency enter the tree. Stdout discipline and invalid-input handling are where workers rot
   (spec §5.4, §13, §18.7).
2. **BC→RL transition recipe.** No verified small-scale precedent exists
   (`research_rl_approaches.md`, open question 2). Expect iteration at the training stage.
3. **Learner throughput on M2 (MPS/CPU).** Unmeasured anywhere in the literature; the env is not
   the bottleneck, the learner will be. Benchmark before committing to model sizes.
4. **Phase 1b expansion** (wide, flying, special targeting). Deliberately deferred; re-audit the
   `simple_v1` assumptions before widening the allowlist (spec §4.4).
