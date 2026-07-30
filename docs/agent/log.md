---
title: "Project log — fheroes2 agent environment"
type: log
tags: [log, agent-env]
updated: 2026-07-30
---

> **What this is.** The dated record of what was done, in what order, with the evidence each step
> produced. It is deliberately *separate* from [[START_HERE]], which explains the system as it
> stands now. Read this when you need history — "why is it like this?", "when did that number
> change?", "what did we already try?". Newest entries at the bottom.

## 2026-07-26 — Phase 0 (M3 MacBook)

Spec v0.3 written and source-cross-checked. Headless battle spike built
(`agent_play/spike/`), 7/7 automated experiments passing from a clean clone.

Four findings overturned the spec (detail in [[START_HERE#What we learned that changed the plan]]):
headless battles need **no game assets**; the world seed needs **no engine change**; a second
entry point needs **no CMake refactor**; the repo has **two build systems**.

Measured (indicative only): ~5,000 episodes/s, 12.9 MB peak RSS, 2,000 sequential arenas in one
process. Report: `local_source_audit.md`.

## 2026-07-27 — Target-machine verification (Mac mini M2) and Milestone 1

**Phase 0 reproduced on the target hardware.** 7/7 in both the `-O3` and the
`FHEROES2_WITH_DEBUG` (`-O0 -g`) builds. Map seed `2227197244` and spike digest
`2cfd42cb104aa5e7` identical to the M3 baseline **and** identical between build types — battle
resolution is stable across optimization levels.

*Correction recorded:* the `src/dist` Makefile never defines `-DNDEBUG`, so its "Release" build
already had `assert()` live. Every Phase 0 run on both machines had been exercising the
`ApplyAction` asserts; the runbook's earlier "Release compiles out the asserts" rationale holds
only for CMake Release builds.

*Portability fix:* `build_spike.sh` broke on a repo path containing spaces
(`/Volumes/External Drive/…`); flag lists are now bash arrays.

**Benchmark written** — `benchmark_m2.md` (Mode A only): ~4,566 eps/s tiny-melee (M2 ≈ 9 % slower
than M3), 12 MB RSS, 10 ms process startup, scaling linear to 4 workers then flat (+17 % from 4→8).
Recommended default: **4 workers**.

**CMake regression** — configures and builds to 100 %; both CMake and Makefile binaries launch to
the main menu against a local `devdata/` root. Discovered: a `FHEROES2_DATA` root also needs the
repo's own `files/data/resurrection.h2d`, or startup throws ~9 s in.

**Sanitizers** — full `FHEROES2_WITH_ASAN=1` build (implies UBSan), spike relinked with matching
flags. 1,900 episodes across five compositions: **zero reports**, all deterministic (~700 eps/s,
207 MB RSS under instrumentation).

**Milestone 1 complete** (`verify_m1.sh` 4/4). Shared `Battle::computeBattleSeed` extracted to
`battle_seed.{h,cpp}`; agent library under `src/fheroes2/agent/` (scenario fixtures + validation,
SHA-256 `agent_terminal_v1` digest, headless AI-vs-AI runner); worker entry point in
`src/agent_worker/`. Ten identical runs per fixture; two fresh processes byte-identical.
`m1_tiny_melee` reproduces the historical map/combat seeds (`2227197244` / `1356111745`).

**RL literature consolidated** — `research_rl_approaches.md` (23 sources, 25 claims verified by
3-vote adversarial verification, 23 confirmed). Produced **ADR 0001** (observation profiles) and
**ADR 0002** (fixed canonical action space + mask), the latter amending spec §10.4 after finding
that no verified codebase consumes variable-length candidate lists.

## 2026-07-28 — Milestone 2

**Decision hook and passive teacher logging** (`verify_m2.sh` 8/8).

`Battle::DecisionController` added as an optional constructor parameter to `Arena`, dispatched in
the full-fledged branch of `UnitTurn`, with the observer firing *before* the command stream
updates the combat RNG. Automatic bad-morale and pending-UI actions are never intercepted.
`GetEngineDecisionIndex()` counts full-fledged decisions (good-morale re-decisions count
separately).

Agent side: `snapshotCommand` (copy-decode via `GetNextValue`), `PassiveTeacherRecorder`,
SHA-256 `agent_decisions_v0` decision-stream digest, JSONL `TrajectoryWriter`
(`agent_passive_v0` — no wall-clock fields, so files are byte-identical across processes), worker
`--trajectory-dir`.

**Inertness evidence:** spike digest and all five golden fixture digests unchanged, *including*
with the recorder actively attached.

**Documentation:** `implementation_report.md` (review inventory) and **ADR 0003** (versioned YAML
configs; every artifact embeds its resolved config + hash + commit). The Battle Only UI item was
downgraded to optional QA / accepted risk — it is a normal-game regression checkbox, not a
training prerequisite.

## 2026-07-29 — Minimap research, reference vault, Milestone 3

**Coarse-minimap question researched** — `research_minimap_observations.md` (20 sources, 24/25
claims verified). Produced **ADR 0004**: a semantic `planes_v1` modality joins the schema; true
pixel rendering is rejected for the training environment. Key evidence: SC2's feature layers were
never RGB; Griddly's vector observer matches pixel observers on performance at ~14× throughput.

**Reference vault built** — `references/` with 43 local source copies (~59 MB), 15 Obsidian
notes, index, and consolidated summary; reproducible via `fetch_references.sh`.

**Milestone 3 complete** (`verify_m3.sh` 8/8) — *the project's top risk closed*.

Command legality lifted **verbatim** out of `battle_action.cpp` (the three anonymous-namespace
helpers plus the `checkParameters` lambdas of `ApplyAction{Move,Attack,Skip}`) into
`battle_action_validation.{h,cpp}`; the engine now executes through the lifted functions. The
attack gate became `resolveAttackCommand()`, returning the resolved target/direction the lambda
used to discard.

Machine-generated 71-monster capability audit → `simple_v1` allowlist
(`python/fheroes2_agent/data/monster_capabilities_v1.json`); scenario validation enforces it.
Canonical `Discrete(793)` action space with mask and candidates from **one** enumeration.
Built-in teacher matching: **100 % coverage, 116/116 decisions**, min candidates ≥ 5.

Golden digests unchanged with enumeration running at every decision — proving the candidate
generator consumes no randomness.

## 2026-07-30 — Documentation restructure

Split the reference-style content out of the entry point: [[START_HERE]] became a teaching
document (notation, concepts, architecture, current state), this log took the chronology, and
`concepts/` gained six primers ([[determinism-seeds-and-digests]], [[battle-turn-dispatch]],
[[legal-actions-and-masking]], [[observation-design]], [[command-encoding-and-snapshots]],
[[teacher-coverage-and-behavior-cloning]]). `references/summary.md` restructured the same way.

---

## Running verification history

Every gate, every time it was run at HEAD, has passed. Current expected output:

| Gate | Expected |
|---|---|
| `agent_play/spike/verify_phase0.sh` | `7 passed, 0 failed` · seed `2227197244` · digest `2cfd42cb104aa5e7` |
| `agent_play/verify_m1.sh` | `4 passed, 0 failed` · `deterministic=yes` |
| `agent_play/verify_m2.sh` | `8 passed, 0 failed` |
| `agent_play/verify_m3.sh` | `8 passed, 0 failed` · `teacher_coverage=complete` |
| `cmake --build build/cmake-regression` | builds to 100 % |
