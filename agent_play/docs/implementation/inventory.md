# Implementation report — `agent-env` branch

> Review-oriented inventory of everything implemented on this branch, as of 2026-07-27. Covers commits `d02c9236d..85c30a11f` (six commits on top of the Phase 0 baseline `b16e6f698`). The spec (§6.1) calls for this file; it will grow per milestone.

## How to review this branch in ~20 minutes

1. Read this file, then skim [[../overview]] for the current status and the runbook.
2. Run the verification suite (expected outputs listed in the matrix below):

   ```bash
   make -C src/dist -j"$(sysctl -n hw.ncpu)"
   ./agent_play/spike/build_spike.sh
   ./agent_play/spike/verify_phase0.sh      # 7 passed — Phase 0 invariants
   ./agent_play/verify_m1.sh                # 4 passed — Milestone 1 exit criteria
   ```

3. Review the commits in order, each is self-contained:

   | Commit | What it is | Review focus |
   |---|---|---|
   | `d02c9236d` | M2-mini verification + Mode A benchmark | docs + `bench_m2.sh`; no engine code |
   | `098041b06` | Runbook completion: CMake regression, sanitizers | build-script changes only |
   | `22867a26e` | Engine refactor: `Battle::computeBattleSeed` extraction | the only behavior-relevant engine diff, see below |
   | `21f1fb1ed` | Milestone 1: agent library, worker, tests | the bulk of new code |
   | `a0e1540a2` | Doc polish | trivial |
   | `85c30a11f` | Verified RL research + ADRs 0001/0002 | design review, read the ADRs |
   | `45f73397b` | Implementation report + ADR 0003 (config management) | docs only |
   | `e155f9b22` | Engine seam: optional `Battle::DecisionController` in `Arena::UnitTurn` | the second behavior-relevant engine diff, see below |
   | `8b884582c` | Milestone 2: `CommandSnapshot`, passive teacher logging, `verify_m2.sh` | agent-side code + gate |
   | `dc486827d` | Minimap research + ADR 0004 | docs only |
   | `18f4f6ef1` | Obsidian reference vault (43 local sources) | docs only |
   | `d689c4e3e` | Engine refactor: legality lifted into `battle_action_validation.{h,cpp}` | the third behavior-relevant engine diff, verbatim lift, digest-proven |
   | `7015980ba` | Milestone 3: capability audit, `Discrete(793)` action space, teacher coverage, `verify_m3.sh` | the top-risk milestone |

## Engine-source surface (what could possibly affect the game)

Everything the branch changes under `src/` (enumerate: `git diff master --stat -- src/`):

- `src/fheroes2/battle/battle_main.cpp`, the anonymous-namespace `computeBattleSeed` moved out verbatim (−29/+1 lines). Proof of inertness: Phase 0 digest `2cfd42cb104aa5e7` identical before/after the refactor; 7/7 verification green.
- `src/fheroes2/battle/battle_seed.{h,cpp}`, new shared helper (spec §7.3). Pinned by golden-value unit tests (four engine-derived seeds + sensitivity invariants).
- `src/fheroes2/battle/battle_decision_controller.h` + `battle_arena.{h,cpp}` (Milestone 2), optional `DecisionController *` constructor parameter (default `nullptr` preserves every caller), dispatch + observer in the full-fledged branch of `UnitTurn` (§9.2; observer runs before the command stream updates the combat RNG), and a full-fledged decision counter (`GetEngineDecisionIndex`, §9.4). Proof of inertness: spike digest `2cfd42cb104aa5e7` and all five Milestone 1 fixture digests unchanged; with the passive recorder *attached*, state digests still equal the goldens (the observer perturbs nothing).
- `src/fheroes2/battle/battle_action_validation.{h,cpp}` + `battle_action.cpp` (Milestone 3), the `checkParameters` lambdas of `ApplyAction{Move,Attack,Skip}` and the anonymous-namespace helpers (`calculateAttackTarget`, `calculateAttackDirection`, `checkMoveParams`→`isMoveDestinationValid`) moved verbatim into a shared module; `ApplyActionX` now gates through `isMoveCommandValid`/`isSkipCommandValid`/ `resolveAttackCommand` (the latter returns the resolved target/direction the lambda used to discard). `ERROR_LOG`/`assert` stay at the call sites. Proof of inertness: all golden digests byte-identical across Phase 0 / M1 / M2 / M3 gates.
- `src/fheroes2/agent/` (6 files), entry-point-free library, compiled into the normal executable by both build systems (spec §6.1 sanctions this) but unreachable from game code: no game translation unit includes an `agent_*` header. Launch smoke tests of both binaries pass.
- `src/agent_worker/`, worker `main.cpp` + build script; outside both source globs, cannot affect the game.

Everything else on the branch is docs, scripts under `agent_play/`, or tests.

## Component inventory

| Path | Purpose | Spec § | Tested by |
|---|---|---|---|
| `src/fheroes2/battle/battle_seed.{h,cpp}` | One combat-seed implementation for engine + agent; trajectory-compatibility contract | §7.3 | `test_battle_seed` (14 checks, 4 golden values); digest invariance |
| `src/fheroes2/agent/agent_scenario.{h,cpp}` | C++ scenario struct, §11.1-subset validation, 5-fixture M1 suite | §11 | `test_agent_scenario` (11 checks) |
| `src/fheroes2/agent/agent_digest.{h,cpp}` | `DigestWriter` (LE fixed-width, length-prefixed) + self-contained SHA-256 | §12.5 | `test_agent_digest` (11 checks incl. FIPS vectors, 1M-byte input) |
| `src/fheroes2/agent/agent_battle_runner.{h,cpp}` | Headless AI-vs-AI episode lifecycle; terminal state read pre-destruction; canonical digest `agent_terminal_v1` | §7.2, §8 | `verify_m1.sh` ten-run + cross-process checks |
| `src/agent_worker/main.cpp` + `build_worker.sh` | `fheroes2_agent_worker` CLI (`--runs/--fixture/--list/--quiet`); JSONL protocol replaces it at M4 | §6.1 | `verify_m1.sh` |
| `agent_play/tests/` + `build_and_run_tests.sh` | 36 assert-based checks, relink strategy, sanitizer-aware | §18.2 | self |
| `agent_play/verify_m1.sh` | M1 gate: unit tests + worker build + ten-run determinism + cross-process byte-identity | §20-M1 | self |
| `src/fheroes2/agent/agent_command_snapshot.{h,cpp}` | Typed decode of a `Battle::Command` copy via `GetNextValue()` (never consumes the original); canonical keys | §10.5 | `test_agent_command_snapshot` (16 checks) |
| `src/fheroes2/agent/agent_trajectory.{h,cpp}` | `DecisionRecord`/`EpisodeRecording`, SHA-256 decision-stream digest (`agent_decisions_v0`), JSONL `TrajectoryWriter` (`agent_passive_v0`, no wall-clock fields) | §15 subset | `verify_m2.sh` byte-identity |
| `PassiveTeacherRecorder` in `agent_battle_runner.cpp` + worker `--trajectory-dir` | Passive capture of every full-fledged built-in-AI decision; worker reports decision digests | §15.2 | `verify_m2.sh` |
| `agent_play/verify_m2.sh` | M2 gate: 8 checks, unit tests, golden state digests with recorder attached, one decision digest per fixture × 10 runs, null-controller spike digest, cross-process trajectory byte-identity | §20-M2 | self |
| `src/fheroes2/agent/agent_capabilities.{h,cpp}` + `python/fheroes2_agent/data/monster_capabilities_v1.json` | Machine-generated 71-monster audit; action-space-based `simple_v1` allowlist; scenario profile gate | §4.2, §11.1 | `test_agent_capabilities` (11 checks) |
| `src/fheroes2/agent/agent_action_space.{h,cpp}` | ADR-0002 canonical `Discrete(793)` indexing; one enumeration → legal mask + sorted candidates, validated through `battle_action_validation`; teacher-decision resolver | §10.1–10.4, §10.6 | `test_agent_action_space` (14 checks) + `verify_m3.sh` |
| Coverage audit (recorder + worker `--audit-coverage`, `--capability-audit`) | Per-decision candidate counts and teacher matching; COVERAGE reporting in the determinism verdict | §10.6 | `verify_m3.sh` |
| `agent_play/verify_m3.sh` | M3 gate: 8 checks, unit tests, determinism+coverage verdict, golden digests under enumeration, 100 % coverage, min-candidates, cross-process byte-identity, audit sanity | §20-M3 | self |
| `agent_play/spike/bench_m2.sh` + `agent_play/docs/benchmark_m2.md` | Mode A benchmark deliverable on target hardware | §19 | reproducible via script |
| `agent_play/docs/references/report-rl-approaches.md` | Adversarially verified literature consolidation (23 sources) | — | citations + verification votes inline |
| `agent_play/docs/decisions/000{1,2,3}-*.md` | Accepted ADRs: observation profiles; action space; config management | amend §10, §12 | — |

## Verification matrix (all green as of `85c30a11f` on the M2 mini)

| Property | Command | Expected |
|---|---|---|
| Phase 0 invariants (headless, determinism, reuse, seeds) | `verify_phase0.sh` | `7 passed, 0 failed`, map seed `2227197244`, digest `2cfd42cb104aa5e7` |
| M1 exit: ten identical runs × 5 fixtures | `verify_m1.sh` | `4 passed, 0 failed`, `deterministic=yes` |
| M2 exit: null-controller inertness + deterministic passive logs | `verify_m2.sh` | `8 passed, 0 failed` |
| M3 exit: 100 % teacher coverage + candidates everywhere + digest inertness of enumeration | `verify_m3.sh` | `8 passed, 0 failed`, `teacher_coverage=complete` |
| Cross-machine | (M3 MacBook baseline) | same seed + digest, reproduced |
| Build-type invariance | Debug `-O0` vs Release `-O3` | identical digests, verified |
| Sanitizers | `FHEROES2_WITH_ASAN=1` build + suite | 0 reports (1 900 spike episodes + full M1 suite) |
| Both build systems | Makefile + `cmake --build` | both compile; both binaries reach main menu with `devdata/` |
| Normal-game inertness | digest + launch smoke | unchanged / alive, clean logs |

## Deviations from the spec (all documented)

1. World seed via thread-local RNG reseed, not a `World` API overload, spec §7.2 option 1, adopted by Phase 0; overload remains a deferred cleanup.
2. Baseline is master lineage, not the `1.1.17` tag (audit drift table).
3. Makefile build path is primary; CMake worker target deferred to M4 (spec assumed CMake-only).
4. §10.4 ephemeral action ids amended by ADR 0002 (fixed canonical index + mask).
5. §12 single observation schema extended by ADR 0001 (`full_v1`/`observable_v1`).
6. M1 fixture "longer balanced" reaches only ~5 rounds, engine battles resolve fast at these scales; acceptable until richer scenarios exist.

## Not implemented yet

- M4 (next): JSONL protocol v1, strict scenario JSON parsing (vendored JSON lib per §6.5), blocking external control via the `DecisionController` seam, observation serialization (ADR 0001 profiles + ADR 0004 `planes_v1`), `ENABLE_AGENT` CMake target.
- M4: JSONL worker + protocol v1 + scenario JSON parsing; CMake `ENABLE_AGENT` target.
- M5: Python package, policies, replay, worker pool. M6: hardening + full benchmark modes B/C.
- Runbook §6.3 human item: one Battle Only battle through the real UI, accepted-risk optional QA (battle path digest-proven unchanged; both binaries launch); not a training or data prerequisite. Teacher demonstrations come from the built-in AI, automated, at M2.
