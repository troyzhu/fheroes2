---
title: Project log — fheroes2 agent environment
type: log
updated: 2026-07-30
related_concepts: ["[[../implementation/determinism-seeds-and-digests]]"]
tags: [log, agent-env]
---

> **What this note is.** The dated record of what was done, in what order, and what evidence each step produced. It is separate from [[../README]], which describes the system as it stands. Read this for history: why something is the way it is, when a number changed, what was already tried. Newest entries at the bottom.
>
> Hardware is always named in full ("Apple M2") so it never reads as a milestone number.

## 2026-07-26 — Phase 0 on the Apple M3 MacBook

Specification v0.3 written and cross-checked against the source. Headless battle spike built under `agent_play/spike/`, with 7 of 7 automated experiments passing from a clean clone.

Four findings overturned the specification, detailed in [[../README#What we learned that changed the plan]]: headless battles need no game assets, the world seed needs no engine change, a second entry point needs no CMake refactor, and the repository has two build systems.

Indicative measurements only: about 5,000 episodes/s, 12.9 MB peak resident memory, and 2,000 sequential arenas in one process. The report is `local_source_audit.md`.

## 2026-07-27 — Target-machine verification (Apple M2 Mac mini) and Milestone 1

Phase 0 reproduced on the target hardware, 7 of 7 in both the `-O3` and the `FHEROES2_WITH_DEBUG` (`-O0 -g`) builds. Map seed `2227197244` and spike digest `2cfd42cb104aa5e7` matched the Apple M3 baseline and matched between build types, so battle resolution is stable across optimization levels.

A correction surfaced along the way. The `src/dist` Makefile never defines `NDEBUG`, so its "Release" build already had `assert()` live, and every Phase 0 run on both machines had been exercising the `ApplyAction` assertions. The runbook's earlier claim that Release compiles the assertions out holds only for CMake Release builds.

A portability fix followed: `build_spike.sh` broke on a repository path containing spaces, since the clone lives under `/Volumes/External Drive/`, so flag lists became bash arrays.

Benchmark written as `benchmark_m2.md`, Mode A only. About 4,600 episodes/s on the tiny-melee fixture (the Apple M2 runs roughly 9% slower than the Apple M3), 12 MB resident memory, 10 ms process startup, scaling linearly to four workers and then flattening, gaining only 17% from four to eight. Recommended default: four workers.

CMake regression configured and built to completion, and both the CMake and Makefile binaries launched to the main menu against a local `devdata/` root. Discovered in passing: a `FHEROES2_DATA` root also needs the repository's own `files/data/resurrection.h2d`, or startup throws about nine seconds in.

Sanitizers ran clean. A full `FHEROES2_WITH_ASAN=1` build, which implies UBSan, with the spike relinked using matching flags, produced zero reports across 1,900 episodes and five compositions, all deterministic, at about 700 episodes/s and 207 MB resident memory under instrumentation.

Milestone 1 completed, `verify_m1.sh` passing 4 of 4. `Battle::computeBattleSeed` was extracted to `battle_seed.{h,cpp}`; the agent library under `src/fheroes2/agent/` gained scenario fixtures and validation, the SHA-256 `agent_terminal_v1` digest, and the headless runner; the worker entry point landed in `src/agent_worker/`. Ten identical runs per fixture, and two fresh processes produced byte-identical output. The `m1_tiny_melee` fixture reproduces the historical map and combat seeds, `2227197244` and `1356111745`.

Literature consolidated into [[research-runs/2026-07-27-rl-approaches]] from 23 sources, with 25 claims put through three-vote adversarial verification and 23 confirmed. It produced ADR 0001 on observability profiles and ADR 0002 on the fixed canonical action space with a mask, the latter amending specification §10.4 after the sweep found that no verified codebase consumes variable-length candidate lists.

## 2026-07-28 — Milestone 2

The decision hook and passive teacher logging landed, `verify_m2.sh` passing 8 of 8.

`Battle::DecisionController` became an optional constructor parameter on `Arena`, dispatched in the full-fledged branch of `UnitTurn`, with the observer firing before the command stream updates the combat generator. Automatic bad-morale and pending-interface actions are never intercepted. `GetEngineDecisionIndex()` counts full-fledged decisions, and a good-morale re-decision counts separately.

On the agent side: `snapshotCommand` decoding a copy through `GetNextValue`, the `PassiveTeacherRecorder`, the SHA-256 `agent_decisions_v0` decision-stream digest, the JSONL `TrajectoryWriter` emitting the `agent_passive_v0` schema with no wall-clock fields so files stay byte-identical across processes, and the worker's `--trajectory-dir` flag.

Inertness evidence: the spike digest and all five golden fixture digests were unchanged, including with the recorder actively attached.

Documentation gained `implementation_report.md` as a review inventory, and ADR 0003 on versioned configuration, under which every artifact embeds its resolved configuration, that configuration's hash, and the commit. The Battle Only interface check was downgraded to optional quality assurance at accepted risk, since it is a normal-game regression checkbox rather than a training prerequisite.

## 2026-07-29 — Minimap research, reference vault, Milestone 3

The coarse-minimap question was researched into [[research-runs/2026-07-29-spatial-observations]], from 20 sources with 24 of 25 claims verified. It produced ADR 0004: a semantic `planes_v1` modality joins the schema, and rendered pixels are rejected for the training environment. The decisive evidence was that SC2's feature layers were never RGB, and that Griddly's vector observer matches its pixel observers on task performance at roughly 14 times the throughput.

The reference vault was built under `references/`, holding 43 local source files at about 59 MB, 15 per-work notes, an index, and a consolidated synthesis, all reproducible through `fetch_references.sh`.

Milestone 3 completed, `verify_m3.sh` passing 8 of 8, which closed the project's top risk.

Command legality was lifted verbatim out of `battle_action.cpp`, covering the three anonymous-namespace helpers and the `checkParameters` lambdas of `ApplyAction{Move,Attack,Skip}`, into `battle_action_validation.{h,cpp}`, and the engine now executes through the lifted functions. The attack gate became `resolveAttackCommand()`, which returns the resolved target and direction that the lambda previously computed and discarded.

A machine-generated 71-monster capability audit produced the `simple_v1` allowlist at `python/fheroes2_agent/data/monster_capabilities_v1.json`, and scenario validation now enforces it. The canonical 793-slot action space emits its mask and candidates from one enumeration. Built-in teacher matching reached 100% coverage over 116 of 116 decisions, with a minimum of 5 candidates per decision.

Golden digests were unchanged with enumeration running at every decision, which proves the candidate generator consumes no randomness.

## 2026-07-30 — Documentation restructure

Reference-style content was split out of the entry point. [[../README]] became a teaching document carrying an MDP on-ramp, notation, architecture, and current state; this log took the chronology; and `concepts/` gained six primers: [[../implementation/determinism-seeds-and-digests]], [[../implementation/battle-turn-dispatch]], [[../implementation/legal-actions-and-masking]], [[../implementation/observation-design]], [[../implementation/command-encoding-and-snapshots]], and [[../implementation/teacher-coverage-and-behavior-cloning]].

The rewrite was driven by two instruments rather than taste. The project's `WRITING_STYLE` contract supplies a deterministic lint, which the first draft breached on every file for em-dash density and on nine files for bold density. An independent critical review supplied 25 prioritized defects, of which the load-bearing ones were a contradictory masking constant, a missing MDP on-ramp, an overloaded use of the word "profile" across three distinct axes, and a bullet-grouped bibliography where a scannable table belonged.

Lint counts after the rewrite, across all eleven documents:

| check | worst file | threshold | breach |
|---|---|---|---|
| em-dash per 1,000 characters | 0.90 | 1 | none |
| bold per 1,000 words | 2.89 | 6 | none |
| label-colon bullets | 0 | 2 | none |
| banned constructions | 3 | 0 | exempt, all three are the `play-harness` branch name matching the hype-verb regex |
| question headings | 0 | 0 | none |
| exclamation marks | 0 | 0 | none |
| paragraphs over 160 words | 0 | 0 | none |

A second review round on 2026-07-30 raised four further issues. Hard-wrapped prose at about 90 characters was rendering as forced line breaks in Obsidian, whose default treats a newline as a break, so all 37 documents were rewrapped to one paragraph per line, matching the reference repository's own style at a 200-to-330-character median. Confidence markers written as `[unverified]` and `[gap]` render as unresolved reference links, so they became parenthetical. The conceptual on-ramp was missing its foundation, so `concepts/rl-for-games.md` now states the general vocabulary of RL game environments and `concepts/fheroes2-battles-vs-other-games.md` explains the Heroes battle domain and places it against Heroes III, microRTS, StarCraft, NetHack, Wesnoth, and board games. The open-source codebases behind the evidence had no practical orientation, so `references/repos.md` describes what each one contains and where to look inside it.

The lint was then extended across every document in the tree rather than the eleven restructured ones, which cleared bold and em-dash breaches in the four decision records, the two research reports, the implementation report, the benchmark, the Phase 0 audit, and all fourteen reference notes. The only remaining regex hits are the `play-harness` branch name and a C++ `!=` operator inside inline code.

The primers also moved onto the tutorial template, gaining a motivation section, an intuition drawn from machine learning, and a grounded comparison against the alternatives that were rejected. A primer taught in isolation, with no comparison, counts as incomplete under that contract.

## Running verification history

Every gate has passed on every run at HEAD. Current expected output:

| Gate | Expected |
|---|---|
| `agent_play/spike/verify_phase0.sh` | `7 passed, 0 failed`, seed `2227197244`, digest `2cfd42cb104aa5e7` |
| `agent_play/verify_m1.sh` | `4 passed, 0 failed`, `deterministic=yes` |
| `agent_play/verify_m2.sh` | `8 passed, 0 failed` |
| `agent_play/verify_m3.sh` | `8 passed, 0 failed`, `teacher_coverage=complete` |
| `cmake --build build/cmake-regression` | builds to completion |

## 2026-07-30 — Navigation restructure

A third review round found the structure cumbersome to navigate for its two real purposes, understanding the research and understanding the implementation, and found the two new foundation primers buried under `concepts/` when they are entry-point material rather than reference.

The two were merged into one promoted document, `rl-and-the-battle-domain.md`, sitting beside START_HERE at the top level. It runs in three parts: the vocabulary of RL game environments, the Heroes battle domain restated in that vocabulary with a comparison against seven other environments, and what the comparison implies for the design.

START_HERE gained a "Where to start" table as its first section, routing by intent across six paths, and absorbed the MDP framing into a "The problem in one page" section rather than deferring it.

Both literature reports moved into `references/` as `report-rl-approaches.md` and `report-spatial-observations.md`, so the top level now holds orientation and status only, `concepts/` holds the six implementation deep dives, `decisions/` holds the ADRs, and `references/` holds every piece of literature.

`references/summary.md` gained inline links to the per-work notes, the ADRs, the concept primers, and the two reports, since it previously named sources without linking any of them.

## 2026-07-30 — Panel review and documentation restructure

Three reviewers read the corpus in role: a machine-learning graduate student new to game reinforcement learning, a game-engine developer with no reinforcement-learning background, and a reinforcement-learning researcher new to game environments. They returned roughly sixty findings, of which the following were defects rather than matters of taste.

The engine reviewer found a decode loop published as shipped source that does not work. It re-read `size()` in the loop condition while `GetNextValue()` pops, so it extracted one parameter of two for MOVE and three of five for ATTACK. The real function hoists the bound, and the primer had dropped exactly the line that makes it correct.

The same reviewer found the command-stream update documented inside the fourth branch of `UnitTurn` when it sits outside the branch chain at lines 545 to 547. The consequence is substantive. Automatic bad-morale commands pass through `updatePCG32Stream` while the observer never sees them, and a good-morale command appended inside the apply loop never passes through it at all. Every `battle_arena.cpp` citation in the corpus was also stale, resolving against `master` rather than against the branch whose hook had shifted the lines.

The reinforcement-learning reviewer found the masking-correctness claim to be a non-sequitur. Differentiability is not what licenses masking, since replacing a logit with a constant discards it; what licenses it is that the mask depends on the state and not on the parameters. The PPO failure mode was also described incorrectly, since PPO-clip carries no KL term and the real defect is that the ratio is not one at the current iterate. Further corrections covered the missing initial-state distribution in the tuple, the claim that `full_v1` equals the true state when the generator position is in neither profile, the two-player reduction, the asymmetric-critic bias result, the DAgger description, and the reporting of 116 of 116 without its interval.

The documentation tree moved out of `docs/`, which is the Jekyll source for the project's published website and excluded nothing, into `agent_play/docs/`. It now separates reading material from records: a routing `README.md`, an `overview.md`, the domain primer, then `research/`, `implementation/`, `decisions/`, and an `archive/` holding this log, the benchmarks, the raw research runs, and the fetched sources.
