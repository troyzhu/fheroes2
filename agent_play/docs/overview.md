---
title: fheroes2 agent environment — start here
type: entry-point
status: active
goal: "A deterministic, headless, structured battle environment for fheroes2 that a policy can be trained on"
branch: agent-env
date_started: 2026-07-26
updated: 2026-07-30
related_concepts: ["[[rl/rl-and-the-battle-domain]]", "[[implementation/legal-actions-and-masking]]", "[[implementation/observation-design]]"]
tags: [agent-env, rl-environment, fheroes2, entry-point]
---

> **What this note is.** The front door. It says what the project is, gives the terms and the shape of the system, states where it stands, and tells you how to build and verify it. Two companions carry the depth: [[rl/rl-and-the-battle-domain]] explains reinforcement learning and the Heroes battle domain from scratch, and [[archive/log]] holds the dated history.

## Where to start

Pick the row that matches why you are here.

| If you want to | Read, in order |
|---|---|
| Understand the problem, with no RL or fheroes2 background | [[rl/rl-and-the-battle-domain]], then this note |
| Get it building and see the current state | This note, sections [[#Build and verify]] and [[#Where the project stands]] |
| Understand the research and the evidence | [[research/findings]], then [[research/prior-art]] for the codebases, then [[research/README]] for a specific source |
| Understand what is implemented and how | [[implementation/inventory]] for the inventory, then [[implementation/README]] for how each mechanism works |
| Understand why a decision was taken | `decisions/`, six accepted records amending the specification |
| Follow one inference decision end to end | [[implementation/inference-walkthrough]], a real battle under the microscope |
| Reconstruct history | [[archive/log]] |

## Table of contents

Orientation, in reading order.

- [[#The problem in one page]], the environment stated as a Markov decision process
- [[#Notation]], both vocabularies
  - [[#Mathematical symbols]], the symbol contract and what this project adds
  - [[#Project terms]], episode, decision, digest, fixture, gate
  - [[#What is recap and what is new]], which topics this tree carries and which it assumes
  - [[#Reading Zhao alongside this]], a translation table for that textbook
- [[#Scope]], what Phase 1a covers and what it deliberately excludes
- [[#Architecture at a glance]]
- [[#The five ideas the design rests on]]
- [[#What we learned that changed the plan]]

Practical.

- [[#Where the project stands]], milestones and gate results
- [[#Build and verify]], the commands
- [[#Where everything is]], the map of every document and its purpose
- [[#Decisions not to relitigate]]
- [[#Gotchas that will bite]]
- [[#Remaining risks, in order]]

## The problem in one page

Build a deterministic, headless, structured environment for fheroes2 battles so a policy can be trained on them. The environment reads true engine state and selects from engine-generated legal actions, never pixels and never synthetic input.

Stated as a Markov decision process, which [[rl/rl-and-the-battle-domain]] develops from first principles:

| MDP element | Here |
|---|---|
| State $s$ | The battle position: which stacks stand where, with what counts, hit points, and shots remaining. |
| Observation $o$ | A serialization of $s$: padded entity records, optionally an `11 × 9 × C` plane tensor, filtered by an observability profile. |
| Action $a$ | What the active stack does. One slot of a fixed 793-wide discrete space, with a per-state legality mask; the count derives from the board under [[#Project terms]]. |
| Transition $P(s' \mid s, a)$ | The fheroes2 engine. Stochastic through damage, morale, and luck rolls, but seeded, so a fixed seed makes an episode reproducible. |
| Reward $R$ | Terminal only, chosen in [[decisions/0005-training-and-reward]] and implemented in `python/fheroes2_agent/env.py`: outcome plus survival margin, with the strength-priced variant, opt-in difficulty weighting, and engine-grounded stall semantics. The environment itself still emits outcomes, not rewards. |
| Episode | One battle, from arena construction to a terminal state or round truncation. |
| Policy $\pi(a \mid s)$ | `python/fheroes2_agent/policy.py`, the masked `BattlePolicy`; the C++ environment still ships learner-free, and the whole training stack lives on the Python side, gated by `verify_agent.sh`. |

Two structural facts distinguish this from a Gymnasium environment you would write yourself, and both follow from the engine rather than from taste.

Control is inverted. Nothing calls `env.step(a)`, because `Arena::Turns()` advances a whole round rather than one decision. The engine owns the call stack and calls into our hook at each decision point, so the worker blocks there until an action arrives. A Python-side adapter re-presents a normal `step()` on top of that, the same trampoline PySC2 uses.

Only one episode runs per process at a time. The engine's arena is a file-static singleton whose constructor asserts it is the only live one, so vectorization means multiple processes.

Determinism here is conditional, exactly like `env.reset(seed=k)`. Seed in, episode out. The distribution a policy trains on is over world seeds and army compositions; the five committed fixtures pin regression anchors, not the training set. Within an episode the dynamics remain stochastic from the policy's point of view, since damage, morale, and luck are drawn from a combat-seeded generator that never appears in the observation.

## Notation

Two vocabularies meet in this project and both are fixed here. The mathematical symbols match the owner's own reinforcement-learning study notes so that material written here continues from them rather than running a second vocabulary alongside. The engineering terms are this project's own and are defined nowhere else.

### Mathematical symbols

| Symbol | Meaning | Defined in |
|---|---|---|
| $\mathcal{S}$, $\mathcal{A}$, $\mathcal{A}(s)$ | State space, action space, and the legal actions at $s$ | [[rl/rl-and-the-battle-domain]] |
| $P(s' \mid s, a)$, $R(s, a, s')$ | Transition function and reward | [[rl/rl-and-the-battle-domain]] |
| $\gamma$, $\lambda$ | Discount factor, and the trace parameter in generalized advantage estimation | [[rl/rl-methods]] |
| $\tau$ | Trajectory, one episode of $s_0, a_0, r_1, s_1, \ldots$ | [[rl/rl-methods]] |
| $G_t = \sum_{k \ge 0} \gamma^k r_{t+k+1}$ | Discounted return from step $t$ | [[rl/rl-methods]] |
| $\pi(a \mid s)$, $\pi_\theta(a \mid s)$ | Policy, and the same policy parameterized by $\theta$ | [[rl/rl-methods]] |
| $\pi^{*}$ | The teacher policy, here `AI::BattlePlanner` | [[rl/training-design]] |
| $V^\pi(s)$, $Q^\pi(s, a)$ | State value and action value | [[rl/rl-methods]] |
| $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ | Advantage | [[rl/rl-methods]] |
| $V_\phi$ | Learned critic, parameters $\phi$ distinct from the actor's $\theta$ | [[rl/training-design]] |
| $b(s)$ | Baseline | [[rl/rl-methods]] |
| $d^\pi(s)$ | State distribution induced by $\pi$ | [[rl/rl-methods]] |
| $J(\theta)$ | The objective being maximized | [[rl/rl-methods]] |
| $\Psi_t$ | The scoring term in the shared policy-gradient shape | [[rl/rl-methods]] |
| $\delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ | Temporal-difference residual | [[rl/rl-methods]] |
| $\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$ | Generalized advantage estimate | [[rl/rl-methods]] |
| $\rho_t(\theta) = \pi_\theta(a_t \mid o_t) / \pi_{\theta_{\text{old}}}(a_t \mid o_t)$ | Importance ratio | [[rl/rl-methods]] |
| $\varepsilon$ | PPO clipping half-width | [[rl/rl-methods]] |
| $\epsilon$ | Per-decision error rate of a fitted policy, in the DAgger bound | [[rl/training-design]] |
| $\log$ | Logarithm, spelled this way throughout | convention |

Symbols this project has to add, because the study notes are written for a fully observed single-agent setting with no legality constraints.

The study notes are written for reinforcement learning in general and, more recently, for language models. Each addition below exists because this project is a game environment rather than either.

| Symbol | Meaning | Why the notes have no equivalent |
|---|---|---|
| $o \in \Omega$ | Observation, what the agent actually receives | their standard setting is fully observed |
| $O(o \mid s)$ | Observation function, in general stochastic | same |
| $\pi_\theta(a \mid o)$ | Observation-conditioned policy, the object actually trained here | same |
| $m(o) \in \{0,1\}^{793}$ | Legality mask, with $m_i = 1$ exactly when action $i$ is legal | action masking appears nowhere in the notes |
| $\ell_\theta(o) \in \mathbb{R}^{793}$ | Policy-head logits before masking | same |
| $\rho_0$ | Initial-state distribution, here the scenario and army generator | episodic start distributions do not arise in the bandit framing of language-model RL |
| $\Phi(s)$ | Potential function in potential-based shaping | shaping is discussed without a fixed symbol |
| $\mathcal{D}$ | Dataset of observation and action pairs | the demonstrator is left unnamed |
| $T$ | Episode length in decisions, 5 to 40 here | horizon rarely appears explicitly |
| $c_v$, $c_e$ | Value and entropy coefficients in the combined loss | the losses are given separately |

The importance ratio is $\rho_t(\theta)$ and the clip half-width is $\varepsilon$. Both the RLHF book and the owner's most recent notes on it use these. The older reinforcement-learning cards write the ratio $r_t(\theta)$, following the PPO paper, and use $\epsilon$ for the clip. This tree takes $\rho_t$ and $\varepsilon$ for two reasons: they match the newest of the two sources, and they leave $r$ meaning a reward and nothing else, which matters in a document that writes $r_{t+1}$ constantly.

That frees $\epsilon$ for the per-decision imitation error rate, which is what card `rl-036` already calls it. The two are never used in the same equation.

The discount range is relaxed. Most treatments require $\gamma \in [0, 1)$ so infinite-horizon returns converge. Battles terminate within 5 to 40 decisions, so $\gamma = 1$ is admissible and this tree uses $\gamma \in [0, 1]$. The relaxation is safe only because termination is guaranteed, and it would not carry to the adventure-map problem scoped in [[roadmap]].

### Project terms

| Term | Meaning |
|---|---|
| Episode | One complete battle, from arena construction to terminal state or round truncation. |
| Round | One pass in which every eligible unit acts once, advanced by `Arena::Turns()`. Not an RL step. |
| Decision | One full-fledged unit choice, the branch of `UnitTurn` where the engine consults an external decider. The step boundary, counted by `engine_decision_index`. |
| Arena | `Battle::Arena`, the engine's live battle object. One per process. |
| Force | The engine's battle-side unit container. Terminal state must be read from it before the arena is destroyed. |
| World seed | Our input seed, pinned by reseeding the thread-local generator. Determines the map seed. |
| Map seed | Derived by the engine; drives battlefield obstacle placement. |
| Combat seed | `computeBattleSeed(tileIndex, mapSeed, attacker, defender)`; seeds the arena's damage, morale, and luck generator. |
| Scenario | A fixed battle definition: terrain, tile index, world seed, five explicit slots per side, limits. |
| Fixture | One of the five committed scenarios used as regression anchors. |
| Scenario profile (`creature_field_v1`) | Which battle content is in scope: commander-free armies, open field, no castle, no spells. |
| Capability allowlist (`simple_v1`) | The audit result naming which creatures the action space fully models: single-cell, walking, ordinary targeting. |
| Observability profile | `full_v1` (true state) or `observable_v1` (player-obtainable only). A per-consumer setting. |
| Modality | `entities` or `planes`. Representation, orthogonal to observability. |
| State digest | SHA-256 over canonical terminal state, `agent_terminal_v1`. The determinism test of record. |
| Decision digest | SHA-256 over the recorded decision stream, `agent_decisions_v0`. |
| Candidate | A legal action carrying its canonical index, semantic metadata, and engine-ready command parameters. |
| Legal mask | `uint8[793]`, where entry $i$ is 1 exactly when a candidate with index $i$ exists. |
| Canonical action index | Position in the fixed 793-slot space, stable across states and machines. |
| Teacher | The built-in `AI::BattlePlanner`. The source of demonstrations, never a human. |
| Teacher coverage | Fraction of teacher decisions expressible as a legal canonical action. |
| Gate | A `verify_m*.sh` script, carrying a milestone's pass or fail evidence. |
| Worker | `fheroes2_agent_worker`, the entry point outside both build systems' source globs. |
| episodes/s | Throughput unit for the environment alone, measured with no protocol layer attached. |

Board constants: the battlefield is 11 by 9, so 99 cells, and every cell has six hex neighbours. The 793-slot action space decomposes as $1 + 99 + 99 + 594$: one skip, 99 move-to-cell actions, 99 ranged-attack actions (one per target cell), and $99 \times 6 = 594$ melee actions. Melee needs the factor of six because a melee attack is a pair, which cell to strike and which adjacent hex to strike it from: the attacker walks to that neighbouring hex and ends its turn there, so attacking the same stack from its left or from below leaves the attacker on different cells facing different retaliation and different next turns. [[implementation/legal-actions-and-masking]] walks a concrete example.

### What is recap and what is new

The study notes already cover most of the machinery, which is why [[rl/rl-methods]] states results and gives the load-bearing derivation rather than teaching from scratch. Rows marked absent are where this tree carries material the notes do not, and they are the pages worth reading on their own account.

| Topic | Covered in the notes | Here |
|---|---|---|
| MDP objects, Bellman equations, dynamic programming | `rl-001` to `rl-006`, `rl-048` | instantiated for a battle in [[rl/rl-and-the-battle-domain]] |
| Monte Carlo, temporal difference, TD($\lambda$) | `rl-007` to `rl-012`, `rl-201` | the bias-variance contrast, used to motivate GAE |
| Policy gradient theorem, REINFORCE, natural gradient | `rl-013`, `rl-014`, `rl-016`, `rl-047` | trajectory form and the baseline derivation |
| Generalized advantage estimation | `rl-015`, `rl-054`, `rl-210` | recap, plus the backward recursion and the choice of $\lambda$ for a 5 to 40 step horizon |
| Actor-critic, A2C, V-trace, deterministic gradients | `rl-017` to `rl-019`, `rl-049` | recap |
| Trust regions, TRPO, PPO | `rl-029`, `rl-030`, `rl-060`, `rl-203` | recap, plus what masking does to the ratio, which is new |
| Value-based methods, DQN family, distributional RL | `rl-026` to `rl-028`, `rl-046`, `rl-056` | why they are not the first path for this problem |
| Function approximation, deadly triad, fitted Q | `rl-020` to `rl-022` | assumed |
| Exploration | `rl-023` to `rl-025`, `rl-050`, `rl-200` | surveyed and set aside |
| Monte Carlo estimation, score-function estimators, control variates, importance and rejection sampling | `rl-008`, `rl-009`, `rl-053`, and the statistics notes behind them | assumed, and used as the reading frame for [[rl/rl-methods]] Part 1 |
| Imitation, behavior cloning, DAgger, inverse RL | `rl-036`, `rl-037` | [[rl/training-design]] gives the architecture, masked loss, mixing schedule, and hyperparameters for this teacher |
| Offline RL | `rl-033` to `rl-035`, `rl-059` | surveyed and set aside |
| Planning, MCTS, model-based RL | `rl-051`, `rl-052`, `rl-057`, `rl-204` | why the door is kept open but not walked through |
| Policy-gradient family shape, RLOO, GRPO, loss aggregation | the RLHF math companion, ch. 6a to 6d | [[rl/rlhf-transfer]] works out what applies to battles |
| Verifiable rewards, difficulty filtering of the training set | beyond where the companion reaches | [[rl/rlhf-transfer]], read from the open-source book; it supplies an acceptance criterion for $\rho_0$ |
| Overoptimization measured against a KL budget | beyond where the companion reaches | [[rl/rlhf-transfer]], which turns it into a reporting protocol |
| Evaluation variance, hillclimbing, contamination | beyond where the companion reaches | [[rl/rlhf-transfer]], applied to the fixture set |
| Legal-action masking | absent | [[implementation/legal-actions-and-masking]], the main new RL content here |
| Elo and TrueSkill for ranking agents | absent | [[rl/rl-and-the-battle-domain]] Part 3 |
| Asymmetric actor-critic, privileged critics | absent | [[implementation/observation-design]], including the bias result |
| Truncation against termination bootstrapping | absent | [[rl/rl-methods]], and it constrains the Milestone 4 protocol |
| Episode-length normalization bias | absent for episodes | [[rl/rlhf-transfer]] derives it from the token-length version |
| Game-environment engineering | absent | all of [[implementation/README]], the bulk of this repository |
| The fheroes2 battle domain | absent | [[rl/rl-and-the-battle-domain]] Part 2 |

### Reading Zhao alongside this

Zhao's *Mathematical Foundations of Reinforcement Learning* is cited in the owner's notes for its treatment of the Bellman operator as a $\gamma$-contraction, and it uses a different and internally consistent set of symbols. This table translates, so a chapter can be read against this tree without re-deriving anything.

| Here | Zhao |
|---|---|
| $V^\pi(s)$, $Q^\pi(s, a)$ | $v_\pi(s)$, $q_\pi(s, a)$ |
| $A^\pi(s, a)$ | $\delta_\pi(s, a)$, deliberately sharing a letter with the TD error that estimates it |
| $\pi_\theta(a \mid s)$ | $\pi(a \mid s, \theta)$ |
| $P(s' \mid s, a)$, $R(s, a, s')$ | $p(s' \mid s, a)$, $p(r \mid s, a)$ |
| $\rho_0$ | $d_0$ |
| $d^\pi(s)$ | $d_\pi(s)$ stationary, $\eta(s)$ in the policy gradient theorem |
| $J(\theta)$ for an episodic task | $\bar v_\pi^{\,0}$, one of three metrics it distinguishes |
| $V_\phi(s)$ | $v(s, w)$ |
| $\log$ | $\ln$ |

One of its observations is worth borrowing without its notation. It derives the variance-minimizing baseline as $Q^\pi$ weighted by $\lVert \nabla_\theta \log \pi_\theta \rVert^2$ rather than $V^\pi$, which is why [[rl/rl-methods]] calls the state value the practical choice rather than the optimal one.

## Scope

The scope below is Phase 1a only, and it is narrow on purpose. The intended end state covers the adventure map as well, meaning hero movement, resource collection, recruitment, and town management, with the battle policy invoked as a component. [[roadmap]] records that plan, why the battle came first, and the research still owed before any of it is designed.

Phase 1a aims at a trustworthy substrate, so the environment itself contains no learner, no language model, no screenshot parsing, and no interface automation.

| In scope (`creature_field_v1` with `simple_v1`) | Deferred to Phase 1b | Excluded from this branch |
|---|---|---|
| One to five stacks per side, optional per-side commanders (`--attacker-hero`) | Flyers | Adventure-map control |
| Open field, fixed tile index 1; wide walkers opt-in (`--allow-wide`, `wide_v1`) | All-adjacent attacks | Castles, sieges, towers |
| Single-cell walking creatures, shooters including blocked ones | Area shots, unusual ranged attacks | Heroes, spells, artifacts |
| MOVE, ATTACK, SKIP | Spell casting | Screenshots, mouse, keyboard |
| Both sides engine-driven and hook-interceptable | Retreat and surrender as actions | Rendered pixels, which live on the `play-harness` branch |

That last row names a separate project in this repository, where Claude plays through the real interface using frame dumps and an input pipe. The two efforts stay independent.

## Architecture at a glance

```
        ┌── engine (behavior unchanged, digest-proven) ───────────────────┐
        │  Arena::Turns ─► UnitTurn ─┬─ pending UI / standing / bad morale │
        │                            └─ FULL DECISION ──► DecisionController
        │                                    │                  ▲          │
        │                    battle_action_validation ◄──────────┘          │
        │                    (one legality implementation)                  │
        └───────────────────────────────┬───────────────────────────────────┘
                                        │
        ┌── src/fheroes2/agent/ (compiled in, entry-point-free) ───────────┐
        │  scenario ─► runner ─► recorder ─► trajectory (JSONL)            │
        │  capabilities (the simple_v1 allowlist)                          │
        │  action_space: ONE enumeration ─► legal mask + candidates        │
        │  digest (SHA-256 over a canonical serialization)                 │
        └───────────────────────────────┬───────────────────────────────────┘
                                        │
        src/agent_worker/  ──►  [Milestone 4: JSONL protocol]  ──►  Python
```

Three properties hold it together. Engine behavior is unchanged, since every engine edit is either a verbatim lift or an optional hook that is inert when unused, and each was accepted only on unchanged digests. Legality has one implementation, so a mask cannot disagree with what the engine accepts. Everything is hashed, so drift is loud.

## The five ideas the design rests on

Each idea has a primer; read the one covering whatever you are about to touch.

**Determinism.** A battle is a pure function of world seed and army composition. We pin the world seed by reseeding the thread-local generator, derive the combat seed through one shared helper, and hash the outcome. Digest equality is how every engine change on this branch was proven safe. See [[implementation/determinism-seeds-and-digests]].

**Turn dispatch.** `Turns()` advances a whole round, so an RL step cannot be a call into the engine. The engine calls us, at exactly one branch of `UnitTurn`, and the observer must run before the command stream perturbs the random generator. See [[implementation/battle-turn-dispatch]].

**Legal actions and masking.** A fixed 793-slot space plus a per-state boolean mask, which remains a correct policy gradient and is empirically the difference between a 0% and an 82–91% win rate in controlled ablations elsewhere. One enumeration yields both the mask and the candidate list. See [[implementation/legal-actions-and-masking]].

**Observation design.** Structured state only: padded entity records plus an optional semantic plane tensor, filtered by an observability profile. Pixels are excluded, costing roughly 14 times more with no measured benefit and undoing the asset-free core. See [[implementation/observation-design]].

**Teacher coverage.** The engine's own AI plays and we record it. The fraction of its decisions our action space can express is the sharpest completeness test available, and the same recordings are the behavior-cloning dataset. No human play is involved. See [[implementation/teacher-coverage-and-behavior-cloning]].

## What we learned that changed the plan

Four Phase 0 findings overturned the original specification. Do not re-derive them.

Headless battles need no game assets at all: no display, audio, AGG, h2d, or HoMM2 data. Monster stats are hardcoded (`monster_info.cpp:384`) and obstacle setup uses ICN identifiers as plain enum tags (`battle_board.cpp:573`), so battle resolution never touches an asset. This was the specification's top risk, and closing it is why pixels stay out.

### How the state is actually obtained

Worth being concrete, because "headless" invites the assumption that something is being read off a picture. Nothing is rendered and nothing is parsed.

The environment constructs the engine's own battle object with rendering switched off, `Battle::Arena( attackingArmy, defendingArmy, tileIndex, false, generator, controller )`, where the fourth argument is the engine's existing `isShowInterface` flag. That flag is not ours; the game already needs it for its own automatic combat resolution. Everything after it is the same code the game runs when a human watches.

State is then read straight off the C++ objects. `collectForce` walks a `Battle::Force`, and for each `Battle::Unit` calls `GetCount()`, `GetHitPoints()`, `GetHeadIndex()`, and `isValid()`. Those are the same accessors the interface itself calls before drawing a number on screen, so the environment and the display read one source and the environment simply stops earlier.

The proof that no asset is involved is direct. Pointing `FHEROES2_DATA` at a path that does not exist still runs all five fixtures to their usual digests. Rendering is skipped rather than replaced, so this is not a reduced or alternative game format; it is the full battle computation with the drawing left out.

One consequence matters for verification. Because it is the same code path, a battle watched through the interface and the same battle run headless under the same seed are the same computation, which is what makes a side-by-side comparison a meaningful check rather than an approximate one. [[roadmap#The state-extraction gap, closed]] builds a gate on that.

The world seed needs no engine change, because `Rand::CurrentThreadRandomDevice()` (`rand.cpp:85`) returns a mutable reference to a `thread_local PCG32`. The proposed `World` API overload became a deferred cleanup rather than a prerequisite.

A second entry point needs no CMake refactor. Linking a new `main` against the game objects minus `fheroes2.o` worked on the first attempt with no undefined symbols, so the non-entry object set is already library-shaped.

This repository has two build systems, CMake and a plain Makefile under `src/dist`, and the Makefile path is the one in regular use. Whether the agent target supports both is still open, and Milestone 4 settles it.

Full evidence, with a 25-row assumption table, is in `local_source_audit.md`.

## Where the project stands

[[roadmap#The milestones]] carries the milestone table, with the exit criterion and gate result for each. It is the single place; this section holds only what is true of the branch right now.

Measured throughput is about 4,600 episodes/s on the Apple M2 target machine for the tiny-melee fixture with no protocol layer attached, at 12 MB resident memory, scaling linearly to four worker processes. The learner, not the environment, will be the bottleneck.

The branch is `agent-env`, taken from `master` and pushed to `origin`. Engine-source changes are limited to two verbatim lifts (`battle_seed`, `battle_action_validation`), one optional hook (`DecisionController`), one opt-in render seam for replays (a null-by-default render observer on the display, a defaulted interface flag on the runner, and a raceless-captain art case, see [[implementation/replay-rendering]]), one query seam on the built-in AI (a public `BattlePlanner::queryUnitTurn`, the digest-inert teacher probe behind DAgger relabeling), and the additive `src/fheroes2/agent/` library. Enumerate them with `git diff master --stat -- src/`.

## Build and verify

```bash
# Toolchain, once
xcode-select --install
brew bundle --file script/macos/Brewfile     # gettext, sdl2, sdl2_mixer, sdl2_image
brew install cmake                           # not in the Brewfile; only for the CMake path

# Source
git clone git@github.com:troyzhu/fheroes2.git && cd fheroes2
git switch agent-env

# Build, then run all four gates
make -C src/dist -j"$(sysctl -n hw.ncpu)"
./agent_play/spike/build_spike.sh
./agent_play/spike/verify_phase0.sh    # Phase 0 invariants
./agent_play/verify_m1.sh              # 5 passed, deterministic runner
./agent_play/verify_m2.sh              # 8 passed, hook inertness and passive logs
./agent_play/verify_m3.sh              # 9 passed, legal actions and full teacher coverage
./agent_play/verify_agent.sh           # 11 passed, the Python training stack end to end
./agent_play/lint_docs.sh              # docs: style, links, and fact checks
./agent_play/verify_memory.sh          # agent memory still describes reality
```

Two numbers carry the determinism claim: map seed `2227197244` and spike digest `2cfd42cb104aa5e7`. Both are machine-independent and have reproduced across three working trees, two machines, and both optimization levels. A mismatch is a real finding, so stop and investigate before building on top of it.

Useful extras:

```bash
./src/agent_worker/fheroes2_agent_worker --list                        # fixtures
./src/agent_worker/fheroes2_agent_worker --runs 1 --audit-coverage     # coverage report
./src/agent_worker/fheroes2_agent_worker --runs 1 --trajectory-dir OUT # teacher trajectories
./src/agent_worker/fheroes2_agent_worker --capability-audit caps.json  # regenerate the audit
./agent_play/experiments/render_replay.py REPLAY.json OUT.mp4          # recorded episode -> real-engine video
FHEROES2_WITH_ASAN=1 make -C src/dist -j8 && FHEROES2_WITH_ASAN=1 ./agent_play/spike/build_spike.sh
```

## Where everything is

The tree has two documents at the top, then four directories. `README.md` routes, this file orients, and [[roadmap]] says where the project is going. Everything else lives in the directory that matches what it is.

| Path | What it is | Read when |
|---|---|---|
| `agent_play/docs/README.md` | routing index, the only file GitHub renders by default | to find anything |
| `agent_play/docs/overview.md` | this file. The problem, both vocabularies, scope, state, build, and this map | first |
| `agent_play/docs/roadmap.md` | where the project is aimed, what each phase involves, what research is owed | to see what is deliberately not built yet |
| `agent_play/docs/rl/` | the learning side. Domain, methods, training design, scenario distribution, RLHF transfer | before designing or training a policy |
| `agent_play/docs/implementation/` | the environment side. What exists, plus a primer per built mechanism | to review or extend the code |
| `agent_play/docs/decisions/` | accepted decision records, each with its options and trade-offs | before implementing the area one touches |
| `agent_play/docs/research/` | consolidated findings, prior art, and a note per source | to consult or extend the evidence base |
| `agent_play/docs/archive/` | dated history, benchmarks, raw research runs, fetched sources | when you need why or when, never as a reading path |

Two directories are reading paths and two are lookups. `rl/` and `implementation/` are written to be read through; `decisions/` and `research/` are written to be looked up, and `archive/` is provenance rather than prose.

Outside the documentation tree:

| Path | What it is |
|---|---|
| `agent_play/fheroes2_agent_system_spec_v0.3.md` | the full design document. Where a decision record disagrees with it, the record wins |
| `agent_play/lint_docs.sh` | the documentation gate, style contract plus wikilink resolution |
| `agent_play/verify_m*.sh` | the milestone verification gates |
| `agent_play/spike/README.md` | Phase 0 spike usage and limits |

## Decisions not to relitigate

Agent work lives on `agent-env`, branched from `master` rather than from `play-harness`. Both trees produce the identical battle digest, which proves the interface patch inert, but the baseline stays clean regardless.

The baseline is the current `master` lineage rather than the `1.1.17` tag the specification pinned. The tag is 42 commits behind and every specification-critical battle file is byte-identical.

The world seed is pinned by reseeding (specification §7.2, option 1); the narrow `World` overload remains a deferred cleanup. The spike's FNV digest stays as a historical baseline while the environment uses SHA-256, and the two are intentionally not comparable.

Legality is extracted, never re-derived. The tactical AI and the human interface already carry their own near-duplicates of those rules, and a third copy is forbidden.

Rendered pixels are permanently out of the training environment (ADR 0004).

## Gotchas that will bite

One arena per process. `battle_arena.cpp:74` holds a file-static pointer and the constructor asserts it is null, so each arena must be destroyed before the next, and parallelism means processes.

Input `Army` objects are not synchronized after a battle, so terminal state must be read from the `Force` objects before the arena is destroyed.

`Battle::Command` stores parameters in reverse and its accessor pops, so decode a copy rather than the live command. See [[implementation/command-encoding-and-snapshots]].

The Makefile build never defines `NDEBUG`, so `assert()` is live even at `-O3`. A CMake `Release` build does define it, which makes benchmark numbers from the two build systems non-comparable.

Run `make -C src/dist clean` after every upstream sync, because the `-MD` depfiles hard-code header paths and a rename breaks incremental builds with `No rule to make target`.

A `FHEROES2_DATA` root needs the repository's own `files/data/resurrection.h2d` in addition to the GOG extraction, or startup throws about nine seconds in (`h2d.cpp:126`).

Repository paths may contain spaces, since this clone lives under `/Volumes/External Drive/`. Build scripts must pass flag lists as bash arrays, and because macOS ships bash 3.2, an empty array under `set -u` needs the `${arr[@]+"${arr[@]}"}` form.

If the Makefile build fails in the `.pot` step, put Homebrew's `gettext` ahead of pyenv on `PATH`.

## Remaining risks, in order

The historical top risk, legal-action generation, is closed, with validators extracted, full teacher coverage, and no live-arena probing. What remains:

Per-decision state extraction was the highest remaining risk, because everything downstream waited on it, and it was closed on 2026-08-03 ahead of the milestone that owned it. A decision record now carries the observation, the legal-action list and the teacher's chosen index, which is one supervised sample, and 2,000 episodes produce 45,380 of them. What Milestone 4 still owes is the `observable_v1` profile; the `planes` modality landed 2026-08-07 (worker `--planes`, `encode_planes`, the conv-fusion arm, capacity-controlled ablation). [[roadmap#The state-extraction gap, closed]] carries the detail.

Terminal state extraction was in the same position until 2026-08-03 and is now checked. `verify_m1.sh` asserts eight invariants per fixture that must hold whatever the battle was, which closes the specific hole that every gate proving byte-identical digests could not: a systematically wrong extraction would have been perfectly deterministic and would have passed all of them.

The protocol and JSON surface arriving in Milestone 4, where a strict parser boundary and a vendored dependency enter the tree, and where stdout discipline and invalid-input handling decide whether the worker stays healthy.

The transition from behavior cloning past the teacher: cloning, DAgger and search distillation all ran in-house and converge to the demonstrator rather than beyond it, so exceeding the built-in AI in bare weights remains the open problem ([[archive/experiments/2026-08-07-overnight-champion-mixture]]).

Learner throughput on Apple silicon, unmeasured anywhere in the literature at relevant model sizes.

The Phase 1b expansion covering wide units, flyers, and special targeting, which requires re-auditing the `simple_v1` assumptions before the allowlist widens.
