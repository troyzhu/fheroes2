---
title: Scope and roadmap
type: roadmap
updated: 2026-07-30
related_concepts: ["[[rl/rl-and-the-battle-domain]]", "[[overview]]"]
tags: [agent-env, roadmap, scope]
---

# Scope and roadmap

Everything built so far concerns battles only, and deliberately so. This note records what the wider goal is, why the battle came first, what the later phases involve, and which questions have to be answered with evidence before any of them starts. It exists so that the current narrow scope is not mistaken for the intended end state.

[[overview]] is the companion to this file. It says what exists and how to build it; this says what does not exist yet and what each next step is waiting on.

## Table of contents
- [[#The milestones]]
- [[#Milestone 4 in detail]]
- [[#Where the project is aimed]]
- [[#Why the battle came first]]
- [[#Phase 1b, wider battles]]
- [[#Phase 2, the navigation and management agent]]
- [[#Sequencing]]

## The milestones

The single milestone table. Exit criteria are the specification's own (§20); the gate column names the script that decides whether the criterion is met, which is what makes each row falsifiable rather than a status opinion.

| Milestone | Exit criterion | Gate | State |
|---|---|---|---|
| 0, local audit and headless spike | The initialization and asset path are known | `spike/verify_phase0.sh`, 7 of 7 | Done, reproduced on target hardware |
| 1, deterministic runner | Ten identical runs match | `verify_m1.sh`, 4 of 4 | Done |
| 2, decision hook and passive logging | Built-in behaviour unchanged under a null controller, and passive logs replay deterministically | `verify_m2.sh`, 8 of 8 | Done |
| 3, `simple_v1` legal actions | Every supported fixture has valid candidates, at 100% teacher coverage | `verify_m3.sh`, 8 of 8, coverage 116 of 116 | Done, top project risk closed |
| 4, JSONL worker | Scripted stdin and stdout tests control both sides without a single invalid command | `verify_m4.sh`, not yet written | Next |
| 5, Python environment and replay | Golden trajectories reproduce across fresh and reused workers | `verify_m5.sh`, not yet written | Not started |
| 6, hardening and benchmark | The definition of done in specification §22 passes | `verify_m6.sh`, not yet written | Not started |
| Optional QA, one Battle Only battle through the interface | Owner's call, normal-game regression only | none | Accepted risk, not a training prerequisite |

Per-component detail for what is already built, meaning which file satisfies which specification section and what tests it, is in [[implementation/inventory]]. That is a different object from this table: it maps code, where this maps milestones.

Nothing under `rl/` appears above, because training a policy is not a milestone of the environment. It begins after 6, and [[decisions/0005-training-and-reward]] governs it.

## Milestone 4 in detail

The next one, so the level of detail is higher than for the rest.

| Deliverable | What it means in practice | Specification |
|---|---|---|
| Worker target | A dedicated target outside the fheroes2 source glob | §6.1, §6.2 |
| Protocol v1 | One JSON object per line on stdout and nothing else on that stream, with diagnostics on stderr | §13 |
| Scenario parsing | Reject an unknown key rather than defaulting it, and name a JSON path and a stable error code in every rejection | §11 |
| JSON dependency | A pinned, vendored, permissively licensed parser, off by default in the normal build | §6.5 |
| Blocking external control | The existing hook waits for an `act` message and copies engine-owned commands into the turn | §5.4 |
| Observation serialization | Both observability profiles and the `planes` modality, from one state source | [[decisions/0001-observation-profiles]], [[decisions/0004-spatial-observation-modality]] |
| Failure handling | On malformed input emit an error frame and stay alive; on end of stdin skip safely and unwind | §5.4 |
| Termination reason | The protocol must distinguish a battle that ended from one that hit the round limit | [[decisions/0005-training-and-reward]] |

### Ordering, and why

The parser and the worker target come first, because everything else is tested through them. Blocking control comes next and is the riskiest change, since it turns the existing observer hook into something that can deadlock. Observation serialization is last and is the largest, but it is additive and testable in isolation once the protocol carries frames at all.

### What the gate has to prove

`verify_m4.sh` does not exist yet. Writing it is part of the milestone rather than a follow-up, on the pattern the earlier gates set. At minimum it has to show that a scripted session drives an episode to termination with no invalid command, that a malformed line produces an error frame and a live worker rather than a crash, that closing stdin unwinds cleanly, that a rejected scenario names its JSON path and error code, that the terminal digest under external control matches the digest the same scenario produces under the built-in AI when the external policy replays the AI's own choices, and that the terminal-state invariants below hold on every fixture.

### The state-extraction gap this milestone has to close

Worth stating separately, because it is the foundation everything after it rests on and it is currently the weakest verified part of the project.

Per-decision state extraction does not exist. `DecisionRecord` holds an engine decision index, a unit id, and the chosen commands, and the source says so at `src/fheroes2/agent/agent_trajectory.h`, that the `agent_passive_v0` schema carries no observations, no legal-action lists, and no teacher matching. The consequence for [[decisions/0005-training-and-reward]] is concrete and easy to miss. Behaviour cloning needs observation and action pairs, and Milestone 2 recorded 116 actions with nothing about the board beside them, so the recorded decisions cannot train anything as they stand. The data is recoverable rather than lost, since episodes are reproducible from a seed and can be re-run once an observation emitter exists, but the emitter is a Milestone 4 deliverable and nothing downstream of it can start first.

Terminal state extraction was in the same position and is now checked, as of 2026-08-03. Every gate proved the digest was stable and none asserted what it held, so a systematically wrong extraction would have been perfectly deterministic and would have passed all of them. `verify_m1.sh` now asserts eight invariants on every fixture, chosen so they hold whatever the battle was and therefore need no golden value and no oracle. A golden value would only have locked in whatever the implementation happens to do.

The invariants are that a living stack holds at least one creature and a living creature at least one hit point, that a side with no stacks is zero on every field, that a reported victory or defeat agrees with which side still has stacks, that a decided battle does not leave both sides standing, and that rounds and decisions are at least one. Each was verified to fire by injecting the corresponding corruption, including the pre-death read that motivated the check.

### Open before it starts

The two-build-system question. `ENABLE_AGENT` covers the CMake path only, and the specification requires deciding explicitly whether the worker target is also wired into the `src/dist` Makefile or whether that path is declared unsupported for agent builds. Leaving it undecided strands whichever machine lacks CMake, and this machine builds through the Makefile.

Whether `AI::BattlePlanner` can be queried without advancing the arena or consuming combat randomness. This is the precondition for the DAgger stage in [[decisions/0005-training-and-reward]], and that record asks for it to be settled during Milestone 4 while the protocol is being built, because the answer decides whether that stage exists at all.

## Where the project is aimed

The end state is an agent that plays fheroes2, which means two quite different problems joined together.

A battle is a small, turn-based, nearly fully observed tactical problem on 99 hex cells with at most ten stacks. That is the problem the current environment solves, and [[rl/rl-and-the-battle-domain]] describes it in full.

The adventure map is everything outside a battle. A hero moves across a large map under fog of war, picks up resources and artifacts, visits locations, recruits creatures, manages towns, and decides which battles to start in the first place. It is a strategy problem rather than a tactical one, and almost every property that makes the battle tractable is absent from it.

An agent that plays the game needs both, and the battle agent is a component the adventure agent invokes rather than a smaller version of it.

## Why the battle came first

Three reasons, in order of weight.

The battle is the part where a substrate can be made trustworthy. It is deterministic under a seed, it runs at roughly 4,600 episodes per second, it has a competent built-in teacher, and its legal action set can be enumerated exactly. Those four properties made it possible to build an environment whose correctness is demonstrable rather than assumed, which is what Milestones 1 through 3 did.

The battle is also where a mistake is cheapest. Getting the action space or the observation schema wrong in a self-contained tactical problem costs a rewrite of one module. Getting it wrong across the whole game costs the project.

Finally, the only comparable shipped system, vcmi-gym for Heroes III, is battle-only. There is a precedent to learn from at battle scope and none at adventure scope, which is itself informative about relative difficulty.

## Phase 1b, wider battles

The next expansion stays inside battles and lifts the `simple_v1` restrictions. Wide two-cell units, flying movement, two-cell and all-adjacent attacks, area shots, and eventually heroes with spells and castle sieges.

The blocking work is the capability audit and the action indexing. The current canonical space keys melee actions on a target cell and a direction, which presumes a single-cell attacker. A two-cell unit has no single head cell for that purpose, so the indexing needs revisiting before wide units enter. Spells multiply the action space by targets and parameters, which is the point where a flat masked space stops being obviously right and a factorized or pointer head earns consideration.

## Phase 2, the navigation and management agent

This is the part that is intended but not yet designed. Recording it now so the design is not improvised later.

The adventure agent covers hero movement across the map, resource and artifact collection, visiting locations, recruiting creatures, town building and management, and choosing which battles to fight. The battle agent becomes a subroutine it calls.

### Why it is a different problem, not a bigger one

Every simplifying property of the battle disappears.

Observability inverts. The battle is effectively fully observed. The adventure map has genuine fog of war, so the problem becomes partially observed in the real sense, and the recurrence, belief states, and memory architectures that battles let us skip become mandatory. This is the single largest change.

The horizon lengthens by orders of magnitude. A battle is 5 to 40 decisions. A campaign map is thousands of turns of decisions whose consequences arrive much later, which is a credit-assignment problem the battle never poses.

The action space stops being enumerable in the same way. Movement targets, building choices, recruitment quantities, and trade decisions produce a large and structurally heterogeneous space, closer to StarCraft's than to the current 793 slots.

The reward stops being obvious. A battle has a natural terminal signal in winning. A campaign has resource curves, map control, army growth, and eventual victory, on wildly different timescales. See [[decisions/0005-training-and-reward]].

Determinism gets harder. Map generation, random map events, and creature growth all draw randomness across a long horizon, so the seeding discipline that works for one battle needs extending.

The engine seam is unknown. Milestones 1 through 3 depended on finding a clean hook inside `Arena::UnitTurn`. Whether the adventure-map turn loop offers an equivalent seam, and whether its AI can be intercepted the same way, has not been investigated. `AI::Planner` and the kingdom-level planners in `src/fheroes2/ai/` are the place to look.

### Where it sits among comparable environments

On the axes in [[rl/rl-and-the-battle-domain]], the adventure map moves from the fheroes2-battle row to somewhere near StarCraft II and the Lux AI competitions. Turn-based rather than real-time, which helps, but fogged, long-horizon, hierarchical, and with a heterogeneous action space, which does not.

The natural structural answer is hierarchy, with a strategic policy choosing where to go and what to build, and the existing battle policy invoked when a fight starts. That is an assumption rather than a conclusion. Hierarchical reinforcement learning has a mixed record, and a flat policy over a well-designed action abstraction sometimes beats an explicit hierarchy.

### The research this needs before it starts

The claim that hierarchy is right, and the choice of algorithm, observation encoding, and reward for a long-horizon fogged strategy game, all need the same treatment the battle environment got: a credibility-gated literature sweep with adversarial verification, not a design improvised from intuition. The questions worth putting to it:

1. What has actually been tried for turn-based 4X and strategy games, and what shipped? Freeciv, Civilization, Polytopia, Lux AI, and any Heroes-family adventure-map work.
2. Does hierarchical reinforcement learning outperform a flat policy over temporally extended actions in this class of game, and under what conditions?
3. How do published systems handle fog of war at this scale, meaning recurrence against belief states against learned world models?
4. How is reward shaped for long-horizon strategy without teaching the wrong objective?
5. How is a tactical sub-policy composed with a strategic one, and is the battle policy trained jointly or frozen?
6. What does the fheroes2 adventure-map AI expose that could serve as a teacher, in the way `AI::BattlePlanner` does for battles?

Until that sweep is run and consolidated, no adventure-map design decision should be recorded as settled.

## Sequencing

The battle environment is finished as a substrate only when Milestones 4 through 6 are done, meaning the protocol, the Python client, and the hardening and benchmark work. Training a battle policy comes after that, and it is where the reward and algorithm choices in [[decisions/0005-training-and-reward]] first meet reality.

Phase 1b and Phase 2 both wait on that, for the same reason the battle came first. A second environment built before the first one has trained anything would be a second untested substrate.

## Related

- [[overview]], current state and the milestone table.
- [[rl/rl-and-the-battle-domain]], the battle domain and the axes this note compares against.
- [[decisions/0005-training-and-reward]], the training and reward design, including what is deliberately still open.
- [[research/findings]], the evidence base the battle decisions rest on.
