---
title: Battle turn dispatch — a primer
aliases:
  - turn-dispatch
  - unit-turn
  - decision-hook
tags:
  - agent-env
  - primer
concept: engine turn loop and the external decision seam
domain: RL environment engineering
grounded_in: "this repo's Milestone 2; battle_arena.cpp"
depth: quick
updated: 2026-07-30
---

# Battle turn dispatch — a primer

An fheroes2 battle advances in rounds, not in policy steps, so nothing in the engine corresponds
to a Gymnasium `step()`. This primer traces how a round actually runs, identifies the one branch
where an external policy may decide, and explains why that placement forces a blocking worker
process rather than a callable environment object.

## Motivation

An RL environment needs a single-decision boundary. The engine offers none: `Arena::Turns()`
advances an entire round in which every eligible unit acts, and returns `void`. Calling it
repeatedly from Python gives round-level granularity, which is far too coarse, since a five-stack
battle makes roughly ten decisions per round.

The naive fix, running the engine on a second thread and pausing it between decisions, trades a
clean call stack for cross-thread synchronization inside a codebase that never expected it. The
approach taken instead leaves the engine's own control flow untouched and inserts a hook where the
engine already asks an external party what to do.

## The idea in one sentence

The engine keeps the call stack and calls into us at the one branch of `Arena::UnitTurn` where it
would otherwise consult the built-in AI or the human interface, and we block there until an action
arrives.

## Intuition

The polarity is the inversion an ML engineer meets in training frameworks. Writing a bare training
loop, you call `model.forward()`. Handing the loop to a framework, the framework calls your
`on_batch_end` callback instead, and your code lives inside its stack rather than above it.

Gymnasium's contract is the first shape: you call `env.step(a)` and the environment returns.
fheroes2 is the second: the engine drives, and our hook is the callback. The adapter that restores
a normal `step()` for Python runs the engine in a separate process, parks it inside the hook while
the chosen action travels over stdio, and lets it continue. PySC2 and the Battle for Wesnoth
environment use the same trampoline.

Two consequences fall straight out of that. An episode cannot be rewound or forked mid-flight,
because the state lives in a live C++ call stack rather than in a snapshot. Vectorization is
process-level, not thread-level.

## How it works

Two loops nest.

`Arena::Turns()` (`battle_arena.cpp:552`) advances one round. It increments the turn counter,
calls `NewTurn()` on both forces, then repeatedly selects the next eligible unit and calls
`UnitTurn` on it until nobody can act or the battle ends. Castle towers and the catapult also act
here.

`Arena::UnitTurn` (`battle_arena.cpp:460`) loops until the chosen unit's turn is over, taking
exactly one of four branches per iteration: pending interface actions, a unit that is standing,
dead, or immovable, a bad-morale automatic action, or a full-fledged action. Only the fourth is a
decision. The first three are bookkeeping, and a hook that intercepted them would be changing the
rules rather than playing the game.

The ordering inside the fourth branch is the load-bearing detail. Once the actions are chosen,
`UnitTurn` updates the combat generator's stream from the command sequence
(`battle_arena.cpp:517`), then applies each command, then removes dead units and may append a
good-morale re-decision. Observation has to happen before the stream update, because afterward the
random state no longer matches the state the decision was made in. Our observer hook is called
immediately after the actions are chosen and immediately before that update.

The interface carries three methods.

| Method | When it fires | Contract |
|---|---|---|
| `handlesDecision` | fourth branch only | return true to take over this decision |
| `chooseActions` | when it claimed the decision | must append at least one valid action |
| `observeChosenActions` | every fourth-branch decision | read-only, before the stream update |

A null controller, the default for every existing caller, leaves the engine bit-identical, which
is what the golden digests verify.

Every fourth-branch decision increments `Arena::GetEngineDecisionIndex()`. Good morale can hand
the same unit a second decision within one turn, and it receives its own index, because for
learning purposes it is a separate choice made in a different state.

## Comparison with alternatives

| Approach | Engine change | Step granularity | Cost | When preferred |
|---|---|---|---|---|
| Blocking hook in `UnitTurn` (ours) | one optional parameter | exact decision | one process per episode | An engine whose loop already consults an external decider |
| Call `Turns()` per step | none | whole round | coarse, unusable for control | Never for per-unit control |
| Second thread, pause and resume | none in the loop, heavy elsewhere | exact decision | cross-thread synchronization in code that never expected it | When the engine cannot be recompiled |
| Refactor the loop into a coroutine | large, invasive | exact decision | high risk of behavior change | A greenfield engine, or one already written around a generator |
| Replay from a command log | none | replay only | no live control | Debugging and strict replay |

The blocking hook wins here because the engine already had the seam. Branch four exists precisely
so the loop can ask somebody what to do, so we answer instead of the built-in AI without
restructuring anything.

## When to use it

Use the hook for any decision the engine treats as full-fledged. Never intercept automatic morale,
pending interface actions, towers, or the catapult, and never use it for spells or siege
machinery, which sit outside the `creature_field_v1` scenario profile.

## Key terms

- Round: one pass in which every eligible unit acts once, advanced by `Arena::Turns()`.
- Decision: one full-fledged unit choice, the fourth branch of `UnitTurn`, and the step boundary.
- Engine decision index: monotonic counter over full-fledged decisions, including morale re-decisions.
- Null controller: the absent hook, which preserves the engine's original behavior exactly.

## Why it came up here

This structure explains two properties that would otherwise look arbitrary. The environment is a
blocking worker rather than a library call, and parallelism means multiple processes, because the
arena is a file-static singleton (`battle_arena.cpp:73`) whose constructor asserts that it is the
only live one.

## What this does not say

Nothing here covers spell casting, siege machinery, or retreat and surrender. The hook never sees
them by design.

## Go deeper

- [[determinism-seeds-and-digests]] — what the command stream does to the random generator.
- [[legal-actions-and-masking]] — what a decision may legally contain.
- `docs/agent/decisions/0002-action-space.md` — the action interface the hook carries.
