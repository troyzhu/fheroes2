---
title: Battle turn dispatch
aliases:
  - turn-dispatch
  - unit-turn
  - decision-hook
tags:
  - agent-env
  - primer
concept: engine turn loop and the external decision seam
domain: engine integration
grounded_in: "battle_arena.cpp and battle_decision_controller.h on branch agent-env"
depth: quick
updated: 2026-07-30
---

# Battle turn dispatch

An fheroes2 battle advances in rounds, and nothing in the engine corresponds to a single agent decision. This primer traces how a round runs, identifies the one branch where an external policy may decide, and states the invariants a controller must respect. Line numbers refer to branch `agent-env`, where the hook itself has shifted them relative to `master`.

## Motivation

A reinforcement learning environment needs a boundary at which exactly one decision is made and control returns to the learner. The engine offers none. `Arena::Turns()` advances an entire round in which every eligible unit acts, and returns `void`. Driving it from Python gives round-level granularity, far too coarse when a five-stack battle makes roughly ten decisions per round.

Running the engine on a second thread and pausing it between decisions trades a clean call stack for cross-thread synchronization inside a codebase that never expected it. The approach taken instead leaves the engine's control flow intact and inserts a hook where the engine already asks an external party what to do.

## The idea in one sentence

The engine keeps the call stack and calls into us at the one branch of `Arena::UnitTurn` where it would otherwise consult the built-in AI or the human interface, and we block there until an action arrives.

## Intuition

Control is inverted relative to the usual environment API. A Python reinforcement learning environment normally exposes `step(action)`, so the learner owns the loop and pulls. A game engine owns its own loop and pushes, calling out to whatever supplies decisions. Our hook is that callee, four frames deep in engine code, and the process blocks inside it.

An episode therefore cannot be rewound or forked mid-flight, because the state lives in a live C++ call stack rather than in a snapshot, and parallelism is process-level.

## How it works

`Arena::Turns()` (`battle_arena.cpp:580`) advances one round. It increments the turn counter, calls `NewTurn()` on both forces, then repeatedly selects the next eligible unit and calls `UnitTurn` until nobody can act or the battle ends. Castle towers and the catapult also act here. `Turns()` does not run the battle. The caller's loop does, and `battle_main.cpp:366-368` is the model the worker reimplements.

`Arena::UnitTurn` (`battle_arena.cpp:463`) rolls morale on entry, then loops until the unit's turn is over. Each iteration takes one of four branches.

| Branch | Condition | Reachable headless |
|---|---|---|
| 1. Pending interface actions | `_interface->getPendingActions` | no, `_interface` is null |
| 2. Standing, dead, or immovable | speed is `STANDING` | yes |
| 3. Bad morale | `MORALE_BAD` set | yes, emits an automatic `MORALE` command |
| 4. Full-fledged action | otherwise | yes, this is the decision |

Only branch four is a decision. The hook lives there and nowhere else.

### The ordering that matters, and where it actually sits

After the branch chain closes, `UnitTurn` folds the chosen commands into the combat generator's stream and then applies them.

```
513   ++_engineDecisionIndex            (branch 4 only)
517   handlesDecision  ─┐
522   assert(!actions.empty())          (branch 4 only)
541   observeChosenActions ─┘
543   end of the branch chain
545   std::accumulate over actions ──► updatePCG32Stream
547   _randomGenerator.setStream(...)   ← OUTSIDE the branch chain
549   while (!actions.empty()) { ApplyAction; removeDeadUnits; ... }
```

The stream update at 545-547 sits at the same nesting level as the branch chain, so it runs for **every** branch, not only for decisions. Two consequences follow that the hook cannot see.

The bad-morale branch emits a `MORALE` command, and `updatePCG32Stream` has a live case for `MORALE` (`battle_command.cpp:55`). Automatic bad-morale commands therefore perturb the generator's stream while `observeChosenActions` never observes them. Anyone reconstructing the generator's trajectory from the recorded decision stream alone will diverge on any battle containing a demoralized stack. The recorded state digests remain correct, because they are taken from real terminal state rather than reconstructed.

A good-morale command is appended inside the inner apply loop, after the fold at 545 has already run, so it never passes through `updatePCG32Stream` at all. Bad morale moves the stream and good morale does not.

`board.removeDeadUnits()` is called at 553, inside `while (!actions.empty())`, so it runs after every applied command rather than once per turn. An observer must not assume the board is stable across a multi-command turn.

A good-morale command clears `TR_MOVED`, which causes the next outer iteration to re-enter branch four. That is a new decision with its own index, not a continuation of the previous one.

### The interface, and how to fail safely

```cpp
class DecisionController {
    virtual bool handlesDecision( const Arena &, const Unit & currentUnit ) const = 0;
    virtual void chooseActions( Arena &, const Unit & currentUnit, Actions & output ) = 0;
    virtual void observeChosenActions( const Arena &, const Unit &, const Actions & );
};
```

`chooseActions` receives a non-const `Arena &` because the engine's legality helpers are non-const in effect, warming a pathfinder cache. A controller must still not apply commands, consume combat randomness, or mutate game state.

Installation is a defaulted raw pointer on the `Arena` constructor (`battle_arena.cpp:347`). The controller must outlive the arena; the arena does not own it.

The failure mode worth memorizing is a hang rather than a crash. If `chooseActions` claims a decision and appends nothing, the only guard is `assert( !actions.empty() )` at line 522, which is a no-op under `NDEBUG`. With no actions the unit neither acts nor ends its turn, and `while ( !endOfTurn )` spins forever. Unwinding safely means appending `Command::SKIP` for the current unit, never appending an empty action list.

A null controller, the default for every existing caller, leaves the engine bit-identical, which the golden digests verify.

Every branch-four decision increments `Arena::GetEngineDecisionIndex()` at line 513, which happens **before** `handlesDecision` is consulted at 517, and happens identically with or without a controller attached. That last property is why the counter is digest-safe.

## Comparison with alternatives

| Approach | Engine change | Step granularity | Cost | When preferred |
|---|---|---|---|---|
| Blocking hook in `UnitTurn` (ours) | one constructor parameter, one conditional | exact decision | one process per episode | An engine whose loop already consults an external decider |
| Call `Turns()` per step | none | whole round | unusable for per-unit control | Never |
| Second thread, pause and resume | none in the loop, heavy elsewhere | exact decision | cross-thread synchronization in code that never expected it | When the engine cannot be recompiled |
| Refactor the loop into a coroutine | large, invasive | exact decision | high risk of behavior change | A greenfield engine |
| Replay from a command log | none | replay only | no live control | Debugging and strict replay |

Branch four already dispatches to an external decider, so the hook adds one constructor parameter and one conditional.

## When to use it

Use the hook for decisions the engine treats as full-fledged. Never intercept automatic morale, towers, or the catapult. Branch one is structurally unreachable headless, since `_interface` is null.

## Key terms

- Round, one pass in which every eligible unit acts once, advanced by `Arena::Turns()`.
- Decision, one full-fledged unit choice, the fourth branch of `UnitTurn`, and the step boundary.
- Engine decision index, a monotonic counter over full-fledged decisions, incremented before the controller is consulted.
- Null controller, the absent hook, which preserves the engine's original behavior exactly.

## Why it came up here

This structure explains why the environment is a blocking worker rather than a library call, and why parallelism means multiple processes. The arena is a pointer in an anonymous namespace (`battle_arena.cpp:74`) whose constructor asserts it is the only live one (`battle_arena.cpp:356`). That assertion is compiled out under `NDEBUG`, so the single-arena rule is enforced only in builds that keep assertions. See the warning in [[determinism-seeds-and-digests]].

Terminal state has no hook. The read must happen in the caller's driver loop between `BattleValid()` going false and the arena leaving scope, because the destructor only clears the global pointer.

## What this does not say

Nothing here covers spell casting, siege machinery, or retreat and surrender, which the hook never sees. It also does not describe how a blocked worker receives its action, which is protocol work scheduled for Milestone 4.

## Go deeper

- [[determinism-seeds-and-digests]], what the command stream does to the generator, and who owns it.
- [[legal-actions-and-masking]], what a decision may legally contain.
- [[command-encoding-and-snapshots]], how the observed commands are decoded.
