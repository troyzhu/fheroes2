---
title: "Battle turn dispatch: rounds, unit turns, and where the agent hooks in"
type: concept-primer
depth: quick
grounded-in: "fheroes2 agent-env branch (Milestone 2)"
related_concepts: ["[[determinism-seeds-and-digests]]", "[[legal-actions-and-masking]]", "[[command-encoding-and-snapshots]]"]
tags: [concept, engine, control-flow, agent-env]
---

> **What this is.** How an fheroes2 battle actually advances, why "one RL step" does not
> correspond to any engine function, and the precise seam where an external policy is allowed to
> decide. Read before editing `battle_arena.cpp`.

## The one-sentence version

`Arena::Turns()` advances a whole **round** (every unit acts once), so the only place an agent can
make a single decision is *inside* `Arena::UnitTurn`, at the one branch where the engine would
otherwise ask the built-in AI or a human.

## The two nested loops

**Round loop — `Arena::Turns()`** (`battle_arena.cpp:552`). Increments the turn counter, calls
`NewTurn()` on both forces, then repeatedly picks the next eligible unit and calls `UnitTurn` on
it until nobody can act or the battle ends. Castle towers and the catapult act here too.

**Unit-turn loop — `Arena::UnitTurn`** (`battle_arena.cpp:460`). For the chosen unit, it loops
until the unit's turn is over, and on each iteration takes exactly one of four branches:

1. **Pending UI actions** (auto-combat toggles etc.) — already happened, handled first.
2. **Standing / dead / immovable** — turn ends.
3. **Bad morale** — the engine emits an automatic `MORALE` command.
4. **A full-fledged action** — the unit will really decide something.

Only branch 4 is a *decision*. Branches 1–3 are bookkeeping, and an agent that intercepted them
would be changing the game's rules rather than playing it.

## The critical ordering inside branch 4

After the actions are chosen, `UnitTurn` does three things **in this order**:

```
1. update the PCG32 stream from the chosen command sequence   (battle_arena.cpp:517)
2. apply each command                                          (ApplyAction, :522)
3. remove dead units, maybe append a good-morale re-decision
```

Step 1 is why *observation must happen before it*: once the command stream has perturbed the RNG,
you are no longer looking at the state the decision was made in. Our
`DecisionController::observeChosenActions` hook is called immediately after the actions are
chosen and immediately before step 1 — that placement is the whole design.

## The hook contract

`Battle::DecisionController` (`battle_decision_controller.h`) has three methods:

| Method | When | Rule |
|---|---|---|
| `handlesDecision` | Branch 4 only | Return true to take over this decision |
| `chooseActions` | If it claimed the decision | Must append ≥1 valid action |
| `observeChosenActions` | **Every** branch-4 decision | Read-only; runs before the RNG stream update |

A null controller — the default for every existing caller — leaves the engine bit-identical,
which is exactly what the golden digests verify.

**Decision identity.** Every branch-4 decision increments `Arena::GetEngineDecisionIndex()`.
Good morale can hand the *same unit* a second decision in the same turn; it gets its own index,
because for learning purposes it is a genuinely separate choice made in a different state.

## Why it matters here

This structure is why the environment is **worker-blocking** rather than step-driven: Python
cannot call "step" — the engine calls *us*, deep inside its own call stack, and we block there
until an action arrives. It is also why parallelism means multiple *processes*: the arena is a
file-static singleton (`battle_arena.cpp:73`) that asserts it is the only live one.

## What this does *not* say

Nothing here covers spell casting, siege machinery, or retreat/surrender — all outside the
`creature_field_v1` profile. The hook deliberately never sees them.

## See also
- [[determinism-seeds-and-digests]] — what the command stream does to the RNG.
- [[legal-actions-and-masking]] — what a decision may legally contain.
