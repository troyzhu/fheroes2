---
title: "Command encoding: reversed parameters, snapshots, and canonical keys"
type: concept-primer
depth: quick
grounded-in: "fheroes2 agent-env branch (Milestone 2, spec §3.8/§10.5)"
related_concepts: ["[[battle-turn-dispatch]]", "[[determinism-seeds-and-digests]]"]
tags: [concept, engine, serialization, agent-env]
---

> **What this is.** How `Battle::Command` actually stores its parameters (not how it looks), the
> trap that costs you a day if you skip it, and how we turn a live command into a typed,
> loggable record without disturbing it.

## The one-sentence version

`Battle::Command` derives from `std::vector<int>` but stores parameters **in reverse**, consuming
them from the back — so anything that reads a command by ordinary iteration is silently wrong,
and every snapshot must decode a *copy*.

## The trap

Constructing `Command(ATTACK, attackerUID, defenderUID, moveCell, targetCell, direction)` pushes
those five values so that `GetNextValue()` — which pops from the *back* — returns them in
semantic order. The raw vector is therefore reversed relative to what you would expect:

```cpp
const Battle::Command command( Battle::Command::MOVE, 7, 34 );
// command[0] == 34, command[1] == 7      <- raw order is REVERSED
// snapshotCommand(command).params == {7, 34}   <- semantic order
```

Worse, `GetNextValue()` **mutates**: it pops. Decode a live command in place and you have
destroyed the command the engine was about to apply. And `operator>>` is private, so the decoder
cannot use it.

## The pattern that works

Copy, then decode the copy:

```cpp
Battle::Command scratch = command;      // trivial copy: it is a vector<int>
for ( size_t i = 0; i < scratch.size(); ++i )
    params.push_back( scratch.GetNextValue() );
```

That is `fheroes2::agent::snapshotCommand`, which additionally decodes the semantic fields per
command type (MOVE: unit, cell · ATTACK: attacker, defender, move cell, target cell, direction ·
SKIP: unit · MORALE: unit, polarity) and leaves `params` as the authoritative record for anything
else.

## Canonical keys

Each snapshot renders to a stable string — `move:7:34`, `attack:1:6:34:45:3`, `skip:9` — used in
trajectories and for human-readable diffs. Keys are semantic, so they survive changes to internal
storage.

## The other asymmetry: the RNG stream

`Command::updatePCG32Stream` reads the ATTACK command **positionally** (`at(2)`, `at(3)`,
`at(4)`), not through `GetNextValue()`, and deliberately ignores the target cell and direction —
because the AI and a human may encode the *same* attack differently and must not diverge the RNG.

So the same command is read two different ways by two different subsystems. Keep the snapshot
decoder and any stream logic in separate, individually tested functions; do not "unify" them.

## Why it matters here

Snapshots are how the passive teacher log is built (Milestone 2) and how a teacher decision is
matched to a canonical action index (Milestone 3). Both run *before* the command is applied, on
copies, which is exactly why recording changes no outcome — proven by unchanged golden digests
with the recorder attached.

## What this does *not* say

Snapshot equality is **semantic** equality, not raw-command equality. Two commands that encode
the same attack differently produce the same resolved action but different raw vectors — which is
why the spec distinguishes *semantic replay* from *strict command replay*, and why state-digest
equivalence is the authoritative determinism test.

## See also
- [[battle-turn-dispatch]] — where snapshots are taken.
- [[determinism-seeds-and-digests]] — the decision-stream digest built from them.
