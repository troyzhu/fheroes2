---
title: Command encoding and snapshots
aliases:
  - command-encoding
  - command-snapshot
tags:
  - agent-env
  - primer
concept: decoding Battle::Command without disturbing it
domain: engine integration
grounded_in: "battle_command.{h,cpp} and agent_command_snapshot.cpp on branch agent-env"
depth: quick
updated: 2026-07-30
---

# Command encoding and snapshots

`Battle::Command` looks like a vector of integers and behaves like a stack. Reading one the obvious way is both wrong and destructive. This primer shows the actual storage order, the copy-then-decode pattern that reads a command safely, and the canonical key form used in recorded trajectories.

## Motivation

Recording what the built-in AI chose requires reading its commands before they are applied. Two plausible readings fail.

Iterating the vector yields the parameters backward, because the constructor pushes them so that the accessor can pop from the back. Calling the accessor on the live object appears to work and then destroys the command the engine was about to execute, since `GetNextValue()` mutates. The stream-relevant accessor `operator>>` is private, so a decoder cannot borrow it.

## The idea in one sentence

Copy the command, decode the copy, and leave the original untouched.

## Intuition

The object is a stack wearing a vector's clothes. Push then pop reverses order, so raw index 0 holds the last parameter rather than the first.

## How it works

Constructing `Command(ATTACK, attackerUID, defenderUID, moveCell, targetCell, direction)` stores those five values so that `GetNextValue()`, which pops from the back, returns them in semantic order. The fold that does this is a right-to-left assignment sequence at `battle_command.h:118-124`. The raw vector is therefore reversed.

```cpp
const Battle::Command command( Battle::Command::MOVE, 7, 34 );
// command[0] == 34, command[1] == 7          raw order is reversed
// snapshotCommand(command).params == {7, 34} semantic order
```

The safe decode copies first, which is cheap because the type derives from `std::vector<int>`. The loop bound must be captured **before** the loop, because `GetNextValue()` pops and therefore shrinks the container on every call. Re-reading `size()` in the condition would terminate the loop after roughly half the parameters.

```cpp
Battle::Command scratch = command;

const size_t parameterCount = scratch.size();   // hoisted: GetNextValue() pops
params.reserve( parameterCount );
for ( size_t i = 0; i < parameterCount; ++i ) {
    params.push_back( scratch.GetNextValue() );
}
```

That is `fheroes2::agent::snapshotCommand` (`agent_command_snapshot.cpp:31-40`). It additionally decodes the semantic fields per command type, covering MOVE with unit and cell, ATTACK with attacker, defender, move cell, target cell and direction, SKIP with unit, and MORALE with unit and polarity, while `params` remains the authoritative record for anything else.

Each snapshot renders to a stable key such as `move:7:34`, `attack:1:6:34:45:3`, or `skip:9`. Keys are semantic, so they survive changes to internal storage.

### The same command is read two ways, on purpose

`Command::updatePCG32Stream` reads an ATTACK command positionally, at raw indices 2, 3 and 4, and ignores the target cell and direction outright (`battle_command.cpp:42-51`). The engine's own comment gives the reason. The built-in AI and a human may encode the same attack differently, and the random stream must not diverge between them.

One object therefore has two legitimate readings. Keep the snapshot decoder and any stream arithmetic in separate, individually tested functions, and never reuse one for the other.

## Comparison with alternatives

| Approach | Mutates the original | Needs engine change | When preferred |
|---|---|---|---|
| Copy then decode (ours) | no | no | Any read of a live command |
| Decode the original in place | yes, destroys it | no | Never |
| Add const accessors to `Command` | no | small upstream patch | If upstream would take the patch |
| Read raw indices positionally | no | no | Only inside stream arithmetic, which is defined positionally |

Copying wins because the type is already a vector, so the copy is trivial and no engine change is needed to obtain a correct semantic read.

## When to use it

Snapshot every command you intend to log, match, or compare. Never snapshot by iteration, and never reuse a snapshot decoder for stream arithmetic.

## Key terms

- Snapshot, a typed decoded record of a command in semantic parameter order.
- Canonical key, the stable string form of a snapshot, used in trajectories and diffs.
- Semantic order, the parameter order the constructor documents, recovered by popping.
- Stream arithmetic, the positional read that advances the combat generator, deliberately partial.

## Why it came up here

Snapshots build the passive teacher log and map a teacher decision onto a canonical action index. Both run before the command is applied, on copies, which is why recording changes no outcome, as the unchanged golden digests confirm.

## What this does not say

Snapshot equality is semantic, not raw. Two commands encoding the same attack differently produce the same resolved action and different raw vectors, which is why the specification separates semantic replay from strict command replay and treats state-digest equivalence as authoritative.

## Go deeper

- [[battle-turn-dispatch]], where snapshots are taken in the turn loop, and which commands never reach the hook.
- [[determinism-seeds-and-digests]], the decision-stream digest built from them.
