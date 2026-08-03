---
title: Concept primers, index
type: moc
updated: 2026-07-30
related_concepts: ["[[../overview]]", "[[../rl/rl-and-the-battle-domain]]"]
tags: [concept, index, moc, agent-env]
---

> **What this note is.** Six deep dives on the mechanisms this project actually implements. They assume the vocabulary of [[../rl/rl-and-the-battle-domain]], which explains reinforcement learning and the Heroes battle domain from scratch, and they assume no C++.

These are reference material rather than a course. Read the one covering whatever you are about to touch.

| Primer | Answers | Read before | Length |
|---|---|---|---|
| [[determinism-seeds-and-digests]] | Which seeds control a battle, what gets hashed, why every refactor is accepted on a digest | Touching `Rand::` or `computeBattleSeed` | about 5 min |
| [[battle-turn-dispatch]] | How rounds and unit turns advance, where an agent may decide, why control is inverted | Editing `battle_arena.cpp` or the decision hook | about 5 min |
| [[legal-actions-and-masking]] | Why masking, how the 793-slot space is laid out, why mask and candidates share one enumeration | Working on the action space or a policy head | about 7 min |
| [[observation-design]] | Entities against planes, `full_v1` against `observable_v1`, why never pixels, the MDP rule | Implementing observation serialization | about 7 min |
| [[command-encoding-and-snapshots]] | Why `Battle::Command` reads backwards, how to snapshot one safely, canonical keys | Reading or logging engine commands | about 4 min |
| [[teacher-coverage-and-behavior-cloning]] | Who the teacher is, why coverage proves completeness, the cloning ladder | Working on demonstrations or training | about 6 min |

Each follows the same shape: motivation, the idea in one sentence, an intuition drawn from machine learning, the mechanism, a comparison against the alternatives that were rejected, key terms, and a closing section naming the boundary of the claim.

## Related

- [[../rl/rl-and-the-battle-domain]], the conceptual entry point these primers build on.
- [[../overview]], the system as it stands.
- [[../overview#Notation]], the symbol contract used throughout.
- [[inventory]], what is actually built and how it was verified.
- [[../research/findings|Literature synthesis]], the evidence behind these ideas.
- `../decisions/`, the ADRs these concepts justify.
