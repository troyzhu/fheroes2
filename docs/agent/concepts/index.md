---
title: Concept primers, index
type: moc
updated: 2026-07-30
related_concepts: ["[[../START_HERE]]"]
tags: [concept, index, moc, agent-env]
---

> **What this note is.** The catalogue of ground-up explanations behind the agent environment.
> Assumes MDPs and policy gradients; assumes no fheroes2 or C++ knowledge. Read
> [[../START_HERE#Notation and key terms]] first, since the primers use those terms without
> redefining them.

| Primer | Answers | Depth | Read before |
|---|---|---|---|
| [[determinism-seeds-and-digests]] | Which seeds control a battle, what gets hashed, why every refactor is accepted on a digest | quick, about 5 min | Touching `Rand::` or `computeBattleSeed` |
| [[battle-turn-dispatch]] | How rounds and unit turns advance, where an agent may decide, why control is inverted | quick, about 5 min | Editing `battle_arena.cpp` or the decision hook |
| [[command-encoding-and-snapshots]] | Why `Battle::Command` reads backwards, how to snapshot one safely, canonical keys | quick, about 4 min | Reading or logging engine commands |
| [[legal-actions-and-masking]] | Why masking, how the 793-slot space is laid out, why mask and candidates share one enumeration | quick, about 7 min | Working on the action space or a policy head |
| [[observation-design]] | Entities against planes, `full_v1` against `observable_v1`, why never pixels, the MDP rule | quick, about 7 min | Implementing observation serialization |
| [[teacher-coverage-and-behavior-cloning]] | Who the teacher is, why coverage proves completeness, the cloning ladder | quick, about 6 min | Working on demonstrations or training |

Each primer follows the same shape: motivation, the idea in one sentence, an intuition drawn from
machine learning, the mechanism, a comparison against the alternatives that were rejected, key
terms, and a closing section naming the boundary of the claim.

## Related

- [[../START_HERE|START_HERE]], the system as it stands, with the notation table.
- [[../log|Project log]], dated history.
- [[../references/summary|Literature synthesis]], the evidence behind these ideas.
- `../decisions/`, the ADRs these concepts justify.
