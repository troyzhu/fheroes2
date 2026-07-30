---
title: Concept primers, index
type: moc
updated: 2026-07-30
related_concepts: ["[[../START_HERE]]"]
tags: [concept, index, moc, agent-env]
---

> **What this note is.** The catalogue of ground-up explanations behind the agent environment. It assumes no fheroes2 knowledge and no C++, and it does not assume reinforcement learning either, since the first two primers build that vocabulary from scratch.

Read in order the first time. The first primer fixes the general vocabulary, the second instantiates it for this game and places the game among its neighbors, and the remaining six go deep on the parts of the system that were hard to get right.

| Order | Primer | Answers | Length |
|---|---|---|---|
| 1 | [[rl-for-games]] | What an RL game environment is: state, action, reward, transition, policy, observability, and the axes along which such environments differ | about 12 min |
| 2 | [[fheroes2-battles-vs-other-games]] | What a Heroes battle is, the same thing in RL terms, and how it compares to Heroes III, microRTS, StarCraft, NetHack, and Wesnoth | about 10 min |
| 3 | [[determinism-seeds-and-digests]] | Which seeds control a battle, what gets hashed, why every refactor is accepted on a digest | about 5 min |
| 4 | [[battle-turn-dispatch]] | How rounds and unit turns advance, where an agent may decide, why control is inverted | about 5 min |
| 5 | [[legal-actions-and-masking]] | Why masking, how the 793-slot space is laid out, why mask and candidates share one enumeration | about 7 min |
| 6 | [[observation-design]] | Entities against planes, `full_v1` against `observable_v1`, why never pixels, the MDP rule | about 7 min |
| 7 | [[command-encoding-and-snapshots]] | Why `Battle::Command` reads backwards, how to snapshot one safely, canonical keys | about 4 min |
| 8 | [[teacher-coverage-and-behavior-cloning]] | Who the teacher is, why coverage proves completeness, the cloning ladder | about 6 min |

Read by task once the vocabulary is in place. Touching `Rand::` or `computeBattleSeed` means primer 3; editing `battle_arena.cpp` or the decision hook means primer 4; working on the action space or a policy head means primer 5; implementing observation serialization means primer 6; reading or logging engine commands means primer 7; working on demonstrations or training means primer 8.

Each primer follows the same shape: motivation, the idea in one sentence, an intuition drawn from machine learning, the mechanism, a comparison against the alternatives that were rejected, key terms, and a closing section naming the boundary of the claim.

## Related

- [[../START_HERE|START_HERE]], the system as it stands, with the notation table.
- [[../references/repos|Repository orientation]], what the codebases behind the evidence contain and where to look inside them.
- [[../references/summary|Literature synthesis]], the evidence behind these ideas.
- [[../log|Project log]], dated history.
- `../decisions/`, the ADRs these concepts justify.
