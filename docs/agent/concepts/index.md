---
title: "Concept primers — index"
type: moc
tags: [concept, index, agent-env]
updated: 2026-07-30
---

# Concept primers

Ground-up explanations of the ideas the agent environment rests on. Each is self-contained,
intuition-first, and grounded in this repo's actual code. Written to be read before touching the
area they describe — [[../START_HERE|START_HERE]] links to them at the point each idea first
matters.

| Primer | Answers | Read before |
|---|---|---|
| [[determinism-seeds-and-digests]] | Which seeds control a battle, what we hash, why every refactor is accepted on a digest | Touching anything under `Rand::` or `computeBattleSeed` |
| [[battle-turn-dispatch]] | How rounds and unit turns advance, where an agent may decide, why observation must precede the RNG stream update | Editing `battle_arena.cpp` or the decision hook |
| [[command-encoding-and-snapshots]] | Why `Battle::Command` reads backwards, how to snapshot one safely, canonical keys | Reading or logging engine commands |
| [[legal-actions-and-masking]] | Why masking, how `Discrete(793)` is laid out, why the mask and candidates share one enumeration | Working on the action space or a policy head |
| [[observation-design]] | Entities vs planes, `full_v1` vs `observable_v1`, why never pixels, the Markov rule | Implementing observation serialization |
| [[teacher-coverage-and-behavior-cloning]] | Who the teacher is, why coverage is a completeness proof, the BC→RL ladder | Working on demonstrations or training |

**Conventions.** Frontmatter carries `type`, `depth`, and `related_concepts`; `[[wikilinks]]`
resolve when `docs/agent/` (or the repo root) is opened as an Obsidian vault. Each primer ends
with a *"what this does not say"* section — the boundary of the claim is part of the content.

## See also
- [[../START_HERE|START_HERE]] — the system as it stands, with the notation table.
- [[../log|Project log]] — dated history.
- [[../decisions/0001-observation-profiles|ADRs]] — the decisions these concepts justify.
- [[../references/summary|Literature synthesis]] — the evidence behind them.
