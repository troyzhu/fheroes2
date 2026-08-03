---
title: "ADR 0002, fixed canonical action space with a legality mask"
type: adr
status: accepted
updated: 2026-08-03
related_concepts: ["[[../implementation/legal-actions-and-masking]]", "[[../rl-and-the-battle-domain]]"]
tags: [adr, action-space, masking, agent-env]
---

# ADR 0002 — Fixed canonical action space with legal mask, candidates derived from one enumeration

- Status: accepted 2026-07-27, implemented at Milestone 3
- Implementation: built and verified. `src/fheroes2/agent/agent_action_space.h` defines `actionSpaceSize = 1 + 99 + 99 + 99*6 = 793`, with `actionSkipIndex = 0`, `actionMoveBase = 1`, `actionRangedBase = 100`, `actionMeleeBase = 199`. Legality comes from `src/fheroes2/battle/battle_action_validation.{h,cpp}`, lifted verbatim from the engine. `verify_m3.sh` passes 8/8 with 116/116 teacher coverage.
- Evidence: spec §10 (legal-action generation), [[../archive/research-runs/2026-07-27-rl-approaches]] §3, [[../research/works/invalid-action-masking]], [[../research/works/vcmi-gym]], [[../research/works/gym-microrts]]
- Mechanism detail: [[../implementation/legal-actions-and-masking]], and [[../implementation/command-encoding-and-snapshots]] for the engine-side encoding

## Table of contents
- [[#Context]]
- [[#The sub-problem]]
- [[#Options considered]]
- [[#A worked example, both interfaces]]
- [[#Why this one, and what it cost]]
- [[#Decision]]
- [[#Consequences]]

## Context

Spec §10 defines legal actions as an engine-enumerated, per-decision list of candidates with contiguous ephemeral `action_id`s assigned after sorting (§10.4). The external policy picks an id from that variable-length list.

The verified literature is unanimous on a different interface for the learning side:

- Every verified codebase exposes a fixed discrete action space plus a boolean legal mask (vcmi-gym: flat `Discrete(2312)` over the 165-hex board, verified 2-1; MicroRTS: per-cell factorized components, verified 3-0). No verified project consumes variable-length candidate lists directly.
- Legal-action masking over a fixed space is provably a valid policy gradient (Huang & Ontañón, FLAIRS 2022) and empirically decisive (unmasked full-game microRTS PPO: 0.0 cumulative win rate; fully masked: 0.82–0.91; penalties collapse as the space grows). Verified 3-0.
- Standard tooling assumes fixed spaces. CleanRL's `CategoricalMasked` is a masked categorical distribution in a single-file PPO reference implementation, and sb3-contrib's `MaskablePPO` is the Stable-Baselines3 contributed variant that accepts a legality mask. Both are described in [[../implementation/legal-actions-and-masking#Alternatives]].
- vcmi-gym's factorized multi-head variant failed to converge; its flat-masked space shipped.
- AlphaStar-style pointer selection over an enumerated candidate set is the architectural home of our current design, viable later, heavyweight now. Verified 3-0.

Meanwhile the candidate list itself remains valuable: it carries semantic metadata for the protocol, teacher-action matching (§10.6), debugging, and any future pointer-network head. And the engine-side enumeration through shared non-mutating resolvers (§10.2) remains the top project risk regardless of representation, this ADR changes the *interface*, not that work.

## The sub-problem

What shape does the action interface present to a learner?

The engine can enumerate exactly which commands are legal at a decision point, so legality is not in question. The question is the container. A policy network needs an output layer of fixed width, and the set of legal actions changes every decision, so something has to reconcile those two facts. This record chooses that reconciliation. It does not decide how legality is computed, which is the resolver-extraction work, nor what the policy head looks like architecturally.

## Options considered

| Option | What it is | For | Against |
|---|---|---|---|
| Variable-length candidate list, per-decision ids | The learner receives the legal actions and picks an index into that list | What spec §10.4 originally specified, and carries semantic metadata naturally | Index $k$ means a different action at every decision, so a softmax over it learns nothing stable. No verified project consumes this directly |
| Fixed flat space plus legality mask (chosen) | One integer per action for all time, with a boolean mask per state | Stable indexing, one softmax, works with standard tooling unmodified | Space is mostly illegal at any state, and every future capability must fit the fixed layout |
| Factorized multi-head | Split an action into independent components, each with its own softmax | Shrinks a combinatorial space to a few hundred logits | vcmi-gym's factorized variant failed to converge; its flat masked space is what shipped |
| Pointer network over candidates | Attention selects among a variable-length candidate set | The general answer when the candidate set is genuinely unbounded | Heavyweight, and the architectural home of the original design rather than a near-term option |
| Penalize illegal actions with negative reward | Keep the space unmasked, teach legality | No masking machinery | Documented to collapse as the illegal fraction grows. Unmasked full-game microRTS PPO reaches 0.0 cumulative win rate against 0.82 to 0.91 masked |

## A worked example, both interfaces

One decision, the same one used in [[../implementation/legal-actions-and-masking#A worked index]]. Our stack stands on cell 34, an enemy occupies cell 35 immediately to its right, and the stack can reach three empty cells.

Under the original specification, the worker emits a variable-length list and the learner picks a position in it.

```json
{ "candidates": [
    { "action_id": 0, "type": "SKIP" },
    { "action_id": 1, "type": "MOVE",  "cell": 22 },
    { "action_id": 2, "type": "MOVE",  "cell": 23 },
    { "action_id": 3, "type": "MOVE",  "cell": 45 },
    { "action_id": 4, "type": "MELEE", "target": 35, "direction": 2 } ] }
```

Under this record, the same enumeration emits a fixed-width mask, and every candidate carries its canonical index instead of a position.

```
legal_mask: uint8[793], zero except
  index   0  → SKIP
  index  23  → MOVE to cell 22        (1 + 22)
  index  24  → MOVE to cell 23        (1 + 23)
  index  46  → MOVE to cell 45        (1 + 45)
  index 411  → MELEE cell 35, dir 2   (199 + 6·35 + 2)
```

### Why the first one cannot be learned from

Nothing is wrong with that list as a message. What breaks is using it as a policy's output space, and the reason is that `action_id` 2 means "move to cell 23" only at this decision. One step later, with a different set of reachable cells, `action_id` 2 is some other action entirely.

A softmax head assigns a weight to each output position and updates it from experience. If position 2 denotes a different action at every decision, the gradient for that weight is an average over unrelated actions, and there is nothing stable for it to converge to. The network can only learn from the accompanying metadata, which means reading the semantics of a variable-length set, which is the pointer architecture in the options table rather than a flat head.

Under the canonical indexing, slot 411 means striking cell 35 from its left in every state, in every episode, on every machine. That is what makes a fixed head trainable, and it is also what makes a recorded trajectory comparable across runs, since the stored index is meaningful without replaying the decision that produced it.

### What the mask costs here

Five of 793 slots are legal in this example, so more than 99% of the output is masked. That sounds wasteful and is the standard objection to flat spaces. The measured answer is that masking's time-to-solve stays roughly flat as the illegal fraction grows, which is what makes the trade acceptable rather than merely tolerable.

## Why this one, and what it cost

The decisive evidence is that masking over a fixed space is provably a valid policy gradient rather than a heuristic, because the mask depends on the state alone and never on the parameters, which makes the masked softmax a well-formed parameterized policy (Huang and Ontañón, FLAIRS 2022, Proposition 1). [[../implementation/legal-actions-and-masking]] carries the argument and the correct implementation. Nothing comparable is true of the penalty option, which is why it is rejected outright rather than ranked.

The factorized option was rejected on the one directly comparable data point available. vcmi-gym is the only shipped Heroes-family battle environment, it tried both, and the flat masked space is what it shipped.

Two costs were accepted knowingly. The space is sparse, with typically 5 to 30 of 793 slots legal, so most of the output layer is masked at any decision; the evidence says masking's time-to-solve stays roughly flat as the illegal fraction grows, which is what makes that tolerable. And the fixed layout keys melee actions on a target cell plus a direction, which presumes a single-cell attacker. Two-cell units have no single head cell for that purpose, so the indexing needs revisiting before wide units enter, and [[../roadmap]] records that as blocking work for Phase 1b.

The candidate list is not discarded. One engine enumeration emits both products, so the mask and the list cannot disagree, and the list continues to carry semantic metadata for the protocol, teacher matching, and any future pointer head.

## Decision

1. Define a fixed canonical action indexing for `simple_v1` over the 11×9 board (99 cells), on the order of 10³ actions (vcmi-gym's scale). The exact layout is fixed at Milestone 3, but the shape is: a small set of global actions (SKIP/WAIT-class) + per-cell MOVE actions + per-cell-per-direction MELEE actions + per-target RANGED actions. 
   Indexing is a pure function of the board geometry and action taxonomy, so it is stable across states. Its version is documented by a schema version tag.
2. One engine enumeration, two products.

   The Milestone 3 candidate generator (§10.3) remains the single source of legality and, per decision, emits: (a) a boolean legal mask over the canonical space, and (b) the candidate list, each candidate carrying its canonical index as its `action_id` (replacing §10.4's per-decision contiguous ids; the sort order stays for display/logging).
3. The protocol's `act` message selects by canonical index. Validation is unchanged: a selection must match an outstanding candidate, else a recoverable error (§5.4).
4. Trajectories store the canonical semantic key AND the canonical index; the index is now meaningful across states, which simplifies passive-teacher datasets for BC.
5. Factorized or pointer heads remain compatible: both consume the same mask/candidates.

## Consequences

- Spec amendments at implementation time: §10.4 (indexing), §12.1 (mask exposure in observations), §13.5/§13.6 (decision/act messages), §15 (trajectory records).
- Milestone 4's Python client can hand the mask directly to CleanRL-style `CategoricalMasked` or MaskablePPO with no adapter.
- No change to Milestones 1–2 artifacts, the canonical digest, or the §10.2 resolver-extraction plan (still the top risk).
