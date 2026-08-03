---
title: "ADR 0001, dual observation profiles"
type: adr
status: accepted
updated: 2026-08-03
related_concepts: ["[[../implementation/observation-design]]", "[[../rl-and-the-battle-domain]]", "[[0004-spatial-observation-modality]]"]
tags: [adr, observation, observability, agent-env]
---

# ADR 0001 — Dual observation profiles: `full_v1` and `observable_v1`

- Status: accepted 2026-07-27
- Implementation: not built. No occurrence of `full_v1`, `observable_v1`, or `observation_profile` exists in `src/fheroes2/agent/` or `python/` as of 2026-07-31. This record binds Milestone 4, which is where observation serialization lands.
- Evidence: spec §12 (observation schema), [[../archive/research-runs/2026-07-27-rl-approaches]] §2, [[../research/findings]], user requirement 2026-07-27 ("the more realistic agent mode should have a partially observable state")
- Engine grounding: `src/fheroes2/battle/battle_interface.cpp`, the right-click information path, verified present
- Mechanism detail: [[../implementation/observation-design]]

## Table of contents
- [[#Context]]
- [[#The sub-problem]]
- [[#Options considered]]
- [[#A worked example]]
- [[#Why this one, and what it cost]]
- [[#What the choice does not buy]]
- [[#Decision]]
- [[#Consequences]]

## Context

Observation schema v1 (spec §12) serializes the full true engine state, including engine-internal values a player never sees, most notably `engine_strength`, which comes from the built-in AI's own evaluator. There is no notion of "what a player could actually observe."

Three facts shape the decision:

1. Creature-only fheroes2 battles are informationally symmetric. The battle UI's right-click info (`battle_interface.cpp`, `Cursor::WAR_INFO` → `Dialog::ArmyInfo`) shows the full stat sheet for any unit, own or enemy, with no ownership gating. Real hidden information in HoMM2 begins later: enemy hero mana, and adventure-map fog of war.
2. The proven pattern for dual modes is one schema with a switch, not two schemas. MicroRTS-Py exposes full observability by default and partial observability via a constructor flag that appends visibility planes to the same tensor (verified 3-0).
3. A full-state mode stays valuable even when the deployed policy is restricted, since oracle critics, teacher matching, and debugging consume it. This is the asymmetric actor-critic idea, meaning a critic that sees privileged state while the actor sees only what will be available at deployment, defined in [[../rl-methods#Partial observability]] and weighed in [[../implementation/observation-design]]. Searched but unverified at our game class, so kept as an option rather than a dependency.

Additionally, vcmi-gym's hard-won Markov discipline (verified 3-0). A problem is Markov when the current state carries everything needed to predict what comes next, which is what lets a policy look only at the present; [[../rl-and-the-battle-domain]] develops this. The discipline that follows is that every attribute influencing battle dynamics must either appear in the observation or be removed from the dynamics. vcmi-gym chose removal (it deleted morale/luck/terrain effects). We choose exposure.

## The sub-problem

One question only. When the environment serializes a battle state, does it emit everything the engine knows, or only what a player could obtain through the game's own interface?

This is a question about the observation function $O$ in the sense of [[../rl-and-the-battle-domain]]. It is not a question about the state, which is fixed by the engine, nor about the digest, nor about representation, which is [[0004-spatial-observation-modality]]. What makes it non-trivial is that the answer changes the problem class: a policy that sees the full state faces a Markov decision process, and one that sees a strict function of it faces a partially observed problem where a memoryless policy is not in general sufficient.

## Options considered

| Option | What it is | For | Against |
|---|---|---|---|
| Full state only | Serialize everything, including the built-in AI's own evaluations | Simplest, strongest signal, matches spec §12 as written | Trains a policy on information no deployed agent could have, so a reported win rate would not be honest |
| Player-observable only | Serialize only what the battle interface exposes | Honest by construction | Forecloses oracle critics and teacher matching, and cannot be reversed without re-running every episode |
| Two separate schemas | One message type per mode | Each is clean in isolation | Two serializers, two parsers, two digests, and drift between them is a silent correctness failure |
| One schema, tagged fields (chosen) | One shape, with `oracle`-tagged fields omitted under the restricted profile | One serializer, one digest, profiles distinguishable on the wire | Requires a field-tag discipline that has to be maintained as fields are added |

## A worked example

One living stack, serialized both ways. Fields are those of the spec §12.3 unit record.

Under `full_v1`:

```json
{
  "uid": 2, "side": "attacker", "monster_id": 1,
  "count": 42, "total_hit_points": 42, "top_creature_hit_points": 1,
  "attack": 1, "defense": 1, "base_speed": 3, "effective_speed": 3,
  "shots_left": 0, "head_cell": 34, "is_wide": false,
  "engine_strength": 42.0
}
```

Under `observable_v1`, the same record with one key gone:

```json
{
  "uid": 2, "side": "attacker", "monster_id": 1,
  "count": 42, "total_hit_points": 42, "top_creature_hit_points": 1,
  "attack": 1, "defense": 1, "base_speed": 3, "effective_speed": 3,
  "shots_left": 0, "head_cell": 34, "is_wide": false
}
```

Everything a player could read off the battle interface survives, including the opposing side's full statistics, because the game's own right-click information panel shows those without ownership gating. What disappears is `engine_strength`, which is the built-in AI's own valuation of the stack and appears nowhere in the interface. The force summary loses `engine_strength_sum` for the same reason.

That single field is the whole disagreement, and it matters more than its size suggests. A policy trained on `full_v1` and cloned from `AI::BattlePlanner` can read the teacher's own evaluation of every stack, which makes matching the teacher easier than matching it honestly would be, so teacher agreement would look better than deployment warrants. At deployment the field does not exist, so the policy would be acting on inputs it never saw in training.

### Why omitted rather than zero-filled

The key is removed, not set to `0.0`. Zero-filling would make the two profiles indistinguishable on the wire, and worse, a policy could not tell the AI valuing a stack at zero apart from the field being withheld. Omission makes the profile a readable property of the message.

### What the other options would have produced

Two separate schemas would give two serializations of one state, and the digest is defined over the full state regardless of profile, so the question of which serialization the digest covers has no good answer. One of them would have to be designated canonical, at which point the second is a filtered view of the first, which is this decision with extra machinery.

Player-observable only would emit the second record above and nothing else, ever. Recovering `engine_strength` later would mean re-running every episode, whereas filtering it out of a full record costs a key lookup.

### What stays in both, and why

`shots_left`, `effective_speed`, and every morale, luck, and spell-effect field appear under both profiles even though some are not obvious from the interface at a glance. The rule is not visibility but influence: an attribute that changes the transition distribution and is not observed makes the observation process non-Markov, so it must either be observed or removed from the dynamics. See [[../implementation/observation-design]].

## Why this one, and what it cost

Three considerations decided it, in order of weight.

The reversibility argument is the strongest. Recording full state and filtering on the way out is cheap, while recording filtered state and wishing for more later means re-running every episode. The asymmetric cost of being wrong points one way.

The two-schemas option was rejected on evidence rather than taste. MicroRTS-Py exposes full observability by default and partial observability through a constructor flag that appends visibility planes to the same tensor, which is the shipped precedent for a switch rather than a fork (verified 3-0). Two schemas would also give two digests, and the digest is the project's determinism test of record.

The cost accepted is a maintenance obligation. Every new observation field has to be classified as oracle or observable at the moment it is added, and there is no mechanism that forces this. A field added without a tag defaults to visible, which is the unsafe direction.

## What the choice does not buy

Naming a profile `observable_v1` does not make the problem fully observed under `full_v1`. The combat generator's position sits in the state and is serialized under neither profile, so both are formally partially observed, and [[../implementation/observation-design]] carries that argument. The profiles differ by a small set of tagged fields, not by problem class.

The asymmetric setup this enables, meaning a critic on `full_v1` and an actor on `observable_v1`, is available but not recommended. A state-value critic used to form advantages for an observation-conditioned actor gives a biased gradient in a partially observed problem (Baisero and Amato, 2022), and our own sweep found no verified claim supporting the naive split.

## Decision

1. Scenario schema v1 gains a field:

   ```json
   "observation_profile": "full_v1" | "observable_v1"      // default: "full_v1"
   ```

2. One schema, filtered fields. `observable_v1` is `full_v1` minus fields tagged `oracle`: engine-internal evaluations (`engine_strength`, `engine_strength_sum`) and anything else a player cannot obtain from the UI. Field tags live in the observation serializer; the JSON shape stays identical (oracle fields are omitted, never zero-filled, so the two profiles are distinguishable on the wire).
3. The canonical state digest (spec §12.5) is always computed over the full state, regardless of profile. Replay integrity and trajectory comparison never depend on what a policy was allowed to see.
4. Markov rule: every dynamics-affecting attribute the engine tracks (morale state, luck state, shots left, spell effects, retaliation availability…) is present in BOTH profiles. Stochastic transitions (morale/luck procs, damage rolls) are fine, unobserved *state* is not.
5. Future partial observability extends `observable_v1`, MicroRTS-style (visibility annotations within the same schema), when hero mana or adventure-map scope arrives. It does not fork the schema.

## Consequences

- Trajectory headers record the profile; BC/RL policies train on `observable_v1` while critics and teacher-matching may consume `full_v1` from the same worker without protocol changes.
- Spec §12 needs a field-tag column (oracle vs observable) in the unit record table when Milestone 2 implements observation serialization.
- The digest implementation from Milestone 1 (`agent_terminal_v1`) is unaffected.
