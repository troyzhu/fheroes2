# ADR 0001 — Dual observation profiles: `full_v1` and `observable_v1`

- Status: accepted 2026-07-27
- Context: spec §12 (observation schema), `research_rl_approaches.md` §2, user requirement 2026-07-27 ("the more realistic agent mode should have a partially observable state")

## Context

Observation schema v1 (spec §12) serializes the full true engine state, including engine-internal values a player never sees, most notably `engine_strength`, which comes from the built-in AI's own evaluator. There is no notion of "what a player could actually observe."

Three facts shape the decision:

1. Creature-only fheroes2 battles are informationally symmetric. The battle UI's right-click info (`battle_interface.cpp`, `Cursor::WAR_INFO` → `Dialog::ArmyInfo`) shows the full stat sheet for any unit, own or enemy, with no ownership gating. Real hidden information in HoMM2 begins later: enemy hero mana, and adventure-map fog of war.
2. The proven pattern for dual modes is one schema with a switch, not two schemas. MicroRTS-Py exposes full observability by default and partial observability via a constructor flag that appends visibility planes to the same tensor (verified 3-0).
3. A full-state mode stays valuable even when the deployed policy is restricted, oracle critics, teachers and debugging consume it (asymmetric actor-critic literature; searched but unverified at our game class, kept as an option, not a dependency).

Additionally, vcmi-gym's hard-won "Markov discipline" (verified 3-0): every attribute that influences battle dynamics must either appear in the observation or be removed from the dynamics. vcmi-gym chose removal (it deleted morale/luck/terrain effects). We choose exposure.

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
