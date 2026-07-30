# ADR 0002 — Fixed canonical action space with legal mask, candidates derived from one enumeration

- Status: accepted 2026-07-27 (implementation lands with Milestone 3)
- Context: spec §10 (legal-action generation), [[references/report-rl-approaches]] §3

## Context

Spec §10 defines legal actions as an engine-enumerated, per-decision list of candidates with contiguous ephemeral `action_id`s assigned after sorting (§10.4). The external policy picks an id from that variable-length list.

The verified literature is unanimous on a different interface for the learning side:

- Every verified codebase exposes a fixed discrete action space plus a boolean legal mask (vcmi-gym: flat `Discrete(2312)` over the 165-hex board, verified 2-1; MicroRTS: per-cell factorized components, verified 3-0). No verified project consumes variable-length candidate lists directly.
- Legal-action masking over a fixed space is provably a valid policy gradient (Huang & Ontañón, FLAIRS 2022) and empirically decisive (unmasked full-game microRTS PPO: 0.0 cumulative win rate; fully masked: 0.82–0.91; penalties collapse as the space grows). Verified 3-0.
- Standard tooling (CleanRL `CategoricalMasked`, sb3-contrib MaskablePPO) assumes fixed spaces.
- vcmi-gym's factorized multi-head variant failed to converge; its flat-masked space shipped.
- AlphaStar-style pointer selection over an enumerated candidate set is the architectural home of our current design, viable later, heavyweight now. Verified 3-0.

Meanwhile the candidate list itself remains valuable: it carries semantic metadata for the protocol, teacher-action matching (§10.6), debugging, and any future pointer-network head. And the engine-side enumeration through shared non-mutating resolvers (§10.2) remains the top project risk regardless of representation, this ADR changes the *interface*, not that work.

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
