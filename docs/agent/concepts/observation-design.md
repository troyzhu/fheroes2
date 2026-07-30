---
title: "Observation design: entities, planes, profiles, and why never pixels"
type: concept-primer
depth: standard
grounded-in: "fheroes2 agent-env branch (ADR 0001, ADR 0004)"
related_concepts: ["[[legal-actions-and-masking]]", "[[determinism-seeds-and-digests]]"]
tags: [concept, observations, pomdp, agent-env]
---

> **What this is.** The four axes of our observation design — *what* is represented (entities vs
> spatial planes), *how much* is revealed (full vs observable), why rendered pixels are excluded,
> and the one rule that ties it together (the Markov property).

## The one-sentence version

The agent sees structured engine state — a padded entity list plus, optionally, a semantic
`11×9×C` plane tensor derived from that same state — filtered by an observability profile, and
never a rendered image.

## Axis 1 — representation: entities and planes

**Entity list.** One record per unit stack: id, position, count, HP, speed, shots, status flags.
Variable-length in principle, so it is emitted as **fixed slots with an explicit NULL category**
for empties (padding is the standard answer at this scale; entity-transformers are the upgrade
path, not a prerequisite).

**Spatial planes.** The same state rasterised onto the board: `11×9` cells × C typed channels
(occupancy per side, unit class, count/HP fractions, passability, later reachability/threat).
This is what a CNN policy consumes.

They are complementary, not alternatives — the production precedent (AlphaStar) feeds *both*
into one core, and even scatters entity embeddings into the plane stack. Our schema therefore
treats them as independently toggleable **modalities** (`entities`, `planes`), following the
PySC2/Griddly "one game state, multiple observers" pattern.

## Axis 2 — observability: `full_v1` vs `observable_v1`

A **profile** decides how much of the true state the policy sees:

- `full_v1` — everything, including engine-internal values like `engine_strength` (the built-in
  AI's own evaluator output). Useful for oracle critics, teachers, and debugging.
- `observable_v1` — only what a player could obtain from the UI. Fields tagged `oracle` are
  omitted (not zero-filled, so the two are distinguishable on the wire).

**Engine fact that shapes this:** creature-only battles are informationally *symmetric*. The
battle UI shows the full stat sheet for any unit, own or enemy, with no ownership gating
(`Cursor::WAR_INFO` → `Dialog::ArmyInfo`). So today the two profiles differ only by oracle
fields. Real hidden information — enemy hero mana, adventure-map fog — arrives later, and
`observable_v1` extends to cover it without forking the schema.

**The digest always covers full state**, regardless of profile. Replay integrity must not depend
on what a policy was allowed to see.

## Axis 3 — why not pixels

Three independent reasons, in increasing order of finality:

1. **Cost.** Semantic planes vs rendering measured ~14× throughput apart (Griddly: ~72,800 vs
   ~5,000 FPS) with *no* measured performance benefit to pixels.
2. **Precedent.** SC2's "minimap" was never RGB — DeepMind shipped synthetic feature layers, on
   the stated rationale that agents should not spend capacity learning to read numbers off a
   screen.
3. **Architecture.** Our headless core loads **zero game assets**. Rendering would drag the
   display/AGG stack back in, undoing the finding the whole environment is built on.

So a "minimap" here means a *semantic plane tensor*, not an image. Anything pixel-real lives on
the separate `play-harness` branch.

## The rule underneath all of it: the Markov property

If an attribute influences battle dynamics, it must either be **in the observation** or **out of
the dynamics**. A hidden-but-active attribute makes the environment non-Markov, and a policy
trained on it is learning from a state that does not explain its own transitions.

The reference project (vcmi-gym) hit this and chose *removal* — it deleted morale, luck, and
terrain effects from the game rather than half-observe them. **We chose exposure**: those
mechanics stay live, and their state fields appear in *both* profiles. Stochastic transitions are
fine; unobserved state is not.

## Why it matters here

These decisions are already committed (ADR 0001, ADR 0004) and bind Milestone 4's observation
serializer. Getting them right *before* the protocol ships avoids a schema break later — the
plane emitter costs nothing when a policy ignores it.

## What this does *not* say

It does not settle whether a CNN over planes actually beats an entity-transformer or an MLP at
`11×9` scale — no published ablation exists at this size. That is a deliberate in-house
experiment at the training milestone, which is precisely why both modalities exist.

## See also
- [[legal-actions-and-masking]] — the action side of the same interface.
- [[determinism-seeds-and-digests]] — why the digest ignores profiles.
