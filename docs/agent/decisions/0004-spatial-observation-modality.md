# ADR 0004 — Semantic spatial-plane observation modality; true pixel rendering rejected

- Status: accepted 2026-07-29 (plane layout specified with Milestone 3/4 observation serialization; emitter lands there)
- Context: user proposal 2026-07-29 (coarse "minimap" view for the agent); [[references/report-spatial-observations]]; extends ADR 0001 (observation profiles)

## Context

The owner proposed giving the agent a small, coarse, pixel-style minimap plus auxiliary structured information, as a complementary observation mode. The verified literature answer (24/25 claims confirmed): the instinct is right, the pixels are not —

- SC2LE/PySC2's "minimap" is a synthetic semantic rasterization of game state (typed categorical/scalar planes at configurable resolution), never captured render output; DeepMind's stated rationale is that agents shouldn't burn capacity reading numbers out of pixels.
- Griddly ships a non-pixel VECTOR observer next to three pixel renderers over one game state: final RL performance is consistent across representations in its 150-experiment baseline, and the vector observer is ~14× faster than rendering.
- AlphaStar's spatial encoder consumes semantic planes; its scatter connections inject entity embeddings *into* those planes, spatial and entity modalities coexist and fuse.
- Our headless core loads zero game assets (Phase 0's headline result). A true rendered minimap would reintroduce the display/AGG dependencies the whole architecture exists to avoid.

## Decision

1. No rendered pixels in the training environment, ever. Anything pixel-real stays on the separate `play-harness` branch. (Rejecting this permanently for the env; revisit only with evidence the literature currently does not contain.)
2. The observation schema gains a second modality axis, orthogonal to ADR 0001's profiles:

   ```json
   "observation_profile":   "full_v1" | "observable_v1",
   "observation_modalities": ["entities", "planes"]        // default: ["entities"]
   ```

   Modalities are independently toggleable (PySC2/Griddly precedent);

requesting `planes` adds a `planes` section to each observation, filtered by the active profile like every other field.
3. `planes_v1` layout (fixed 11×9 board-index grid; exact channel list frozen at Milestone 3/4 implementation): per-side occupancy, unit-class identity, count fraction, HP fraction, speed, shooter flag/shots fraction, passability/obstacles; reachability and threat channels join once the Milestone 3 resolvers exist. Channels are typed categorical or scalar-normalized, SC2 feature-layer style.

   The tensor is derived from the same engine state that feeds the entity list, one source of truth, one digest (unchanged, still over full state).
4. Hex convention documented, not assumed: planes use the engine's own 11×9 `Battle::Board` cell indexing (row-offset hex). No published evidence favors any hex rasterization convention (verified gap), so we standardize on the engine's and record it in the schema.
5. Policy-head choice is deferred to the training milestone, CNN over planes vs entity-transformer vs MLP at this board size is an open question the literature does not answer; the plane emitter costs nothing when unused, and having both modalities enables AlphaStar-style hybrid/scatter fusion as well as the in-house ablation.

## Consequences

- Spec §12 observation schema gains the modality field and `planes` section; the JSONL protocol (Milestone 4) carries it from day one, no schema break later.
- Trajectory headers record requested modalities (dataset-compatibility axis, spec §15.4).
- `configs/env/` (ADR 0003) exposes profile and modalities as config keys.
- The state digest and all existing verification gates are untouched.
