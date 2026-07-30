# Coarse-minimap / hybrid observations — verified literature consolidation

> Produced 2026-07-29 by the same adversarial research pipeline as `research_rl_approaches.md`: 5 search angles → 20 sources fetched → 96 claims extracted → top 25 verified (3 votes each) → 24 confirmed, 1 refuted, 0 unverified. Question: should the fheroes2 battle agent get a coarse, pixel-like minimap view in addition to structured state, and if so, how? The design decision extracted from this report is `decisions/0004-spatial-observation-modality.md`.

## The one-paragraph answer

The verified literature strongly supports semantic per-cell feature planes over rendered pixels for this genre: SC2LE deliberately shipped categorical/scalar "feature layers" instead of RGB; Griddly's non-pixel VECTOR observer matches its pixel observers on final RL performance while running ~14× faster; and NLE and AlphaStar both build their coarse spatial maps *synthetically from structured state*, never from the game renderer. The proven hybrid pattern is entity list + spatial planes + scalar vector, fused by concatenation (AlphaStar's Core, NLE's baseline), optionally with entity embeddings *scattered* into the planes (AlphaStar). The "one game state, multiple independently-toggleable observers" API shape has two shipped precedents (PySC2's feature/RGB/raw interfaces; Griddly's per-player + global observers). For an 11×9 board: add the minimap as a derived semantic-plane modality in the observation schema, and reject true pixel rendering, every precedent treats it as an optional, ~10×-costlier add-on with no verified performance benefit, and it would reintroduce exactly the display/asset dependencies the headless core was built to exclude.

## 1. Feature layers vs raw pixels (SC2LE/PySC2 — the canonical precedent)

- SC2LE launched with no RGB at all: the API generated semantic feature layers; all baselines (Atari-net, FullyConv, FullyConv LSTM) trained on 64×64 feature layers; RGB arrived later as an optional interface. Verified 3-0. *("the StarCraft II API does not currently render RGB pixels. Rather, it generates a set of 'feature layers'…")*
- Feature layers are a synthetic top-down rasterization of game state, not captured pixels: configurable N×M resolution; split into a detailed "screen" and a lower-information whole-map "minimap"; every plane typed categorical (unit type, owner, visibility, player_relative) or scalar (HP, height map). Verified 3-0 (five merged claims; note the doc's "~25 layers" is stale, current code defines 38).
- DeepMind's stated rationale institutionalizes the hybrid: structured tensors accompany the planes because *"agents aren't expected to learn to read text and numbers from pixels, especially at low resolution"*. Verified 3-0.
- (gap) The promised raw-RGB-vs-feature-layer comparison was never published in a verifiable form; no claim on RGB training cost/failure survived. (AlphaStar itself used the raw/feature interface, not RGB.)

## 2. Hybrid fusion — how spatial + structured coexist

- **AlphaStar's spatial encoder consumes a 128×128 *semantic* minimap** (camera, scattered_entities, height_map, visibility, creep, owners, alerts, pathable, buildable, not RGB), downsampled by three stride-2 convs + ResBlocks. Verified 3-0 against the Nature supplementary (mirror) and the mini-AlphaStar reproduction.
- Scatter connections are the canonical entity→spatial fusion: per-entity embeddings projected to 32 dims and scattered into a map plane at each unit's location, feeding the spatial CNN alongside terrain planes. Verified 3-0.
- Three modalities, one core: entity-transformer output + spatial-CNN output + scalar encodings concatenated into one tensor for the LSTM core, spatial and structured *coexist*, neither replaces the other. Verified 3-0.
- Ablation caveat (2-1 vote, medium confidence): Nature Fig. 3f/h did ablate the architecture components, but the openly visible text quotes no component-level effect sizes, those sit behind the paywall/supplementary. We can cite *that* scatter connections were ablated, not by how much they helped.

## 3. Multiple observers over one game state (the API shape)

- Griddly ships exactly this: three Vulkan pixel renderers (SPRITE_2D, BLOCK_2D, ISOMETRIC) plus a non-pixel VECTOR observer, binary per-cell object-presence planes with optional one-hot ownership/rotation and per-object variable channels, all selectable by configuration, with independently configurable per-player observers (possibly partial) *and* a global all-seeing observer on the same engine instance. Verified 3-0 (code-confirmed in `VectorObserver.cpp`). This is our `full_v1`/`observable_v1` split extended with modalities, already shipped in another engine.
- Performance parity: in Griddly's 150-experiment baseline (10 games × 5 levels × {Vector, Block, Sprite}), final RL performance was mostly consistent across representations, no systematic advantage for pixels. Verified 3-0; medium confidence (single-source, library authors' own baseline).
- Throughput: vector output ~72,790 FPS vs rendered ~5,023 FPS (~14.5×) on the same environments, the one surviving quantified rendering-vs-planes comparison (2020 hardware, Vulkan renderer; ratio is architecture-specific). Verified 3-0. *(A tempting companion claim, NLE 14.4K vs ALE 0.9K steps/s, was refuted 0-3; do not reuse it.)*
- NLE at small scale: NetHack's observation is a 21×79 symbolic grid + stats vector + text, no pixels, and the baseline net embeds per-cell IDs, runs CNNs over the embedding planes including a dedicated 9×9 egocentric-crop CNN, and fuses with an MLP-encoded stats vector: practice evidence that CNN-over-semantic-planes is standard at board sizes comparable to our 11×9. Verified 3-0.

## 4. What did NOT survive (honest gaps)

No verified evidence emerged on: MicroRTS/Lux/Battlecode synthetic-rasterization specifics, FiLM or broadcast-vector-into-planes ablations, the RogueNet-vs-IMPALA-CNN parameter comparison (it survived in the *previous* report's extraction but not this run's verified top-25), controlled CNN-vs-MLP-vs-entity-transformer ablations at ~11×9 scale, sub-cell/anti-aliased detail benefits, or hex-grid rasterization conventions. These are open questions, the first is cheaply answerable in-house at the training milestone by swapping policy heads over the same planes.

## Recommendation (synthesis — adopted as ADR 0004)

1. Reject a true rendered minimap. Asset/display dependencies return, ~10× throughput cost precedent, zero verified benefit anywhere.
2. **Add a coarse *semantic* minimap as a derived modality**: a fixed-layout 11×9×C plane tensor (occupancy per side, unit class, count/HP fractions, passability/obstacles; reachability and threat planes once Milestone 3's resolvers exist) rasterized from the *same* state that feeds the entity list, nearly free to emit, and exactly what a policy CNN or AlphaStar-style scatter fusion would consume.
3. Expose it PySC2/Griddly-style: observation *modalities* (`entities`, `planes`) independently toggleable and orthogonal to the `full_v1`/`observable_v1` *profiles* (planes are filtered by whichever profile is active). One schema, one JSONL protocol, multiple observers.
4. Timing: specify the plane layout during observation-schema design (Milestone 3/4) so the protocol carries it without a schema break; defer the CNN-vs-transformer policy-head choice to the training milestone, an unused plane emitter costs nothing.

## Sources (20 fetched; key ones)

SC2LE (arXiv:1708.04782) + PySC2 repo/docs · AlphaStar Nature 2019 + detailed-architecture supplementary mirror + AlphaStar Unplugged (arXiv:2308.03526) · Griddly (arXiv:2011.06363, docs, `VectorObserver.cpp`) · NLE (arXiv:2006.13760, repo) · MiniHack docs · Gym-μRTS (arXiv:2105.13807) · Lux AI 2021 winner repo · entity-neural-network/rogue-net.
