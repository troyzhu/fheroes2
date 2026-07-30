---
title: "AlphaStar, Grandmaster level in StarCraft II (Nature 2019) + architecture supplementary"
type: paper
authors: Vinyals et al. (DeepMind)
year: 2019
quality: primary
urls:
  - https://www.nature.com/articles/s41586-019-1724-z
  - https://github.com/chengyu2/learning_alpha_star/blob/master/detailed-architecture.txt
  - https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf
  - https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/
related_papers:
  - "AlphaStar Unplugged (arXiv:2308.03526), offline RL benchmark, corroborates architecture"
  - "mini-AlphaStar (arXiv:2104.06890), reproduction, corroborates supplementary"
runs: [rl-approaches, minimap-observations]
tags: [reference, starcraft, entity-transformer, scatter-connections, behavior-cloning, hybrid-fusion]
local: ["files/alphastar-detailed-architecture.txt", "files/alphastar-nature-landing.html", "files/alphastar-deepmind-blog.html", "files/alphastar-unformatted.pdf", "files/arxiv-2308.03526.pdf", "files/arxiv-2104.06890.pdf"]
---

# AlphaStar (Nature 2019)

The high-capacity ceiling of structured-state design, and the strongest evidence that behavior cloning before RL works: the purely supervised agent reached 87 % win rate vs the built-in Elite bot (ablation ladder 0 → 7 → 36 (+pointer net) → 71 (+transformer) → 87 % (+scatter connections)).

Verified claims anchored here:

- Entity list (≤512 units) processed by transformer self-attention; deep LSTM core for partial observability; autoregressive action heads + recurrent pointer network over the entity set (3-0).
- Spatial encoder consumes a 128×128 semantic minimap (camera, scattered_entities, height_map, visibility, creep, owners, alerts, pathable, buildable), not RGB, downsampled to 16×16 by three stride-2 convs + ResBlocks (3-0, supplementary mirror corroborated by mini-AlphaStar).
- Scatter connections: per-entity embeddings projected to 32 dims and scattered into a map plane at each unit's location, the canonical entity→spatial fusion (3-0).
- Three modalities (entity/spatial/scalar) concatenated into one LSTM core, spatial and structured coexist (3-0).
- Ablation caveat (2-1, medium): Fig 3f/h ablated these components but open text quotes no component-level effect sizes (paywall); cite the fact of the ablation, not magnitudes.

Where we use it: BC-first validation ([[../../archive/research-runs/2026-07-27-rl-approaches]] §4), pointer-head compatibility in [[../../decisions/0002-action-space]], hybrid fusion pattern in [[../../decisions/0004-spatial-observation-modality]].

Related: [[sc2le]], [[pysc2]], [[entity-based-rl]]
