---
title: "Griddly — a platform for AI research in games (multi-observer engine)"
type: project
authors: Bamford et al.
year: 2020-2023
arxiv: "2011.06363"
quality: primary
urls:
  - https://arxiv.org/abs/2011.06363
  - https://griddly.readthedocs.io/en/latest/getting-started/observation%20spaces/index.html
  - https://griddly.readthedocs.io/en/latest/rllib/intro/index.html
  - https://github.com/Bam4d/Griddly
runs: [minimap-observations]
tags: [reference, multi-observer, semantic-planes, throughput, grid-games]
local: ["files/arxiv-2011.06363.pdf", "files/griddly-README.md", "files/griddly-observation-spaces.html", "files/griddly-rllib-intro.html"]
---

# Griddly

The strongest evidence base for semantic planes over rendered pixels, and the shipped "multiple observers over one game state" API.

Verified claims anchored here (all 3-0 unless noted):

- Four+ observer types on one engine: three Vulkan pixel renderers (SPRITE_2D, BLOCK_2D, ISOMETRIC) plus the non-pixel VECTOR observer (binary per-cell presence planes + optional one-hot ownership/rotation/variables, code-confirmed in `VectorObserver.cpp`).
- Per-player observers (possibly partial) plus an always-all-seeing global observer, each independently configurable, the full_v1/observable_v1 split with modalities, shipped.
- Performance parity: 150-experiment baseline (10 games × 5 levels × {Vector, Block, Sprite}), results "consistent across all the representations"; no systematic pixel advantage (medium confidence: authors' own single-source baseline).
- Throughput: vector ~72,790 FPS vs rendered ~5,023 FPS (~14.5×), the only surviving quantified rendering-vs-planes comparison (2020 hardware, Vulkan; architecture-specific).

Where we use it: the decisive evidence in [[../decisions/0004-spatial-observation-modality]] (planes yes, pixels no).

Related: [[ref-pysc2]], [[ref-nle]]
