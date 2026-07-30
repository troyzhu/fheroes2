---
title: "The NetHack Learning Environment (+ MiniHack, NLE-revisited)"
type: project
authors: Küttler et al. (FAIR)
year: 2020
arxiv: "2006.13760"
quality: primary
urls:
  - https://arxiv.org/abs/2006.13760
  - https://github.com/heiner/nle
  - https://minihack.readthedocs.io/en/latest/getting-started/observation_spaces.html
  - https://iclr-blogposts.github.io/2026/blog/2026/revisiting-the-nle/
runs: [minimap-observations]
tags: [reference, symbolic-observation, small-board-cnn, hybrid-fusion]
local: ["files/arxiv-2006.13760.pdf", "files/nle-README.md", "files/minihack-observation-spaces.html", "files/nle-revisited-blog.html"]
---

# NetHack Learning Environment (NLE)

Practice evidence for symbolic multi-component observations and CNN-over-semantic-planes at small board scale.

Verified claims anchored here (3-0, two merged claims):

- Observations are symbolic, never pixels: a 21×79 grid of glyph IDs with parallel char/color/special planes, plus a stats vector, message text, and inventory tensors (pixel views exist only via external wrappers).
- The baseline agent embeds per-cell glyph IDs into learned vectors, runs CNNs over the embedding planes including a dedicated 9×9 egocentric-crop CNN (a board patch comparable to our 11×9), encodes stats with an MLP, and fuses by concatenation + MLP into an LSTM, the hybrid coarse-spatial + vector architecture over a synthetic, non-rendered raster.
- Caveat: practice evidence, not a controlled CNN-vs-MLP-vs-transformer ablation. A tempting NLE-vs-ALE throughput figure (14.4K vs 0.90K steps/s) was refuted 0-3, do not cite it.

Where we use it: [[../research_minimap_observations]] §3; supports CNN heads over `planes_v1` in [[../decisions/0004-spatial-observation-modality]].

Related: [[ref-griddly]], [[ref-alphastar]]
