---
title: "StarCraft II: A New Challenge for Reinforcement Learning (SC2LE)"
type: paper
authors: Vinyals, Ewalds, Bartunov, et al. (DeepMind & Blizzard)
year: 2017
arxiv: "1708.04782"
quality: primary
urls:
  - https://arxiv.org/abs/1708.04782
runs: [minimap-observations]
tags: [reference, starcraft, feature-layers, observation-design]
local: ["files/arxiv-1708.04782.pdf"]
---

# SC2LE (StarCraft II Learning Environment)

The canonical precedent for semantic "minimap" observations. The launch API deliberately exposed no RGB pixels at all: observations are *feature layers*, coarse spatial rasterizations of game state drawn by a synthetic top-down orthogonal camera at configurable N×M resolution, split into a detailed "screen" and a whole-map "minimap", every plane typed categorical (unit type, owner, visibility, player_relative) or scalar (HP, height map).

Verified claims anchored here (all 3-0):

- "the StarCraft II API does not currently render RGB pixels. Rather, it generates a set of 'feature layers'", RGB arrived only in a later release; all baselines (Atari-net, FullyConv, FullyConv LSTM) trained on 64×64 feature layers.
- Structured non-spatial tensors accompany the planes because "agents aren't expected to learn to read text and numbers from pixels, especially at low resolution", the API itself institutionalizes the hybrid planes+vector pattern.
- (gap) The promised raw-RGB-vs-feature-layer comparison was never published in verifiable form.

Where we use it: [[../research_minimap_observations]]; design template for [[../decisions/0004-spatial-observation-modality]] (`planes_v1`).

Related: [[ref-pysc2]], [[ref-alphastar]], [[ref-griddly]]
