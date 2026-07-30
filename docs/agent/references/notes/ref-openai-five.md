---
title: "Dota 2 with Large Scale Deep Reinforcement Learning (OpenAI Five)"
type: paper
authors: OpenAI (Berner et al.)
year: 2019
arxiv: "1912.06680"
quality: primary
urls:
  - https://arxiv.org/abs/1912.06680
runs: [rl-approaches]
tags: [reference, structured-observations, dota, scale]
local: ["files/arxiv-1912.06680.pdf"]
---

# OpenAI Five (Dota 2)

The other giant-scale precedent for **structured (non-pixel) observations**: OpenAI Five
consumed semantic data arrays (~16,000 mostly float/categorical values per timestep) rather
than rendered frames, with the rationale that rendering every frame across all training games
would have multiplied compute requirements many-fold.

**Verification status**: extracted in run 1 (structured-arrays rationale) but **not among the
verified top-25** — cite the paper directly for specifics; the direction (structured beats
pixels on cost at scale) is consistent with everything that *was* verified
([[ref-sc2le]], [[ref-griddly]]).

**Where we use it**: cost rationale background in [[../research_rl_approaches]] §1 (marked as
gap there) and [[../research_minimap_observations]].

Related: [[ref-alphastar]], [[ref-sc2le]]
