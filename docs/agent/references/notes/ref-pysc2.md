---
title: "PySC2 — DeepMind's StarCraft II environment (code + docs)"
type: codebase
year: 2017-2024
quality: primary
urls:
  - https://github.com/google-deepmind/pysc2
  - https://github.com/google-deepmind/pysc2/blob/master/docs/environment.md
runs: [minimap-observations]
tags: [reference, starcraft, multi-observer, observation-design]
local: ["files/pysc2-README.md", "files/pysc2-environment.md"]
---

# PySC2 (code and environment docs)

The shipped realization of [[ref-sc2le]], and a direct precedent for **one game state, multiple
independently-toggleable observers**: feature layers, RGB pixels, and raw-unit observations are
three separate interfaces, each with configurable resolution; feature layers are the default and
RGB is an optional add-on (`rgb_screen_size` defaults to `None`; `AgentInterfaceFormat` requires
at least one interface enabled).

**Verified claims anchored here** (all 3-0):

- Independent enable/disable of feature/RGB/raw interfaces over one game state.
- `features.py` types every layer SCALAR or CATEGORICAL; current code defines 38 layers (the
  "~25" in older docs is stale); some layers were later added to the minimap set.
- Feature layers are rendered "from a top down orthogonal camera, as opposed to a perspective
  camera" — synthetic rasterization, not captured render output.

**Where we use it**: the modality-toggle API shape of
[[../decisions/0004-spatial-observation-modality]] and the profile/modality orthogonality in
[[../decisions/0001-observation-profiles]].

Related: [[ref-sc2le]], [[ref-griddly]]
