---
title: "MicroRTS-Py — Farama reference implementation of Gym-µRTS"
type: codebase
year: 2021-2025
quality: primary
urls:
  - https://github.com/Farama-Foundation/MicroRTS-Py
runs: [rl-approaches, minimap-observations]
tags: [reference, microrts, action-masking, partial-observability, trueskill, evaluation]
local: ["files/microrts-py-README.md"]
---

# MicroRTS-Py (codebase)

The reference implementation behind [[gym-microrts]]. Deprecated by Farama in Aug 2025, treat as a canonical but frozen reference, not a live dependency.

Verified claims anchored here (all 3-0):

- Observation `Box(0,1,(h,w,29),int32)` binary one-hot planes (HP 5, resources 5, owner 3, unit type 8, action 6, terrain 2).
- Dual observation modes via constructor flag: `partial_obs=True` appends two visibility planes (29 → 31 channels), one schema, not two (the pattern ADR 0001/0004 adopt).
- Action space `MultiDiscrete(7hw)` per-cell factorized ("gridnet"); PPO scripts implement `CategoricalMasked` (`torch.where(mask, logits, -1e8)`) and the README directs users to cite Huang & Ontañón for the masking method.
- Evaluation: `league.py` TrueSkill league with uncertainty-based stopping (`while sigma > 1.4`), leaderboard by `mu − 3·sigma`, mixing scripted bots and checkpoints.

Where we use it: masking implementation to copy, TrueSkill eval protocol, the `partial_obs`-flag precedent for [[../../decisions/0001-observation-profiles]].

Related: [[gym-microrts]], [[invalid-action-masking]]
