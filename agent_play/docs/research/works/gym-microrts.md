---
title: "Gym-µRTS: Toward Affordable Full Game Real-time Strategy Games Research with Deep RL"
type: paper
authors: Huang, Ontañón, Bamford, Grela
year: 2021
arxiv: "2105.13807"
quality: primary
urls:
  - https://arxiv.org/abs/2105.13807
runs: [rl-approaches, minimap-observations]
tags: [reference, microrts, action-masking, factorized-actions, single-machine, evaluation]
local: ["files/arxiv-2105.13807.pdf"]
---

# Gym-µRTS (paper)

The single-machine feasibility proof and the masking-ablation evidence base.

Verified claims anchored here (all 3-0):

- Masking is make-or-break: PPO with action composition but *no* mask scored 0.0 cumulative win rate; partial mask (action type only, PySC2/SMAC-style) 0.32; full per-component mask 0.82–0.91. Mask every component.
- Factorized ("composed") action heads: 8 independent softmax components; ~301 logits instead of ~50 M joint actions (paper's own printed arithmetic is sloppy, ~334-341 actual, reproduced faithfully).
- State of the art on one modest machine: 91 % cumulative win rate vs all 13 past competition bots (incl. champion CoacAI) in ~60–63 h on 1 GPU / 3 vCPU / 16 GB RAM (single-map; CUDA, no Apple-silicon data exists).
- Observation: per-cell one-hot feature planes (h, w, 27), no entity list, the "semantic minimap" in degenerate form.
- Eval protocol: 4 seeds, best-seed 100 games/bot under step cap + opponent-diverse training mix (18 CoacAI + 2×3 weaker bots); single-opponent agents lose to simple rushes.

Where we use it: masking requirement + eval design in [[../../archive/research-runs/2026-07-27-rl-approaches]], hardware feasibility for the M2 mini, [[../../decisions/0002-action-space]].

## The tabula-rasa reading, verified 2026-08-08

Strictly from scratch, confirmed in paper and code for the owner's survey: no imitation anywhere, and the working configuration is instructive in both directions. The experiment-level reward is densely shaped, six weighted event terms with the weights $[10, 1, 1, 0.2, 1, 4]$ over win-loss, resource, worker, building, attack and combat-unit events, while the environment's own default is nearly sparse, so the shaping was a deliberate experimental choice, not an accident of the framework. The opponent mix leans on the strongest scripted bot, eighteen of twenty-four environments against the 2020 champion, and the paper's own appendix reports pure self-play performing worse, the ecology point: train against the incumbent when one exists. The line continues past the paper: RAISocketAI, the first deep-learning winner of the IEEE competition in six editions, kept the recipe and added phase-scheduled reward annealing from shaped toward sparse win-loss with separate value heads, plus a map-size transfer curriculum, at 1.5 billion steps over about seventy GPU-days.

Related: [[microrts-py]], [[invalid-action-masking]], [[vcmi-gym]], [[lux-ai-competitions]]
