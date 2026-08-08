---
title: "vcmi-gym / MMAI — HoMM3 battle RL environment and shipped combat AI"
type: project
year: 2023-2026
quality: primary
urls:
  - https://github.com/smanolloff/vcmi-gym
  - https://smanolloff.github.io/projects/vcmi-gym/
  - https://github.com/vcmi-mods/mmai
  - https://vcmi.eu
runs: [rl-approaches, minimap-observations]
tags: [reference, homm, rl-environment, action-masking, prior-art]
local: ["files/vcmi-gym-README.md", "files/vcmi-gym-blog.html", "files/mmai-README.md"]
---

# vcmi-gym / MMAI

The closest prior art in existence: a Gymnasium-compatible RL environment for Heroes of Might and Magic III battles on the open-source VCMI engine, whose trained models shipped in the real game as the experimental MMAI combat AI in VCMI 1.7.0 (2025-12-24). No fheroes2/HoMM2 RL environment exists, our project would be the first.

Verified claims anchored here (votes from the research pipeline):

- Trained models shipped in VCMI 1.7.0 via `vcmi-mods/mmai` (3-0).
- Observation: flat 12,685-float Box, 20 padded stack slots × 98 floats + 165 hexes × 65 floats, one-hot "categorical explicit with NULL category" (3-0; documented v3 env, docs lag code, copy patterns, not constants).
- Action space: flat fixed `Discrete(2312)` + boolean legal mask; the factorized multi-head variant failed to converge (2-1, medium).
- Training stack: CleanRL-inspired single-file masked implementations (MPPO/MPPG/MPPO-DNA/ MQRDQN), Ray PBT + W&B; SB3 prototyped then deliberately dropped (3-0).
- Observation design lesson: attributes that can't be represented were removed from the dynamics (morale/luck/terrain) to preserve the Markov property; v12+ moved to a 165-node/7-edge-type GENConv graph (3-0).
- Eval: ~75 % vs StupidAI / ~45 % vs BattleAI initially; v12 GNN ~65 % vs BattleAI; ~5 days / 2.5 M battles / $45 GPU per model (self-reported; 2-1 on the eval details).

Where we use it: [[../../archive/research-runs/2026-07-27-rl-approaches]] (template for schemas, masking, stack), [[../../decisions/0002-action-space]] (fixed space + mask), calibration that beating the strong scripted AI is a multi-iteration goal.

## The tabula-rasa reading, verified 2026-08-08

Re-read at commit `819ab32` for the owner's from-scratch survey, and the answer is strict: no imitation, no behavior cloning, no human data anywhere in the tree, random initialization throughout, which makes this the genre's strongest existence proof that a HoMM battle AI can be trained from nothing. What it spends instead of demonstrations is reward engineering and iteration time. The reward is dense and hand-shaped: a fixed per-step penalty, a progressive per-round penalty that squares after round ten (their anti-stall device, where ours is the terminal stalemate pricing), per-step net-damage and net-army-value differentials, a terminal army-value efficiency term relative to the starting balance, and truncation punished as losing the whole army. Opponents mix scripted bots, frozen own models, and dual-environment self-play over procedurally generated maps and 4,096-army pools, the league-lite shape. Current results after roughly three years of single-machine iteration: v15 at 72 percent against BattleAI and above 88 in mirrors, v13 shipped as VCMI 1.7's experimental MMAI.

The project's own failure ledger is as useful as its successes: DreamerV3 learned nothing, model-based I2A was impractical at 2,312 actions, LSTM, attention and residual variants all underperformed the flat network, removing the action mask collapsed the agent into spamming Defend, and MCTS was ruled out because VCMI cannot rewind battle state, the same arena constraint we have. Against our program the contrast is exact: they bought from-scratch viability with dense shaping and years of iteration; we bought terminal-only reward purity with an imitation start, and the genre record ([[botbowl-competitions]], [[pokemon-selfplay-from-scratch]]) says those are the two working corners of the same trade.

Related: [[gym-microrts]], [[invalid-action-masking]], [[alphastar]], [[botbowl-competitions]], [[pokemon-selfplay-from-scratch]]
