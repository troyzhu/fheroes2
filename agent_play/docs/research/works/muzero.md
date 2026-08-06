---
title: "MuZero (Nature 2020)"
type: reference
source: https://arxiv.org/abs/1911.08265
read: 2026-08-06
runs: []
tags: [reference, search, learned-model, value]
local: ["files/arxiv-1911.08265.pdf"]
---

# MuZero (Nature 2020)

Schrittwieser et al. remove the simulator from the AlphaZero loop: a learned latent model predicts reward, value and policy, and search runs inside it. The headline transfer for this project is inverted, because we already have the thing MuZero replaces. Our simulator is exact, deterministic and costs milliseconds through prefix replay, so learning a model buys nothing here; what MuZero shares with AlphaZero and does need is a value function trained on the search's own play, which the 2026-08-06 probe measured as this project's missing piece rather than its model.

The paper's practical detail that does transfer is the training target: value targets come from search returns rather than from the raw outcome, so the value learns what search believes about a position instead of what a single trajectory happened to produce, and it is trained jointly with the policy rather than as a probe on a frozen trunk. Both differ from this project's current critic, which is a 193-parameter head on a frozen imitation trunk fitted to Monte-Carlo returns; [[../../rl/training-design#Search leaves and the value question, measured 2026-08-06]] records that as the diagnosis of the failed value-leaf search.

Where we use it: the recorded requirements for a usable leaf evaluator, and the reason a learned model is explicitly not on this project's path.

Related: [[alphazero]], [[mcts-regularized-policy-optimization]], [[alphastar-unplugged]], [[expert-iteration]]
