---
title: "MCTS as Regularized Policy Optimization (ICML 2020)"
type: reference
source: https://arxiv.org/abs/2007.12509
read: 2026-08-06
runs: []
tags: [reference, search, puct, policy-optimization]
local: ["files/arxiv-2007.12509.pdf"]
---

# MCTS as Regularized Policy Optimization (ICML 2020)

Grill et al. show that AlphaZero's empirical visit distribution approximates the solution of a regularized policy-optimization problem, the prior acting as the regularizer and the search values as the objective. Two consequences transfer directly to this project.

The first is diagnostic. Visit counts approximate that solution only when the simulation budget is large relative to the action set; at small budgets the visit distribution is a poor approximation and the paper's remedy is to use the computed solution rather than the counts. Our root search runs 32 simulations against 5 to 30 legal actions, which is squarely the small-budget regime, so the visit-count target this project distills is the weaker of the two available signals.

The second is architectural. The regularized view makes explicit that search improves on the prior only to the extent the values it backs up are informative. With rollout values that means the rollout policy's quality, and with leaf values it means the value network's accuracy, which is the measured failure of the 2026-08-06 value probe: a leaf evaluator explaining 0.09 of return variance ranks candidates worse than the prior alone.

Where we use it: the interpretation of the value-leaf probe in [[../../rl/training-design#Search leaves and the value question, measured 2026-08-06]], and the recorded upgrade path from visit counts to the regularized solution if search distillation continues.

Related: [[alphazero]], [[uct]], [[expert-iteration]], [[muzero]]
