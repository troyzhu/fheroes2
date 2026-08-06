---
title: "Off-Policy Deep RL Without Exploration (ICML 2019)"
type: reference
source: https://arxiv.org/abs/1812.02900
read: 2026-08-05
runs: []
tags: [reference, offline-rl, extrapolation-error, value-estimation]
local: ["files/arxiv-1812.02900.pdf"]
---

# Off-Policy Deep RL Without Exploration (ICML 2019)

Fujimoto et al. name the failure this project measured on 2026-08-05: extrapolation error, value estimates queried on state-action pairs the training data never covered, which off-policy methods then maximize into, producing confident nonsense. Their repair, BCQ, constrains the improved policy to actions the data supports.

The named phenomenon in our stack: the behavior critic fitted on teacher play carries an optimistic bias of $+0.32$ and negative explained variance on student-played states (`critic_calibration.py`), which is extrapolation error made visible by simply evaluating where the data was not. Every proposed consumer of the critic inherits it: a GAE baseline misleads on exactly the states an exploring policy visits, a search leaf evaluator inflates off-data branches (the collapse mechanism [[alphastar-unplugged]] observed), and only rollout returns, which query the engine instead of the estimate, are immune.

Where we use it: the value-estimation analysis in [[../../rl/training-design#The behavior value, measured where it would be spent]] and the design rule that search here scores branches by rollout, not by critic.

Related: [[double-q-overestimation]], [[one-step-offline-rl]], [[alphastar-unplugged]]
