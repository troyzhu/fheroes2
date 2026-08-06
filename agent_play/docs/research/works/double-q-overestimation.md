---
title: "Deep RL with Double Q-learning (AAAI 2016)"
type: reference
source: https://arxiv.org/abs/1509.06461
read: 2026-08-05
runs: []
tags: [reference, value-estimation, overestimation, dqn]
local: ["files/arxiv-1509.06461.pdf"]
---

# Deep RL with Double Q-learning (AAAI 2016)

Van Hasselt et al., the classic result that bootstrapped value learning with a max operator systematically overestimates, because the same noisy estimate both selects and evaluates the maximizing action, and that the overestimation is not harmless but degrades the learned policies; decoupling selection from evaluation (Double DQN) removes most of it.

Relevance here is by contrast as much as by warning. Our critic is fitted on Monte-Carlo discounted returns of recorded episodes, no bootstrapping and no max, so this particular bias mechanism is absent, and the optimism we measured on student states comes from distribution shift instead ([[bcq-extrapolation]]). The warning becomes live the moment anything here bootstraps or maximizes over value estimates: a TD-trained critic, a Q-head over the 793 actions, or search backups that take maxima over noisy leaf values, which is one more reason the search probe scores by full rollouts.

Where we use it: the value-estimation analysis in [[../../rl/training-design#The behavior value, measured where it would be spent]], as the boundary between the bias we have and the bias we would acquire.

Related: [[bcq-extrapolation]], [[one-step-offline-rl]]
