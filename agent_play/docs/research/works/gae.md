---
title: "Generalized Advantage Estimation (ICLR 2016)"
type: reference
source: https://arxiv.org/abs/1506.02438
read: 2026-08-06
runs: []
tags: [reference, credit-assignment, advantage, gae]
local: ["files/arxiv-1506.02438.pdf"]
---

# Generalized Advantage Estimation (ICLR 2016)

Schulman et al., the canonical treatment of the credit-assignment dial the owner's critique turns on. The trajectory-level advantage (every action inheriting the episode's whole return) sits at one end, $\lambda = 1$: unbiased and maximally smeared, a bad action redeemed later still reinforced. Per-step temporal-difference credit sits at the other, $\lambda = 0$: sharply localized and exactly as biased as the value function is wrong. GAE's $\lambda$ interpolates, trading the smearing against the critic's error.

The transfer caveat is this project's measured critic: with the behavior value at $-0.13$ explained variance and $+0.32$ optimistic bias on student-visited states, low-$\lambda$ credit localizes toward an estimate that misleads exactly off-distribution, so the interpolation buys sharpness with the wrong currency here until the value is fitted on the states being credited. Search-based per-decision values (`credit_assignment.py`) sidestep the dial entirely by paying rollouts for locality instead of bias.

Where we use it: the credit-assignment analysis in [[../../rl/reward-design#The dense family]] and the measured mis-signing rates of trajectory credit.

Related: [[one-step-offline-rl]], [[bcq-extrapolation]], [[alphastar-unplugged]]
