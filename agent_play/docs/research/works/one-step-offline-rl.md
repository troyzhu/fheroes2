---
title: "Offline RL Without Off-Policy Evaluation (NeurIPS 2021)"
type: reference
source: https://arxiv.org/abs/2106.08909
read: 2026-08-05
runs: []
tags: [reference, offline-rl, one-step, behavior-value]
local: ["files/arxiv-2106.08909.pdf"]
---

# Offline RL Without Off-Policy Evaluation (NeurIPS 2021)

Brandfonbrener et al. show that one step of policy improvement against the behavior value function, $\hat{Q}^\beta$ fitted on the data with no off-policy evaluation at all, matches or beats iterative offline-RL methods across standard benchmarks. The mechanism argument: iterative methods must evaluate policies increasingly far from the data, where value estimates degrade, while the one-step method only ever queries the value on the distribution it was fitted on.

This is the theory piece behind the recipe [[alphastar-unplugged]] found empirically at StarCraft scale, and behind [[metamon]]'s offline improvement past its demonstrators. Its warning is quantitative in our own stack: the refitted behavior critic explains 0.30 of return variance on held-out teacher play and less than zero on student-played states with a systematic optimistic bias ([[../../rl/training-design#The behavior value, measured where it would be spent]]), which is precisely the off-distribution degradation that breaks multi-step methods.

Where we use it: the recorded discipline that every improvement operator here, DAgger rounds, search-as-teacher, or advantage-weighted training, takes one step from fresh data rather than iterating against a frozen estimate.

Related: [[alphastar-unplugged]], [[bcq-extrapolation]], [[metamon]]
