---
title: "Uncertainty-based offline RL with diversified Q-ensembles (NeurIPS 2021)"
type: reference
source: https://arxiv.org/abs/2110.01548
read: 2026-08-06
runs: []
tags: [reference, offline-rl, ensembles, uncertainty]
local: ["files/arxiv-2110.01548.pdf"]
---

# Uncertainty-based offline RL with diversified Q-ensembles (NeurIPS 2021)

An, Moon, Kim and Song get pessimism from disagreement: train many Q networks, act on the clipped minimum, and out-of-distribution actions are penalized automatically because the ensemble disagrees exactly where data is absent. With enough diversified members this outperforms the explicit-penalty methods without modeling the data distribution at all.

Where we use it: the remedies survey in [[../../rl/off-support-and-offline-improvement]]. It is the uncertainty-family representative there, and its mechanism explains a fact our lab already produced, that two same-recipe value fits can disagree wildly off-distribution (the unseeded-lottery incident in [[../../rl/value-estimation-lab]]), which is the signal an ensemble harvests deliberately.

Related: [[cql]], [[double-q-overestimation]], [[bcq-extrapolation]]
