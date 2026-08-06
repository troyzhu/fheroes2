---
title: "Conservative Q-Learning (NeurIPS 2020)"
type: reference
source: https://arxiv.org/abs/2006.04779
read: 2026-08-06
runs: []
tags: [reference, offline-rl, pessimism, q-learning]
local: ["files/arxiv-2006.04779.pdf"]
---

# Conservative Q-Learning (NeurIPS 2020)

Kumar, Zhou, Tucker and Levine train the Q to be deliberately pessimistic: an added regularizer pushes Q down on actions the policy would like and up on actions the data actually took, so the expected value under the learned Q lower-bounds the true value. Improvement against a lower bound is safe by construction, which is the pessimism family's whole bet: rather than keeping the policy near the data, make the estimator too gloomy to be exploited off it.

Where we use it: the remedies survey in [[../../rl/off-support-and-offline-improvement]]. Our measured failure, a behavior Q at 0.853 on-support whose top-5 re-ranking collapses play ([[../../rl/value-estimation-lab]]), is precisely the exploitation CQL's lower bound exists to prevent; the note's verdict there explains why we prefer support-side remedies first at our scale.

Related: [[bcq-extrapolation]], [[iql]], [[edac-ensembles]], [[one-step-offline-rl]]
