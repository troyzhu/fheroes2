---
title: "Implicit Q-Learning (ICLR 2022)"
type: reference
source: https://arxiv.org/abs/2110.06169
read: 2026-08-06
runs: []
tags: [reference, offline-rl, expectile, avoid-query]
local: ["files/arxiv-2110.06169.pdf"]
---

# Implicit Q-Learning (ICLR 2022)

Kostrikov, Nair and Levine remove the off-support query instead of penalizing it: the paper's own framing is a method that never needs to evaluate actions outside the dataset. The value is fitted by expectile regression over observed state-action pairs, an upper expectile standing in for "the best supported action here" without naming one, and the policy is extracted afterwards by advantage-weighted regression. Improvement comes from the function approximator generalizing across states, not from querying unseen actions.

Where we use it: the remedies survey in [[../../rl/off-support-and-offline-improvement]], as the cleanest structural answer to the failure our lab measured, since a query that never happens cannot extrapolate. Its extraction step is exactly [[awr]], which is the piece we can test first on our own corpora.

Related: [[awr]], [[cql]], [[bcq-extrapolation]], [[one-step-offline-rl]]
