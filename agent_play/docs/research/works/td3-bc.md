---
title: "TD3+BC, a minimalist offline baseline (NeurIPS 2021)"
type: reference
source: https://arxiv.org/abs/2106.06860
read: 2026-08-06
runs: []
tags: [reference, offline-rl, behavior-regularization]
local: ["files/arxiv-2106.06860.pdf"]
---

# TD3+BC, a minimalist offline baseline (NeurIPS 2021)

Fujimoto and Gu show that adding a behavior-cloning term to an online algorithm's policy update, plus normalizing the data, matches the state-of-the-art offline methods at half the compute. The message is less the specific algorithm than the finding that the simplest support-keeping device, a cloning anchor inside the loss, buys most of what elaborate machinery buys.

Where we use it: the remedies survey in [[../../rl/off-support-and-offline-improvement]], and as external confirmation of a choice this project made before reading it, since our stage-3 design already anchors reinforcement learning on the cloning loss ([[../../rl/training-design]]) for exactly this reason.

Related: [[bcq-extrapolation]], [[awr]], [[cql]], [[dppo-trust-region]]
