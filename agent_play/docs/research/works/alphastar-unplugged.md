---
title: "AlphaStar Unplugged (2023)"
type: reference
source: https://arxiv.org/abs/2308.03526
read: 2026-08-05
runs: []
tags: [reference, starcraft, offline-rl, behavior-value, mcts, one-step]
local: ["files/arxiv-2308.03526.pdf"]
---

# AlphaStar Unplugged (2023)

Large-scale offline reinforcement learning on StarCraft II human games, from the AlphaStar group. Vendored since the first research sweep and only shallowly cited until the 2026-08-05 review, when it turned out to answer three live questions at once with the largest comparable evidence base there is.

Claims read directly from the local PDF:

- Their best agents follow a two-step recipe: fit the behavior policy and the behavior value function, then apply one step of policy improvement using that value, during training or at inference. Every multi-step off-policy variant they tried failed to beat the unconditional behavior-cloning baseline; the successful family is exactly the one-step regime of Brandfonbrener et al. 2021 ([[one-step-offline-rl]]).
- MCTS at inference time over the learned prior and value is stable and improves play (their MZS-MCTS agent, following Hubert et al. 2021); MCTS at training time collapses the policy. Repeated application of MCTS improvement against the fixed behavior value degrades: the searched version of the once-improved policy is worse than the once-improved policy itself, because iterating detaches the improvement from the distribution the value understands.
- Their strongest offline agents reach a 90 percent win rate against the published AlphaStar behavior-cloning agent, so the offline route past a demonstrator is real at scale, not a small-domain artifact.

Where we use it: the 2026-08-05 review ([[../../archive/experiments/2026-08-05-dagger-and-battlefield-transfer]]). Our measured results mirror theirs at small scale: unanchored PPO eroding the clone matches their multi-step failures, the non-monotone second DAgger round matches their iterated-improvement degradation, and our critic measuring worse than the mean on student states ([[../../rl/training-design#The behavior value, measured where it would be spent]]) is the mechanism their MCTS-at-training collapse exploits. It is the strongest external argument for one improvement step at a time, each step re-grounded in fresh data.

Related: [[alphastar]], [[metamon]], [[one-step-offline-rl]], [[bcq-extrapolation]]
