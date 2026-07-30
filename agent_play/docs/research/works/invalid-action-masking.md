---
title: "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms + tooling"
type: paper
authors: Huang, Ontañón
year: 2022 (FLAIRS-35)
arxiv: "2006.14171"
quality: primary
urls:
  - https://arxiv.org/abs/2006.14171
  - https://github.com/vwxyzjn/invalid-action-masking
  - https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
  - https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
runs: [rl-approaches]
tags: [reference, action-masking, ppo, theory]
local: ["files/arxiv-2006.14171.pdf", "files/sb3-maskable-ppo.html", "files/ppo-implementation-details.html"]
# note: the vwxyzjn/invalid-action-masking repo ships no README.md (manifest: FAILED); the paper PDF is the substantive source
---

# Invalid action masking (theory + canonical implementations)

The theoretical foundation for our entire action interface.

Verified claims anchored here (3-0, three merged claims):

- Masking is a valid policy gradient, not a hack (Proposition 1): the mask is a state-dependent differentiable transform of the logits, satisfying the Sutton et al. (2000) policy gradient theorem.
- Correct implementation: replace invalid logits with a large negative constant (−1e8) *before* softmax, gradients w.r.t. invalid logits become exactly zero, and apply the mask at both sampling and gradient time (sample-only "naive" masking blows up PPO's KL).
- Scaling ablations: masking's time-to-solve stays roughly flat as the invalid-action space grows; penalty-based legality (r ∈ {0,−0.01,−0.1,−1}) collapses on ≥10×10 maps, with r=−1 consistently worst. Never use penalties for legality.
- Canonical code: MicroRTS-Py `CategoricalMasked`; sb3-contrib MaskablePPO documents this paper as its basis. (The PPO-implementation-details blog is the companion practice reference for PPO's 37 implementation details.)

Where we use it: [[../../decisions/0002-action-space]] (mask over fixed canonical space); Milestone-5 training stack choice.

Related: [[gym-microrts]], [[microrts-py]], [[vcmi-gym]]
