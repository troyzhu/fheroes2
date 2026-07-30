---
title: "Asymmetric / privileged-critic RL under partial observability (three papers)"
type: paper-group
year: 2017-2022
quality: primary
urls:
  - https://arxiv.org/abs/2105.11674
  - https://arxiv.org/abs/1710.06542
  - https://arxiv.org/abs/2110.05038
runs: [rl-approaches]
tags: [reference, partial-observability, asymmetric-critic, pomdp]
local: ["files/arxiv-2105.11674.pdf", "files/arxiv-1710.06542.pdf", "files/arxiv-2110.05038.pdf"]
---

# Asymmetric actor-critic / POMDP baselines (paper group)

The literature behind our `full_v1`-as-oracle option:

- Unbiased Asymmetric Reinforcement Learning under Partial Observability (Baisero & Amato, AAMAS 2022, arXiv:2105.11674), shows the common "critic sees full state, actor sees observations" variant is theoretically biased and derives an unbiased history-state critic.
- Asymmetric Actor Critic for Image-Based Robot Learning (Pinto et al., 2017, arXiv:1710.06542), the original privileged-critic training setup.
- Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs (Ni et al., 2021, arXiv:2110.05038), recurrence as the default POMDP treatment before reaching for belief states.

Verification status, important: these were fetched and are on-topic, but no claim about asymmetric/oracle-critic setups survived to verification in either run (an explicitly marked gap). They justify *keeping the full-state profile available* (near-zero cost), not any performance expectation. Creature-only HoMM2 battles are near-fully observable anyway (hidden info ≈ RNG, not fog), so POMDP machinery may simply be unnecessary until hero/fog scope.

Where we use it: [[../decisions/0001-observation-profiles]] context; open question 1 of [[../research_rl_approaches]].

Related: [[ref-microrts-py]] (partial_obs flag), [[ref-alphastar]] (LSTM core)
