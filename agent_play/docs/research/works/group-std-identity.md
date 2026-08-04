---
title: "GRPO, Dr. GRPO, and DAPO Are Three Operations on One Number: The Group-Standard-Deviation Identity"
type: paper
authors: Bay, Yearick
year: 2026
arxiv: "2607.00152"
quality: primary
urls:
  - https://arxiv.org/abs/2607.00152
tags: [reference, grpo, advantage-normalization, difficulty-bias, group-baseline]
---

# The group-standard-deviation identity

Read on 2026-08-04, found while checking a result that looked impossible: two arms of a five-way comparison, leave-one-out and Dr. GRPO under the divergence trust region, had produced runs identical to three decimals in every statistic. The identity explaining that turned out to be this paper's Proposition 1, published five weeks earlier.

## The claim

For binary verifier rewards, the per-prompt GRPO update factors exactly into a direction and a scalar (their Theorem 1):

$$g = \sigma\,(\bar s_+ - \bar s_-), \qquad \sigma = \frac{\sqrt{k(G-k)}}{G}$$

where $k$ of $G$ sampled responses are correct and $\bar s_\pm$ are the mean score directions of the correct and incorrect sets. The scalar is the group's reward standard deviation, so the same number GRPO divides by is the size of the learning signal the prompt produced. The three methods are then one dial: GRPO divides the advantage by $\sigma$, Dr. GRPO removes the division, and DAPO discards the $\sigma = 0$ groups.

## Proposition 1, which this project re-derived empirically

Group-mean centering equals the leave-one-out baseline up to a constant: for any rewards with $G \ge 2$,

$$R_i - \mu = \frac{G-1}{G}\,\big(R_i - b_i\big), \qquad b_i = \tfrac{1}{G-1}\textstyle\sum_{j \ne i} R_j$$

so Dr. GRPO's advantage and the leave-one-out advantage differ only by $G/(G-1)$, which does not depend on the sample. They attribute the absorption of that constant to the learning rate. In this codebase the absorption is exact rather than approximate, because `normalize_advantages` divides the batch by its own spread, which carries the same constant. The two modes therefore issue bit-identical updates whenever the advantage floor does not bind, which is what the identical runs were. Four unit tests in `python/tests/test_objectives.py` pin this, including that a binding floor is the one thing separating the modes and that studentization survives normalization because it divides each group by a different number.

## The silent-group rate, and a filter this project already had

A group is silent, carrying zero advantage everywhere, with probability $p^G + (1-p)^G$, and DAPO's dynamic sampling is exactly the operation of discarding that mass. `train_group` drops groups whose returns are exactly equal, implemented here before reading the paper, so the project's degenerate-group filter is DAPO's filter under another name. The observed drop rate on the 140-matchup pool was 9 to 10 percent of groups.

## What does not transfer, and it is the interesting part

Their setting has binary rewards, which bounds the dial. With $G = 8$, a non-silent group's $\sigma$ lies between $\sqrt{7}/8 \approx 0.33$ and $0.5$, so GRPO's division is at most a modest reweighting across prompt difficulties, and the argument against it is about bias, the question-level difficulty bias of Liu et al. (2025), not about magnitude.

This project's reward is continuous, the margin-weighted terminal reward, so a group can disagree by arbitrarily little without being silent, and $1/\sigma$ is unbounded. The same division that is a bounded difficulty reweighting under binary rewards is unbounded noise amplification under continuous ones. That is the measured $+0.068 \pm 0.021$ cost of studentization on the contested matchup, and it is the same failure the batch-level advantage floor closes one level down. The exact-zero drop that protects a binary-reward trainer does not protect a continuous-reward one, because the dangerous groups are the nearly-silent, not the silent.

## Verdict for this project

Cite Proposition 1 for the leave-one-out equivalence rather than presenting it as new, treat `loo` and `drgrpo` as one arm wherever the floor does not bind, and keep both the silent-group drop and the floor, which handle the $\sigma = 0$ and $\sigma \to 0$ ends of the same hazard.

## Related

- [[../../rl/rlhf-transfer#Critic-free baselines, which this project should seriously consider]], where the estimators are laid out.
- [[../../archive/experiments/2026-08-03-training-runs]], the runs that hit the identity empirically.
- [[dppo-trust-region]], the other 2026 result tested here, on the trust-region axis of the same trainer.
