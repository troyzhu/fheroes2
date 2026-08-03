---
title: "Rethinking the Trust Region in LLM Reinforcement Learning (DPPO)"
type: paper
authors: Qi, Zhou, Liu, Pang, Du, Lin, Lee
year: 2026 (ICML, PMLR 306)
arxiv: "2602.04879"
quality: primary
urls:
  - https://arxiv.org/abs/2602.04879
  - https://github.com/sail-sg/Stable-RL
tags: [reference, ppo, trust-region, importance-sampling, divergence]
---

# DPPO, the ratio as a one-sample estimate of divergence

Sea AI Lab and NUS, the group behind Dr. GRPO. Read on 2026-07-30 after the owner flagged it as an alternative perspective on the importance ratio. It is, and the perspective survives translation to this project.

## The claim

PPO's clipping condition constrains a noisy single-sample estimate of policy divergence rather than the divergence itself. The identity making that precise is their equation (11):

$$D_{\mathrm{TV}}\big(\mu(\cdot \mid s_t) \,\|\, \pi(\cdot \mid s_t)\big) = \tfrac{1}{2}\,\mathbb{E}_{a_t \sim \mu}\big[\lvert \rho_t - 1 \rvert\big]$$

So $\lvert \rho_t - 1 \rvert \le \varepsilon$ is a one-sample Monte Carlo estimate of $2 D_{\mathrm{TV}}$, evaluated at whichever action happened to be drawn. TRPO constrained the divergence; PPO constrains a single draw from the random variable whose mean is that divergence.

## Why that estimate is biased in a way that matters

The ratio depends on the behavior probability, so equal ratios correspond to wildly unequal amounts of moved probability mass. Their worked example, at a fixed state with two actions:

| | $\mu(a \mid s)$ | $\pi(a \mid s)$ | $\rho$ | mass moved |
|---|---|---|---|---|
| low-probability action | $10^{-4}$ | $10^{-2}$ | 100 | about $10^{-2}$ |
| high-probability action | $0.99$ | $0.80$ | 0.808 | $0.19$ |

The first is clipped hard at any sane $\varepsilon$ while contributing almost nothing to the divergence. The second sits inside a clip range of $\varepsilon = 0.2$ while moving nineteen times more mass. PPO therefore over-penalizes low-probability actions, which slows learning and specifically suppresses the exploratory ones, and under-penalizes high-probability actions, which permits the updates that actually destabilize training.

They note Clip-Higher and CISPO identify the same symptom and treat it heuristically, by raising the upper clip bound or by keeping the gradient of clipped tokens, without addressing the ratio-against-divergence mismatch itself.

## What they propose

Divergence proximal policy optimization keeps PPO's asymmetric masking structure but conditions the mask on a real divergence rather than on the sampled ratio. Writing $D = D(\mu(\cdot \mid s_t) \,\|\, \pi(\cdot \mid s_t))$ and $\delta$ for a threshold, their equation (12) is

$$M_t^{\mathrm{DPPO}} = \begin{cases} 0 & \text{if } (\hat A_t > 0 \text{ and } \rho_t > 1 \text{ and } D > \delta) \text{ or } (\hat A_t < 0 \text{ and } \rho_t < 1 \text{ and } D > \delta) \\ 1 & \text{otherwise} \end{cases}$$

with the objective $L^{\mathrm{DPPO}}_\mu(\pi) = \mathbb{E}\big[\sum_t M_t^{\mathrm{DPPO}} \rho_t \hat A_t\big]$. Substituting $D = \lvert \rho_t - 1 \rvert$ recovers PPO exactly, which makes the two directly comparable. The mask blocks an update only when it is already moving away from the trust region, so updates pulling the ratio back toward one are never blocked.

Their theory adapts the policy improvement bound to the finite-horizon undiscounted setting, since the usual $\tfrac{1}{1-\gamma}$ factor diverges at $\gamma = 1$. Theorem 3.2 bounds $\mathcal{J}(\pi) - \mathcal{J}(\mu)$ below by the surrogate minus a term in $D_{\mathrm{TV}}^{\max}$, with a tighter average-divergence form. Pinsker's inequality, $D_{\mathrm{TV}}^2 \le \tfrac12 D_{\mathrm{KL}}$, licenses using KL instead.

## The cost, which is their problem and not ours

Computing $D$ exactly means summing over the full action distribution at every state. Over a vocabulary of $10^5$ tokens that is memory-prohibitive, so most of their methodology section builds approximations: a binary collapse to sampled-against-rest, and a top-$K$ reduction. Both are lower bounds on the true divergence.

This project's action space is 793 slots with typically 5 to 30 legal after masking. The exact divergence over the legal set is a handful of floating-point operations, so the approximations are unnecessary here and the exact form is available. That inverts the usual direction of transfer, where a language-model technique arrives with costs a small problem cannot pay.

## Verdict for this project

Worth testing, with the magnitude honestly uncertain. See [[../../rl/rlhf-transfer#The ratio is a one-sample estimate of a divergence you can afford to compute]].

The pathology is driven by the dynamic range of the behavior probabilities, which in a language model spans $10^{-5}$ to $0.99$. A masked categorical over 5 to 30 legal actions, initialized from behavior cloning, has a far narrower range, so the effect should be milder here than the paper measures. What makes it worth trying anyway is that the remedy costs almost nothing at this action-space size, and that the failure it prevents is the one this project is most exposed to, namely a destabilizing early update from a cloned policy that is already confident.

## What does not transfer

Their motivating instability is training-inference mismatch, where the sampling engine and the trainer produce different token probabilities from identical parameters. This project runs one process with one code path and a seeded generator, so that source of mismatch does not exist. The argument above stands on the ratio-against-divergence mismatch alone, which is independent of it.

## Related

- [[../../rl/rl-methods#Why the step must be constrained]], where the clip is derived.
- [[../../rl/rlhf-transfer]], which works out what applies to battles.
- [[invalid-action-masking]], the masking result this would compose with.
