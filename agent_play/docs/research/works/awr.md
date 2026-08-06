---
title: "Advantage-Weighted Regression (2019)"
type: reference
source: https://arxiv.org/abs/1910.00177
read: 2026-08-06
runs: []
tags: [reference, offline-rl, weighted-regression, support]
local: ["files/arxiv-1910.00177.pdf"]
---

# Advantage-Weighted Regression (2019)

Peng, Kumar, Zhang and Levine reduce policy improvement to supervised regression on the data's own actions: maximum likelihood where each taken action is weighted by the exponentiated advantage, $\exp(\hat A / \beta)$, derived as expectation-maximization on the RL objective. Nothing is ever queried off support, because the only actions in the loss are the ones the data took; the estimator's job shrinks to ranking observed decisions, not imagined ones.

Where we use it: the recommended first experiment of [[../../rl/off-support-and-offline-improvement]]. Our pipeline is already supervised regression on taken actions with per-sample weights, so AWR is one weighting away: the 0.856 bestiary value ([[../../rl/value-estimation-lab]]) supplies $\hat A = G - V_\phi(o)$ per decision, and the arm distills the champion corpus with those weights against the plain recipe, battery-gated. [[alphastar-unplugged]]'s one-step recipe is this idea at scale.

Related: [[iql]], [[one-step-offline-rl]], [[alphastar-unplugged]], [[gae]]
