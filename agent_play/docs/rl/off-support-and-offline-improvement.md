---
title: "The off-support problem, and how offline improvement gets around it"
type: study
updated: 2026-08-06
related_concepts: ["[[value-estimation-lab]]", "[[rl-methods]]", "[[../research/works/bcq-extrapolation]]", "[[../research/works/one-step-offline-rl]]"]
tags: [agent-env, rl, offline-rl, study]
---

# The off-support problem, and how offline improvement gets around it

The owner asked for the clean version of one claim: that a value estimator failing off its data is the known central problem of offline reinforcement learning, with a literature of remedies. This page states the problem once, sorts the remedies into four families with one vendored primary each, and ends with which of them this project should try and in what order. Every source cited here is local under `../research/works/`, and every failure named as ours is measured in [[value-estimation-lab]].

## The problem, stated once

Improvement means preferring an action the data did not take, and the estimator is the only witness for what that action would have earned. Off its support the estimator is guessing, function approximation makes the guesses smooth and confident, and any maximization then selects for the largest guessing errors rather than the best actions. [[../research/works/bcq-extrapolation]] named this extrapolation error; our lab produced the cleanest miniature of it we could ask for, a behavior $Q$ reading 0.853 explained variance on the actions it saw whose top-five re-ranking collapses play from 0.512 to 0.263, one action off support.

The problem is structural rather than statistical: more data on the same distribution does not fix it, because the queries that matter are off that distribution by definition. Every offline method is therefore some answer to one question, where does the counterfactual come from.

## The four families

**Constrain to support.** Never prefer an action the data cannot vouch for. [[../research/works/bcq-extrapolation]] filters candidates through a generative model of the data; [[../research/works/td3-bc]] shows a plain cloning term inside the policy loss buys most of the same protection at a fraction of the machinery; [[../research/works/awr]] goes furthest and puts only taken actions in the loss, weighted by $\exp(\hat A/\beta)$, so improvement is a re-weighting of what happened rather than a bet on what did not. [[../research/works/one-step-offline-rl]] and [[../research/works/alphastar-unplugged]] add the iteration discipline: one improvement step against the behavior value, because a second step queries the value where the first step moved the policy.

**Be pessimistic where data is absent.** Let the estimator answer off-support queries, but bias it low there so maximization retreats to support. [[../research/works/cql]] builds the lower bound explicitly with a regularizer; improvement against a lower bound cannot be exploitation. The cost is a tuning burden and, at small scale, the risk of drowning the signal in the penalty.

**Never ask the question.** Restructure the algorithm so no unseen action is ever evaluated. [[../research/works/iql]] fits values by expectile regression over observed pairs only, the upper expectile standing in for the best supported action without naming it, then extracts the policy by advantage-weighted regression. The off-support query is not answered carefully, it is absent.

**Measure the uncertainty.** Train an ensemble and act on its clipped minimum; members agree on-support and disagree off it, so the penalty appears exactly where data is missing. [[../research/works/edac-ensembles]] shows this alone can beat the explicit penalties. Our unseeded-lottery incident, two same-recipe value fits disagreeing completely off-distribution, is this signal encountered by accident; an ensemble farms it on purpose.

## A fifth answer, and why it is ours

Search manufactures the counterfactual instead of estimating it: a rollout per candidate is a real measurement of an action the dataset never took, which is why rollout-scored search improves play here while every fitted estimator so far has not. Its per-candidate values are also the one dataset on which a learned evaluator could be trained without the off-support problem, since every labeled action was actually played, and the prior-anchored soft target of [[../research/works/mcts-regularized-policy-optimization]], $\bar\pi(a) \propto \pi_{\theta}(a) \exp(Q_{\text{search}}(a)/\lambda)$, keeps the distillation on support by construction. The collector records these values as of 2026-08-06 (`search_values` on decision records), and the owner proposed the softmax form independently before the paper was consulted.

## What this project runs, in order

First the advantage-weighted arm, because it is one weighting away from the existing supervised pipeline: the 0.856 bestiary value supplies $\hat A = G - V_\phi(o)$ per teacher decision, taken actions only, one step, battery-gated against the plain recipe.

That arm has now run, and the verdict is negative with a structural reason worth more than the number. At $\beta = 1$ the weighted arm loses to its paired unweighted twin nearly everywhere, held-out 0.558 against 0.619 and the Thunk ladder 0.677 against 0.854 with the top rung at 0.12 against 0.54, one seed, `awr_distill.json` and `battery_awr.json` in the archive.

The mechanism AWR needs is missing from this data. Advantage weighting improves by preferring better actions at the same state, which requires the behavior to have tried different actions there, and our demonstrator is a deterministic planner that takes exactly one action per state. With no within-state diversity to select over, the weights can only reweight whole episodes toward surprising outcomes, which is a mixture perturbation, and the ladder's collapse is the same mixture fragility every reweighting arm of the night block showed. The lesson transfers: one-step methods presuppose coverage not just of states but of alternatives, [[../research/works/one-step-offline-rl]]'s setting has stochastic behavior for exactly this reason, and the soft-target route is unaffected because search values genuine alternatives itself.

Second the soft-target distillation from search values, the fifth answer above, piloted on a fresh both-sides collection. Pessimism and ensembles wait until either support-side arm shows the estimator is the binding constraint rather than the data, and IQL's expectile machinery is recorded as the fallback if bootstrapped values are ever needed, since it is the variant that never queries off data. Each arm reports under the two-families contract of [[rl-methods#Two families, and the rule against mixing them]], and every claim of improvement carries the built-in AI's column beside it.
