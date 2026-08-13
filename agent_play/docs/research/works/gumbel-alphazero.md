---
title: "Policy improvement by planning with Gumbel (ICLR 2022)"
type: reference
source: https://openreview.net/forum?id=bERaNdoegnO
read: 2026-08-11
runs: []
tags: [reference, mcts, bandit, simple-regret, distillation, policy-target]
local: []
---

# Policy improvement by planning with Gumbel (ICLR 2022)

Danihelka, Guez, Schrittwieser and Silver rebuild the parts of AlphaZero that are heuristic, and every one of the four things they replace is something this project either does or was about to try. The paper's own framing is that AlphaZero's mechanisms are tuned for large simulation budgets and "perform poorly" when the budget is small relative to the action count, which is the regime here.

## Why UCB is the wrong bandit at the root

This is the result that bears hardest. PUCB was designed to minimise *cumulative* regret, which is the right objective when a node's value feeds a parent. The root has no parent: "no ancestors are dependent upon the evaluation of the root node, and the performance of the Monte-Carlo tree search therefore only depends upon the final recommended action at the root node, and not upon the intermediate actions selected during search". So the root is a *simple* regret problem, and the paper cites Bubeck et al. (2011), Hay and Russell (2011) and Tolpin and Shimony (2012) as having argued exactly that. Their replacement at the root is Sequential Halving (Karnin et al. 2013), chosen partly because it "does not have problem-dependent hyperparameters" and was easier to tune than UCB-E and UCB$\sqrt{\cdot}$.

Our search is a single ply, so the root *is* the whole search. That makes the mismatch total rather than partial: there is no subtree whose estimates would benefit from cumulative-regret behaviour, and the pure-exploration bandit literature applies directly with none of the tree complications the paper still has to handle.

## The counterexample against acting on the top-$n$ prior actions

Section 3.2 is three lines and worth carrying whole. Take $q = (0, 0, 1)$ and $\pi = (0.5, 0.3, 0.2)$. The policy's own value is $\sum_a \pi(a) q(a) = 0.2$. With two simulations spent on the two most probable actions $\{0, 1\}$, the search returns $\arg\max_{a \in \{0,1\}} q(a) = 0$, worth $0$. Searching made the agent *worse than not searching*. The fix is to sample the candidate set without replacement using the Gumbel-Top-$k$ trick and reuse the same Gumbel vector when picking the winner, which restores a policy-improvement guarantee whenever the action values are correctly evaluated.

## Visit counts are not a sound policy target at small budgets

The paper is explicit that AlphaZero's visit-count target "does not guarantee a policy improvement, especially when using small numbers of simulations", citing Grill et al. (2020). This matters directly, because the visit-count target is the obvious thing to reach for once a value-derived target is measured to be degenerate, and the literature says it is unsound in precisely the regime that would motivate reaching for it.

What replaces it is a completed-Q target. Unvisited actions do not keep whatever estimate they happen to carry; they are assigned the policy's own value:

$$\mathrm{completedQ}(a) = \begin{cases} q(a) & \text{if } N(a) > 0 \\ v_\pi = \sum_a \pi(a) q(a) & \text{otherwise} \end{cases}$$

and the improved policy is $\pi' = \operatorname{softmax}\bigl(\text{logits} + \sigma(\mathrm{completedQ})\bigr)$, distilled by $L(\pi) = D_{\mathrm{KL}}(\pi' \,\|\, \pi)$, which "trains all actions, not only the action $A_{n+1}$". Their $\sigma$ is not a fixed temperature but $\sigma(\hat q(a)) = (c_{\text{visit}} + \max_b N(b)) \, c_{\text{scale}} \, \hat q(a)$, so the weight on the measured values grows with how much the search actually looked, at $c_{\text{visit}} = 50$ and $c_{\text{scale}} = 1.0$.

## How it lines up with what is measured here

Three of our own measurements become predictions of this paper rather than surprises.

Coverage forcing spends the first rollout on every candidate and then hands the remainder to UCB, and it measured negative at every budget, losing more as the budget grew. Read against Sequential Halving that is the first phase of the right algorithm with the rest of it missing: it pays the full breadth cost, 26 of 48 rollouts on this corpus, and then abandons the schedule that would have made the breadth pay by progressively rejecting halves.

The soft-target family measured degenerate, $\bar\pi \propto \text{prior} \cdot \exp(Q/\lambda)$ carrying a median 97 percent of its mass on one action. That functional form is the same family as the paper's $\pi'$, which suggests the form was never the problem. What differs is that our version leaves poorly-visited actions at whatever single rollout gave them, and completed Q-values exist precisely to stop that.

Two thirds of the time the highest-$Q$ action on our corpus had been visited once or less, on 2.47 rollouts per candidate. Completion is the paper's answer to exactly that failure: an action nobody looked at gets $v_\pi$ and therefore "zero advantage", rather than a lucky sample that a soft target would then chase.

## What this does not settle

The paper's search has a learned value network at the leaves; ours rolls out the policy to termination, so our $q(a)$ is an unbiased but high-variance Monte Carlo return rather than a network estimate, and the variance is what the completion is being asked to absorb. Whether completion helps as much when the noise is sampling noise rather than approximation error is not something the paper answers.

Their budgets are also spent on a tree. At one ply, Sequential Halving's phases and our rollout cost interact differently, and the halving schedule assumes candidates can be compared on equal visit counts, which coverage forcing already approximates badly.

The Sequential Halving root has since been built and measured: `_sequential_halving` in `python/fheroes2_agent/search.py` behind `--allocator`, swept against PUCT by `agent_play/experiments/allocator_scaling.py`, where a one-experiment advantage at thirty-two playouts did not survive its own budget sweep and PUCT stays the default; what replicates is the mechanism, roughly triple the visit entropy at every budget. The completed-Q target remains unbuilt. So the paper's root-bandit half is measured here and its policy-target half is not, and the budget log carries the sweep.

<!-- verify
grep    python/fheroes2_agent/search.py :: _sequential_halving
exists  agent_play/experiments/allocator_scaling.py
-->
