---
title: "UCT: Bandit based Monte-Carlo Planning (ECML 2006)"
type: reference
source: http://ggp.stanford.edu/readings/uct.pdf
read: 2026-08-05
runs: []
tags: [reference, mcts, ucb, planning]
local: ["files/uct-kocsis-szepesvari-2006.pdf"]
---

# UCT: Bandit based Monte-Carlo Planning (ECML 2006)

Kocsis and Szepesvári's paper is the origin of the popular UCB variant of Monte-Carlo tree search: treat action selection at each internal node as a bandit and pick the child maximizing $\bar{X}_a + C \sqrt{\ln N / n_a}$, the UCB1 rule applied recursively, with rollouts supplying the leaf values. The two results that matter here: the probability of selecting a suboptimal action at the root vanishes as simulations grow, and the value estimates converge, so search is an anytime improvement operator whose quality scales with budget.

The modern descendant is PUCT as used by AlphaZero ([[alphazero]]), which replaces the log bonus with a prior-weighted one, $Q(a) + c \, P(a) \sqrt{N} / (1 + n_a)$, so a learned policy focuses the search. That is the form `python/fheroes2_agent/search.py` implements at the root, with rollout returns for $Q$ and the clone as $P$.

Where we use it: the search-as-improvement-operator entry in [[../../rl/rl-methods#Search as an improvement operator]] and the 2026-08-05 search probe, which measures whether root-PUCT lifts play on the matchups the policy loses.

Related: [[alphazero]], [[stratega]], [[alphastar-unplugged]]
