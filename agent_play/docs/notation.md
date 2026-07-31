---
title: Notation
type: reference
updated: 2026-07-30
related_concepts: ["[[rl-and-the-battle-domain]]", "[[rl-methods]]", "[[training-design]]", "[[rlhf-transfer]]"]
tags: [agent-env, rl, notation, reference, entry-point]
---

# Notation

This page fixes the symbols used everywhere else in this documentation. It is self-contained: nothing here requires a file outside this repository, and no page in this tree depends on one.

The symbols are not a house style. They match the owner's existing reinforcement-learning study notes, so that material written here reads as a continuation of those rather than as a second vocabulary held alongside the first. Where a note is the origin of a convention, it is cited by its identifier, as `rl-014` for the policy gradient theorem. Those identifiers are provenance only. Every definition needed to read this tree is restated here or in [[rl-methods]].

Readers who want the domain rather than the symbols should start at [[rl-and-the-battle-domain]] and treat this page as a lookup.

## The core symbols

| Symbol | Meaning | Defined in |
|---|---|---|
| $\mathcal{S}$, $\mathcal{A}$, $\mathcal{A}(s)$ | State space, action space, and the legal actions at $s$ | [[rl-and-the-battle-domain]] |
| $P(s' \mid s, a)$, $R(s, a, s')$ | Transition function and reward | [[rl-and-the-battle-domain]] |
| $\gamma$, $\lambda$ | Discount factor, and the trace parameter in generalized advantage estimation | [[rl-methods]] |
| $\tau$ | Trajectory, one episode of $s_0, a_0, r_1, s_1, \ldots$ | [[rl-methods]] |
| $G_t = \sum_{k \ge 0} \gamma^k r_{t+k+1}$ | Discounted return from step $t$ | [[rl-methods]] |
| $\pi(a \mid s)$, $\pi_\theta(a \mid s)$ | Policy, and the same policy parameterized by $\theta$ | [[rl-methods]] |
| $\pi^{*}$ | The teacher policy, here `AI::BattlePlanner` | [[training-design]] |
| $V^\pi(s)$, $Q^\pi(s, a)$ | State value and action value | [[rl-methods]] |
| $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ | Advantage | [[rl-methods]] |
| $V_\phi$ | Learned critic, parameters $\phi$ distinct from the actor's $\theta$ | [[training-design]] |
| $b(s)$ | Baseline | [[rl-methods]] |
| $d^\pi(s)$ | State distribution induced by $\pi$ | [[rl-methods]] |
| $J(\theta)$ | The objective being maximized | [[rl-methods]] |
| $\Psi_t$ | The scoring term in the shared policy-gradient shape | [[rl-methods]] |
| $\delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ | Temporal-difference residual | [[rl-methods]] |
| $\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$ | Generalized advantage estimate | [[rl-methods]] |
| $\rho_t(\theta) = \pi_\theta(a_t \mid o_t) / \pi_{\theta_{\text{old}}}(a_t \mid o_t)$ | Importance ratio | [[rl-methods]] |
| $\varepsilon$ | PPO clipping half-width | [[rl-methods]] |
| $\epsilon$ | Per-decision error rate of a fitted policy, in the DAgger bound | [[training-design]] |
| $\log$ | Logarithm, spelled this way throughout | convention |

## Symbols this project adds

The study notes are written for reinforcement learning in general and, more recently, for language models. Each addition below exists because this project is a game environment rather than either.

| Symbol | Meaning | Why the notes have no equivalent |
|---|---|---|
| $o \in \Omega$ | Observation, what the agent actually receives | their standard setting is fully observed |
| $O(o \mid s)$ | Observation function, in general stochastic | same |
| $\pi_\theta(a \mid o)$ | Observation-conditioned policy, the object actually trained here | same |
| $m(o) \in \{0,1\}^{793}$ | Legality mask, with $m_i = 1$ exactly when action $i$ is legal | action masking appears nowhere in the notes |
| $\ell_\theta(o) \in \mathbb{R}^{793}$ | Policy-head logits before masking | same |
| $\rho_0$ | Initial-state distribution, here the scenario and army generator | episodic start distributions do not arise in the bandit framing of language-model RL |
| $\Phi(s)$ | Potential function in potential-based shaping | shaping is discussed without a fixed symbol |
| $\mathcal{D}$ | Dataset of observation and action pairs | the demonstrator is left unnamed |
| $T$ | Episode length in decisions, 5 to 40 here | horizon rarely appears explicitly |
| $c_v$, $c_e$ | Value and entropy coefficients in the combined loss | the losses are given separately |

## Two symbols worth pausing on

The importance ratio is $\rho_t(\theta)$ and the clip half-width is $\varepsilon$. Both the RLHF book and the owner's most recent notes on it use these. The older reinforcement-learning cards write the ratio $r_t(\theta)$, following the PPO paper, and use $\epsilon$ for the clip. This tree takes $\rho_t$ and $\varepsilon$ for two reasons: they match the newest of the two sources, and they leave $r$ meaning a reward and nothing else, which matters in a document that writes $r_{t+1}$ constantly.

That frees $\epsilon$ for the per-decision imitation error rate, which is what card `rl-036` already calls it. The two are never used in the same equation.

The discount range is relaxed. Most treatments require $\gamma \in [0, 1)$ so infinite-horizon returns converge. Battles terminate within 5 to 40 decisions, so $\gamma = 1$ is admissible and this tree uses $\gamma \in [0, 1]$. The relaxation is safe only because termination is guaranteed, and it would not carry to the adventure-map problem scoped in [[roadmap]].

## Where this tree sits against the existing notes

The study notes already cover most of the machinery, which is why [[rl-methods]] states results and gives the load-bearing derivation rather than teaching from scratch. Rows marked absent are where this tree carries material the notes do not, and they are the pages worth reading on their own account.

| Topic | Covered in the notes | Here |
|---|---|---|
| MDP objects, Bellman equations, dynamic programming | `rl-001` to `rl-006`, `rl-048` | instantiated for a battle in [[rl-and-the-battle-domain]] |
| Monte Carlo, temporal difference, TD($\lambda$) | `rl-007` to `rl-012`, `rl-201` | the bias-variance contrast, used to motivate GAE |
| Policy gradient theorem, REINFORCE, natural gradient | `rl-013`, `rl-014`, `rl-016`, `rl-047` | trajectory form and the baseline derivation |
| Generalized advantage estimation | `rl-015`, `rl-054`, `rl-210` | recap, plus the backward recursion and the choice of $\lambda$ for a 5 to 40 step horizon |
| Actor-critic, A2C, V-trace, deterministic gradients | `rl-017` to `rl-019`, `rl-049` | recap |
| Trust regions, TRPO, PPO | `rl-029`, `rl-030`, `rl-060`, `rl-203` | recap, plus what masking does to the ratio, which is new |
| Value-based methods, DQN family, distributional RL | `rl-026` to `rl-028`, `rl-046`, `rl-056` | why they are not the first path for this problem |
| Function approximation, deadly triad, fitted Q | `rl-020` to `rl-022` | assumed |
| Exploration | `rl-023` to `rl-025`, `rl-050`, `rl-200` | surveyed and set aside |
| Imitation, behavior cloning, DAgger, inverse RL | `rl-036`, `rl-037` | [[training-design]] gives the architecture, masked loss, mixing schedule, and hyperparameters for this teacher |
| Offline RL | `rl-033` to `rl-035`, `rl-059` | surveyed and set aside |
| Planning, MCTS, model-based RL | `rl-051`, `rl-052`, `rl-057`, `rl-204` | why the door is kept open but not walked through |
| Policy-gradient family shape, RLOO, GRPO, loss aggregation | the RLHF math companion, ch. 6a to 6d | [[rlhf-transfer]] works out what applies to battles |
| Legal-action masking | absent | [[implementation/legal-actions-and-masking]], the main new RL content here |
| Elo and TrueSkill for ranking agents | absent | [[rl-and-the-battle-domain]] Part 3 |
| Asymmetric actor-critic, privileged critics | absent | [[implementation/observation-design]], including the bias result |
| Truncation against termination bootstrapping | absent | [[rl-methods]], and it constrains the Milestone 4 protocol |
| Episode-length normalization bias | absent for episodes | [[rlhf-transfer]] derives it from the token-length version |
| Game-environment engineering | absent | all of [[implementation/README]], the bulk of this repository |
| The fheroes2 battle domain | absent | [[rl-and-the-battle-domain]] Part 2 |

## A note on other textbook notation

Zhao's *Mathematical Foundations of Reinforcement Learning* is cited in the owner's notes for its treatment of the Bellman operator as a $\gamma$-contraction, and it uses a different and internally consistent set of symbols. This table translates, so a chapter can be read against this tree without re-deriving anything.

| Here | Zhao |
|---|---|
| $V^\pi(s)$, $Q^\pi(s, a)$ | $v_\pi(s)$, $q_\pi(s, a)$ |
| $A^\pi(s, a)$ | $\delta_\pi(s, a)$, deliberately sharing a letter with the TD error that estimates it |
| $\pi_\theta(a \mid s)$ | $\pi(a \mid s, \theta)$ |
| $P(s' \mid s, a)$, $R(s, a, s')$ | $p(s' \mid s, a)$, $p(r \mid s, a)$ |
| $\rho_0$ | $d_0$ |
| $d^\pi(s)$ | $d_\pi(s)$ stationary, $\eta(s)$ in the policy gradient theorem |
| $J(\theta)$ for an episodic task | $\bar v_\pi^{\,0}$, one of three metrics it distinguishes |
| $V_\phi(s)$ | $v(s, w)$ |
| $\log$ | $\ln$ |

One of its observations is worth borrowing without its notation. It derives the variance-minimizing baseline as $Q^\pi$ weighted by $\lVert \nabla_\theta \log \pi_\theta \rVert^2$ rather than $V^\pi$, which is why [[rl-methods]] calls the state value the practical choice rather than the optimal one.

## Related

- [[rl-and-the-battle-domain]], the vocabulary applied to a battle, and the comparison with other game environments.
- [[rl-methods]], every technique this documentation names, with a verdict on each.
- [[rlhf-transfer]], what the language-model reinforcement-learning literature contributes here.
- [[training-design]], the architecture, losses, and hyperparameters that use these symbols concretely.
