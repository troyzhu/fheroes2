---
title: Notation, and how this tree extends the RL wiki
type: reference
updated: 2026-07-30
related_concepts: ["[[rl-and-the-battle-domain]]", "[[rl-methods]]", "[[training-design]]"]
tags: [agent-env, rl, notation, reference, entry-point]
---

# Notation, and how this tree extends the RL wiki

This page fixes the symbols used everywhere else in this documentation, and fixes them against an external source rather than inventing a house style. That source is the repository owner's personal reinforcement-learning wiki, an Obsidian vault under Dropbox at `Papers/wiki`, which carries roughly six thousand concept notes and already defines nearly every technique this project uses. Note names below are given in code font, as `concepts/policy-gradient-theorem` and similar. They are deliberately not wikilinks, because that vault is a separate Obsidian vault and a link across the boundary would not resolve.

Three commitments follow. Symbols match the wiki, so nothing has to be mentally translated when moving between the two. Concepts the wiki already covers are named and pointed at rather than re-derived, which is what [[#Section 3, coverage against the wiki]] is for. And the handful of things the wiki does not cover are called out explicitly, because those are the parts of this tree that carry new material rather than recap.

Readers who want the domain rather than the symbols should start at [[rl-and-the-battle-domain]] and treat this page as a lookup.

## Section 1, the shared symbols

These carry the same meaning here as in the wiki. The last column names the note that defines the concept, so an unfamiliar symbol can be chased to a full treatment instead of a one-line gloss.

| Symbol | Meaning | Defined in the wiki at |
|---|---|---|
| $\mathcal{S}$, $\mathcal{A}$, $\mathcal{A}(s)$ | State space, action space, and the legal actions at $s$ | `concepts/markov-decision-process` |
| $P(s' \mid s, a)$ | Transition function | `concepts/markov-decision-process` |
| $R(s, a, s')$ | Reward on one transition | `concepts/markov-decision-process` |
| $\gamma$ | Discount factor | `concepts/markov-decision-process` |
| $\tau$ | Trajectory, one episode of $s_0, a_0, r_1, s_1, \ldots$ | `concepts/policy-gradient-theorem` |
| $G_t = \sum_{k \ge 0} \gamma^k r_{t+k+1}$ | Discounted return from step $t$ | `concepts/returns-rl` |
| $\pi(a \mid s)$, $\pi_\theta(a \mid s)$ | Policy, and the same policy parameterized by $\theta$ | `concepts/policy-gradient` |
| $V^\pi(s)$ | State value | `concepts/value-function` |
| $Q^\pi(s, a)$ | Action value | `concepts/q-function` |
| $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ | Advantage | `concepts/advantage-function` |
| $V_\phi$ | Learned critic, parameters $\phi$ distinct from the actor's $\theta$ | `concepts/actor-critic` |
| $b(s)$ | Baseline | `concepts/policy-gradient-theorem` |
| $d^\pi(s)$ | State distribution induced by $\pi$ | `concepts/policy-gradient-theorem` |
| $J(\theta)$ | The objective being maximized | `concepts/policy-gradient-theorem` |
| $\delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ | Temporal-difference residual | `concepts/temporal-difference` |
| $\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$, and $\lambda$ | Generalized advantage estimate and its trace parameter | `concepts/generalized-advantage-estimation` |
| $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ | Probability ratio | `concepts/ppo-clip` |
| $\epsilon$ | PPO clipping half-width | `concepts/ppo-clip` |
| $\log$ | Logarithm, spelled this way throughout | wiki convention |

## Section 2, symbols this project adds

The wiki is written for reinforcement learning in general and for language models in particular. Each addition below exists because this project is a game environment rather than either of those.

| Symbol | Meaning | Why the wiki has no equivalent |
|---|---|---|
| $o \in \Omega$ | Observation, what the agent actually receives | the wiki's standard setting is fully observed |
| $O(o \mid s)$ | Observation function, in general stochastic | same |
| $\pi_\theta(a \mid o)$ | Observation-conditioned policy, the object actually trained here | same |
| $m(o) \in \{0,1\}^{793}$ | Legality mask, with $m_i = 1$ exactly when action $i$ is legal | the wiki has no action-masking note at all |
| $\ell_\theta(o) \in \mathbb{R}^{793}$ | Policy-head logits before masking | same |
| $\rho_0$ | Initial-state distribution, here the scenario and army generator | episodic start distributions do not arise in the wiki's bandit framing of language-model RL |
| $\Phi(s)$ | Potential function in potential-based shaping | `concepts/reward-shaping` discusses it without fixing a symbol |
| $\pi^{*}$ | The teacher policy, here `AI::BattlePlanner` | `concepts/behavior-cloning` leaves the demonstrator unnamed |
| $\mathcal{D}$ | Dataset of observation and action pairs | same |
| $\varepsilon$ | Per-decision error rate of a fitted policy, used in the DAgger bound | same |
| $T$ | Episode length in decisions, 5 to 40 here | horizon rarely appears explicitly in the wiki |
| $c_v$, $c_e$ | Value and entropy coefficients in the combined PPO loss | the wiki gives the losses separately |

## Section 3, coverage against the wiki

What is recap and what is new. Rows naming a wiki note are assumed known, and this tree links back rather than re-deriving them; where it does restate one, that is for a reader of this repository who does not have the wiki, and the wiki note remains the fuller treatment. Rows marked absent carry material the wiki does not have, and they are where this tree is worth reading on its own account.

| Topic | Your wiki | Here |
|---|---|---|
| MDP objects, returns, value and action values | `markov-decision-process`, `value-function`, `q-function`, `returns-rl` | [[rl-and-the-battle-domain]] instantiates each one for a battle |
| Bellman equation, value and policy iteration | `bellman-equation`, `value-iteration`, `policy-iteration` | assumed and not used, since this project is model-free |
| Monte Carlo and temporal-difference estimation | `monte-carlo-rl`, `temporal-difference` | [[rl-methods]] uses the bias-variance contrast when motivating GAE |
| Policy gradient theorem, REINFORCE, baselines | `policy-gradient-theorem`, `reinforce-algorithm` | [[rl-methods]] gives the trajectory form and the baseline derivation |
| Actor-critic, advantage | `actor-critic`, `advantage-function` | recap only |
| GAE | `generalized-advantage-estimation` | recap, plus the $\lambda$ choice for a 5 to 40 step horizon |
| Trust regions, TRPO, PPO clipping | `trust-region-methods`, `trpo`, `ppo-clip` | recap, plus what masking does to the ratio, which is new |
| Importance sampling, off-policy correction | `importance-sampling-rl`, `off-policy-correction` | recap |
| Value-based methods, DQN, QR-DQN | `dqn`, `qr-dqn`, `distributional-rl` | [[rl-methods]] records why they are not the first path for this problem |
| Distributed actor-critic, IMPALA, V-trace | `impala`, `v-trace` | recap, with a note that this environment is too fast to need them yet |
| Planning, MCTS, AlphaZero, MuZero | `monte-carlo-tree-search`, `alphazero`, `muzero` | [[rl-methods]] records why the door is kept open but not walked through |
| Imitation, behavior cloning, DAgger | `behavior-cloning`, `imitation-learning`, `dagger` | [[training-design]] gives the architecture, the masked loss, the mixing schedule, and the hyperparameters for this teacher |
| Reward shaping, potential-based shaping | `reward-shaping`, `sparse-reward`, `reward-hacking` | [[rl-methods]] derives the invariance; [[decisions/0005-training-and-reward]] chooses |
| Entropy regularization | `entropy-regularization` | recap, plus that it must be computed over the legal set only |
| Partial observability | `pomdp-detail`, `recurrent-policy` | [[implementation/observation-design]] applies it to two concrete profiles |
| Self-play, opponent pools, multi-agent | `self-play-rl`, `multi-agent-rl` | recap, plus the speed-queue semi-MDP wrinkle |
| Offline RL, inverse RL, exploration, curricula | `offline-rl`, `inverse-reinforcement-learning`, `exploration-in-rl`, `curriculum-learning` | surveyed and set aside in [[rl-methods]] Part 2 |
| Legal-action masking | absent | [[implementation/legal-actions-and-masking]], the main new RL content in this tree |
| Elo and TrueSkill for ranking agents | absent | [[rl-and-the-battle-domain]] Part 3, and [[research/findings]] on protocol |
| Asymmetric actor-critic, privileged critics | absent | [[implementation/observation-design]], including the bias result |
| Truncation against termination bootstrapping | absent | [[rl-methods]], and it constrains the Milestone 4 protocol |
| Frame stacking and memoryless-policy limits | absent | [[rl-methods]] Part 3 |
| Game-environment engineering | absent | all of [[implementation/README]], which is the bulk of this repository |
| The fheroes2 battle domain | absent | [[rl-and-the-battle-domain]] Part 2 |

Two adjacent wiki notes are worth reading beside this tree even though they are not about notation. `reading-notes/mathematics-of-games` and `reading-notes/case-study-llm-and-games` are the closest existing material to this project's subject.

## Section 4, cautions

Three places where a careless reading goes wrong.

The letter $r$ is overloaded and stays that way. It is a reward in $r_{t+1}$ and the probability ratio in $r_t(\theta)$. Both the PPO paper and `concepts/ppo-clip` write it this way, so changing it here would create a mismatch worse than the ambiguity it removed. The ratio always carries an explicit $(\theta)$, which is enough to disambiguate in practice.

The discount range is relaxed. Most treatments require $\gamma \in [0, 1)$ so that infinite-horizon returns converge. Battles here terminate within 5 to 40 decisions, so $\gamma = 1$ is admissible and this documentation uses $\gamma \in [0, 1]$. The relaxation is safe only because termination is guaranteed, and it would not carry over to the adventure-map problem scoped in [[roadmap]].

Actor and critic parameters are named separately but are not independent. The critic is $V_\phi$ and the actor is $\pi_\theta$, following the wiki, yet the architecture in [[training-design]] shares a trunk between them, so the two parameter sets overlap. Where a loss is written $L(\theta, \phi)$ it is optimized jointly over the union.

## Section 5, if you are reading Zhao

The book *Mathematical Foundations of Reinforcement Learning* uses a different and internally consistent set of symbols from the ones above. This table translates, so a chapter can be read against this tree without re-deriving anything.

| Here and in the wiki | Zhao |
|---|---|
| $V^\pi(s)$, $Q^\pi(s, a)$ | $v_\pi(s)$, $q_\pi(s, a)$ |
| $A^\pi(s, a)$ | $\delta_\pi(s, a)$, deliberately sharing a letter with the TD error it is estimated by |
| $\pi_\theta(a \mid s)$ | $\pi(a \mid s, \theta)$ |
| $P(s' \mid s, a)$, $R(s, a, s')$ | $p(s' \mid s, a)$, $p(r \mid s, a)$ |
| $\rho_0$ | $d_0$ |
| $d^\pi(s)$ | $d_\pi(s)$ stationary, $\eta(s)$ in the policy gradient theorem |
| $J(\theta)$ for an episodic task | $\bar v_\pi^{\,0}$, one of three metrics it distinguishes in §9.2 |
| $V_\phi(s)$ | $v(s, w)$ |
| $\log$ | $\ln$ |
| definitions written with $=$ | $\doteq$ |

Two of the book's choices are worth borrowing as ideas even while keeping the wiki's symbols. Its $\delta_\pi$ makes visible that the temporal-difference error is a sampled estimate of the advantage, which is why one critic network suffices for A2C. And its Box 10.1 derives the actual variance-minimizing baseline, which is $Q^\pi$ weighted by $\lVert \nabla_\theta \log \pi_\theta \rVert^2$ rather than $V^\pi$, a point `concepts/advantage-function` gestures at with the word approximately.

## Related

- [[rl-and-the-battle-domain]], the vocabulary applied to a battle, and the comparison with other game environments.
- [[rl-methods]], every technique this documentation names, with a verdict on each.
- [[training-design]], the architecture, losses, and hyperparameters that use these symbols concretely.
- [[decisions/0005-training-and-reward]], the training and reward decisions, including what is deliberately still open.
