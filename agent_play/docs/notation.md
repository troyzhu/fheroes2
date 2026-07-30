---
title: Notation, and how this tree extends the standard treatment
type: reference
updated: 2026-07-30
related_concepts: ["[[rl-and-the-battle-domain]]", "[[rl-methods]]", "[[training-design]]"]
tags: [agent-env, rl, notation, reference, entry-point]
---

# Notation, and how this tree extends the standard treatment

This page fixes the symbols used everywhere else in this documentation, and fixes them against an external source rather than inventing a house style. That source is Shiyu Zhao, *Mathematical Foundations of Reinforcement Learning* ([book PDF](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/Book-all-in-one.pdf)), referred to below as the book. The intent is that this material reads as a continuation of notes already taken on that text, not as a second vocabulary that has to be held alongside the first.

Three commitments follow. Anything the book defines is used here with the book's meaning and is not redefined. Anything this project needs that the book does not cover is introduced explicitly, together with the reason it is absent there. And the few genuine departures are listed in [[#Section 4, departures and collisions]] with their justification, so no departure is silent.

Readers who want the domain rather than the symbols should start at [[rl-and-the-battle-domain]] and treat this page as a lookup.

## Section 1, symbols inherited from the book

These carry the book's meaning. The last column points at where the book introduces them, so an unfamiliar symbol can be chased to its original treatment rather than to a paraphrase here.

| Symbol | Meaning | Book |
|---|---|---|
| $\mathcal{S}$ | State space | Ch. 2 |
| $\mathcal{A}(s)$ | Actions available at $s$, abbreviated $\mathcal{A}$ where the dependence is not at issue | Ch. 2 |
| $\mathcal{R}$ | Reward set | Ch. 2 |
| $S_t, A_t, R_{t+1}$ | Random variables: state and action at step $t$, and the reward that arrives after acting | §2.3 |
| $s, a, r, s'$ | Realizations of those variables | Ch. 2 |
| $p(s' \mid s, a)$ | State transition model | §2.4 |
| $p(r \mid s, a)$ | Reward model | §2.4 |
| $\pi(a \mid s)$ | Policy | §2.4 |
| $\pi(a \mid s, \theta)$ | Policy parameterized by $\theta$, written with a comma rather than as $\pi_\theta$ | §9.3 |
| $\gamma$ | Discount rate | §2.3 |
| $G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$ | Discounted return from step $t$ | §2.3 |
| $v_\pi(s) \doteq \mathbb{E}[G_t \mid S_t = s]$ | State value | §2.3 |
| $q_\pi(s, a)$ | Action value | Ch. 2 |
| $\delta_\pi(s, a) \doteq q_\pi(s, a) - v_\pi(s)$ | Advantage function | §10.2 |
| $\delta_t = r_{t+1} + \gamma v(s_{t+1}, w) - v(s_t, w)$ | Temporal-difference error, the sampled stand-in for the advantage | §10.2 |
| $J(\theta)$ | The metric being maximized | §9.2 |
| $\bar v_\pi^{\,0} = \mathbb{E}_{S \sim d_0}[v_\pi(S)]$ | Average state value under a start distribution that does not depend on $\pi$ | §9.2 |
| $d_0$ | Initial-state distribution | §9.2 |
| $d_\pi(s)$ | Stationary state distribution under $\pi$ | §9.2 |
| $\eta(s)$ | The state distribution appearing in the policy gradient theorem | §9.3 |
| $b(s)$ | Baseline subtracted from the action value | §10.2 |
| $w$ | Parameters of a value approximator, as in $v(s, w)$ | §10.2 |
| $\alpha_\theta,\ \alpha_w$ | Actor and critic step sizes | §10.2 |
| $h(s, a, \theta)$ | Action preference feeding a softmax policy | §9.3 |
| $\ln$ | Natural logarithm, the book's spelling, used here in place of $\log$ | §9.3 |
| $\doteq$ | Equality by definition | throughout |

## Section 2, what this project maximizes, in the book's terms

The book gives three metrics in §9.2 and this project uses one of them. A battle terminates, an episode is one battle drawn from a scenario and army generator, and that generator does not depend on the policy. The metric is therefore $\bar v_\pi^{\,0}$, the average state value taken over a fixed start distribution, with $d_0$ standing for the generator. The stationary-distribution metrics $\bar v_\pi$ and $\bar r_\pi$ belong to the continuing setting and are not what is optimized here.

Writing the objective as $J(\theta) = \bar v_\pi^{\,0} = \mathbb{E}_{S \sim d_0}[v_\pi(S)]$ makes a consequence visible that is easy to lose. A reported win rate is a statement about $d_0$ every bit as much as about $\theta$, so a scenario generator is part of the objective rather than part of the test harness. [[decisions/0005-training-and-reward]] records that the generator is currently five fixed fixtures used as regression anchors, which is not yet a training distribution, and treats defining it as a prerequisite for any reported number meaning anything.

## Section 3, symbols this project adds

The book works throughout with a fully observed, single-agent problem in which every action is available and the horizon is infinite. Each addition below exists because one of those assumptions fails here. The last column names the assumption that breaks.

| Symbol | Meaning | Absent from the book because |
|---|---|---|
| $o \in \Omega$ | Observation, what the agent actually receives | its agent sees $s$ |
| $O(o \mid s)$ | Observation function, in general stochastic | same |
| $\pi(a \mid o, \theta)$ | Observation-conditioned policy, the object actually trained here | same |
| $m(o) \in \{0,1\}^{793}$ | Legality mask, with $m_i = 1$ exactly when action $i$ is legal | its $\mathcal{A}(s)$ is a set; here it has to become a tensor |
| $\ell(o, \theta) \in \mathbb{R}^{793}$ | Network logits before masking, the book's $h(s, a, \theta)$ gathered into a vector | it keeps preferences scalar and per-action |
| $\hat A_t$ | Sampled advantage estimator, here generalized advantage estimation | it estimates $\delta_\pi$ by the one-step residual only |
| $\lambda$ | Trace parameter trading bias against variance in $\hat A_t$ | generalized advantage estimation is not covered |
| $\rho_t(\theta)$ | Importance ratio $\pi(a_t \mid o_t, \theta) \,/\, \pi(a_t \mid o_t, \theta_{\text{old}})$ | it writes importance weights as $p_0(x)/p_1(x)$ with no per-step symbol |
| $\epsilon$ | Clipping half-width in the PPO surrogate | PPO is not covered |
| $\Phi(s)$ | Potential function in potential-based shaping | reward design is not covered |
| $\pi^{*}$ | The teacher policy, here `AI::BattlePlanner` | imitation is not covered |
| $\mathcal{D}$ | Dataset of observation and action pairs | same |
| $\varepsilon$ | Per-decision error rate of a fitted policy | same |
| $\tau$ | Trajectory, the sequence $s_0, a_0, r_1, s_1, \ldots$ | it reasons state by state rather than over trajectories |
| $T$ | Episode length in decisions, 5 to 40 here | it works in the infinite-horizon setting |
| $c_v,\ c_e$ | Value and entropy coefficients in a combined loss | a combined actor-critic loss is not formed |

## Section 4, departures and collisions

Four places where a careless reading across sources goes wrong.

The letter $r$ is a reward, always. The PPO literature writes the importance ratio as $r_t(\theta)$, which collides with that. This documentation writes the ratio $\rho_t(\theta)$ instead, so a reader arriving from Schulman et al. (2017) should read our $\rho_t(\theta)$ as their $r_t(\theta)$. The letter $\rho$ is free for this because the initial-state distribution is $d_0$ here rather than $\rho_0$, which is itself a change from earlier revisions of these documents and was made to match the book.

The letter $\delta$ carries two meanings and the book intends both. It is the exact advantage function $\delta_\pi(s, a)$, and it is the temporal-difference error $\delta_t$, which is a sampled stand-in for the first. That identification is worth keeping rather than smoothing away, since it is the reason a single critic network suffices for A2C. Where an advantage is estimated across several steps rather than one, this documentation writes $\hat A_t$, following the source that defines generalized advantage estimation, and states that its target is $\delta_\pi$.

The discount range is relaxed. The book states $\gamma \in (0, 1)$, which is what its infinite-horizon convergence arguments need. Battles here terminate within 5 to 40 decisions, so $\gamma = 1$ is admissible and this documentation uses $\gamma \in [0, 1]$. The relaxation is safe only because termination is guaranteed, and it would not carry over to the adventure-map problem scoped in [[roadmap]].

Sampled quantities keep lowercase letters. The book distinguishes $\delta_\pi(S, A)$ over random variables from $\delta_t(s_t, a_t)$ over samples, as in its (10.7) against (10.8). The same convention holds here, so an expression in capitals is a statement about the expectation and an expression in lowercase is what an implementation computes on a batch.

## Section 5, coverage map

Which material is assumed from the book and which is added here. Rows carrying a chapter are treated as known and are not re-derived; a claim of that kind in this tree links back rather than restating. Rows marked absent are the delta, and they are the pages worth reading.

| Topic | Book | Here |
|---|---|---|
| MDP objects, returns, state and action values | Ch. 1 to 2 | [[rl-and-the-battle-domain]] instantiates each one for a battle |
| Bellman equation, optimality, value and policy iteration | Ch. 2 to 4 | assumed, and not used, since this project is model-free |
| Monte Carlo and temporal-difference estimation | Ch. 5, 7 | [[rl-methods]] uses the bias-variance contrast when motivating GAE |
| Stochastic approximation | Ch. 6 | assumed |
| Value function approximation, DQN | Ch. 8 | [[rl-methods]] weighs DQN variants and records why they are not the first path here |
| Policy gradient theorem, REINFORCE, softmax policies | Ch. 9 | [[rl-methods]] restates the theorem and derives the baseline result in trajectory form |
| Actor-critic, A2C, importance sampling, off-policy actor-critic | Ch. 10 | the point where [[rl-methods]] picks up and continues to PPO |
| Trust regions and the PPO clipped surrogate | absent | [[rl-methods]], applied in [[training-design]] |
| Generalized advantage estimation | absent | [[rl-methods]] |
| Legal-action masking | absent | [[implementation/legal-actions-and-masking]] |
| Partial observability, belief states, recurrence | absent | [[rl-methods]], [[implementation/observation-design]] |
| Imitation, behavior cloning, DAgger | absent | [[training-design]] |
| Potential-based reward shaping | absent | [[rl-methods]], decided in [[decisions/0005-training-and-reward]] |
| Self-play, opponent mixtures, rating and evaluation | absent | [[rl-and-the-battle-domain]], [[rl-methods]] |
| Environment engineering, determinism, action encoding | absent | [[implementation/README]] |

## Section 6, a reading order from the book

Someone who has worked through the book to Chapter 10 already holds most of Part 1 of [[rl-methods]]. The efficient path is to read [[rl-and-the-battle-domain]] for the domain and the six MDP objects as this project instantiates them, then jump into [[rl-methods]] at trust regions, which is the first thing the book does not cover, and read forward from there. Parts 2 to 5 of that page are a landscape and can be read in any order.

Someone who has not read the book can still read this tree straight through. Nothing here depends on holding a proof from the book in mind, and the pieces that matter are restated at the point of use.

## Related

- [[rl-and-the-battle-domain]], the vocabulary applied to a battle, and the comparison with other game environments.
- [[rl-methods]], every technique this documentation names, derived, with a verdict on each.
- [[training-design]], the architecture, losses, and hyperparameters that use these symbols concretely.
- [[decisions/0005-training-and-reward]], the training and reward decisions, including what is deliberately still open.
