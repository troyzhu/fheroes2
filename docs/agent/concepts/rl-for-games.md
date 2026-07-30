---
title: Reinforcement learning for games — a primer
aliases:
  - rl-for-games
  - rl-basics
  - mdp
tags:
  - agent-env
  - primer
concept: the shared vocabulary of RL game environments
domain: reinforcement learning
grounded_in: "standard RL formalism; docs/agent/references/ for the game-specific practice"
depth: standard
updated: 2026-07-30
---

# Reinforcement learning for games — a primer

Every RL game environment is described with the same handful of objects, and every design decision in this project is a choice about one of them. This primer fixes that vocabulary once, states what each object is in a game setting, and names the axes along which game environments differ from each other. Read it before anything else here; the fheroes2 specifics build directly on these terms in [[fheroes2-battles-vs-other-games]].

## Motivation

Documentation for game environments usually assumes either the RL vocabulary or the game, and rarely bridges the two. That gap makes design choices look arbitrary. Whether an environment blocks or is called, whether an action is one integer or five, and whether a value is hidden from the agent all sound like implementation trivia until they are stated as choices about the state, the action space, and the observation function.

The formalism is worth the two pages because it makes the choices comparable. Once StarCraft, microRTS, NetHack, and a Heroes battle are all written as the same tuple, the differences between them become measurable rather than anecdotal, and a decision taken in one can be argued for or against in another.

## Notation

| Symbol | Meaning |
|---|---|
| $s \in \mathcal{S}$ | State: everything the simulator needs to continue the game. |
| $a \in \mathcal{A}$ | Action: one choice by the acting player. |
| $\mathcal{A}(s) \subseteq \mathcal{A}$ | The legal actions in state $s$. Usually a small subset. |
| $P(s' \mid s, a)$ | Transition function: how the world evolves. Deterministic when it puts all mass on one $s'$. |
| $R(s, a, s')$ | Reward: the scalar the agent maximizes. |
| $\gamma \in [0, 1]$ | Discount factor, trading immediate against future reward. |
| $\tau$ | Trajectory, the sequence $s_0, a_0, r_1, s_1, \ldots$ of one episode. |
| $G_t = \sum_{k \ge 0} \gamma^k r_{t+k+1}$ | Return: the discounted sum of future reward from step $t$. |
| $o = O(s)$ | Observation: what the agent actually receives, which may hide part of $s$. |
| $\pi(a \mid s)$ | Policy: the distribution over actions the agent follows. |
| $V^\pi(s)$, $Q^\pi(s,a)$ | Value functions: expected return from a state, or from a state-action pair. |
| $m \in \{0,1\}^{|\mathcal{A}|}$ | Legality mask, with $m_i = 1$ exactly when action $i$ lies in $\mathcal{A}(s)$. |

## The idea in one sentence

A game environment is a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ plus an observation function $O$, and every environment design decision is a decision about one of those six objects.

## The environment: four objects

**State.** Everything the simulator needs to continue. In a battle that is which units stand where, with what counts and hit points, whose turn it is, and the random generator's position. The last item matters: if the generator state is part of $s$ but never observed, the agent sees a world whose randomness it cannot account for, which is normal and fine, but it must be a deliberate choice rather than an accident.

**Action.** One choice by the acting player. The shape of $\mathcal{A}$ is the single most consequential design decision in a game environment, because it decides what the policy's output layer looks like. Four shapes recur. A flat discrete space enumerates every action as one integer, which is simple and works up to roughly $10^4$ entries. A factorized space splits an action into independent components, each with its own softmax, which is how a space of $10^7$ joint actions becomes a few hundred logits. A parameterized space picks a discrete type and then continuous parameters. A pointer space selects among a variable-length set of candidates by attention, which is the general answer when the candidate set is genuinely unbounded.

**Transition.** How the world evolves. Games are usually stochastic through damage rolls, critical hits, or hidden shuffles. Stochasticity is not the same as unpredictability from bad engineering: a seeded generator gives a stochastic $P$ that is nonetheless exactly reproducible, which is what makes replay and regression testing possible.

**Reward.** The scalar being maximized. Game environments typically offer a sparse terminal signal, meaning win or lose at the end, which is unbiased but hard to learn from, or a shaped signal such as damage dealt minus damage taken, which learns faster and risks teaching the wrong objective. The choice is a modeling decision and does not have to be made when the environment is built.

## The policy and what it may see

The policy maps observations to a distribution over actions. Two properties of that mapping drive most of the architecture.

**Observability.** If the agent sees the full state, the problem is a Markov decision process and a memoryless policy suffices. If it sees a function of the state, $o = O(s)$, the problem is a partially observed MDP, and a memoryless policy is provably insufficient in general, so agents add recurrence, frame stacking, or an explicit belief state. Many game environments sit in between: a battle where all units are visible is effectively fully observed, while a strategy game with fog of war is not.

A useful consequence is the asymmetric actor-critic pattern. Because the critic exists only during training, it may read privileged full state while the actor reads only what will be available at deployment. That requires the environment to expose both views, which is a reason to build the distinction into the observation schema rather than into the training code.

**Legality.** Almost every game restricts which actions are available in a state, and the standard mechanism is masking. The illegal entries of the policy's output are set to a large negative constant before the softmax, so they receive no probability and no gradient. Masking is not a heuristic: it is a state-dependent differentiable transform of the logits, so the masked update remains a valid policy gradient. The alternative of penalizing illegal actions with negative reward is well documented to collapse as the illegal fraction grows. See [[legal-actions-and-masking]].

## The axes along which game environments differ

These are the dimensions worth checking before borrowing a design from another environment.

| Axis | Range | Why it matters |
|---|---|---|
| Timing | turn-based against real-time | Real-time forces action delays and frame skipping; turn-based gives a clean decision boundary. |
| Turn order | alternating against simultaneous | Simultaneous play makes the opponent's action part of the transition and rules out naive search. |
| Players | single-agent against two-player zero-sum | Two-player training needs an opponent policy, which raises self-play and league scheduling. |
| Board and unit scale | a few cells against thousands of units | Sets whether entity lists, planes, or both are practical. |
| Observability | full against fogged | Decides whether recurrence is needed at all. |
| Stochasticity | deterministic against dice-driven | Deterministic transitions make planning methods far cheaper. |
| Action-space size | tens against millions | Decides flat, factorized, or pointer output. |
| Legal-action access | exposed by the engine against derived by the environment | Deriving it independently risks divergence from the engine's own rules. |
| Simulator speed | hundreds against hundreds of thousands of steps per second | Decides whether the environment or the learner is the bottleneck. |
| Built-in opponent | none against a competent scripted AI | A scripted AI supplies demonstrations and an evaluation baseline for free. |

## How agents are usually trained on games

Four families cover most published results, and they are complementary rather than exclusive.

Imitation from a teacher, meaning supervised learning on recorded decisions, is the cheapest way to reach competent play when a scripted AI or human replays exist. It plateaus at the teacher's level and suffers distribution shift once the student visits states the teacher never did, which DAgger-style correction addresses by relabeling the student's own trajectories. See [[teacher-coverage-and-behavior-cloning]].

Policy-gradient reinforcement learning, in practice usually PPO with masking, is the workhorse for discrete game actions. It needs many environment steps, so simulator speed matters, and it needs a diverse opponent mixture or it overfits to one adversary.

Planning and search, meaning Monte Carlo tree search and its learned variants, exploits a fast, copyable simulator to look ahead instead of learning a reflex. It is strongest where the transition is deterministic and the branching factor is modest.

Self-play and league training generate their own curriculum for two-player games, at the cost of scheduling complexity and the risk of cyclic strategies that beat each other in rotation.

## Evaluation

Game agents are compared by win rate against a fixed pool of opponents under a fixed seed set, and by rating systems such as Elo or TrueSkill when many agents must be ordered. The seeded fixed pool answers whether a change helped; the rating league answers how a checkpoint compares to everything else built so far. Reporting either without the seed set and the opponent list makes the number unreproducible.

## What makes a good environment, independent of the agent

Three properties matter more than they sound. Determinism under a seed, so trajectories replay and regressions are detectable. Speed, so the learner rather than the simulator is the bottleneck. Engine-sourced legality, so the mask cannot disagree with what the simulator will accept. An environment missing any of the three can still train an agent, but debugging it becomes guesswork.

## Key terms

- Episode: one complete play-through, from reset to a terminal state or a truncation limit.
- Trajectory: the recorded sequence of states, actions, and rewards in an episode.
- Return: discounted cumulative reward, the quantity a policy maximizes.
- Rollout: running the current policy to collect trajectories.
- On-policy: learning from data generated by the current policy, as PPO does.
- Off-policy: learning from data generated by a different or older policy.
- Distribution shift: the mismatch between the states a teacher visited and those the student visits.
- Sparse reward: a signal that arrives only at the end of an episode.

## Why it came up here

Every design record in this project cites one of these objects. The observability profiles are a choice about $O$, the canonical action space and mask are a choice about $\mathcal{A}$ and $\mathcal{A}(s)$, the seed discipline is a choice about reproducing $P$, and the deliberate absence of a reward in Phase 1a is a choice to defer $R$ until the substrate is trustworthy.

## What this does not say

This is vocabulary, not method. It does not recommend an algorithm, and it does not cover the mathematics of policy-gradient estimators, value-function approximation, or exploration, all of which are standard and well covered elsewhere.

## Go deeper

- [[fheroes2-battles-vs-other-games]] — what these objects are in a Heroes battle, and how it compares to other environments.
- [[legal-actions-and-masking]] — the action-space decision in detail.
- [[observation-design]] — the observation-function decision in detail.
- `../references/summary.md` — what the literature establishes about these choices.
