---
title: Reinforcement learning and the fheroes2 battle domain
type: primer
updated: 2026-07-30
related_concepts: ["[[README]]", "[[implementation/legal-actions-and-masking]]", "[[implementation/observation-design]]"]
tags: [agent-env, primer, rl, entry-point]
---

> **What this note is.** The conceptual entry point. Part 1 fixes the vocabulary of reinforcement learning for games from scratch. Part 2 explains what a Heroes of Might and Magic II battle is, restates it in that vocabulary, and places it against the environments this project borrows from. Part 3 draws out what the comparison implies for the design. It assumes no reinforcement learning, no fheroes2 knowledge, and no C++. Read it before [[README]] if either half is unfamiliar; read [[README]] first if you only need the build and the current state.

## Table of contents
- [[#Part 1 — the vocabulary]]
- [[#Notation]]
- [[#The environment: four objects]]
- [[#The policy and what it may see]]
- [[#The axes along which game environments differ]]
- [[#Part 2 — the fheroes2 battle domain]]
- [[#The game, for someone who has never played it]]
- [[#The same thing in RL terms]]
- [[#How it compares]]
- [[#What is unusually easy here]]
- [[#What is unusually awkward here]]
- [[#Part 3 — what this implies]]
- [[#How agents are usually trained on games]]
- [[#Evaluation]]
- [[#Implications for this project]]
- [[#Key terms]]

## Part 1 — the vocabulary

Reinforcement learning studies an agent that repeatedly looks at a situation, picks an action, and receives a number scoring how that turned out. Nobody supplies the right answer. The agent learns by acting and watching what the number does, which is what separates it from supervised learning, where every training example arrives with its correct label attached.

A turn-based battle fits that loop directly, and the sections below name each part of it. The names are worth learning because the rest of this documentation uses them constantly. The observability profiles are a choice about the observation function, the fixed action space and its legality mask are a choice about the action set and its legal subset, and the deliberately undefined reward is a choice to leave one part of the loop empty until the environment underneath it is trustworthy.

Naming the parts also makes environments comparable. Once StarCraft, microRTS, NetHack, and a Heroes battle are written as the same tuple, their differences are measurable rather than anecdotal, and an argument settled in one can be carried into another.

One property is assumed throughout and is easiest to state now. A problem is Markov when the current state carries everything needed to predict what comes next, so the history that produced that state adds nothing. The assumption is what allows a policy to look only at the present. Much of the observation design in this project exists either to keep it true or to be explicit about where it fails.

## Notation

| Symbol | Meaning |
|---|---|
| $s \in \mathcal{S}$ | State: everything the simulator needs to continue the game. |
| $a \in \mathcal{A}$ | Action: one choice by the acting player. |
| $\mathcal{A}(s) \subseteq \mathcal{A}$ | The legal actions in state $s$. Usually a small subset. |
| $P(s' \mid s, a)$ | Transition function: how the world evolves. Deterministic when it puts all mass on one $s'$. |
| $R(s, a, s')$ | Reward, the scalar arriving on one transition. The quantity a policy maximizes is the expected return $J(\pi)$, not $R$ itself. |
| $\gamma \in [0, 1)$ | Discount factor, trading immediate against future reward. A value of 1 is admissible only for episodic tasks that are guaranteed to terminate, which battles are. |
| $\tau$ | Trajectory, the sequence $s_0, a_0, r_1, s_1, \ldots$ of one episode. |
| $G_t = \sum_{k \ge 0} \gamma^k r_{t+k+1}$ | Return, the discounted sum of future reward from step $t$. Episodes here are finite, so the sum runs to termination. |
| $o = O(s)$ | Observation: what the agent actually receives, which may hide part of $s$. |
| $\pi(a \mid s)$ | Policy, the distribution over actions the agent follows. When the agent sees an observation rather than the state, it is $\pi(a \mid o)$, or $\pi(a \mid h)$ over the history. |
| $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$, $Q^\pi(s,a)$ | Value functions, the expected return under $\pi$ from a state, or from a state and action. |
| $m \in \{0,1\}^{\lvert \mathcal{A} \rvert}$ | Legality mask, with $m_i = 1$ exactly when action $i$ lies in $\mathcal{A}(s)$. |
| $\rho_0$ | Initial-state distribution. Here it is the scenario and army generator, and it defines what a reported win rate means. |
| $\Omega$ | The observation space, the set $o$ is drawn from. |
| $J(\pi) = \mathbb{E}_{s_0 \sim \rho_0}[G_0]$ | The objective a policy maximizes. |

An environment is the tuple $(\mathcal{S}, \mathcal{A}, P, R, \rho_0)$, extended to a partially observed problem by an observation space $\Omega$ and an observation function $O$, which in general is stochastic and written $O(o \mid s, a)$. The discount $\gamma$ belongs to the agent's objective rather than to the environment, alongside $R$. 

Almost every environment design decision is a decision about one of those objects. $\rho_0$ is the one most often left implicit.

## The environment: four objects

State is everything the simulator needs to continue. In a battle that is which units stand where, with what counts and hit points, whose turn it is, and the random generator's position. The last item matters. If the generator state is part of $s$ but never observed, the agent sees a world whose randomness it cannot account for, which is normal and fine, but it must be a deliberate choice rather than an accident.

Action is one choice by the acting player, and the shape of $\mathcal{A}$ is the single most consequential design decision in a game environment, because it decides what the policy's output layer looks like. The shapes that recur are these. A flat discrete space enumerates every action as one integer, which is simple and works up to roughly $10^4$ entries. A factorized space splits an action into independent components, each with its own softmax, which is how a space of $10^7$ joint actions becomes a few hundred logits. A parameterized space picks a discrete type and then continuous parameters. A pointer space selects among a variable-length set of candidates by attention, which is the general answer when the candidate set is genuinely unbounded.

Transition is how the world evolves. Games are usually stochastic through damage rolls, critical hits, or hidden shuffles. Stochasticity is not the same as unpredictability from bad engineering. A seeded generator leaves $P$ unchanged and makes the sampler reproducible, which is what makes replay and regression testing possible. See [[implementation/determinism-seeds-and-digests]].

Reward is the per-transition scalar. What a policy maximizes is the expected return $J(\pi)$, not $R$ itself. Game environments typically offer a sparse terminal signal, meaning win or lose at the end, which is unbiased but hard to learn from, or a shaped signal such as damage dealt minus damage taken, which learns faster and risks teaching the wrong objective. The choice is a modeling decision and does not have to be made when the environment is built.

## The policy and what it may see

The policy maps observations to a distribution over actions, and two properties of that mapping drive most of the architecture.

Observability decides the problem class. If the agent sees the full state, the problem is a Markov decision process and a memoryless policy suffices. If it sees a function of the state, $o = O(s)$, the problem is a partially observed MDP, and a memoryless policy is provably insufficient in general, so agents add recurrence, frame stacking, or an explicit belief state. Many game environments sit in between. A battle where all units are visible is close to fully observed, while a strategy game with fog of war is not.

A useful consequence is the asymmetric actor-critic pattern. Because the critic exists only during training, it may read privileged full state while the actor reads only what will be available at deployment. That requires the environment to expose both views, which is a reason to build the distinction into the observation schema rather than into the training code. See [[implementation/observation-design]].

Legality is the second property. Almost every game restricts which actions are available in a state, and the standard mechanism is masking. The illegal entries of the policy's output are set to a large negative constant before the softmax, so they receive no probability and no gradient.
 Masking is not a heuristic, and the reason is narrower than it first appears. Replacing a logit with a large negative constant is not a differentiable function of that logit; it discards it.

What licenses the method is that the mask depends on $s$ alone and never on the policy parameters, so the masked softmax is itself a well-formed parameterized policy $\pi^{\text{mask}}_\theta$ over $\mathcal{A}(s)$. The usual estimator is then an unbiased gradient of $J(\pi^{\text{mask}}_\theta)$, which is the objective over legal play. The policy class being optimized has changed, so a policy trained with a mask is undefined behavior if the mask is removed at deployment. The alternative of penalizing illegal actions with negative reward is well documented to collapse as the illegal fraction grows. See [[implementation/legal-actions-and-masking]].

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

## Part 2 — the fheroes2 battle domain

## The game, for someone who has never played it

Two armies meet on a fixed battlefield. Each army holds up to five stacks, where a stack is a number of identical creatures acting as one unit, so "fifty peasants" is one stack that moves and attacks together and loses members as it takes damage.

The field is a grid of 11 by 9 cells drawn as hexes, so each cell has six neighbors rather than eight. Attackers start on the left, defenders on the right.

Play proceeds in rounds. Within a round every stack that can act does so once, in order of speed, and its options are to move, to attack an adjacent enemy, to shoot a distant one if it has ammunition and nothing adjacent is blocking it, or to skip. A melee attack usually draws a retaliation from the defender. Damage depends on the attacker's attack rating against the defender's defense rating, multiplied by the number of creatures in the stack, and rolled within a range, so outcomes are stochastic. Morale and luck can grant an extra action or a damage bonus at random. The battle ends when one side has no living stacks.

Phase 1a narrows this. No heroes, so no spells and no leadership bonuses. No castle, so no walls, moat, or towers. And only creatures with ordinary action spaces, meaning single-cell, walking, and without special targeting.

## The same thing in RL terms

| Object | In a Heroes battle |
|---|---|
| State $s$ | Position, count, hit points, remaining shots, and status of every stack, plus whose turn it is and the combat generator's position. |
| Action $a$ | What the active stack does: move to a cell, attack a specific enemy from a specific direction, shoot, or skip. |
| Legal set $\mathcal{A}(s)$ | Typically 5 to 30 actions, out of a fixed space of 793. |
| Transition $P$ | The engine, stochastic through damage rolls, morale, and luck, but exactly reproducible under a fixed combat seed. |
| Reward $R$ | Deliberately undefined in Phase 1a. The terminal outcome and surviving force are recorded so an objective can be chosen later. |
| Observation $o$ | Structured records, either the true state or the player-obtainable subset, never pixels. |
| Episode | One battle: roughly 5 to 40 decisions in the fixtures measured so far. |
| Players | Two, opposed, ordered by a speed queue rather than strictly alternating. Fixing an opponent policy induces a single-agent MDP whose transition absorbs that opponent; the induced problem is non-stationary as soon as the opponent changes, which is the regime self-play creates. Because the queue can give one side several consecutive decisions, the induced problem is a semi-MDP, and the discount between a player's own successive decisions is $\gamma^k$ for a random $k$. |

The decision boundary is a unit's turn rather than a round, so a five-stack battle produces about ten decisions per round.

The environment is close to fully observed in the sense that matters for the units. What a player could obtain from the game's own interface includes any unit's full statistics regardless of ownership, so no unit attribute is secret. The combat generator's position is a different matter. It sits in $s$, it is serialized in neither observation profile, and it therefore leaves both profiles formally partially observed. That is benign only to the extent that the draws consumed per decision do not depend on anything the policy can influence, which is an assumption worth stating rather than an established fact.

## How it compares

| Environment | Timing | Board | Units per side | Action space | Observability | Stochastic | Speed |
|---|---|---|---|---|---|---|---|
| fheroes2 battle (this project) | turn-based, alternating | 99 hex cells | 1 to 5 stacks | 793, masked | effectively full | yes, damage and morale | ~4,600 episodes/s |
| Heroes III battle ([[research/works/vcmi-gym\|vcmi-gym]]) | turn-based, alternating | 165 hex cells | up to 7 stacks | 2,312, masked | effectively full | yes | not published |
| microRTS ([[research/works/gym-microrts\|gym-µRTS]]) | real-time, simultaneous | 16×16 grid | dozens | factorized per cell | full or fogged by flag | mostly deterministic | very high |
| StarCraft II ([[research/works/pysc2\|PySC2]]) | real-time, simultaneous | large minimap | hundreds | huge, structured | fogged | yes | low |
| NetHack ([[research/works/nle\|NLE]]) | turn-based, single agent | 21×79 glyphs | one hero | ~100 discrete | fogged, partial | yes | high |
| Battle for Wesnoth ([[research/works/arlinbfw\|ARLinBfW]]) | turn-based, alternating | hex map | several | small discrete | full on the map | yes | low |
| Chess and Go | turn-based, alternating | 64 or 361 cells | fixed | hundreds to thousands | full | no | very high |
| Dota 2 ([[research/works/openai-five\|OpenAI Five]]) | real-time, simultaneous | large map | five heroes | large, parameterized | fogged | yes | low |

For what each of those codebases actually contains and where to look inside it, see [[research/prior-art]].

## What is unusually easy here

The scale is small in every dimension that normally forces machinery. With at most ten stacks and 99 cells, a padded entity list fits comfortably and an entity transformer is an optimization rather than a necessity. The action space is about 793 entries, so a flat masked softmax works and factorization buys nothing.

Observability is effectively complete, so recurrence, belief states, and frame stacking are unnecessary at this stage, which removes an entire class of architecture decisions that dominate work on fogged games.

Turn-based alternating play means one decision at a time and no action-delay modeling, a large simplification against every real-time environment in the table.

The engine is fast and deterministic under a seed, at roughly 4,600 episodes per second on the target machine, so the learner will be the bottleneck rather than the simulator, and planning methods stay viable later.

A competent scripted opponent already exists inside the game. It plays both sides for free, supplying both a demonstration source and an evaluation baseline without writing either. See [[implementation/teacher-coverage-and-behavior-cloning]].

## What is unusually awkward here

The constraints below show up rarely in published environments. Both come from embedding in a real game engine rather than a purpose-built simulator.

The engine exposes no legal-action API. Its validation logic lived inside the functions that execute commands, so the environment either extracts that logic or re-derives battle legality and risks disagreeing with the engine. This was the project's largest risk and it is why the validators were lifted into a shared module rather than reimplemented. See [[implementation/legal-actions-and-masking]].

Control is inverted and the arena is a singleton. The engine advances a whole round per call and owns the call stack, so the environment blocks inside a hook rather than exposing a callable step, and only one battle can exist per process, which makes parallelism process-level. See [[implementation/battle-turn-dispatch]].

A third and milder awkwardness is that the stochasticity is coarse. Morale and luck can grant an entire extra action, so a single lucky roll changes a turn's structure rather than nudging a number, which makes variance across seeds larger than damage rolls alone would suggest.

## Part 3 — what this implies

## How agents are usually trained on games

The families below cover most published results, and they are complementary.

Imitation from a teacher, meaning supervised learning on recorded decisions, is the cheapest way to reach competent play when a scripted AI or human replays exist. In practice it lands below the teacher, because error compounds once the student leaves the teacher's state distribution. DAgger addresses that by rolling out the student, having the expert label the states the student actually visited, and training on the union of all data collected so far, which is the aggregation the name refers to. The reduction gives a regret bound linear in the horizon where plain cloning is quadratic (Ross, Gordon and Bagnell, 2011). It requires an expert that can be queried at arbitrary states, which is a real precondition here rather than a formality.

Policy-gradient reinforcement learning, in practice usually PPO with masking, is the workhorse for discrete game actions. It needs many environment steps, so simulator speed matters, and it needs a diverse opponent mixture or it overfits to one adversary.

Planning and search, meaning Monte Carlo tree search and its learned variants, exploits a fast copyable simulator to look ahead instead of learning a reflex. It is strongest where the transition is deterministic and the branching factor is modest.

Self-play and league training generate their own curriculum for two-player games, at the cost of scheduling complexity and the risk of cyclic strategies that beat each other in rotation.

## Evaluation

Game agents are compared by win rate against a fixed pool of opponents under a fixed seed set, and by rating systems such as Elo or TrueSkill when many agents must be ordered. A win rate is a statement about $\rho_0$ and the opponent pool, so both belong in the report. Ratings additionally assume a transitive ordering of strength, which is exactly what cyclic self-play strategies violate, so where cycles are the concern the pairwise win-rate matrix has to be reported alongside the scalar. The seeded fixed pool answers whether a change helped; the rating league answers how a checkpoint compares to everything built so far. Reporting either without the seed set and the opponent list makes the number unreproducible.

Three environment properties matter independently of the agent. Determinism under a seed, so trajectories replay and regressions are detectable. Speed, so the learner rather than the simulator is the bottleneck. Engine-sourced legality, so the mask cannot disagree with what the simulator will accept. An environment missing any of the three can still train an agent, but debugging it becomes guesswork.

## Implications for this project

The comparison explains why our decisions diverge from the environments we borrow evidence from. We use a flat masked action space rather than microRTS's factorized one because 793 entries do not need factoring. We keep both an entity list and an optional plane tensor rather than committing, because at this scale neither is expensive and no published ablation settles which wins on an 11 by 9 board. We ship an observability profile despite full observability today, because the seam is free now and expensive to retrofit when hero mana and fog arrive. And we invest in seed and digest discipline more heavily than comparable projects do, because a fast deterministic engine makes that discipline cheap and it is the only affordable proof that engine edits changed nothing.

Every design record in this project cites one of the six objects from Part 1. The observability profiles are a choice about $O$ ([[decisions/0001-observation-profiles|ADR 0001]]), the canonical action space and mask are a choice about $\mathcal{A}$ and $\mathcal{A}(s)$ ([[decisions/0002-action-space|ADR 0002]]), the seed discipline is a choice about reproducing $P$, and the deliberate absence of a reward in Phase 1a is a choice to defer $R$ until the substrate is trustworthy.

## Key terms

- Episode: one complete play-through, from reset to a terminal state or a truncation limit.
- Trajectory: the recorded sequence of states, actions, and rewards in an episode.
- Return: discounted cumulative reward, the quantity a policy maximizes.
- Rollout: running the current policy to collect trajectories.
- On-policy: the learner keeps the collecting policy close to the policy being updated, so an importance-sampling correction stays low variance. PPO is on-policy in this sense, since it reuses each batch for several epochs under a clipped ratio rather than for a single gradient step.
- Off-policy: learning from data generated by a different or older policy.
- Distribution shift: the mismatch between the states a teacher visited and those the student visits.
- Sparse reward: a signal that arrives only at the end of an episode.
- Policy gradient: adjusting the policy's parameters in the direction that raises expected return, estimated from sampled trajectories.
- PPO: proximal policy optimization, the policy-gradient method used in most of the work cited here. It reuses each batch of experience for several passes while clipping how far the policy may move.
- Logits: the raw scores a network emits per action, before they are turned into probabilities.
- Softmax: the function turning logits into probabilities, by exponentiating and normalizing. Setting a logit very negative therefore drives its probability to zero.
- Actor and critic: the actor is the policy that chooses actions; the critic estimates value and is used to reduce the variance of the actor's updates. The critic exists only during training.
- Behavior cloning: supervised learning of a policy from recorded expert decisions, treating the expert's action as the label.
- Gymnasium: the common Python interface for RL environments, where the learner calls `step(action)` and receives an observation and reward.
- Monte Carlo tree search: a planning method that looks ahead by simulating many possible continuations, rather than learning a reflex.
- Stack: a group of identical creatures acting as a single unit, the atom of a Heroes battle.
- Round: one pass in which every eligible stack acts once, ordered by speed.
- Retaliation: the defender's automatic counter-attack after a melee strike.
- Shooter: a stack with ranged ammunition, which reverts to melee when an enemy is adjacent.
- Morale and luck: random effects granting an extra action or bonus damage.

## What this does not say

Part 1 is vocabulary rather than method, so it does not recommend an algorithm and does not cover policy-gradient estimators, value-function approximation, or exploration, all standard and well covered elsewhere. Part 2 covers the battle domain only. The fheroes2 adventure map, with fog of war, resource management, and a much larger action space, is a different problem that would sit far closer to StarCraft in the comparison table.

## Go deeper

- [[rl-methods]] — every technique named here, derived, from the policy gradient through PPO and the alternatives.
- [[README]] — the system as it stands, the build, and the current state.
- [[implementation/README|Concept primers]] — how each implemented mechanism works.
- [[research/findings|Literature synthesis]] — what the evidence establishes.
- [[research/prior-art|Repository orientation]] — the codebases behind the comparison.
