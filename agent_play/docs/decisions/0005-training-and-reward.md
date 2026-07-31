---
title: "ADR 0005 — Training algorithm and reward design"
type: adr
status: partially accepted
updated: 2026-07-30
related_concepts: ["[[../rl-and-the-battle-domain]]", "[[../implementation/teacher-coverage-and-behavior-cloning]]"]
tags: [adr, training, reward, agent-env]
---

# ADR 0005 — Training algorithm and reward design

- Status: algorithm choice accepted; reward design deliberately open, with the decision criteria fixed here
- Context: [[../research/findings]], [[../implementation/teacher-coverage-and-behavior-cloning]], user question 2026-07-30
- Techniques: [[../rl-methods]] defines every method named below, with its equation and our verdict.
- Mechanics: [[../training-design]] carries the architecture, losses, hyperparameters, and the full alternatives analysis. This record states the decisions and their reasons only.

## Context

Milestones 1 through 3 built an environment and said almost nothing about what would train on it. That was intentional while the substrate was unproven, but it left two questions unanswered that a reader is entitled to ask, and this record answers the first and scopes the second.

The environment ships with no reward function at all. Phase 1a records the terminal outcome and the surviving force and stops there. That is a real design choice rather than an omission, and the reasoning is that a reward encodes what you want the agent to do, which is a modeling decision that should not be frozen while the thing underneath it is still being verified. The cost of deferring is nothing, since the recorded terminal state is sufficient to define most candidate rewards after the fact.

Where this record names a technique without teaching it, [[../training-design]] teaches it. That document gives the network architecture, the cross-entropy objective cloning minimizes and why it is masked, the DAgger iteration written out with its mixing schedule and aggregation step, the PPO surrogate with its masking integration, starting hyperparameter tables for both stages, and a compared-alternatives table at every choice point.

## Decision, part one: the training algorithm

Accepted. The staging is imitation first, then masked policy-gradient reinforcement learning against a mixture of opponents.

Stage 1, behavior cloning from the built-in AI. The teacher is `AI::BattlePlanner`, it plays both sides of every headless battle, and Milestone 2 already records its decisions. Supervised learning on those decisions is the cheapest route to competent play, and it is validated at the strongest scale available, where AlphaStar's supervised stage reached 87% against its game's strongest built-in opponent before any reinforcement learning. Two qualifications carry over from the review of that evidence. The target is $\pi(a \mid o)$, conditioned on what the deployed policy will actually receive, not on full state. And the analogy is imperfect, because there the demonstrator and the opponent were different agents while here they are the same, so cloning can at best approach the teacher.

Stage 2, DAgger-style correction. Plain cloning degrades once the student visits states the teacher never did, at a rate quadratic in the horizon where DAgger is linear (Ross, Gordon and Bagnell, 2011). The precondition is an expert answerable at arbitrary student-visited states. Whether `AI::BattlePlanner` can be queried without advancing the arena or consuming combat randomness is open and load-bearing, and it should be settled early, because the answer determines whether stage 2 is available at all.

Stage 3, masked PPO. Proximal policy optimization with legality masking is the workhorse for discrete game actions and the method behind every comparable result in our corpus. The mask must be applied when sampling and again when recomputing log-probabilities, or the ratio is not one at the current iterate and the clipping window is miscentered.

Implementation. A single-file CleanRL-style implementation rather than a framework, which is what the one shipped comparable system used, and which keeps device placement under our control on Apple silicon. `sb3-contrib`'s `MaskablePPO` is the fallback if that proves inconvenient.

Opponent mixture. Training against a single opponent produces agents that lose to simple strategies. Train against a mixture of the engine AI's configurations from the start.

Not chosen, and why. Value-based methods such as masked DQN variants are viable and were used by the comparable system, but policy-gradient methods have the stronger evidence base for masked discrete spaces and compose more naturally with the imitation stages. Planning methods such as MCTS and MuZero remain attractive because the simulator is fast and deterministic, and the environment deliberately keeps that door open, but they are not the first thing to try. Self-play leagues are premature until a policy exists that is worth playing against.

## Decision, part two: reward design

Open by intent. What is fixed here is the criteria and the candidates, so the eventual choice is made against a standard rather than by whichever shaping happened to train first.

### The candidates

Sparse terminal reward. Plus one for a win, minus one for a loss, zero otherwise. Unbiased, in the sense that it encodes exactly the objective and nothing else, and hardest to learn from because the signal arrives once per episode. Battles here are 5 to 40 decisions, which is short enough that sparse reward is genuinely viable, unlike in long-horizon domains where it is hopeless.

Terminal reward weighted by margin. The win or loss, scaled by surviving force, so that winning while preserving an army scores better than winning pyrrhically. Closer to what a player actually wants, since a battle is one episode inside a campaign and the surviving army carries forward. Still terminal, so it keeps the sparse signal's honesty while carrying more information.

Shaped per-decision reward. Damage dealt minus damage taken, or the change in an army-strength estimate, delivered every decision. Learns fastest and is the most dangerous, because it teaches the proxy rather than the objective. A shaped reward that rewards damage will trade a stack to deal damage when retreating was correct.

Potential-based shaping. Shaping expressed as the difference of a potential function over states, which provably leaves the optimal policy unchanged (Ng, Harada and Russell, 1999) while still densifying the signal. The principled version of the previous option, and the right form to use if shaping is used at all.

### The criteria for choosing

The choice is made against these, in order.

1. The objective must survive. Any shaping must be potential-based or demonstrably not change the optimal policy. A proxy that changes what the agent is optimizing for is rejected regardless of how fast it learns.
2. Terminal-first, shaped only if needed. Start with the margin-weighted terminal reward. Add shaping only if learning demonstrably stalls, and report both.
3. It must compose with the campaign. A battle reward that does not reflect the surviving army's value in the wider game will teach behavior that is locally optimal and globally wrong. This is the strongest argument for the margin-weighted variant over the pure win-loss signal.
4. It must be defined over recorded state. Anything the reward needs must already be in the terminal record, or the record changes first.

### What must be settled alongside it

Truncation against termination. When an episode ends because a round limit was reached rather than because the battle ended, the value of the final state has to be bootstrapped rather than treated as zero, or returns are biased low. The environment must tell the learner which case occurred, and the protocol in Milestone 4 has to carry that distinction. This is the most common environment-side reinforcement-learning bug and it is cheap to prevent now.

The discount. $\gamma$ belongs to the objective alongside the reward and is deferred with it. Short episodes make a value near 1 defensible.

The initial-state distribution $\rho_0$. A win rate is a statement about the scenario and army generator as much as about the policy. That generator is currently five fixed fixtures used as regression anchors, which is not a training distribution. Defining it is a prerequisite for any reported result meaning anything, and it is presently the largest undocumented modeling choice in the project.

## Consequences

Milestone 4's protocol must carry the termination reason and enough terminal state to compute any of the candidate rewards, which it already plans to.

The reward remains absent from the environment. It belongs to the training configuration under ADR 0003, so changing it is a configuration change with a recorded hash rather than a code change.

Stage 2 depends on an unanswered engine question, so the DAgger feasibility check should happen during Milestone 4 while the protocol is being built, not after.

## What this record does not decide

It says nothing about network architecture, which is a separate choice waiting on the observation modalities, and nothing about the adventure-map agent, whose reward problem is substantially harder and is scoped in [[../roadmap]].
