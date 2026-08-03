---
title: "ADR 0005 — Training algorithm and reward design"
type: adr
status: partially accepted
updated: 2026-07-30
related_concepts: ["[[../rl-and-the-battle-domain]]", "[[../implementation/teacher-coverage-and-behavior-cloning]]"]
tags: [adr, training, reward, agent-env]
---

# ADR 0005 — Training algorithm and reward design

- Status: algorithm choice accepted, reward design deliberately open with the decision criteria fixed here
- Implementation: nothing here is built. No learner exists in this repository by design, and Milestone 2 has recorded 116 teacher decisions across the fixture set, which is the dataset stage 1 would start from. Everything below binds the training milestones, not the current code.
- Evidence: [[../research/findings]], [[../research/works/alphastar]], [[../implementation/teacher-coverage-and-behavior-cloning]], user question 2026-07-30
- Techniques: [[../rl-methods]] defines every method named below, with its equation and our verdict.
- Mechanics: [[../training-design]] carries the architecture, losses, hyperparameters, and the full alternatives analysis. This record states the decisions and their reasons only.
- Transfer: [[../rlhf-transfer]] works out what the language-model reinforcement-learning literature contributes here, and supplies evidence for two things this record left open.

## Context

Milestones 1 through 3 built an environment and said almost nothing about what would train on it. That was intentional while the substrate was unproven, but it left two questions unanswered that a reader is entitled to ask, and this record answers the first and scopes the second.

The environment ships with no reward function at all. Phase 1a records the terminal outcome and the surviving force and stops there. That is a real design choice rather than an omission, and the reasoning is that a reward encodes what you want the agent to do, which is a modeling decision that should not be frozen while the thing underneath it is still being verified. The cost of deferring is nothing, since the recorded terminal state is sufficient to define most candidate rewards after the fact.

Where this record names a technique without teaching it, [[../training-design]] teaches it. That document gives the network architecture, the cross-entropy objective cloning minimizes and why it is masked, the DAgger iteration written out with its mixing schedule and aggregation step, the PPO surrogate with its masking integration, starting hyperparameter tables for both stages, and a compared-alternatives table at every choice point.

## The sub-problem

How does a policy get from random initialization to competent play, given a competent scripted teacher that plays both sides of every episode for free?

The unusual feature of this problem is the teacher. `AI::BattlePlanner` is not a weak baseline; it is the opponent the agent will be measured against, and Milestone 2 already records every decision it makes. That makes the question less "which reinforcement-learning algorithm" and more "how much can be taken from the teacher before reinforcement learning is needed at all, and what does the handover look like".

This record answers that. It does not choose the network architecture, which waits on the observation modalities, and it does not decide the reward, which is part two.

## Decision, part one: the training algorithm

Accepted. The staging is imitation first, then masked policy-gradient reinforcement learning against a mixture of opponents.

### Options considered

| Option | What it is | For | Against |
|---|---|---|---|
| Reinforcement learning from scratch | Ignore the teacher, learn from reward alone | No dependence on teacher quality, and no ceiling at the teacher's level | Discards a free competent demonstrator, and early exploration in a 793-slot space with a sparse terminal reward is the expensive part |
| Imitation only | Clone the teacher and stop | Cheapest competent policy, fully supervised, needs no reward at all | Cannot exceed the teacher, and compounds error on states the teacher never visited |
| Imitation then policy gradient (chosen) | Clone, then improve by masked PPO against an opponent mixture | Skips the expensive exploration phase, and retains the ability to exceed the teacher | Two stages to build, and the handover can destroy what cloning bought |
| Imitation then value-based improvement | Clone, then masked DQN or a distributional variant | Sample efficient through replay, and used by the one shipped comparable system | The cloned encoder transfers but the head does not, since softmax logits are shift-invariant within a state while action values are not, and they encode teacher frequency rather than return. Using demonstrations for a value function needs a margin loss rather than a warm start. [[../rl-methods#Value-based methods]] |
| Planning, MCTS or a learned model | Search using the simulator | The simulator is fast and seed-reproducible, which is exactly what search wants | Heavy to build, and premature before any policy exists. The environment deliberately keeps the door open |
| Self-play league | Populations of agents playing each other | The answer at the top of this genre | Needs a policy worth playing against, which does not yet exist |

Terms used above are defined in [[../rl-methods]], which gives each one its equation and a verdict.

Stage 1, behavior cloning from the built-in AI. The teacher is `AI::BattlePlanner`, it plays both sides of every headless battle, and Milestone 2 already records its decisions. Supervised learning on those decisions is the cheapest route to competent play, and it is validated at the strongest scale available, where AlphaStar's supervised stage reached 87% against its game's strongest built-in opponent before any reinforcement learning. Two qualifications carry over from the review of that evidence. The target is $\pi(a \mid o)$, conditioned on what the deployed policy will actually receive, not on full state. And the analogy is imperfect, because there the demonstrator and the opponent were different agents while here they are the same, so cloning can at best approach the teacher.

Stage 2, DAgger-style correction. DAgger stands for dataset aggregation, and it exists to fix one specific failure of plain cloning. A cloned policy is trained on states the teacher visits, but once deployed it visits its own states, and its first mistake lands it somewhere the training data never covered, where its error is larger, which produces the next mistake. DAgger closes that loop by rolling out the student, asking the teacher what it would have done at each state the student actually reached, adding those answers to the dataset, and retraining on the union. The gain is not marginal: cloning error compounds as $O(\epsilon T^2)$ in the horizon while DAgger achieves $O(\epsilon T)$ (Ross, Gordon and Bagnell, 2011), where $\epsilon$ is the per-decision error rate. [[../training-design#Stage 2, DAgger]] writes out the iteration with its mixing schedule and aggregation step.

The precondition is an expert answerable at arbitrary student-visited states, which is a stronger requirement than an expert that plays. Whether `AI::BattlePlanner` can be queried without advancing the arena or consuming combat randomness is open and load-bearing, and it should be settled early, because the answer determines whether stage 2 is available at all.

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

Potential-based shaping. A shaping term of the restricted form $F(s, a, s') = \gamma \Phi(s') - \Phi(s)$ for some potential $\Phi$ over states. Because that form telescopes along any trajectory, it adds a constant depending only on the start state, so it cannot change which policy is optimal (Ng, Harada and Russell, 1999) while still delivering a signal at every decision. [[../rl-methods#Part 4, reward shaping]] carries the derivation. This is the principled version of the previous option, and the right form to use if shaping is used at all.

### The criteria for choosing

The choice is made against these, in order.

1. The objective must survive. Any shaping must be potential-based or demonstrably not change the optimal policy. A proxy that changes what the agent is optimizing for is rejected regardless of how fast it learns.
2. Terminal-first, shaped only if needed. Start with the margin-weighted terminal reward. Add shaping only if learning demonstrably stalls, and report both. This now has outside support rather than resting on the short horizon alone. A win or loss computed by the engine is a verifiable reward in the sense of [[../rlhf-transfer#The chapter that transfers almost completely]], and the strongest recent results in that regime train on sparse binary outcomes rather than on step-level process rewards. Reporting both means plotting the shaped proxy and the win rate against divergence from the cloned checkpoint, not against training steps.
3. It must compose with the campaign. A battle reward that does not reflect the surviving army's value in the wider game will teach behavior that is locally optimal and globally wrong. This is the strongest argument for the margin-weighted variant over the pure win-loss signal.
4. It must be defined over recorded state. Anything the reward needs must already be in the terminal record, or the record changes first.

### What must be settled alongside it

Truncation against termination. When an episode ends because a round limit was reached rather than because the battle ended, the value of the final state has to be bootstrapped rather than treated as zero, or returns are biased low. The environment must tell the learner which case occurred, and the protocol in Milestone 4 has to carry that distinction. This is the most common environment-side reinforcement-learning bug and it is cheap to prevent now.

The discount. $\gamma$ belongs to the objective alongside the reward and is deferred with it. Short episodes make a value near 1 defensible.

The initial-state distribution $\rho_0$. A win rate is a statement about the scenario and army generator as much as about the policy. That generator is currently five fixed fixtures used as regression anchors, which is not a training distribution. Defining it is a prerequisite for any reported result meaning anything, and it is presently the largest undocumented modeling choice in the project.

It now has an acceptance criterion, which it did not when this record was written. A generated scenario carries gradient only when the policy neither always wins nor always loses it, and under a group or leave-one-out baseline that is an identity rather than a heuristic, since equal returns across a group make every advantage in it zero. Published practice in the verifiable-reward setting filters training problems to roughly a 20 to 80 percent solve rate for this reason, and [[../rlhf-transfer#Difficulty filtering, and what it settles]] carries the argument. Two things follow. The generator needs a difficulty control whose effect is measured as a win rate over sampled scenarios rather than asserted from army sizes. And the target band moves as the policy improves, which makes this a curriculum rather than a fixed distribution.

A held-out seed set is a separate obligation. The five fixtures are used continuously during development, so a win rate reported on them is a number that development has been optimizing against. Evaluation seeds have to be fixed in advance and excluded from training before any headline number is quoted.

## Consequences

Milestone 4's protocol must carry the termination reason and enough terminal state to compute any of the candidate rewards, which it already plans to.

The reward remains absent from the environment. It belongs to the training configuration under ADR 0003, so changing it is a configuration change with a recorded hash rather than a code change.

Stage 2 depends on an unanswered engine question, so the DAgger feasibility check should happen during Milestone 4 while the protocol is being built, not after.

## What this record does not decide

It says nothing about network architecture, which is a separate choice waiting on the observation modalities, and nothing about the adventure-map agent, whose reward problem is substantially harder and is scoped in [[../roadmap]].
