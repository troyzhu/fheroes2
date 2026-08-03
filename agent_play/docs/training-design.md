---
title: Training design
type: design
updated: 2026-07-30
related_concepts: ["[[notation]]", "[[decisions/0005-training-and-reward]]", "[[rl-and-the-battle-domain]]", "[[implementation/legal-actions-and-masking]]"]
tags: [agent-env, training, design]
---

# Training design

Techniques named here are defined in [[rl-methods]], which derives the chain from the policy gradient to PPO and surveys every alternative. This document says how a battle policy is actually fitted: what the network consumes and emits, what loss each stage minimizes, which algorithm optimizes it, which hyperparameters it starts from, and what the alternatives were at each choice. [[decisions/0005-training-and-reward]] records the decisions; this is the reasoning and the mechanics behind them. Nothing here is implemented yet, so every number is a starting point to be measured rather than a tuned result.

## Table of contents
- [[#The learning problem]]
- [[#Notation]]
- [[#The policy network]]
- [[#Stage 1, behavior cloning]]
- [[#Stage 2, DAgger]]
- [[#Stage 3, masked PPO]]
- [[#Why this order]]
- [[#Open questions before any of this runs]]

## The learning problem

A policy maps an observation to a distribution over the 793 canonical actions, of which typically five to thirty are legal. Two different objectives fit that same network in sequence. Imitation fits it to reproduce a teacher's choices, which is a supervised classification problem. Reinforcement learning then fits it to maximize return, which is not.

The teacher is the engine's own `AI::BattlePlanner`, described in [[implementation/teacher-coverage-and-behavior-cloning]]. It plays both sides of every headless battle, so demonstration data costs only the time to run episodes.

## Notation

These follow the contract in [[notation]], which matches the owner's RL wiki and lists what this project adds on top of it.

| Symbol | Meaning |
|---|---|
| $o$ | Observation, the serialized battle state the policy receives |
| $a \in \{0, \ldots, 792\}$ | Canonical action index |
| $m(o) \in \{0,1\}^{793}$ | Legality mask for the state behind $o$ |
| $\ell_\theta(o) \in \mathbb{R}^{793}$ | Network logits |
| $\pi_\theta(a \mid o)$ | Masked policy, $\operatorname{softmax}$ over $\ell_\theta$ with illegal entries set to $-10^8$ |
| $V_\phi(o)$ | Critic, sharing a trunk with the actor, so $\theta$ and $\phi$ overlap |
| $\pi^{*}$ | The teacher policy |
| $\mathcal{D}$ | Dataset of observation and action pairs |
| $\hat A_t$ | Advantage estimate at step $t$, targeting $A^\pi$, computed by GAE |
| $\rho_t(\theta)$ | Probability ratio $\pi_\theta(a_t \mid o_t) \,/\, \pi_{\theta_{\text{old}}}(a_t \mid o_t)$ |
| $T$ | Episode length in decisions, 5 to 40 here |
| $\epsilon$ | Per-decision error rate of a fitted policy |

## The policy network

The observation has two parts, so the encoder does too. Entity records are ten fixed slots of per-stack fields, and the optional plane tensor is $11 \times 9 \times C$ over the board. Both are described in [[implementation/observation-design]].

### The encoder, smallest first

The starting architecture is deliberately small, because the domain is small and because a large network on 116 recorded decisions would memorize rather than generalize.

Each of the ten entity slots is embedded by a shared multilayer perceptron, roughly two layers of 128 units, applied per slot. Sharing weights across slots enforces the permutation structure that matters here: a stack's meaning comes from its fields, not from which slot it happens to occupy. The slot embeddings are then pooled, by concatenation initially because ten slots is few enough that order can be fixed and learned, with mean or attention pooling as the fallback if concatenation proves brittle to slot permutation.

Scalar features that belong to neither side, such as the round number and whose turn it is, pass through a small multilayer perceptron and are concatenated to the pooled embedding.

The plane tensor, when enabled, goes through a small convolutional stack of two or three layers with 32 to 64 channels and no downsampling, since $11 \times 9$ is already tiny. Its flattened output concatenates with the rest.

A shared trunk of one or two layers of 256 units produces the representation. From it, a linear head emits 793 logits, and a second linear head emits a scalar value estimate used only in stage 3.

Total parameters land around 200,000 to 500,000. For comparison, the microRTS agent that reached state of the art in this genre used a network of similar order, and the entity-transformer work that matched a vision baseline on Procgen did so with roughly fifty times fewer parameters than the baseline.

> [!note]- Why the hex board does not get a plain square convolution without care
> The board is $11 \times 9$ but its adjacency is hexagonal, six neighbors rather than eight, with row-offset geometry. A square $3 \times 3$ kernel therefore covers a receptive field that does not match the game's adjacency, including two cells that are not neighbors and missing none. That is not fatal, since the network can learn to discount the spurious positions, but it is a reason to treat the plane modality as an experiment rather than an obvious win, and a reason the entity encoder is the baseline rather than the convolutional one.

### Alternatives considered

| Encoder | Argument for | Argument against | Verdict |
|---|---|---|---|
| Shared per-slot multilayer perceptron with pooling | Matches the data's structure, small, fast | Fixed slot count, no interaction modeling between stacks | Baseline |
| Transformer over entity tokens | Models stack interactions directly, handles variable counts | Overkill at ten entities, more to tune | Upgrade path if interactions matter |
| Convolution over planes | Spatial structure explicit, strong precedent in grid games | Hex mismatch above, redundant with entities | Experiment, second |
| Flat multilayer perceptron over concatenated raw fields | Simplest possible | Discards permutation structure, wastes capacity | Ablation control only |
| Graph network over cells | Encodes reach and threat exactly | Heaviest to build and debug | Only if the simpler encoders plateau |

## Stage 1, behavior cloning

### What it is

Behavior cloning treats imitation as supervised classification. Each teacher decision gives a training pair, an observation and the action the teacher chose, and the network is fitted to predict that action. It ignores that the data are sequential and that the policy's own mistakes change which states it later sees, which is exactly the weakness stage 2 addresses.

### The loss

Maximum likelihood over teacher actions, which for a categorical policy is cross-entropy:

$$\mathcal{L}_{\text{BC}}(\theta) = -\frac{1}{|\mathcal{D}|} \sum_{(o, a^{*}) \in \mathcal{D}} \log \pi_\theta(a^{*} \mid o)$$

The detail that matters is that $\pi_\theta(a \mid o)$ is the masked policy, not a raw softmax over 793 outputs. The mask is applied before the log, so the normalization runs over the legal set alone and the loss never asks the network to suppress actions that were already impossible. Training against an unmasked softmax would spend most of the gradient signal teaching legality that the environment already supplies, and would produce a network whose probabilities are wrong the moment masking is switched on at evaluation.

> [!derivation]- Why masked cross-entropy is the right likelihood, and what it implies about the ceiling
> The teacher's action is always legal, so $m_{a^{*}}(o) = 1$ and the masked likelihood is well defined. Writing $\mathcal{A}(o)$ for the legal set,
> $$\pi_\theta(a^{*} \mid o) = \frac{\exp \ell_{a^{*}}}{\sum_{a' \in \mathcal{A}(o)} \exp \ell_{a'}}$$
> so minimizing $\mathcal{L}_{\text{BC}}$ is maximum likelihood over a categorical distribution whose support is exactly the legal set. The minimum of $\mathcal{L}_{\text{BC}}$ is the conditional entropy $H(a^{*} \mid o)$ of the teacher's own policy given the observation. Two consequences follow. If the teacher is deterministic given full state but our observation drops something it conditions on, that entropy is strictly positive and the loss cannot reach zero however large the network. Residual loss should therefore not be read as underfitting without first checking what the teacher sees and the student does not, which for `full_v1` fields such as `engine_strength` is a real gap.

### Training algorithm and starting hyperparameters

AdamW, because it is the default that works and decoupled weight decay is the better-behaved regularizer for small networks on small data.

| Hyperparameter | Start | Reasoning |
|---|---|---|
| Optimizer | AdamW | Standard; decoupled weight decay |
| Learning rate | $3 \times 10^{-4}$ | The usual Adam starting point for networks this size |
| Schedule | Cosine decay to $10^{-5}$ | Cheap, few epochs, avoids a late-training plateau |
| Weight decay | $10^{-2}$ | The data are small, so regularization matters more than usual |
| Batch size | 256 decisions | Large enough for stable gradients, small enough to be many steps per epoch |
| Epochs | Until validation loss stops improving, patience 5 | The dataset size is not yet known, so a fixed epoch count would be arbitrary |
| Gradient clipping | Global norm 0.5 | Carried over from the PPO stage for consistency |
| Validation split | By scenario, not by decision | Splitting by decision leaks, because decisions within one episode are highly correlated |

That last row is the one most easily got wrong. Two decisions from the same battle share almost all of their state, so a random decision-level split reports a validation accuracy that is close to training accuracy and means nothing. The split has to be at the level of whole scenarios or whole seeds.

### Data, and the size problem

The current recorded corpus is 116 decisions across five fixtures, which is a regression anchor rather than a training set. Cloning needs orders of magnitude more, and getting it costs only compute, since the environment runs at roughly 4,600 episodes per second and the teacher plays itself.

The real constraint is diversity rather than volume. Ten thousand episodes of the same five fixtures would give a large dataset covering a tiny region of state space. The scenario generator, currently undefined and flagged in [[decisions/0005-training-and-reward]] as the largest open modeling choice, is what determines whether the data are worth collecting.

### What success looks like

Two metrics, measuring different things. Top-1 agreement with the teacher on held-out scenarios says whether the fit worked. Win rate against the teacher says whether the fit is useful, and it is the one that matters. A cloned policy that agrees with the teacher 90% of the time can still lose consistently, because the 10% may be concentrated in the decisions that decide battles.

### Alternatives considered

| Approach | Argument for | Argument against | Verdict |
|---|---|---|---|
| Behavior cloning (chosen) | Cheapest path to competence; teacher already free | Compounding error off-distribution; bounded by teacher | Stage 1 |
| Train from scratch with PPO | No teacher bias; can exceed the teacher | Far more samples; wastes an available expert | Rejected as a first step, retained as an ablation |
| Offline reinforcement learning on teacher data | Can exceed the teacher without new interaction | Needs reward, which is deferred; harder to tune | Reconsider once a reward exists |
| Inverse reinforcement learning | Recovers the teacher's implicit objective | Much heavier; the reward here is ours to choose, not to recover | Rejected |

## Stage 2, DAgger

### What it is, concretely

DAgger stands for Dataset Aggregation, and the name is the algorithm. Plain cloning trains on states the teacher visits, but the trained student visits its own states, where it was never taught. DAgger closes that loop by collecting the student's own states and asking the teacher to label them.

Each iteration $i$ does four things. It trains a policy $\hat\pi_i$ on the accumulated dataset. It rolls out a mixture policy $\pi_i = \beta_i \pi^{*} + (1 - \beta_i)\hat\pi_i$, which follows the teacher with probability $\beta_i$ and the student otherwise. It records the observations that rollout visited and asks the teacher what it would have done at each, giving new pairs $\mathcal{D}_i$. It then sets $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_i$ and repeats.

The mixing coefficient starts near 1 and decays to 0, typically $\beta_i = p^{i}$ with $p$ around 0.5, so early iterations stay near the teacher's distribution and later ones train on the student's own. The aggregation is the essential part: training only on the newest batch each round oscillates, while training on the union is what makes the procedure a no-regret online learning algorithm.

### Why it is worth the extra machinery

Cloning error compounds. A policy with per-decision error $\epsilon$ that makes a mistake early lands in a state unlike anything it trained on, where its error is higher, and the damage accumulates over the episode. The classical result is that plain cloning has regret growing as $O(\epsilon T^{2})$ in the horizon, while DAgger achieves $O(\epsilon T)$ (Ross, Gordon and Bagnell, 2011).

With $T$ between 5 and 40 here, that quadratic term is far less punishing than in the long-horizon settings where DAgger was developed. The honest expectation is that DAgger helps but is not transformative at battle scope, and that it becomes important at adventure-map scope where $T$ is in the thousands.

### The precondition, which is open

DAgger requires querying the teacher at arbitrary states the student produced. Whether `AI::BattlePlanner` can be asked "what would you do here" without advancing the arena or consuming combat randomness is unresolved, and it is a genuine engineering question rather than a formality, because the arena is a process singleton and the legality helpers read process-global state. If the answer is no, the fallback is to replay a student trajectory into a fresh arena and query at the divergence point, which costs an episode replay per label.

## Stage 3, masked PPO

### The objective

PPO maximizes a clipped surrogate that keeps each update near the policy that collected the data:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( \rho_t(\theta)\, \hat A_t, \ \operatorname{clip}\big(\rho_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon\big)\, \hat A_t \right) \right]$$

with the full loss adding a value term and an entropy bonus:

$$L(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_v\, \hat{\mathbb{E}}_t\big[(V_\phi(o_t) - \hat G_t)^2\big] - c_e\, \hat{\mathbb{E}}_t\big[H(\pi_\theta(\cdot \mid o_t))\big]$$

Advantages come from generalized advantage estimation, which trades bias against variance through $\lambda$ and is derived in [[rl-methods#GAE]].

Two integration details decide whether this works. The mask must be applied when sampling and again when recomputing log-probabilities during the update, or $\rho_t \neq 1$ at $\theta = \theta_{\text{old}}$ and the clipping window is centered on the wrong point. And the entropy bonus must be computed over the legal set only, since entropy over 793 outputs where 760 are masked is dominated by the mask rather than by the policy's actual indecision.

### Pre-fitting the critic on teacher play

Owner proposal, 2026-08-03. Stage 1 trains a policy head and leaves the value head at initialization, which wastes something the setup gives away for free.

The teacher plays both sides of every headless battle, so a rollout yields an observation sequence and a terminal outcome without any learner involved. Regressing a value head on the realized returns of those rollouts is Monte Carlo policy evaluation, a supervised problem, and it fits $V^{\pi^{*}}$, the value of the teacher's policy against a teacher opponent.

The reason this is worth doing here rather than being a generic nicety is an asymmetry in data. The policy head is limited by how many teacher decisions have been recorded, currently 116, and getting more means recording more. The value head needs only observation and return pairs, and the environment produces those at roughly 4,600 episodes per second from a seed. Value data is effectively unlimited where policy data is not.

It also removes a weakness at the moment it is most dangerous. Early stage-3 updates are worst when the critic is uninformative, because the advantage is then mostly noise, and that is exactly when the cloned policy's competence is most at risk of being destroyed. See [[rlhf-transfer#A reference policy this project does have, and an argument against using it]] for the other half of that problem.

The prerequisite is a reward, since a value is defined relative to one. That is open in [[decisions/0005-training-and-reward]], but not blocking, because the terminal record already carries the winner and the surviving force, so returns for the margin-weighted terminal candidate can be computed retroactively without re-running anything.

The policy mismatch does not break it, which is the part worth being clear about. $V^{\pi^{*}}$ is not $V^{\pi_\theta}$, and after stage 3 begins the learner drifts away from the teacher. A baseline only has to be independent of the action to leave the gradient unbiased, by the control-variate argument in [[rl-methods#Baselines, and why variance is the real enemy]], and $V^{\pi^{*}}(o)$ is a function of the observation alone. Using it is therefore unbiased however far the learner has drifted. It buys less variance reduction than the correct critic would, nothing worse. Bias enters only through bootstrapping, meaning through $\lambda < 1$, which is the trade that parameter already controls.

Three caveats worth recording.

The value is conditioned on the opponent. Fitted on teacher against teacher, it estimates returns against one opponent, while [[decisions/0005-training-and-reward]] requires training against a mixture of engine configurations. That is acceptable for an initialization and wrong for a frozen baseline, so the value loss must keep training it.

The observation profile has to match the actor's. Fitting on `full_v1` while the actor consumes `observable_v1` reintroduces exactly the asymmetric-critic bias that [[decisions/0001-observation-profiles]] declines to recommend.

Near the start of an episode the fitted value is dominated by the army matchup rather than by tactics, since two identical policies resolve an opening position mostly according to who has the stronger force. That is useful rather than a defect, because subtracting it removes scenario difficulty from the advantage, which is the same variance the leave-one-out baseline in [[rlhf-transfer#Critic-free baselines, which this project should seriously consider]] removes by sampling instead of by regression. The two are alternatives addressing one problem, and pre-fitting is the cheaper of them to try first.

### Starting hyperparameters

These are the community defaults, adjusted for a small fast environment. Every one is a starting point.

| Hyperparameter | Start | Reasoning |
|---|---|---|
| Learning rate | $3 \times 10^{-4}$, annealed | PPO's usual value |
| Clip range $\varepsilon$ | 0.2 | Standard; lower it if the policy moves too fast after cloning |
| Discount $\gamma$ | 0.99 | Episodes are short, so near-undiscounted is defensible |
| GAE $\lambda$ | 0.95 | Standard bias-variance point |
| Rollout length | 256 decisions per worker | Several episodes per rollout at this episode length |
| Parallel workers | 4 | The measured process-scaling sweet spot on the target machine |
| Update epochs | 4 | Standard; fewer if the ratio drifts far from 1 |
| Minibatches | 4 | Standard |
| Value coefficient $c_v$ | 0.5 | Standard |
| Entropy coefficient $c_e$ | 0.01, decayed | Starting from a cloned policy, less exploration pressure is needed |
| Gradient clipping | Global norm 0.5 | Standard |
| Initialization | Cloned weights for the actor, value head pre-fitted per below | The point of stage 1, extended to the critic |

The interaction worth watching is between the cloned initialization and the entropy bonus. A well-cloned policy is confident, so a standard entropy coefficient will actively push it back toward uniformity over the legal set and can undo stage 1 in the first few updates. Starting lower and decaying, or warming up the policy loss, is the expected adjustment.

### Alternatives considered

| Algorithm | Argument for | Argument against | Verdict |
|---|---|---|---|
| Masked PPO (chosen) | Strongest evidence base for masked discrete actions; composes with cloning | On-policy, so sample-hungry | Stage 3 |
| Masked DQN or QR-DQN | Off-policy, replays data, used by the comparable shipped system | Weaker fit with an imitation warm start; harder with large action spaces | Reconsider if sample cost dominates |
| IMPALA or other distributed actor-critic | Scales across many workers | Built for scales we do not have; four workers is not distributed | Rejected |
| MuZero or MCTS-based planning | Exploits a fast deterministic simulator; strong where branching is modest | Much heavier; the arena singleton blocks cheap state copying today | Deliberately kept open, not first |
| Self-play league | Generates its own curriculum | Premature before a policy worth playing exists; risks cyclic strategies | Later |

## Why this order

Cloning first because the teacher is free, competent, and already recorded, and because a policy that starts from competence needs far less exploration than one starting from noise. DAgger second because it fixes cloning's specific failure at low cost once the query precondition is settled. Reinforcement learning third because it is the only stage that can exceed the teacher, and because it is the most expensive, so it should start from the best initialization available.

The staging also fails gracefully. If stage 3 never beats the teacher, stage 1 still produced a working policy and the environment is still validated. If the DAgger precondition turns out to be unsatisfiable, stages 1 and 3 remain intact.

## Open questions before any of this runs

The scenario generator is undefined, and it determines both the training distribution and what any reported win rate means. It is the first thing to settle.

Whether the teacher can be queried at arbitrary states decides whether stage 2 exists.

Whether the plane modality helps at $11 \times 9$ is unmeasured anywhere, and the hex adjacency mismatch gives a concrete reason to doubt it. This is a cheap in-house ablation once cloning runs.

Learner throughput on Apple silicon is unmeasured at these model sizes. The environment produces roughly 4,600 episodes per second, and whether the learner keeps up decides whether four workers is the right number.

The reward is not chosen. [[decisions/0005-training-and-reward]] fixes the candidates and the criteria, and stage 3 cannot start without it.

## Related

- [[decisions/0005-training-and-reward]], the decision record this document expands.
- [[implementation/teacher-coverage-and-behavior-cloning]], the teacher and the coverage measurement.
- [[implementation/legal-actions-and-masking]], the mask this design depends on throughout.
- [[research/findings]], the evidence behind the algorithm choice.
