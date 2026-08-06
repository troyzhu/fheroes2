---
title: Training design
type: design
updated: 2026-07-30
related_concepts: ["[[../overview#Notation]]", "[[../decisions/0005-training-and-reward]]", "[[rl-and-the-battle-domain]]", "[[../implementation/legal-actions-and-masking]]"]
tags: [agent-env, training, design]
---

# Training design

Techniques named here are defined in [[rl-methods]], which derives the chain from the policy gradient to PPO and surveys every alternative. This document says how a battle policy is actually fitted: what the network consumes and emits, what loss each stage minimizes, which algorithm optimizes it, which hyperparameters it starts from, and what the alternatives were at each choice. [[../decisions/0005-training-and-reward]] records the decisions; this is the reasoning and the mechanics behind them. Nothing here is implemented yet, so every number is a starting point to be measured rather than a tuned result.

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

A policy maps an observation to a distribution over the 793 canonical actions (one skip, 99 move targets and 99 ranged targets for the board's $11 \times 9 = 99$ cells, and $99 \cdot 6 = 594$ melee cell-direction pairs; [[../decisions/0002-action-space]] fixes the layout), of which typically five to thirty are legal. Two different objectives fit that same network in sequence. Imitation fits it to reproduce a teacher's choices, which is a supervised classification problem. Reinforcement learning then fits it to maximize return, which is not.

The teacher is the engine's own `AI::BattlePlanner`, described in [[../implementation/teacher-coverage-and-behavior-cloning]]. It plays both sides of every headless battle, so demonstration data costs only the time to run episodes.

## Notation

These follow the contract in [[../overview#Notation]], which matches the owner's RL wiki and lists what this project adds on top of it.

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

The observation has two parts, so the encoder does too. Entity records are ten fixed slots of per-stack fields, and the optional plane tensor is $11 \times 9 \times C$ over the board. Both are described in [[../implementation/observation-design]].

### The encoder, smallest first

The starting architecture is deliberately small, because the domain is small and because a large network on 116 recorded decisions would memorize rather than generalize.

Each of the ten entity slots is embedded by a shared multilayer perceptron, roughly two layers of 128 units, applied per slot. Sharing weights across slots enforces the permutation structure that matters here: a stack's meaning comes from its fields, not from which slot it happens to occupy. The slot embeddings are then pooled, by concatenation initially because ten slots is few enough that order can be fixed and learned, with mean or attention pooling as the fallback if concatenation proves brittle to slot permutation.

Scalar features that belong to neither side, such as the round number and whose turn it is, pass through a small multilayer perceptron and are concatenated to the pooled embedding.

The plane tensor, when enabled, goes through a small convolutional stack of two or three layers with 32 to 64 channels and no downsampling, since $11 \times 9$ is already tiny. Its flattened output concatenates with the rest.

A shared trunk of one or two layers of 256 units produces the representation. From it, a linear head emits 793 logits, and a second linear head emits a scalar value estimate used only in stage 3.

Total parameters land around 200,000 to 500,000. For comparison, the [[../research/works/microrts-py|microRTS agent]] that reached state of the art in this genre used a network of similar order, and the [[../research/works/entity-based-rl|entity-transformer work]] that matched a vision baseline on Procgen did so with roughly fifty times fewer parameters than the baseline.

### As built, 2026-08-05

The implemented network (`python/fheroes2_agent/policy.py`) sits inside the envelope above: the shared slot encoder is two layers of 96, the four global features pass through one layer of 32, the ten slot embeddings concatenate with it into a 992-wide vector, and the trunk is two layers of 192 feeding the 793-logit policy head and the scalar value head, 396,570 parameters in total. The input is the 634-wide `obs_encoding_v3` vector, ten slots of 63 named per-stack features plus four globals; `FEATURE_NAMES` in `python/fheroes2_agent/encoding.py` is deliberately the single authoritative layout, and [[../decisions/0006-encoding-count-scaling]] fixes the count scaling. Capacity was measured rather than assumed: cloning agreement still rises monotonically from half to double width, so the clone is data-limited rather than capacity-saturated, while reinforcement learning at the current batch size gets slightly worse at double width; [[../archive/experiments/2026-08-04-flip-and-collapse]] carries the run.

Pooling is concatenation, as preferred above at ten slots, and it is now the live suspect rather than a settled choice. On the fully diverse commander-and-horde pool the held-out transfer gain vanished, $+0.007 \pm 0.046$ against $+0.173 \pm 0.039$ on the training split, and the count-extrapolation ablation lost a third of its agreement above the training range under every encoding, so [[../archive/experiments/2026-08-05-diversity-and-encoding]] names the pooling-against-concatenation axis the natural next experiment ahead of any width increase.

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

The real constraint is diversity rather than volume. Ten thousand episodes of the same five fixtures would give a large dataset covering a tiny region of state space. The scenario generator, currently undefined and flagged in [[../decisions/0005-training-and-reward]] as the largest open modeling choice, is what determines whether the data are worth collecting.

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

### The precondition, resolved 2026-08-05

DAgger requires querying the teacher at arbitrary states the student produced, and the answer is yes: the planner can be asked "what would you do here" without perturbing the battle. The code says why (no `Rand::` call site anywhere in the planner, analysis members recomputed from the arena on every call, and the pathfinder cache it warms is the one the action-space enumeration already warms under digest-proven gates), and `planner_query.py` says so empirically: 100 paired episodes across every fixture, all three controlled sides, world seeds, and 20 commander-and-wide pool matchups, with terminal state digests identical whether or not every controlled decision also ran the query. The probe costs about 19 percent wall time and resolved 4,297 of 4,297 teacher choices into `simple_v1`, which is the DAgger label rate.

The mechanism is in place: `--probe-teacher` on the worker emits `teacher_action` per decision through `ExternalDecisionController`'s probe of the planner's public `queryUnitTurn`. Two scope limits. The verdict covers the current scenario space, where commanders carry no spellbook, so the spell-planning path never runs and would need re-verification before spellcasting heroes enter. And the probed planner is the same agent that plays the opponent, so DAgger labels inherit every teacher blind spot that cloning already inherits.

## Stage 3, masked PPO

### The objective

PPO maximizes a clipped surrogate that keeps each update near the policy that collected the data:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( \rho_t(\theta)\, \hat A_t, \ \operatorname{clip}\big(\rho_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon\big)\, \hat A_t \right) \right]$$

with the full loss adding a value term and an entropy bonus:

$$L(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_v\, \hat{\mathbb{E}}_t\big[(V_\phi(o_t) - \hat G_t)^2\big] - c_e\, \hat{\mathbb{E}}_t\big[H(\pi_\theta(\cdot \mid o_t))\big]$$

Advantages come from generalized advantage estimation, which trades bias against variance through $\lambda$ and is derived in [[rl-methods#GAE]].

Two integration details decide whether this works. The mask must be applied when sampling and again when recomputing log-probabilities during the update, or $\rho_t \neq 1$ at $\theta = \theta_{\text{old}}$ and the clipping window is centered on the wrong point. And the entropy bonus must be computed over the legal set only, since entropy over 793 outputs where 760 are masked is dominated by the mask rather than by the policy's actual indecision.

### Pre-fitting the critic on teacher play

Owner proposal, 2026-08-03, now built as `python/fheroes2_agent/train_critic.py` and measured below. Stage 1 trains a policy head and leaves the value head at initialization, which wastes something the setup gives away for free.

The teacher plays both sides of every headless battle, so a rollout yields an observation sequence and a terminal outcome without any learner involved. Regressing a value head on the realized returns of those rollouts is Monte Carlo policy evaluation, a supervised problem, and it fits $V^{\pi^{*}}$, the value of the teacher's policy against a teacher opponent.

The original argument for it was an asymmetry in data, and building it showed that argument to be wrong. The claim was that policy data is limited by how many teacher decisions have been recorded while value data is effectively unlimited. Both come from the same recorded rollouts: the 2,000 episodes below yield 45,380 teacher actions and 45,380 returns, the same rows read two ways, so there is no asymmetry to exploit at this stage. A genuine one appears only once stage 3 begins, since returns can then be computed from the learner's own rollouts while teacher labels cannot. What actually justifies pre-fitting is the measurement below rather than the argument that motivated it.

It also removes a weakness at the moment it is most dangerous. Early stage-3 updates are worst when the critic is uninformative, because the advantage is then mostly noise, and that is exactly when the cloned policy's competence is most at risk of being destroyed. See [[rlhf-transfer#A reference policy this project does have, and an argument against using it]] for the other half of that problem.

The prerequisite is a reward, since a value is defined relative to one. That is open in [[../decisions/0005-training-and-reward]], but not blocking, because the terminal record already carries the winner and the surviving force, so returns for the margin-weighted terminal candidate can be computed retroactively without re-running anything.

The policy mismatch does not break it, which is the part worth being clear about. $V^{\pi^{*}}$ is not $V^{\pi_\theta}$, and after stage 3 begins the learner drifts away from the teacher. A baseline only has to be independent of the action to leave the gradient unbiased, by the control-variate argument in [[rl-methods#Baselines, and why variance is the real enemy]], and $V^{\pi^{*}}(o)$ is a function of the observation alone. Using it is therefore unbiased however far the learner has drifted. It buys less variance reduction than the correct critic would, nothing worse. Bias enters only through bootstrapping, meaning through $\lambda < 1$, which is the trade that parameter already controls.

Three caveats worth recording.

The value is conditioned on the opponent. Fitted on teacher against teacher, it estimates returns against one opponent, while [[../decisions/0005-training-and-reward]] requires training against a mixture of engine configurations. That is acceptable for an initialization and wrong for a frozen baseline, so the value loss must keep training it.

The observation profile has to match the actor's. Fitting on `full_v1` while the actor consumes `observable_v1` reintroduces exactly the asymmetric-critic bias that [[../decisions/0001-observation-profiles]] declines to recommend.

Near the start of an episode the fitted value is dominated by the army matchup rather than by tactics, since two identical policies resolve an opening position mostly according to who has the stronger force. That is useful rather than a defect, because subtracting it removes scenario difficulty from the advantage, which is the same variance the leave-one-out baseline in [[rlhf-transfer#Critic-free baselines, which this project should seriously consider]] removes by sampling instead of by regression. The two are alternatives addressing one problem, and pre-fitting is the cheaper of them to try first. [[scenario-distribution#Three mechanisms, one target]] sets both beside difficulty filtering and shows all three removing the same variance term.

#### What it measures

Full numbers in [[../archive/experiments/2026-08-03-training-runs#Stage 2b, the critic pre-fitted on teacher play]]. On the 2,000 recorded episodes, explained variance on held-out battles moves from $-3.061$ to $+0.835$. The negative figure is not a formality: an untrained head emits near zero while returns average $+0.489$, so it is worse than predicting the mean, and PPO currently spends its first ten iterations with a critic that is actively misleading.

The value head is frozen against the rest of the network, which was not obvious in advance. Fitting end to end reaches $+0.946$ and drags teacher agreement from 0.887 to 0.701, because the trunk is shared and nothing in the value objective preserves the features the policy head reads. That is catastrophic forgetting inside one training run, and it destroys the warm start stage 3 exists to protect.

Freezing has a cost worth stating. A 193-parameter head over a 192-wide trunk is linear regression on the cloned policy's features, so the representation caps the fit: on identical data a trunk cloned to 0.606 agreement reaches $+0.489$ where one cloned to 0.887 reaches $+0.841$. Improving the clone improves the critic for free, and no quantity of value data substitutes for it.

It does not measurably help stage 3, which is worth stating plainly because the argument for it was persuasive and the measurement disagrees. Over 60 cold and 35 pre-fitted seeds the paired difference in final win rate is $+0.033 \pm 0.027$, and both arms solve the training matchup every time. At twenty seeds the pre-fitted arm looked six times less variable and had avoided two collapses; forty further cold seeds produced no more collapses, which put the rate at 2 in 60 and made 0 in 35 exactly what chance predicts. The collapses had a cause of their own, described next.

A matchup every run solves cannot show which run solved it better, so that is evidence about the matchup as much as about the critic. On the 140-matchup pool, which does not saturate, it does show: three paired seeds put the pre-fitted arm $+0.043 \pm 0.012$ ahead on the training matchups, with every pre-fitted seed beating every cold one, and $-0.017 \pm 0.009$ behind on the held-out ones.

That is the shape a better critic should have. A more accurate advantage on the distribution being trained on buys a policy that exploits that distribution more precisely, and nothing about it argues for transfer. Pre-fitting is therefore an optimization aid rather than a generalization aid, which is worth knowing before it is asked to carry more than that.

The value function itself does transfer, which is a separate claim and holds. Fitted on teacher play from the five Milestone 1 fixtures, its value loss on the pool's first rollout is 0.99 against a cold head's 6.17, on army pairs it never saw.

### An amplification in advantage normalization

Found by chasing those two collapses, and unrelated to the critic. Advantage normalization divides a batch by its own spread so the step size does not depend on the reward scale. Once a matchup is solved every episode scores alike, the spread collapses toward zero, and the division rescales what is left, which is value-function error, up to unit variance.

Measured on a calibrated matchup, a raw spread of 0.02 against a healthy 0.3 to 1.0, so the amplification reaches fiftyfold. Four epochs of it drove a policy from a 1.000 win rate to 0.031 within two iterations.

The repair is a floor on the divisor rather than dropping the batch. Dropping is what `train_group` already did for a group whose returns are exactly equal, and it fails twice over: the spread that does the damage is small rather than zero, so the drop never fires before the collapse, and after the collapse the spread is exactly zero again because the policy now loses every episode, so the drop does fire and blocks the updates that would recover it. A floor keeps the sign and ranking of every advantage and turns a degenerate batch into a small update. Both trainers now share it.

It is free, in the sense that can be measured. Across twenty paired seeds on the contested matchup, dropping the two that collapsed leaves eighteen where the floor is worth $-0.001 \pm 0.007$, and it fires on most iterations of those runs. Its whole effect on the mean is the collapses it prevents.

What makes a matchup vulnerable resisted three simple predicates before yielding to a geometric one. A reward spread of exactly zero is not sufficient, a sharp policy is not the discriminator, and contestedness is not either: three further contested matchups ran 36 unfloored seeds with one transient dip between them while spending 0.40 to 0.69 of their iterations in the amplification regime, which turns out to be the background condition of any solved matchup rather than a rare state. The discriminating measurement is the width of the solved region. Perturbing each trained policy with per-tensor parameter noise, the one matchup that produces terminal collapses is the one whose win rate craters at half-scale noise, 0.51 against 0.82 to 1.00 for every other, and the matchups flat at that scale have never dipped in 46 recorded runs. [[../archive/experiments/2026-08-04-flip-and-collapse]] carries the census, the autopsy of the one new dip, which shows the same near-degenerate signature as the original collapses, and the probe.

The floor's role reads differently in that light. It does not prevent knock-offs, whose rate is similar floored and unfloored on the vulnerable matchup; it shrinks the amplified steps, and every floored dip on record recovered while both unfloored terminal collapses did not. That is insurance proportional to exactly how narrow the solved region is, at a measured cost of nothing.

The same hazard sits inside GRPO's studentization, which divides each group by its own spread, and [[rlhf-transfer#Critic-free baselines, which this project should seriously consider]] already records that as the reason Dr. GRPO drops the term. That the identical failure appears in the batch normalization every variant applies afterwards was not noticed until it destroyed a run.

### Measured, 2026-08-03

Stage 1 now exists and has been run. Numbers below are from `python/fheroes2_agent/train_bc.py` on 2,000 recorded episodes.

| Quantity | Value |
|---|---|
| Training decisions | 36,819, from 1,600 episodes |
| Held-out decisions | 8,561, from 400 episodes, split by episode rather than by decision |
| Parameters | 396,570, of which the value head is 193 |
| Held-out teacher agreement | 0.887 |
| Baseline, always the most common teacher action | 0.131 |
| Baseline, uniform over the legal set | 0.079 |
| Wall clock | 13 seconds for 25 epochs on the Apple M2 |

Splitting by episode rather than by decision matters more than it looks. Consecutive decisions inside one battle are nearly the same board, so a decision-level split puts a state and its successor on opposite sides and reports an agreement that has partly memorized the answer.

The same argument goes one level further, and it took until 2026-08-05 to notice. Episodes from one matchup are the same armies replayed under different seeds, so an episode-level split still leaks matchup identity, and every agreement figure in this section is within-matchup generalization across seeds. The fixture-era numbers could not have been anything else, five fixtures shared across all episodes, and on diverse data the same clone measures near 0.86 episode-split and near 0.52 matchup-split. Quote the matchup-split number for anything that faces new armies; [[../archive/experiments/2026-08-05-diversity-and-encoding]] carries the measurement, and ADR 0006's encoding evidence is built on leak-free splits throughout.

Agreement is a ceiling rather than a target. The teacher plays both sides, so a perfect clone equals the teacher and does not beat it, and the minimum achievable loss is the teacher's conditional entropy given what the observation shows rather than zero. Held-out loss flattened near 0.38 while training loss continued to 0.28, which is the mild overfitting expected at this data scale and is why the network was sized down from 626k parameters to 393k.

The size question was re-measured on 2026-08-04 with width as the only variable, and the loss-gap reasoning above does not survive it cleanly. Agreement rises monotonically with width, 0.870 at 140k parameters, 0.887 at 397k, 0.901 at 1.27M, so whatever the loss gap showed, tripling the size costs no held-out agreement and buys some. Reinforcement learning at a 40-iteration pool budget prefers the deployed size, with the 1.27M model worse by 0.042 at about 1.8 standard errors, which is the sample-efficiency price of more parameters on the same batches rather than a ceiling. So capacity binds cloning mildly from above and reinforcement learning not at all at current budgets. [[../archive/experiments/2026-08-04-flip-and-collapse#Capacity, asked and measured]] carries the run.

The data came from replaying the five fixtures under 400 world seeds each. That varies the obstacle layout and the combat seed while holding the army matchup fixed, which is the separation [[scenario-distribution]] requires, and it produced 129 distinct teacher decision streams from the first 200 episodes, so the seeds do change the teacher's behaviour rather than merely its outcomes.

### Stage 3 measured, 2026-08-03

PPO now runs against the blocking worker, starting from the cloned weights. The result that means something and the two that do not are worth separating.

| Matchup | Cloned policy | After PPO | Reading |
|---|---|---|---|
| 5 Peasants against 5 Peasants | 0.646 | 0.979 | The real result. Measured to sit inside the difficulty band before training |
| `m1_tiny_melee`, 50 against 50 | 0.958 | 1.000 | Near the ceiling already, so it shows little |
| `m1_three_stack` | 0.000 | 0.000 | Degenerate. Every rollout scored exactly -1.000 |

The third row is the finding rather than a failure. That matchup is unwinnable, so every rollout returned the same reward, the advantage was identically zero, and fifteen iterations changed nothing. [[scenario-distribution]] predicted this on the argument that equal returns across a batch make every advantage in it zero, and here it is measured.

It also sharpens the claim that a margin-weighted reward keeps hopeless matchups informative. It does so only when the loss is partial. In this matchup the attacker is always wiped out, so the survival term is zero every time and the reward is degenerate whatever its shape. The claim holds for losses that vary in cost, not for a rout.

### The difficulty band is narrow, and mirror matchups have none

Measuring the cloned policy against every fixture put four of five above 0.95 and the fifth at zero. None is in the 20 to 80 percent band, which confirms in measurement what the records already said, that the fixtures are regression anchors rather than a training distribution.

Searching for a band by varying army sizes turned up something the design did not anticipate. In a mirror matchup the win rate is a step function of the count: 50 Peasants beat 70 defenders 96.9 percent of the time and beat 71 zero percent of the time. Damage rolls average out across fifty creatures, so the outcome is decided by arithmetic and play barely matters.

A band therefore needs matchups where variance is large relative to the mean, or where the creature types make positioning decide the fight. Five against five sits at 0.79 because a single unlucky exchange is a fifth of the army. That is the first entry in what a scenario generator has to look like, and it is an empirical constraint rather than a preference.

### Checked against the research corpus, 2026-08-03

The implementation followed the design documents, which were themselves derived from the evidence sweeps. Going back to the primary work notes afterwards found one real omission and two pieces of context that change how the numbers above should be read.

What the corpus had already settled and the implementation followed. A flat masked action space rather than a factorized one, since [[../research/works/vcmi-gym|vcmi-gym]]'s factorized variant failed to converge while its flat masked space shipped. Masking applied at both sampling and log-probability recomputation, with a large negative constant rather than an infinity. Padded entity slots rather than tokenized entity lists, which [[../research/works/entity-based-rl]] records as the choice for a first version with transformers as the upgrade path. A single-file implementation rather than a framework, which is what the one shipped comparable system used. And exposure rather than removal for morale and luck, which is where [[../decisions/0001-observation-profiles]] deliberately departs from vcmi-gym, so both appear in the observation.

The omission. The `obs_encoding_v1` feature vector had no creature identity at all, leaving the policy to infer type from attack, defense, speed and the ability flags. [[../research/works/vcmi-gym]] encodes categories explicitly as one-hot with a NULL for empty slots, which is the practice this should have followed from the start. `obs_encoding_v2` adds a one-hot over the 41 monsters the `simple_v1` allowlist supports, widening an observation from 224 to 634.

It made almost no difference: held-out teacher agreement moved from 0.8867 to 0.8873. That is the honest result and it has an explanation. The fixtures use three creature types whose stat lines already separate them, so identity was redundant information at this data scale. The change is still right, because a roster with creatures sharing a stat line and differing in something unmodelled would need it, and that is what widening past `simple_v1` will produce.

The context that reframes the numbers. vcmi-gym reports roughly five days, 2.5 million battles and $45 of GPU per model, reaching about 45 percent against its strong scripted opponent initially and about 65 percent after moving to a graph encoder. The runs recorded above are a few thousand battles. Three orders of magnitude separate them, so nothing here should be read as evidence about what this architecture reaches, only that the machinery works. Its observation is also 12,685 floats against this one's 634, and includes 165 per-hex vectors where this has no spatial channel at all, since the `planes_v1` modality of [[../decisions/0004-spatial-observation-modality]] is specified and unbuilt.

### Advantage and trust region compared, 2026-08-03

`objectives.py` separates two choices that are usually bundled into one algorithm name. The advantage is the baseline: a learned critic with GAE, a leave-one-out group mean, the full group mean studentized as group-relative optimization does it, or that mean without the studentization as the Dr. GRPO variant prefers. The trust region is what bounds the step: PPO's clip on the sampled ratio, or the divergence mask from [[../research/works/dppo-trust-region]].

Keeping them separate is what makes them comparable, since a run that changes both cannot attribute its result to either.

The divergence trust region is computed exactly here rather than approximated. That paper spends most of its methodology on binary and top-$K$ lower bounds because summing over a $10^5$-token vocabulary at every position is prohibitive. This action space is 793 slots with 5 to 30 legal after masking, so the exact total-variation distance over the legal set costs a handful of operations.

Four runs on a calibrated opening-fight matchup, six Archers and ten Peasants against 121 Peasants, twenty iterations of four groups of eight from the same cloned checkpoint and the same seed:

| Advantage | Trust region | Start | Last five | Best |
|---|---|---|---|---|
| Leave-one-out | ratio | 0.188 | 0.925 | 1.000 |
| Group-relative, studentized | ratio | 0.188 | 0.925 | 0.969 |
| Group-relative, no studentizing | ratio | 0.188 | 0.925 | 1.000 |
| Leave-one-out | divergence | 0.188 | 0.944 | 1.000 |

They are indistinguishable. At 32 episodes an iteration the standard error on a win rate is about 0.05, so the spread between 0.925 and 0.944 is well inside noise, and no ranking can be read from this. The honest conclusion is that at this scale and on this matchup the choice does not matter, and separating them would need many seeds and a task where the cloned policy does not saturate within twenty iterations.

That is a useful negative result rather than a disappointment. It says the machinery is not the bottleneck, and the scenario distribution is, which is the same conclusion [[scenario-distribution]] reached from the variance side.

### The two trust regions disagree about which updates are large

The comparison above found the advantage estimators indistinguishable, and left the trust regions apparently so too. Instrumenting both on the same run says otherwise.

On the Thunk matchup, PPO's clip fired on 7 to 14 percent of samples while the total-variation distance exceeded its threshold on 22 to 40 percent of the same samples. They are not flagging the same updates. That is the measurable form of the claim in [[../research/works/dppo-trust-region]] that the sampled ratio is a poor proxy for the divergence it stands in for, and it holds here despite the argument that the effect should be milder in a masked categorical of five to thirty actions than over a vocabulary.

What this does not show is that either is better. Both runs converged, and separating them on outcome would need the many seeds the comparison above already called for. What it shows is that the quantities differ enough to be worth distinguishing, which is the premise the comparison rests on.

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

The reward is not chosen. [[../decisions/0005-training-and-reward]] fixes the candidates and the criteria, and stage 3 cannot start without it.

## Related

- [[../decisions/0005-training-and-reward]], the decision record this document expands.
- [[../implementation/teacher-coverage-and-behavior-cloning]], the teacher and the coverage measurement.
- [[../implementation/legal-actions-and-masking]], the mask this design depends on throughout.
- [[../research/findings]], the evidence behind the algorithm choice.
