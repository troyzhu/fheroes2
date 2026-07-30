---
title: Reinforcement learning methods, a reference
type: reference
updated: 2026-07-30
related_concepts: ["[[notation]]", "[[training-design]]", "[[rl-and-the-battle-domain]]", "[[decisions/0005-training-and-reward]]"]
tags: [agent-env, rl, reference, methods-map]
---

# Reinforcement learning methods, a reference

Every technique named anywhere in this documentation is defined here, with the equation that carries its content, the reason it exists, and our verdict on it. The documents that use these techniques link here rather than restating them, so a definition lives in one place.

Part 1 derives the chain from the objective to PPO, because those pieces build on each other and the later ones are meaningless without the earlier. Parts 2 to 5 are a landscape, readable in any order. Heavy derivations are folded; the load-bearing equation always stays visible.

## Table of contents
- [[#Part 1, from the objective to PPO]]
- [[#Part 2, the alternatives landscape]]
- [[#Part 3, partial observability]]
- [[#Part 4, reward shaping]]
- [[#Part 5, architecture techniques]]
- [[#What we chose, in one table]]

## Part 1, from the objective to PPO

### The objective

A policy $\pi(a \mid s, \theta)$ is scored by the expected return from a start distribution that does not depend on it. That metric is the book's $\bar v_\pi^{\,0}$, and [[notation]] says why it rather than either stationary-distribution metric is the right one for an episodic task.

$$J(\theta) = \mathbb{E}_{S_0 \sim d_0}\big[v_\pi(S_0)\big] = \mathbb{E}_{S_0 \sim d_0,\ \tau \sim \pi(\theta)}\!\left[\sum_{t=0}^{T-1} \gamma^{t} R_{t+1}\right]$$

Everything in Part 1 is machinery for estimating $\nabla_\theta J$ from sampled play, because that gradient cannot be computed directly. The environment's dynamics appear in the expectation and are unknown to the learner.

### The policy gradient and REINFORCE

The policy gradient theorem says the gradient can be written as an expectation over trajectories the current policy already generates, which is what makes it estimable by sampling.

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi(\theta)}\!\left[\sum_{t} \nabla_\theta \ln \pi(a_t \mid s_t, \theta)\, G_t\right]$$

REINFORCE is the estimator obtained by replacing the expectation with sampled episodes. Its meaning is direct. Increase the log-probability of actions that preceded high return, decrease it for actions that preceded low return, in proportion.

The book states the same theorem over a state distribution rather than over trajectories, as $\nabla_\theta J(\theta) = \mathbb{E}_{S \sim \eta,\, A \sim \pi(S, \theta)}\big[\nabla_\theta \ln \pi(A \mid S, \theta)\, q_\pi(S, A)\big]$ in its (9.9). The trajectory form above is that expression with $G_t$ substituted for $q_\pi(S_t, A_t)$, which is legitimate because $G_t$ is an unbiased sample of it, and with the sum over $\eta$ absorbed into sampling whole episodes.

> [!derivation]- Where the log comes from, and why the dynamics vanish
> The trajectory probability factorizes into terms the policy controls and terms it does not.
> $$p(\tau \mid \theta) = d_0(s_0) \prod_t \pi(a_t \mid s_t, \theta)\, p(s_{t+1} \mid s_t, a_t)$$
> Differentiating $J = \int p(\tau \mid \theta)\, G(\tau)\, \mathrm{d}\tau$ needs $\nabla_\theta p(\tau \mid \theta)$, and the identity $\nabla_\theta p = p \nabla_\theta \ln p$ turns it back into an expectation. Here $G(\tau)$ is the return of a whole trajectory, written with $G$ rather than $R$ because $R$ is reserved for a single step's reward.
> $$\nabla_\theta J = \int p(\tau \mid \theta)\, \nabla_\theta \ln p(\tau \mid \theta)\, G(\tau)\, \mathrm{d}\tau = \mathbb{E}\!\left[\nabla_\theta \ln p(\tau \mid \theta)\, G(\tau)\right]$$
> Taking the log of the factorization turns the product into a sum, and every term that does not depend on $\theta$, meaning $d_0$ and every transition factor $p(s_{t+1} \mid s_t, a_t)$, differentiates to zero.
> $$\nabla_\theta \ln p(\tau \mid \theta) = \sum_t \nabla_\theta \ln \pi(a_t \mid s_t, \theta)$$
> That is why a model of the environment is not required. This same argument is what licenses action masking, since a mask that depends on the state alone is another factor independent of $\theta$.

### Baselines, and why variance is the real enemy

REINFORCE is unbiased and nearly unusable on its own, because its variance is enormous. Subtracting any function of the state leaves it unbiased while reducing that variance.

$$\nabla_\theta J = \mathbb{E}\!\left[\sum_t \nabla_\theta \ln \pi(a_t \mid s_t, \theta)\,\big(G_t - b(s_t)\big)\right]$$

> [!derivation]- Why subtracting a state-dependent baseline changes nothing in expectation
> The cross term vanishes because probabilities sum to one.
> $$\mathbb{E}_{a \sim \pi}\!\left[\nabla_\theta \ln \pi(a \mid s, \theta)\, b(s)\right] = b(s) \sum_a \pi(a \mid s, \theta)\, \frac{\nabla_\theta \pi(a \mid s, \theta)}{\pi(a \mid s, \theta)} = b(s)\, \nabla_\theta \sum_a \pi(a \mid s, \theta) = b(s)\, \nabla_\theta 1 = 0$$
> The condition is that $b$ must not depend on the action. A baseline that peeks at $a$ introduces bias, which is exactly the trap in the asymmetric-critic case discussed in Part 3.

The baseline used in practice is the state value $v_\pi(s) \doteq \mathbb{E}[G_t \mid S_t = s]$, the expected return from a state. Using it produces the advantage.

$$\delta_\pi(s, a) \doteq q_\pi(s, a) - v_\pi(s)$$

Two notes on this. The state value is not quite the variance-minimizing baseline, which the book derives in its Box 10.1 as $q_\pi$ weighted by $\lVert \nabla_\theta \ln \pi \rVert^2$. The state value wins on practicality instead, because a critic is being learned anyway and the optimal baseline would need a second estimator. The advantage is then written $\delta_\pi$ rather than $A^\pi$ because the temporal-difference error two sections below is a sampled estimate of exactly this quantity, and reusing the letter keeps that link visible.

The advantage answers the question the gradient actually needs. Not "was this outcome good" but "was this action better or worse than what this policy usually does here". An action followed by a return of 10 in a state worth 12 should be discouraged, and raw return cannot express that.

### Actor-critic

Learning $v_\pi$ alongside $\pi$ gives actor-critic. The actor is the policy, updated by the gradient above. The critic is an approximator $v(s, w)$ whose only job is to reduce the actor's variance, and it is discarded at deployment, which is what makes privileged critics possible at all.

### Bootstrapping, and the truncation bug

Estimating $v_\pi$ from complete episodes is Monte Carlo, unbiased and high variance. Estimating it from one step plus the current estimate of the next state is temporal-difference learning, biased and much lower variance.

$$v(s_t, w) \leftarrow r_{t+1} + \gamma\, v(s_{t+1}, w)$$

That substitution of an estimate into its own target is bootstrapping. It creates a distinction the environment must respect, and it is the most common environment-side bug in reinforcement learning.

When an episode ends because the task genuinely finished, there is no future, so the target is $r_{t+1}$ alone. When an episode ends because a step limit was reached, the future still exists and was merely cut off, so the target must bootstrap $r_{t+1} + \gamma\, v(s_T, w)$. Treating a truncation as a termination tells the learner that the world ends at the limit, which biases every value estimate downward. This is why [[decisions/0005-training-and-reward]] requires the protocol to report which case occurred, rather than only that the episode is over.

### GAE

Generalized advantage estimation interpolates between the two extremes with a single knob. The starting point is the one-step temporal-difference residual, which is the advantage estimate the book uses in its Algorithm 10.2 and the reason a single critic network suffices there.

$$\delta_t = r_{t+1} + \gamma\, v(s_{t+1}, w) - v(s_t, w)$$

Generalized advantage estimation replaces that single residual with an exponentially weighted sum over the residuals that follow it.

$$\hat A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^{l}\, \delta_{t+l}$$

At $\lambda = 0$ it is the single residual, low variance and biased by whatever the critic gets wrong. At $\lambda = 1$ it telescopes to the Monte Carlo advantage, unbiased and high variance. Values near 0.95 are the usual compromise.

> [!derivation]- Why $\lambda = 1$ recovers Monte Carlo
> Expanding the sum and cancelling adjacent value terms telescopes it.
> $$\sum_{l \ge 0} \gamma^{l} \delta_{t+l} = \sum_{l \ge 0} \gamma^{l}\big(r_{t+l+1} + \gamma\, v(s_{t+l+1}, w) - v(s_{t+l}, w)\big) = G_t - v(s_t, w)$$
> Every intermediate value term appears once positively and once negatively, so only $-v(s_t, w)$ survives, which is the empirical advantage.

### Why the step must be constrained

A policy-gradient step that is too large is not merely inefficient, it is destructive. The gradient is valid only near the current policy, because the data were collected under it. Move far and the collected data no longer describe the policy being updated, so performance can collapse and there is no mechanism to recover, since the next batch is collected by the damaged policy.

Trust-region policy optimization enforces this with an explicit constraint on the KL divergence between old and new policy. It works and is heavy, requiring second-order machinery.

PPO achieves a similar effect with a first-order trick. Define the importance ratio between new and old policy, written $\rho_t(\theta)$ here because $r$ already means a reward. The PPO paper writes the same quantity $r_t(\theta)$.

$$\rho_t(\theta) = \frac{\pi(a_t \mid o_t, \theta)}{\pi(a_t \mid o_t, \theta_{\text{old}})}$$

and optimize the clipped surrogate.

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\Big(\rho_t(\theta)\,\hat A_t,\ \operatorname{clip}\big(\rho_t(\theta), 1-\epsilon, 1+\epsilon\big)\,\hat A_t\Big)\right]$$

The clip removes the incentive to move the ratio beyond $1 \pm \epsilon$. The $\min$ makes the objective pessimistic, so an update that would overshoot gains nothing while an update that would make things worse is still penalized in full. The policy can drift no further than the data support without the optimizer noticing it has stopped improving.

Two consequences bear directly on our design. The ratio must equal 1 at $\theta = \theta_{\text{old}}$, which is why masking has to be applied both when sampling and when recomputing log-probabilities. And PPO-clip carries no KL term, so any KL an implementation reports is a diagnostic, typically used for early stopping.

### The entropy bonus

Added to the loss with a small coefficient, entropy $H(\pi(\cdot \mid o))$ rewards keeping the policy uncertain, which slows premature collapse onto one action before the value estimates are trustworthy. In a masked setting it must be computed over the legal set alone, otherwise it measures the mask rather than the policy's indecision.

## Part 2, the alternatives landscape

### Value-based methods

Rather than parameterizing the policy, learn an action-value approximator $q(s, a, w)$ and act greedily. DQN made this work with deep networks through a replay buffer and a slowly updated target network, both of which exist to stabilize a bootstrapped target that would otherwise chase itself. QR-DQN extends it to learn a distribution over returns rather than a mean, which is more informative under heavy randomness.

The appeal is sample efficiency, since a replay buffer reuses old data indefinitely. The drawback here is that value-based methods compose awkwardly with an imitation warm start, because a cloned policy is not an action-value function, and that large discrete action spaces make the greedy maximization and the target computation more expensive. The one shipped comparable system implemented both families and reported the policy-gradient variants as its production path.

### Distributed actor-critic

IMPALA separates many acting workers from one learner, which makes the data slightly off-policy by the time it arrives. V-trace is the importance-weighting correction that fixes this, clipping weights to trade a little bias for stability. Sample Factory pursues the same goal with an emphasis on single-machine throughput.

These solve a problem we do not have. Four worker processes on one machine is not a distributed system, and the added complexity buys nothing at this scale.

### Planning and model-based methods

Monte Carlo tree search builds a search tree by simulating continuations, using the simulator rather than a learned reflex. AlphaZero combines it with a learned policy and value that guide and truncate the search. MuZero removes the requirement for a simulator by learning a latent dynamics model, which matters when no simulator exists.

Our simulator is fast and deterministic, which is precisely the setting where search is strongest, so this stays deliberately open rather than rejected. The obstacle is engineering rather than principle. Search needs cheap state copying, and the arena is a process singleton whose legality helpers read process-global state, so copying a position is not currently possible. Opening that path means a forward-model interface of the kind Stratega exposes.

### Imitation and offline methods

Behavior cloning and DAgger are covered in [[training-design]]. Two neighbors deserve naming.

Offline reinforcement learning fits a policy from a fixed dataset with no further interaction, and can in principle exceed the demonstrator by stitching together good segments of mediocre trajectories. Its central difficulty is that value estimates for actions absent from the data are unconstrained and tend to be wildly optimistic, which the modern methods address either by penalizing out-of-distribution actions or by never evaluating them. It needs a reward, which we have deferred, so it is a candidate for later rather than now.

Inverse reinforcement learning recovers the reward function a demonstrator appears to be optimizing. It is the wrong tool here, because our reward is ours to choose rather than to infer, and choosing it is a modeling decision we would rather make explicitly.

### Multi-agent training

Naive self-play trains against the current policy and can cycle, since each iteration counters the last without any strategy being genuinely better. Fictitious play trains against the historical average instead, which damps the cycling. Prioritized fictitious self-play samples past opponents by how much trouble they cause, spending compute where the policy is weak. A league adds explicitly specialized agents whose job is to punish specific weaknesses.

All of this is premature until a policy exists that is worth playing against, which is why the first reinforcement stage trains against a mixture of the engine's scripted configurations rather than against itself.

### Hyperparameter search

Population-based training runs many configurations at once and periodically copies the weights and perturbs the hyperparameters of the better performers into the worse. It tunes on a schedule rather than to a fixed value, which suits reinforcement learning where the best learning rate early differs from the best learning rate late. The comparable shipped system used it. It is worth considering only once a single run trains reliably.

### Evaluation

Elo places agents on a scalar scale from pairwise results. TrueSkill extends it with an explicit uncertainty per player, which allows scheduling matches until that uncertainty is small enough to rank confidently rather than playing a fixed number.

Both assume strength is transitive. That assumption is exactly what cyclic strategies violate, so where self-play cycling is a concern the pairwise win-rate matrix has to be reported alongside any scalar rating.

## Part 3, partial observability

The battle is close to fully observed, so none of this is needed yet. The adventure map in [[roadmap]] is genuinely fogged, so all of it becomes relevant then.

Frame stacking concatenates the last $k$ observations. It is the cheapest possible memory and handles short-range hidden state such as velocity, but nothing beyond the window.

Recurrence carries a hidden state across steps, usually with an LSTM or GRU, and in principle summarizes unbounded history. It is the standard answer and complicates training, since sequences must be preserved through the update.

A belief state is the posterior over true states given the history. It is sufficient in the formal sense, meaning a policy on the belief loses nothing, and computing or learning it is usually harder than the control problem itself.

An asymmetric or privileged critic exploits the fact that the critic is discarded at deployment, letting it read state the actor cannot. The caution from Part 1 applies with force. A critic conditioned on state that the actor cannot see correlates with the action through the unobserved component, which breaks the baseline argument and biases the gradient. The unbiased construction conditions on history as well as state. Our own evidence sweep found no verified claim supporting the naive version, so [[implementation/observation-design]] treats it as an available option rather than a default.

## Part 4, reward shaping

Shaping adds an auxiliary signal to speed learning, and the danger is that it changes what the agent is optimizing for. Potential-based shaping is the form that provably does not.

Given any potential $\Phi$ over states, the shaping term

$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$

leaves the optimal policy unchanged, for every transition function and reward.

> [!derivation]- Why potential-based shaping preserves the optimal policy
> Summing the shaping term along a trajectory telescopes.
> $$\sum_{t} \gamma^{t} F(s_t, a_t, s_{t+1}) = \sum_t \big(\gamma^{t+1}\Phi(s_{t+1}) - \gamma^{t}\Phi(s_t)\big) = \gamma^{T}\Phi(s_T) - \Phi(s_0)$$
> The return under the shaped reward is therefore the original return plus a constant determined by the start state, minus a terminal term that vanishes for absorbing states where $\Phi$ is defined as zero. Adding a state-dependent constant to every trajectory from the same start shifts all returns equally, so the ordering over policies is untouched and the argmax is preserved. Any shaping not expressible in this form has no such guarantee, which is why a damage-based bonus can and does change what the optimal policy is.

The practical reading is that shaping should either be potential-based or be treated as a change to the objective, documented as such.

## Part 5, architecture techniques

Attention computes a weighted combination of a set of items where the weights depend on the items, which is what makes it the natural encoder for a variable-length collection of units. A transformer stacks self-attention so that every entity's representation can depend on every other, capturing interactions such as threat and blocking directly.

Scatter connections are the bridge between entity and spatial representations used in AlphaStar. Per-entity embeddings are written into a spatial grid at each entity's location, so a convolutional encoder sees entity-derived features in place rather than only terrain.

A pointer network selects an element of a variable-length input set by attending over it, which is how a policy can choose among candidates whose number changes between states, and is the architectural home of a candidate-list action interface.

## What we chose, in one table

| Technique | Verdict | Where it is used or why not |
|---|---|---|
| Behavior cloning | Chosen, stage 1 | Free competent teacher; see [[training-design]] |
| DAgger | Chosen, stage 2, precondition open | Fixes compounding error; needs a queryable teacher |
| Masked PPO with GAE | Chosen, stage 3 | Strongest evidence for masked discrete actions |
| Entropy bonus over the legal set | Chosen, small and decayed | Cloned start needs less exploration pressure |
| Potential-based shaping | Permitted if shaping is needed | Only form that provably preserves the objective |
| Sparse or margin-weighted terminal reward | Candidates, undecided | See [[decisions/0005-training-and-reward]] |
| DQN and QR-DQN | Not first | Composes poorly with an imitation start |
| IMPALA, Sample Factory | Rejected | Built for scales we do not have |
| MCTS, AlphaZero, MuZero | Open, blocked | Needs a copyable state the arena singleton prevents |
| Offline reinforcement learning | Later | Requires a reward we have deferred |
| Inverse reinforcement learning | Rejected | Reward is ours to choose, not to infer |
| Self-play, fictitious play, leagues | Later | Premature before a policy worth playing exists |
| Population-based training | Later | Tune after a single run is reliable |
| Elo, TrueSkill | Adopted for evaluation | Report the pairwise matrix alongside, cycles break transitivity |
| Frame stacking, recurrence, belief states | Not needed yet | Battle is near fully observed; see [[roadmap]] |
| Asymmetric critic | Option, not default | Naive form is biased; no verified support in our sweep |
| Attention, transformers | Upgrade path | Ten entities does not require it yet |
| Scatter connections, pointer networks | Compatible, unused | Both remain reachable from the current interfaces |

## Related

- [[training-design]], how these are configured for our stages, with hyperparameters.
- [[decisions/0005-training-and-reward]], the decisions and the open reward question.
- [[rl-and-the-battle-domain]], the vocabulary Part 1 assumes.
- [[research/findings]], the verified evidence behind the choices.
