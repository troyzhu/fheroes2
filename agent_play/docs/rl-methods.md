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
  - [[#One shape, many algorithms]], the frame the rest of Part 1 hangs on
  - [[#Behavior cloning is this same update]], why imitation and PPO are one update
- [[#Part 2, the alternatives landscape]]
- [[#Part 3, partial observability]]
- [[#Part 4, reward shaping]]
- [[#Part 5, architecture techniques]]
- [[#What we chose, in one table]]

## Part 1, from the objective to PPO

### Read this part as Monte Carlo estimation

Part 1 is one estimation problem wearing a series of different names. The quantity of interest is $\nabla_\theta J$, it cannot be computed, so it is estimated from samples, and every technique after the first exists to cut the variance of that estimate. Reading it that way is faster than reading it as a sequence of algorithms, because the statistical content is already familiar and only the vocabulary is new.

| Reinforcement learning calls it | Which is |
|---|---|
| the policy gradient, REINFORCE | the score-function estimator, $\nabla_\theta \mathbb{E}[f] = \mathbb{E}[f \nabla_\theta \log p_\theta]$ |
| a baseline $b(s)$ | a control variate with known mean zero |
| using $V^\pi$ as the baseline | picking the control variate most correlated with the return |
| the advantage $A^\pi$ | what is left after that control variate is subtracted |
| bootstrapping | substituting an estimate for the sampled target, trading bias for variance |
| GAE's $\lambda$ | a one-parameter family of estimators indexed by that trade |
| the PPO ratio $\rho_t(\theta)$ | an importance weight |
| the PPO clip | truncated importance sampling |
| a leave-one-out baseline | the jackknife |
| group-relative normalization | studentizing the estimator |

Two consequences of that framing get used later. The variance of a control-variate estimator falls by a factor $1 - \rho_{X,C}^2$ at the optimal coefficient, which makes the choice of baseline a question about correlation with the return rather than about accuracy. And an importance-weighted estimator has finite variance only when the proposal covers the target, which is what the clip is really enforcing.

### The objective

A policy $\pi_\theta$ is scored by the expected return from the initial-state distribution. Battles terminate, so this is the episodic objective rather than an average-reward one, and the start distribution does not depend on the policy.

$$J(\theta) = \mathbb{E}_{s_0 \sim \rho_0}\big[V^\pi(s_0)\big] = \mathbb{E}_{s_0 \sim \rho_0,\ \tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T-1} \gamma^{t} r_{t+1}\right]$$

Everything in Part 1 is machinery for estimating $\nabla_\theta J$ from sampled play, because that gradient cannot be computed directly. The environment's dynamics appear in the expectation and are unknown to the learner.

### One shape, many algorithms

Every policy-gradient method in Part 1 is the same update with a different scoring term.

$$\Delta\theta \propto \Psi_t\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

The gradient factor points in the direction that makes $a_t$ more likely. The scalar $\Psi_t$ decides whether to go that way and how hard. Everything that follows differs only in what gets substituted for $\Psi_t$.

$$\Psi_t \in \Big\{\ \underbrace{G_t}_{\text{REINFORCE}},\quad \underbrace{G_t - b(s_t)}_{\text{with baseline}},\quad Q^\pi(s_t, a_t),\quad \underbrace{A^\pi(s_t, a_t)}_{\text{lowest variance}},\quad \underbrace{r_{t+1} + \gamma V(s_{t+1}) - V(s_t)}_{\text{TD residual}}\ \Big\}$$

Keeping this frame in view is what makes the rest of Part 1 a sequence of substitutions rather than a list of unrelated algorithms. It also explains something about the staging in [[decisions/0005-training-and-reward]] that is otherwise easy to miss, and [[#Behavior cloning is this same update]] returns to it.

### The policy gradient and REINFORCE

The policy gradient theorem says the gradient can be written as an expectation over trajectories the current policy already generates, which is what makes it estimable by sampling.

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t\right]$$

REINFORCE is the estimator obtained by replacing the expectation with sampled episodes. Its meaning is direct. Increase the log-probability of actions that preceded high return, decrease it for actions that preceded low return, in proportion.

The state-distribution form of the same theorem, $\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi,\, a \sim \pi_\theta}\big[\nabla_\theta \log \pi_\theta(a \mid s)\, Q^\pi(s, a)\big]$, is the one your `policy-gradient-theorem` note states. The trajectory form above is that expression with $G_t$ substituted for $Q^\pi(s_t, a_t)$, which is legitimate because $G_t$ is an unbiased sample of it, and with the sum over $d^\pi$ absorbed into sampling whole episodes.

> [!derivation]- Where the log comes from, and why the dynamics vanish
> The trajectory probability factorizes into terms the policy controls and terms it does not.
> $$p_\theta(\tau) = \rho_0(s_0) \prod_t \pi_\theta(a_t \mid s_t)\, P(s_{t+1} \mid s_t, a_t)$$
> Differentiating $J = \int p_\theta(\tau)\, G(\tau)\, d\tau$ needs $\nabla_\theta p_\theta(\tau)$, and the identity $\nabla_\theta p = p \nabla_\theta \log p$ turns it back into an expectation. Here $G(\tau)$ is the return of a whole trajectory, written $G$ rather than $R$ because $R$ is the per-step reward function.
> $$\nabla_\theta J = \int p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)\, G(\tau)\, \mathrm{d}\tau = \mathbb{E}\!\left[\nabla_\theta \log p_\theta(\tau)\, G(\tau)\right]$$
> Taking the log of the factorization turns the product into a sum, and every term that does not depend on $\theta$, meaning $\rho_0$ and every transition factor $P(s_{t+1} \mid s_t, a_t)$, differentiates to zero.
> $$\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$
> That is why a model of the environment is not required. This same argument is what licenses action masking, since a mask that depends on the state alone is another factor independent of $\theta$.

### Baselines, and why variance is the real enemy

REINFORCE is unbiased and nearly unusable on its own, because its variance is enormous. Subtracting any function of the state leaves it unbiased while reducing that variance. This is the control-variate construction: $b(s)$ correlates with the return, its contribution to the gradient has known mean zero, and subtracting it therefore moves variance without moving the estimand.

$$\nabla_\theta J = \mathbb{E}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\big(G_t - b(s_t)\big)\right]$$

> [!derivation]- Why subtracting a state-dependent baseline changes nothing in expectation
> The cross term vanishes because probabilities sum to one.
> $$\mathbb{E}_{a \sim \pi}\!\left[\nabla_\theta \log \pi_\theta(a \mid s)\, b(s)\right] = b(s) \sum_a \pi_\theta(a \mid s)\, \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)} = b(s)\, \nabla_\theta \sum_a \pi_\theta(a \mid s) = b(s)\, \nabla_\theta 1 = 0$$
> The condition is that $b$ must not depend on the action. A baseline that peeks at $a$ introduces bias, which is exactly the trap in the asymmetric-critic case discussed in Part 3.

The baseline used in practice is the state value $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$, the expected return from a state. Using it produces the advantage.

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

Read as a control variate this is the natural choice rather than an arbitrary one. Variance falls by $1 - \rho^2$ where $\rho$ is the correlation between the control variate and the quantity being averaged, so the best baseline is whichever available function of the state correlates most with the return, and the expected return from that state is the obvious candidate.

One caveat, since it is easy to overstate. Standard control-variate theory also scales the correction by $\beta^{*} = \operatorname{Cov}(X, C)/\operatorname{Var}(C)$, and policy gradient fixes $\beta = 1$. That is why the state value is not exactly the variance-minimizing baseline, which is $Q^\pi$ weighted by $\lVert \nabla_\theta \log \pi_\theta \rVert^2$. It wins on practicality instead, because a critic is being learned anyway and the true optimum would need a second estimator. Your `advantage-function` note makes the same point with the word approximately.

### Behavior cloning is this same update

Supervised imitation is not a different family. Maximizing the log-likelihood of a teacher's action is the update above with $\Psi_t \equiv +1$ and the states drawn from the teacher's distribution rather than the policy's.

$$\nabla_\theta\, \mathbb{E}_{(o, a^{*}) \sim \mathcal{D}}\big[\log \pi_\theta(a^{*} \mid o)\big] \;=\; \text{policy gradient with } \Psi_t \equiv +1,\ \ o \sim \mathcal{D}$$

Two differences hide in that line and they are exactly the two things stage 3 adds. A constant positive $\Psi_t$ can only push probability up, so cloning has no mechanism to push a bad action down. The baseline is what creates a negative signal and therefore any contrast at all. And the states come from a fixed dataset rather than from the policy's own play, which is the covariate shift that DAgger addresses. Seeing cloning and PPO as one update with two knobs changed is the cleanest way to understand why [[training-design]] runs them in sequence on one network.

The advantage answers the question the gradient actually needs. Not "was this outcome good" but "was this action better or worse than what this policy usually does here". An action followed by a return of 10 in a state worth 12 should be discouraged, and raw return cannot express that.

### Actor-critic

Learning $V^\pi$ alongside $\pi$ gives actor-critic. The actor is the policy, updated by the gradient above. The critic is an approximator $v(s, w)$ whose only job is to reduce the actor's variance, and it is discarded at deployment, which is what makes privileged critics possible at all.

### Bootstrapping, and the truncation bug

Estimating $V^\pi$ from complete episodes is Monte Carlo, unbiased and high variance. Estimating it from one step plus the current estimate of the next state is temporal-difference learning, biased and much lower variance.

$$V_\phi(s_t) \leftarrow r_{t+1} + \gamma\, V_\phi(s_{t+1})$$

That substitution of an estimate into its own target is bootstrapping. It creates a distinction the environment must respect, and it is the most common environment-side bug in reinforcement learning.

When an episode ends because the task genuinely finished, there is no future, so the target is $r_{t+1}$ alone. When an episode ends because a step limit was reached, the future still exists and was merely cut off, so the target must bootstrap $r_{t+1} + \gamma\, V_\phi(s_T)$. Treating a truncation as a termination tells the learner that the world ends at the limit, which biases every value estimate downward. This is why [[decisions/0005-training-and-reward]] requires the protocol to report which case occurred, rather than only that the episode is over.

### GAE

Generalized advantage estimation interpolates between the two extremes with a single knob. The starting point is the one-step temporal-difference residual, which is on its own the cheapest advantage estimate and the reason a single critic network suffices for A2C.

$$\delta_t = r_{t+1} + \gamma\, V_\phi(s_{t+1}) - V_\phi(s_t)$$

Generalized advantage estimation replaces that single residual with an exponentially weighted sum over the residuals that follow it.

$$\hat A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^{l}\, \delta_{t+l}$$

Written that way it costs $O(T^2)$ to fill in every $\hat A_t$. Pulling the $l = 0$ term out gives a backward recursion that does the whole episode in one $O(T)$ pass, and this is the form an implementation actually uses.

$$\hat A_t = \delta_t + \gamma\lambda\, \hat A_{t+1}, \qquad \hat A_{T-1} = \delta_{T-1}$$

At $\lambda = 0$ it is the single residual, low variance and biased by whatever the critic gets wrong. At $\lambda = 1$ it telescopes to the Monte Carlo advantage, unbiased and high variance. Values near 0.95 are the usual compromise.

> [!derivation]- Why $\lambda = 1$ recovers Monte Carlo
> Expanding the sum and cancelling adjacent value terms telescopes it.
> $$\sum_{l \ge 0} \gamma^{l} \delta_{t+l} = \sum_{l \ge 0} \gamma^{l}\big(r_{t+l+1} + \gamma\, V_\phi(s_{t+l+1}) - V_\phi(s_{t+l})\big) = G_t - V_\phi(s_t)$$
> Every intermediate value term appears once positively and once negatively, so only $-V_\phi(s_t)$ survives, which is the empirical advantage.

### Why the step must be constrained

A policy-gradient step that is too large is not merely inefficient, it is destructive. The gradient is valid only near the current policy, because the data were collected under it. Move far and the collected data no longer describe the policy being updated, so performance can collapse and there is no mechanism to recover, since the next batch is collected by the damaged policy.

Trust-region policy optimization enforces this with an explicit constraint on the KL divergence between old and new policy. It works and is heavy, requiring second-order machinery.

PPO achieves a similar effect with a first-order trick. Define the importance ratio between new and old policy. It is written $\rho_t(\theta)$ here, following the PPO paper and your `ppo-clip` note, so the letter $r$ is overloaded against the per-step reward. Context disambiguates, since the ratio always carries a $(\theta)$.

$$\rho_t(\theta) = \frac{\pi_\theta(a_t \mid o_t)}{\pi_{\theta_{\text{old}}}(a_t \mid o_t)}$$

and optimize the clipped surrogate.

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\Big(\rho_t(\theta)\,\hat A_t,\ \operatorname{clip}\big(\rho_t(\theta), 1-\varepsilon, 1+\varepsilon\big)\,\hat A_t\Big)\right]$$

Stated in estimation terms, the surrogate is an importance-weighted estimate of the objective under $\pi_\theta$ computed from samples drawn under $\pi_{\theta_{\text{old}}}$, and the clip is a truncation of the importance weight. The pathology it guards against is the standard one. An importance-weighted estimator has finite variance only when the proposal covers the target, and its worst case is governed by $\sup_a \pi_\theta(a \mid o) / \pi_{\theta_{\text{old}}}(a \mid o)$, the same supremum that serves as the envelope constant $M$ in rejection sampling. PPO bounds that ratio by construction rather than estimating it.

The clip removes the incentive to move the ratio beyond $1 \pm \varepsilon$. The $\min$ makes the objective pessimistic, so an update that would overshoot gains nothing while an update that would make things worse is still penalized in full. The policy can drift no further than the data support without the optimizer noticing it has stopped improving.

> [!derivation]- Why the surrogate is a policy gradient at all, and which branch the min selects
> The unclipped surrogate is $J^{\text{surr}}(\theta) = \mathbb{E}_{\pi_{\theta_{\text{old}}}}[\rho_t(\theta) \hat A_t]$. Using $\nabla_\theta \rho_t = \rho_t \nabla_\theta \log \pi_\theta$ and $\rho_t = 1$ at $\theta = \theta_{\text{old}}$,
> $$\nabla_\theta J^{\text{surr}}\big|_{\theta = \theta_{\text{old}}} = \mathbb{E}_{\pi_{\theta_{\text{old}}}}\big[\nabla_\theta \log \pi_\theta(a_t \mid o_t)\, \hat A_t\big]\Big|_{\theta_{\text{old}}}$$
> which is exactly the policy gradient. The first step on the surrogate is therefore a true gradient step, and the clip only starts to bite once $\theta$ has moved and $\rho_t \neq 1$. That identity is the whole reason the ratio has to equal one at the start of an epoch, and it is what masking breaks if applied inconsistently.
>
> Which branch the $\min$ picks, and where the gradient vanishes:
>
> | | $\rho_t < 1 - \varepsilon$ | $1 - \varepsilon \le \rho_t \le 1 + \varepsilon$ | $\rho_t > 1 + \varepsilon$ |
> |---|---|---|---|
> | $\hat A_t > 0$ | $\rho_t \hat A_t$ | $\rho_t \hat A_t$ | $(1+\varepsilon)\hat A_t$, gradient 0 |
> | $\hat A_t < 0$ | $(1-\varepsilon)\hat A_t$, gradient 0 | $\rho_t \hat A_t$ | $\rho_t \hat A_t$ |
>
> The two clipped cells are constant in $\theta$, so they contribute no gradient: a good action already pushed far enough up, or a bad one already pushed far enough down, stops contributing. The other two extreme cells stay live, so an action pushed the wrong way still gets full gradient to correct it. The clip is one-sided in effect rather than a symmetric brake.

There is a sharper reading of what the clip is doing, and it matters here. The total-variation divergence between the behavior and current policies at a state satisfies

$$D_{\mathrm{TV}}\big(\pi_{\theta_{\text{old}}}(\cdot \mid o_t) \,\|\, \pi_\theta(\cdot \mid o_t)\big) = \tfrac{1}{2}\,\mathbb{E}_{a_t \sim \pi_{\theta_{\text{old}}}}\big[\lvert \rho_t(\theta) - 1 \rvert\big]$$

so the condition $\lvert \rho_t - 1 \rvert \le \varepsilon$ constrains a one-sample estimate of $2 D_{\mathrm{TV}}$, evaluated at whichever action happened to be drawn. TRPO constrained the divergence itself, and PPO constrains a single draw from the random variable whose mean is that divergence. The substitution introduces a systematic bias, since equal ratios move very unequal amounts of probability mass, and [[rlhf-transfer#The ratio is a one-sample estimate of a divergence you can afford to compute]] works out what it costs and what this project can do about it.

Three consequences bear directly on our design. The ratio must equal 1 at $\theta = \theta_{\text{old}}$, which is why masking has to be applied both when sampling and when recomputing log-probabilities. PPO-clip carries no KL term, so any KL an implementation reports is a diagnostic, typically used for early stopping. And the two KL-shaped quantities that appear around PPO are not the same object and should never be conflated: the trust region is $\pi_\theta$ against $\pi_{\theta_{\text{old}}}$ and is enforced here by the clip, while the leash to a fixed reference policy $\pi_{\text{ref}}$ is a separate regularizer that RLHF adds and that this project does not have, since there is no pretrained reference to stay near.

### The entropy bonus

Added to the loss with a small coefficient, entropy $H(\pi(\cdot \mid o))$ rewards keeping the policy uncertain, which slows premature collapse onto one action before the value estimates are trustworthy. In a masked setting it must be computed over the legal set alone, otherwise it measures the mask rather than the policy's indecision.

## Part 2, the alternatives landscape

### Value-based methods

Rather than parameterizing the policy, learn $Q_\phi(s, a)$ and act greedily. DQN made this work with deep networks through a replay buffer and a slowly updated target network, both of which exist to stabilize a bootstrapped target that would otherwise chase itself. QR-DQN extends it to learn a distribution over returns rather than a mean, which is more informative under heavy randomness.

The appeal is sample efficiency, since a replay buffer reuses old data indefinitely. Two drawbacks apply here, and the first is the one that decided [[decisions/0005-training-and-reward]].

Value-based methods compose awkwardly with an imitation warm start. The claim is often compressed to "a cloned policy is not an action-value function", which is true but hides where the difficulty actually sits, so it is worth separating by part of the network.

The encoder transfers. Whatever representation cloning learned about which stacks matter and how the board is laid out is a function of the observation, and a value network can be initialized from it. Nothing is lost there.

The head does not, and the reason is a specific degeneracy. A softmax head is shift-invariant within a state, so adding any constant $c(o)$ to every logit at $o$ leaves $\pi_\theta(a \mid o)$ unchanged. Cloning therefore never constrains the level of the logits, only their differences inside a state. Action values are not shift-invariant, since $Q^\pi$ has to satisfy a Bellman equation that ties levels across states. The cloned head is undetermined in exactly the dimension a value method needs.

Even the within-state differences are the wrong quantity. They encode how often the teacher picks an action, which is a behavioral frequency, not how much return it earns. Those coincide only if the demonstrator is Boltzmann-rational with respect to our reward at a known temperature, and `AI::BattlePlanner` is a hand-written heuristic planner with no such guarantee. In this project the gap is starker still, because no reward is defined yet, so there is no return for the logits to be calibrated against.

Contrast the policy-gradient path, where the continuation is exact rather than approximate. The cloned network is the object PPO keeps optimizing, with the same parameters and the same functional form, and at $\theta = \theta_{\text{old}}$ the importance ratio is exactly 1, so the first update is a true policy-gradient step by the identity in [[#Why the step must be constrained]]. No conversion, no recalibration, no lost stage.

None of this makes the combination impossible, only different from a warm start. Deep Q-learning from demonstrations (Hester et al., AAAI 2018) does train a value function from demonstration data, but not by reusing a cloned policy. It fits $Q$ directly on the demonstrations with a large-margin classification term that forces the demonstrated action's value above every alternative by a margin, alongside the usual one-step and $n$-step temporal-difference losses. That is a viable route, and it means designing the imitation stage as value pre-training from the start rather than reusing a policy trained for a different purpose.

The second drawback is milder. Large discrete action spaces make the greedy maximization and the bootstrapped target more expensive, since every target needs a maximum over the next state's legal set.

The one shipped comparable system implemented both families and reported the policy-gradient variants as its production path.

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
