---
title: What transfers from RLHF to a battle agent
type: reference
updated: 2026-07-30
related_concepts: ["[[rl-methods]]", "[[training-design]]", "[[../decisions/0005-training-and-reward]]", "[[../overview#Notation]]"]
tags: [agent-env, rl, rlhf, reference]
---

# What transfers from RLHF to a battle agent

Most reinforcement learning written about today is applied to language models, so most of the recent engineering experience lives there. This project has no language model, no preference data, and no pretrained reference policy, which makes it fair to ask what any of that experience is worth here. The answer is that a specific and useful subset transfers, one chapter transfers almost completely, and the rest actively misleads.

The source is Nathan Lambert's *Reinforcement Learning from Human Feedback* ([rlhfbook.com](https://rlhfbook.com)), which is open source, read here in full rather than only where the owner's notes reach. Its Appendix A uses the same symbols this tree does, including $V^\pi$, $Q^\pi$, $A^\pi$, $\pi_\theta$, and $\log$, so nothing below needs translating. Techniques named here are defined in [[rl-methods]]; symbols are fixed in [[../overview#Notation]].

## Table of contents
- [[#The short version]]
- [[#The chapter that transfers almost completely]]
- [[#Critic-free baselines, which this project should seriously consider]]
- [[#The aggregation unit, and an episode-length bias we would otherwise inherit]]
- [[#Where value-network bias actually comes from]]
- [[#A reference policy this project does have, and an argument against using it]]
- [[#The ratio is a one-sample estimate of a divergence you can afford to compute]]
- [[#Overoptimization, and the measurement it demands]]
- [[#Evaluation discipline]]
- [[#What does not transfer]]

## The short version

| Idea | Transfers | Why |
|---|---|---|
| Verifiable rewards, and what follows from them | Almost completely | A win or loss is a verifiable reward in exactly the book's sense |
| Difficulty filtering of the training distribution | Yes, and it is the sharpest finding here | Answers an open question in [[../decisions/0005-training-and-reward]] about the scenario generator |
| Group and leave-one-out baselines instead of a critic | Yes, strongly | Battles are cheap and seed-repeatable, so the sampling this needs is nearly free |
| Aggregation unit, per-decision against per-episode | Yes | Battles vary 5 to 40 decisions, the same length-bias structure as variable completion length |
| The exact statement of value-network bias | Yes | Decides whether a critic fitted on very little data is safe |
| A KL leash to a reference policy | Contested, and the book argues both sides | The cloned policy is a genuine reference, but reasoning practice removed the leash to allow exploration |
| Replacing the clipped ratio with a computed divergence | Yes, and the usual cost objection is absent here | 793 slots with 5 to 30 legal makes the exact divergence nearly free |
| Overoptimization and the proxy-against-gold measurement | Yes | The shaped-reward risk in [[../decisions/0005-training-and-reward]] is the same failure |
| Evaluation variance, contamination, hillclimbing | Yes | The five fixtures cannot be both regression anchors and the reported evaluation |
| Truncated importance sampling for async training | Later | Only once actors and learners are separated, which is not the current design |
| Token-level structure, DPO, preference data, instruction tuning | No | There is no language model and no human preference signal here |

## The chapter that transfers almost completely

The book's reasoning chapter covers reinforcement learning with verifiable rewards, meaning a reward computed by a deterministic function rather than predicted by a learned model. A unit test passes or it does not. An extracted answer equals 77 or it does not.

A battle is won or it is lost, computed by the engine. This project is therefore already doing verifiable-reward reinforcement learning, and three consequences follow directly.

Overoptimization mostly stops being a concern. The failure in RLHF is that a policy pushed hard against a *learned* reward exploits that model's errors. A verifiable reward has no errors to exploit, and the book reports these domains are robust to overoptimization for exactly that reason. The residual risk here is not the win-loss signal but any *shaped* term added beside it, which is a hand-written reward model and does have errors to exploit.

Sparse binary outcome rewards are the practice, not a compromise. The book records that reasoning models train on correct-or-incorrect at the end rather than on step-level process rewards, with auxiliary terms only for format. That is independent support for the terminal-first criterion already in [[../decisions/0005-training-and-reward]], and it is worth more than the usual argument that sparse reward is merely tolerable at short horizons. The strongest recent results in a comparable regime chose it.

The third consequence is large enough to have its own section.

### Difficulty filtering, and what it settles

The book reports that filtering training problems to those the model solves between roughly 20 and 80 percent of the time is essential, because a problem solved every time and a problem never solved both produce no gradient. With a group-relative or leave-one-out baseline this is not a heuristic but an identity: if every rollout in a group earns the same return, the advantage of every member is zero and the batch contributes nothing.

[[../decisions/0005-training-and-reward]] calls the initial-state distribution the largest undocumented modeling choice in the project and leaves it open. This gives it a concrete acceptance criterion. A scenario and army generator is suitable when the policy being trained wins somewhere in a middle band, and unsuitable when it wins almost always or almost never, regardless of how realistic the matchups look.

Two things follow for the design. The generator needs a difficulty parameter that can be measured rather than asserted, and the natural measurement is the win rate of the current policy, or the teacher, over a sample of generated scenarios. And the band moves, because a scenario set that is well calibrated for a freshly cloned policy becomes too easy once training works, which makes this a curriculum rather than a fixed distribution. The five committed fixtures are regression anchors and were never a training distribution; this says what the training distribution has to satisfy.

The cost of ignoring it is specific. Sampling armies uniformly would produce many lopsided matchups, most batches would carry near-zero advantage, and the run would look like a learning-rate problem. [[scenario-distribution]] works the variance argument through, including the case where filtering is the only remedy that helps.

## Critic-free baselines, which this project should seriously consider

PPO needs a critic, and [[training-design]] plans one. The language-model world has largely moved away from that, and the reason applies here more strongly than it does there.

REINFORCE leave-one-out draws $K$ episodes from the same starting state and uses the mean return of the others as the baseline for each. It is the jackknife construction, and it inherits the jackknife's reason for excluding the held-out sample.

$$b_k = \frac{1}{K-1}\sum_{i \neq k} G^{(i)}, \qquad \hat A_k = G^{(k)} - b_k$$

Excluding the sample itself is what keeps this exactly unbiased, because $b_k$ is then independent of $a_k$ and the baseline term vanishes by the control-variate argument in [[rl-methods]]. Include it and the baseline correlates with the very sample it is correcting, which is the same finite-sample bias that appears whenever a control-variate coefficient is estimated from the data it is applied to.

Group-relative optimization does include it, taking the full group mean, and accepts the resulting $O(1/K)$ bias. It then divides by the group standard deviation, which is studentization and is contested for the reason studentization usually is. Each group gets rescaled by its own noisy spread, so a group that happens to be homogeneous has its advantages inflated. The Dr. GRPO variant drops the division.

The reason this matters here is that the objection to it does not apply. In a language model, drawing $K$ completions per prompt is the dominant cost. Here the environment runs at roughly 4,600 episodes per second and a scenario is reproducible from a seed, so drawing $K$ episodes from one starting state is close to free. Against that, the critic must be fitted on very little data, and [[training-design]] already flags 116 recorded decisions as a regime where a network memorizes.

The cost is real and should be stated. A leave-one-out baseline gives one advantage for the whole episode, so every decision receives the same credit, which is coarse when a battle turns on one decision out of thirty. That is the trade in the next section rather than an argument against trying it. The concrete recommendation is that a leave-one-out baseline belongs in the first round of experiments beside the critic, not as a fallback after the critic disappoints.

## The aggregation unit, and an episode-length bias we would otherwise inherit

Whether an advantage is computed per decision or per episode is the same choice as the Markov-decision-process against bandit framing in the language-model literature.

| | Per episode, critic-free | Per decision, critic plus GAE |
|---|---|---|
| Advantage | one scalar for the whole battle | $\hat A_t$ at each decision |
| Credit | every decision gets the same | distributed across decisions by the value function |
| Needs | $K$ episodes per start state | a fitted critic |
| Fits | short battles, weak reward signal | longer battles where one decision decides the outcome |

There is a bias hiding in the per-episode form that the language-model literature had to find the hard way, and this project would inherit it unchanged. If the loss averages per-decision terms within an episode and then averages over episodes, each decision carries weight $1/(B \lvert \tau \rvert)$, so decisions in a short battle are weighted more heavily than decisions in a long one. When episodes run from 5 to 40 decisions that is an eightfold difference in per-decision gradient weight, driven by nothing but episode length.

The fix is to normalize by a decision count rather than per episode, either the batch's actual total or a fixed constant. Dividing by a fixed maximum removes the length dependence entirely and introduces no data-dependent scale, which under an optimizer that already normalizes by a second-moment estimate makes it nearly equivalent to dividing by the batch total. The choice that matters is against per-episode averaging, not between the two fixes.

This is worth recording now because it is invisible until measured and it looks exactly like a hyperparameter problem when it bites.

## Where value-network bias actually comes from

The claim that a critic-free method avoids value-network bias is often stated loosely. The precise version decides whether a critic fitted on very little data is dangerous or merely imprecise.

A value estimate used as a pure baseline, subtracted from a Monte Carlo return, leaves the gradient unbiased no matter how bad the estimate is. That follows from the baseline argument in [[rl-methods]], which needs only that the baseline does not depend on the action. A poor critic costs variance reduction, not correctness.

Bias enters with bootstrapping. Replacing the sampled return with a target such as $r_{t+1} + \gamma V_\phi(s_{t+1})$ substitutes the network's own estimate for the truth, so $\mathbb{E}[\hat A_t] \neq A^\pi$ whenever $V_\phi$ is imperfect. That is what buys the variance reduction, and it is the same trade $\lambda$ controls in generalized advantage estimation.

For this project the reading is direct. A critic trained on very little data is safe as a plain baseline and becomes a bias risk exactly to the extent that $\lambda$ is pushed toward 0. Starting near $\lambda = 0.95$ and treating a lower value as an explicit bias-for-variance decision is the disciplined way to hold that.

## A reference policy this project does have, and an argument against using it

RLHF carries two KL-shaped quantities that [[rl-methods]] warns against conflating. The trust region between $\pi_\theta$ and $\pi_{\theta_{\text{old}}}$ is enforced by the clip. The leash to a fixed reference $\pi_{\text{ref}}$, usually the supervised-fine-tuned model, is a separate regularizer, applied either as a reward penalty or as a loss term.

$$r = r_\theta - \lambda_{\text{KL}}\, \mathcal{D}_{\text{KL}}\big(\pi_\theta(y \mid x) \,\|\, \pi_{\text{ref}}(y \mid x)\big)$$

The obvious reading is that this does not apply here, since there is no pretrained model to stay near. That reading is wrong. After stage 1 of [[../decisions/0005-training-and-reward]] there is a cloned policy that plays competently, and stage 3 can destroy that competence early while value estimates are still poor. The cloned checkpoint is a reference policy in exactly the sense meant, and the book supplies quantitative support: forgetting correlates with the KL divergence between the initial and trained policies at $R^2 = 0.96$.

The same source argues the other way, and honesty requires reporting it. As reasoning training scaled, many systems removed the KL penalty entirely, because it constrains exploration and the verifiable reward removed the failure the penalty was guarding against. This project is in the verifiable-reward regime, which puts it on the side of the argument that drops the leash.

The book also notes a third path that costs nothing. On-policy sampling is itself an implicit regularizer, because updates stay near where the policy already puts probability mass, which is why reinforcement learning forgets less than supervised fine-tuning on distant targets. Stage 3 is on-policy, so some protection is already present without any coefficient.

The resulting recommendation is narrower than the one this page carried before. Do not add a KL leash by default. Instrument the divergence from the cloned checkpoint, watch whether early stage-3 updates degrade win rate against the teacher, and add the penalty only if that degradation appears. If it does, the reward-penalty and loss-term forms are both available and the book treats the choice as minor.

## The ratio is a one-sample estimate of a divergence you can afford to compute

Everything above takes PPO's clip as given. A 2026 result from the group behind Dr. GRPO argues the clip constrains the wrong object, and the argument is worth carrying because the reason it is impractical for language models does not hold here. Full detail in [[../research/works/dppo-trust-region]].

The identity at the centre of it is that the total-variation divergence between behavior and current policy at a state is the mean absolute deviation of the ratio from one.

$$D_{\mathrm{TV}}\big(\pi_{\theta_{\text{old}}}(\cdot \mid o) \,\|\, \pi_\theta(\cdot \mid o)\big) = \tfrac{1}{2}\,\mathbb{E}_{a \sim \pi_{\theta_{\text{old}}}}\big[\lvert \rho(\theta) - 1 \rvert\big]$$

PPO's condition $\lvert \rho_t - 1 \rvert \le \varepsilon$ therefore constrains a one-sample estimate of that divergence, taken at whichever action was drawn. In the vocabulary of [[rl-methods#Read this part as Monte Carlo estimation]], PPO thresholds a single realization of a random variable when the trust region wants its mean.

The bias is systematic rather than merely noisy, because the ratio depends on the behavior probability while the divergence depends on moved mass. Their worked example: an action at $10^{-4}$ raised to $10^{-2}$ has ratio 100 and is clipped hard while moving about $10^{-2}$ of mass, and an action at $0.99$ lowered to $0.80$ has ratio $0.808$, sits inside a clip range of $0.2$, and moves $0.19$. Low-probability actions are over-penalized, which suppresses exactly the exploratory ones, and high-probability actions are under-penalized, which permits the destabilizing updates.

Their fix keeps PPO's asymmetric mask but conditions it on a computed divergence over the whole action distribution rather than on the sampled ratio, blocking an update only when it is already moving away from the trust region and the divergence exceeds a threshold. Substituting $\lvert \rho_t - 1 \rvert$ for that divergence recovers PPO, which makes the two directly comparable.

What makes this interesting here is the cost. Their methodology is mostly about approximating the divergence, because summing over a $10^5$-token vocabulary at every position is memory-prohibitive, so they build binary and top-$K$ lower bounds. This project has 793 slots with typically 5 to 30 legal after masking, so the exact divergence over the legal set is a handful of operations. The approximations that occupy most of their paper are unnecessary, and the exact form is available.

The honest caveat is magnitude. The pathology is driven by the dynamic range of behavior probabilities, which spans several orders of magnitude over a vocabulary and will be much narrower over a masked categorical of 5 to 30 actions. The effect should be milder here than they measure. What keeps it worth testing is that the remedy is nearly free at this size, and that the failure it prevents, a destabilizing early update from a confident cloned policy, is the one this project is most exposed to.

One caution against importing their framing wholesale. Their motivating instability is training-inference mismatch, where the sampling engine and the trainer disagree on probabilities from identical parameters. This project runs one process, one code path, and a seeded generator, so that does not arise. The ratio-against-divergence argument stands independently of it.

## Overoptimization, and the measurement it demands

Overoptimization is the observation that a policy pushed against a proxy improves on the proxy while getting worse at what the proxy was meant to measure. The book separates two forms: quantitative reward overoptimization, where a learned reward's score rises as held-out quality falls, and qualitative degradation, where no metric moves but behavior becomes verbose, sycophantic, or rigid.

Only the first has an analogue here, and only if shaping is used. A shaped per-decision reward is a hand-written reward model, and it fails the same way: a term rewarding damage dealt will trade a stack to deal damage when retreating was correct.

The structural finding worth importing is that overoptimization is measured against a budget, and the budget is KL divergence from the starting policy. Proxy and gold reward track each other for a while and then separate, and where they separate depends on how much KL has been spent. Methods differ in how much they spend, with online reinforcement learning spending more than inference-time selection.

That converts directly into a protocol for this project, and it is cheap. Report the win rate, which is the gold objective and is verifiable, alongside whatever shaped quantity is being optimized, and plot both against the KL divergence from the cloned checkpoint rather than against training steps. Divergence between the two curves is then visible rather than inferred, and the KL axis makes runs with different learning rates comparable. Without the shared axis, a shaped reward that has started teaching the wrong objective looks like a run that is simply training well.

Mitigations the book lists that would carry over are a larger KL penalty and reward ensembling. The one that matters most here is simpler, which is to keep any shaped term potential-based, since that is the only form with a proof that the optimal policy is unchanged.

## Evaluation discipline

The evaluation chapter is about benchmarks for language models and most of it does not apply. Three findings do, and all three bear on how a result from this project should be reported.

Evaluation noise is larger than people assume. The book cites post-training evaluations showing between 0.25 and 1.5 points of standard deviation with the setup held constant, meaning differences smaller than that reflect methodology rather than capability. The analogue here is the seed set. A win rate over a fixed set of seeds carries sampling error that should be reported with it, and a change smaller than that error is not a result.

Hillclimbing on a metric is not the same as evaluating on it. Teams improve against a target benchmark during development and cannot then use it as evidence. This project has a sharper version of the problem, because the five committed fixtures are regression anchors used continuously during development. Using them to report a trained policy's win rate would be reporting a number that development has been optimizing against. A held-out set of scenario seeds, drawn from the same generator and never inspected, has to exist before any headline number is quoted.

Contamination is the same failure at the data level. The battle analogue is training on scenarios drawn from a generator and then evaluating on seeds that generator can also produce, which is legitimate only if the evaluation seeds were fixed in advance and excluded from training. The environment already makes this cheap, since a seed is a small integer and exclusion is a set membership test.

## What does not transfer

Recording these so they are not imported by analogy.

Token-level structure. A completion is a sequence of actions under one reward, which motivates sequence-level ratios and per-token clipping. A battle decision is a genuine action in a genuine state, so the per-decision form is correct here and the machinery built to compensate for the bandit framing is unnecessary.

Discounting conventions. Language-model RLHF sets $\gamma = 1$ because the completion is scored as a whole. Battles terminate quickly enough that a value near 1 is defensible, but for a different reason, and [[../overview#Notation]] records why the range is relaxed at all.

Direct preference optimization and the direct alignment family. These replace a reward model with a closed-form objective on preference pairs. There is no preference data here and no reward model to eliminate.

Reward modeling from preferences, including the Bradley-Terry loss and its shift-invariance, and the reward normalization that invariance forces. The win-loss signal is computed, not learned, so none of it applies.

Instruction tuning, preference data collection, synthetic data pipelines, and character training. These concern getting a language model to a usable starting point and have no analogue in an environment with a scripted teacher.

Rejection sampling and best-of-N, with a caveat about the name. Two unrelated procedures share the phrase and separating them is worth a paragraph, because one of them does carry over.

What the RLHF literature calls rejection sampling generates $N$ completions, keeps the highest-scoring by a learned reward, and fine-tunes on those. There is no envelope and no exactness guarantee, so it is a selection heuristic rather than the classical method. Nothing about it transfers here: there is no learned reward to select by, and the superficially similar move of sampling several actions and picking the best by a value estimate is shallow search, which belongs with the planning methods in [[rl-methods]].

Classical rejection sampling is a different and more useful object. To draw from a target $p$ using a proposal $q$, take $M \ge \sup_x p(x)/q(x)$, sample $x \sim q$ and $u \sim \text{Uniform}(0,1)$, and accept when $u \le p(x)/(M q(x))$. Accepted draws are exact and the acceptance rate is $1/M$. That does not appear anywhere in this project's training loop either, but the constant does.

The envelope constant is the transfer. $\sup_x p(x)/q(x)$ is simultaneously the smallest valid $M$ and the bound on the importance weight $p/q$, so the same supremum that decides whether rejection sampling is efficient decides whether an importance-weighted estimator has usable variance. When it is unbounded, rejection sampling has no valid envelope and importance sampling has infinite variance; these are one failure described twice. Every ratio-bounding device in this document is a response to it. The PPO clip bounds $\pi_\theta / \pi_{\theta_{\text{old}}}$ two-sidedly by construction, and the truncated cap below bounds $\pi_\theta / \mu$ one-sidedly, accepting a known bias for finite variance. Reading them as truncations of the same supremum is the shortest route to why either is needed.

Asynchronous training with truncated importance sampling. This corrects for actors and learners on separate hardware with lagged weights. It becomes relevant only if this project separates them, which the current single-process design does not. The correction to remember if that changes is a one-sided cap on the importance weight, biased upward but bounded in variance, distinct from the two-sided PPO clip.

## A note on symbols

The book's Appendix A agrees with [[../overview#Notation]] on every symbol this tree uses, which is why nothing above needed translating. One difference is worth flagging for anyone reading the book directly. It writes the state distribution $\rho_\pi$ while also using $\rho_t$ for the PPO ratio. This tree keeps $d^\pi$ for the state distribution and $\rho_0$ for the initial-state distribution, leaving $\rho_t(\theta)$ unambiguous.

## Related

- [[rl-methods]], the techniques themselves, derived.
- [[training-design]], the architecture and hyperparameters these ideas would modify.
- [[../decisions/0005-training-and-reward]], the decisions this page proposes revisiting.
- [[../overview#Notation]], the symbol contract.
