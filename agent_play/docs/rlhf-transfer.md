---
title: What transfers from RLHF to a battle agent
type: reference
updated: 2026-07-30
related_concepts: ["[[rl-methods]]", "[[training-design]]", "[[decisions/0005-training-and-reward]]", "[[notation]]"]
tags: [agent-env, rl, rlhf, reference]
---

# What transfers from RLHF to a battle agent

Most of the reinforcement learning written about today is applied to language models, so most of the recent engineering experience lives there. This project has no language model, no preference data, and no pretrained reference policy, which makes it fair to ask what any of that experience is worth here. The answer is that a specific and useful subset transfers, and the rest actively misleads.

Sources are Nathan Lambert's *Reinforcement Learning from Human Feedback* ([rlhfbook.com](https://rlhfbook.com), policy-gradient chapter) and the owner's math companion to it. Both use the notation fixed in [[notation]]. Techniques named here are defined in [[rl-methods]].

## Table of contents
- [[#The short version]]
- [[#Critic-free baselines, which this project should seriously consider]]
- [[#The aggregation unit, and an episode-length bias we would otherwise inherit]]
- [[#Where value-network bias actually comes from]]
- [[#A reference policy this project does have]]
- [[#Overoptimization, restated for a shaped battle reward]]
- [[#What does not transfer]]

## The short version

| Idea | Transfers | Why |
|---|---|---|
| Group and leave-one-out baselines instead of a critic | Yes, strongly | Battles are cheap and seed-repeatable, so the sampling this needs is nearly free here |
| Aggregation unit, per-decision against per-episode | Yes | Battles vary 5 to 40 decisions, which is the same length-bias structure as variable completion length |
| The exact statement of value-network bias | Yes | Decides whether a critic fitted on very little data is safe to use |
| A KL leash to a reference policy | Yes, after cloning | The cloned policy is a genuine reference even though no pretrained one exists |
| Reward overoptimization and normalization | Yes | The shaped-reward risk in [[decisions/0005-training-and-reward]] is the same failure |
| Truncated importance sampling for async training | Later | Only once actors and learners are separated, which is not the current design |
| Token-level structure, DPO, preference data, instruction tuning | No | There is no language model and no human preference signal here |

## Critic-free baselines, which this project should seriously consider

PPO needs a critic, and [[training-design]] plans one. The language-model world has largely moved away from that, and the reason applies here more strongly than it does there.

REINFORCE leave-one-out draws $K$ episodes from the same starting state and uses the mean return of the others as the baseline for each.

$$b_k = \frac{1}{K-1}\sum_{i \neq k} G^{(i)}, \qquad \hat A_k = G^{(k)} - b_k$$

Excluding the sample itself is what keeps this exactly unbiased, because $b_k$ is then independent of $a_k$ and the baseline term vanishes by the argument in [[rl-methods]]. Group-relative optimization instead uses the full group mean, which includes the sample and leaves an $O(1/K)$ bias, and then divides by the group standard deviation, which is contested and which the Dr. GRPO variant drops.

The reason this matters here is that the objection to it does not apply. In a language model, drawing $K$ completions per prompt is the dominant cost. Here the environment runs at roughly 4,600 episodes per second and a scenario is reproducible from a seed, so drawing $K$ episodes from one starting state is close to free. Against that, the critic has to be fitted on very little data, and [[training-design]] already flags 116 recorded decisions as a regime where a network memorizes.

The cost is real and should be stated. A leave-one-out baseline gives one advantage for the whole episode, so every decision in it receives the same credit, which is coarse when a battle turns on one decision out of thirty. That is the trade in the next section rather than an argument against trying it. The concrete recommendation is that a leave-one-out baseline belongs in the first round of experiments beside the critic, not as a fallback after the critic disappoints.

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

The claim that a critic-free method avoids value-network bias is often stated loosely. The precise version is worth having, because it decides whether a critic fitted on very little data is dangerous or merely imprecise.

A value estimate used as a pure baseline, subtracted from a Monte Carlo return, leaves the gradient unbiased no matter how bad the estimate is. That follows from the baseline argument in [[rl-methods]], which needs only that the baseline does not depend on the action. A poor critic costs variance reduction, not correctness.

Bias enters with bootstrapping. Replacing the sampled return with a target such as $r_{t+1} + \gamma V_\phi(s_{t+1})$ substitutes the network's own estimate for the truth, so $\mathbb{E}[\hat A_t] \neq A^\pi$ whenever $V_\phi$ is imperfect. That is what buys the variance reduction, and it is the same trade $\lambda$ controls in generalized advantage estimation.

For this project the reading is direct. A critic trained on very little data is safe to use as a plain baseline and becomes a bias risk exactly to the extent that $\lambda$ is pushed toward 0. Starting near $\lambda = 0.95$ and treating a lower value as an explicit bias-for-variance decision is the disciplined way to hold that.

## A reference policy this project does have

RLHF carries two KL-shaped quantities that [[rl-methods]] warns against conflating. The trust region between $\pi_\theta$ and $\pi_{\theta_{\text{old}}}$ is enforced by the clip. The leash to a fixed reference $\pi_{\text{ref}}$, usually the supervised-fine-tuned model, is a separate regularizer.

The obvious reading is that the second does not apply here, since there is no pretrained model to stay near. That reading is wrong, and the reason is the staging in [[decisions/0005-training-and-reward]]. After stage 1 there is a cloned policy that plays competently, and stage 3 can destroy that competence early while the value estimates are still poor. The cloned checkpoint is a genuine reference policy in exactly the sense RLHF means, and a KL penalty against it, decayed as training proceeds, is a principled way to keep the reinforcement stage from throwing away what imitation bought.

This is a proposal rather than a decision, and it belongs in [[decisions/0005-training-and-reward]] if adopted. What argues for it is that the failure it prevents, an early collapse away from competent play, is the most likely way stage 3 goes wrong. What argues against it is that it adds a coefficient and a schedule to tune, and that the entropy bonus already provides some of the same protection. Measure the collapse before adding the leash.

## Overoptimization, restated for a shaped battle reward

Reward-model overoptimization is the observation that a policy pushed hard against a learned reward improves on that reward while getting worse at what the reward was supposed to measure. The Bradley-Terry model behind those rewards is identified only up to an additive constant, which is why reward normalization is needed for the KL coefficient to mean anything across runs.

This project has no learned reward model, so the identifiability part does not apply. The overoptimization part applies in full and is already the stated worry in [[decisions/0005-training-and-reward]], where a shaped per-decision reward risks teaching the proxy rather than the objective. A shaped reward is a hand-written reward model, and it fails the same way.

Two things carry over as practice. Report the true objective, the win rate, alongside whatever shaped quantity is being optimized, so divergence between them is visible rather than inferred. And keep the shaped term potential-based, since that is the one form with a proof that the optimal policy is unchanged.

## What does not transfer

Recording these so they are not imported by analogy.

Token-level structure. A completion is a sequence of actions under one reward, which motivates sequence-level ratios and per-token clipping. A battle decision is a genuine action in a genuine state, so the per-decision form is correct here and the machinery built to compensate for the bandit framing is unnecessary.

Discounting conventions. Language-model RLHF sets $\gamma = 1$ because the completion is scored as a whole. Battles terminate quickly enough that a value near 1 is defensible, but for a different reason, and [[notation]] records why the range is relaxed at all.

Direct preference optimization and the direct alignment family. These replace a reward model with a closed-form objective on preference pairs. There is no preference data here and no reward model to eliminate.

Instruction tuning, preference data collection, and synthetic data pipelines. These concern getting a language model to a usable starting point and have no analogue in an environment with a scripted teacher.

Asynchronous training with truncated importance sampling. This corrects for actors and learners running on separate hardware with lagged weights. It becomes relevant only if this project separates them, which the current single-process design does not. The correction to remember if that changes is a one-sided cap on the importance weight, biased upward but bounded in variance, distinct from the two-sided PPO clip.

## Related

- [[rl-methods]], the techniques themselves, derived.
- [[training-design]], the architecture and hyperparameters these ideas would modify.
- [[decisions/0005-training-and-reward]], the decisions this page proposes revisiting.
- [[notation]], the symbol contract, including where the RLHF sources differ.
