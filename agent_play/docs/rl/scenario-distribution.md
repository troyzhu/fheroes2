---
title: "The scenario distribution, and why the matchup dominates the outcome"
type: design
updated: 2026-08-03
related_concepts: ["[[../decisions/0005-training-and-reward]]", "[[training-design]]", "[[rlhf-transfer]]", "[[rl-and-the-battle-domain]]"]
tags: [agent-env, scenario, variance, curriculum, evaluation]
---

# The scenario distribution, and why the matchup dominates the outcome

A battle's outcome is decided more by which armies were drawn than by how either side plays. Some matchups are lost before the first decision, against any opponent playing reasonably, and no policy can recover them. This is a property of the game rather than a flaw in the environment, but it has consequences for training and for evaluation that are easy to miss and expensive to discover late.

The pieces of the answer were scattered across four documents, each addressing one symptom without naming the shared cause. This page states the cause, shows that three mechanisms already under consideration all attack the same term, and records what is decided and what is not.

## Table of contents
- [[#Where the variance actually sits]]
- [[#What a hopeless matchup costs, and when it costs nothing]]
- [[#Three mechanisms, one target]]
- [[#Measuring difficulty, and what a scenario even is]]
- [[#Evaluation, where the same problem returns]]
- [[#What is decided and what is open]]

## Where the variance actually sits

Write $Z$ for the scenario drawn from $\rho_0$, meaning terrain, both army compositions, and the seed, and $G$ for the return of one episode. The law of total variance splits the outcome exactly.

$$\operatorname{Var}(G) = \underbrace{\operatorname{Var}_Z\big(\mathbb{E}[G \mid Z]\big)}_{\text{between scenarios}} + \underbrace{\mathbb{E}_Z\big[\operatorname{Var}(G \mid Z)\big]}_{\text{within a scenario}}$$

The first term is scenario difficulty. It is identical for any two policies of equal strength, so it carries no information about the thing being trained or measured. The second term is what play and combat randomness contribute, and only part of that is attributable to the policy at all.

The problem is one of proportion. In this game the first term is large, because a two-to-one advantage in a key stack decides most battles regardless of tactics. Every estimate of policy quality, whether a gradient or a reported win rate, is therefore mostly measuring which scenarios happened to be drawn. With a fixed sampling budget, most of it is spent re-estimating difficulty rather than learning anything about the policy.

Naming the decomposition is the useful step, because every remedy below is a way of removing the first term rather than tolerating it.

## What a hopeless matchup costs, and when it costs nothing

Take a matchup so lopsided that the outcome is a loss whatever the agent does. Under a pure win-or-lose reward, $\mathbb{E}[G \mid Z = z] = -1$ and $\operatorname{Var}(G \mid Z = z) = 0$. It contributes to the between term and nothing to the within term.

That is the worst possible case for learning, and it is worth being precise about why. Under a group or leave-one-out baseline, every rollout from that scenario earns the same return, so every advantage in the group is exactly zero and the batch contributes no gradient while costing full compute. Under a critic, the critic correctly learns to predict a loss, the advantage collapses to noise around zero, and the same thing happens. No algorithm rescues a scenario with no outcome variance, because there is nothing in it to be right or wrong about.

### The reward choice changes this, and ADR 0005 does not say so

Under the margin-weighted terminal reward, the same hopeless matchup stops being dead weight. The return still varies with how much force survives, so $\operatorname{Var}(G \mid Z = z) > 0$ and the scenario teaches something real, namely how to lose cheaply. That is not a consolation prize. A battle is one episode inside a campaign, the surviving army carries forward, and losing while preserving three stacks is genuinely a better outcome than losing while preserving none.

This is an argument for the margin-weighted candidate in [[../decisions/0005-training-and-reward#Options considered]] that the record does not currently make. Its stated case is that the reward should compose with the wider game. The additional case is that it keeps a whole region of the scenario distribution informative instead of degenerate, which matters more the wider the generator's spread.

## Three mechanisms, one target

Three ideas appear in three different documents. All of them remove the between-scenario term, and seeing that is the point of this page.

| Mechanism | Where it is discussed | How it removes the term | Cost |
|---|---|---|---|
| Difficulty filtering of $\rho_0$ | [[rlhf-transfer#Difficulty filtering, and what it settles]] | Declines to draw scenarios where the within term is near zero | Needs a measured difficulty control, and the band moves as the policy improves |
| Leave-one-out or group baseline | [[rlhf-transfer#Critic-free baselines, which this project should seriously consider]] | Conditions on $Z$ directly. Subtracting the group mean removes $\mathbb{E}[G \mid Z]$ exactly rather than estimating it | $K$ episodes per start state, and one advantage shared across every decision in an episode |
| Pre-fitting the critic on teacher play | [[training-design#Pre-fitting the critic on teacher play]] | Regresses $\mathbb{E}[G \mid Z]$ from data and subtracts the fit | Approximate, and conditioned on the opponent it was fitted against |

The leave-one-out route is the only one that removes the term exactly, because conditioning on the scenario is not an estimate. The critic route amortizes across scenarios instead of paying $K$ episodes each time, at the price of being only as good as the fit. Filtering is complementary to both rather than an alternative, since it addresses the case neither can help with, where the within-scenario variance is itself zero.

That last point is worth stating plainly. Conditioning removes difficulty from the advantage, but a scenario with no outcome variance has nothing left after difficulty is removed. Filtering is what keeps those out of the batch in the first place.

## Measuring difficulty, and what a scenario even is

Difficulty has to be measured rather than asserted from army sizes, and doing that requires being careful about what is held fixed.

A `Scenario` in this environment fixes the terrain, the tile index, the world seed, and both `SideSpec` army lists. Because the combat seed is derived from the map seed and the two armies, a fully specified scenario played by a deterministic policy is a single reproducible episode with no outcome distribution at all. Difficulty is therefore not a property of a `Scenario` as the struct defines it.

The quantity that matters is a property of the army matchup, estimated over a distribution of seeds and over the policy's own sampling. Concretely, hold the two `SideSpec` lists fixed, vary the world seed, sample the policy, and take the empirical win rate over the resulting episodes. At roughly 4,600 episodes per second this is cheap enough to do routinely rather than once.

Two consequences follow. Difficulty is policy-relative, so a matchup that is hopeless for a freshly cloned policy may be winnable later, which is why [[../decisions/0005-training-and-reward]] treats the target band as a curriculum rather than a fixed filter. And the generator should expose the army matchup and the seed as separate axes, because collapsing them into one scenario identifier makes the measurement above impossible to express.

## Evaluation, where the same problem returns

Every remedy above concerns training. The same decomposition governs whether a reported number means anything, and here there is a cheap technique the project is not yet using.

### Compare on identical seeds

To compare two policies, evaluate both on the same seed set rather than on independent draws. The quantity of interest is the difference, and

$$\operatorname{Var}\big(G_A - G_B\big) = \operatorname{Var}(G_A) + \operatorname{Var}(G_B) - 2\operatorname{Cov}(G_A, G_B)$$

where the covariance is large and positive precisely because both policies face the same scenario difficulties. Pairing therefore cancels most of the between-scenario term from the comparison. This is the common random numbers technique, it is free here because a scenario is reproducible from a seed, and it costs nothing but the discipline of fixing the seed list. An unpaired comparison throws away that covariance and needs far more episodes for the same resolution.

### Report the distribution, not only its mean

A single win rate over a wide generator is a statement about the generator as much as about the policy, as [[rl-and-the-battle-domain]] notes. Reporting it stratified by measured difficulty band separates two questions that a single number conflates: whether the agent wins the battles it should, and whether it steals any it should not.

### Playing both sides of one matchup

Since a scenario names an attacker and a defender separately, the same army pair can be played from either side. Doing both and reporting the pair controls for army composition at the level of the comparison, in the way that engine testing alternates colours.

The caveat is that the two sides are not symmetric here. Starting positions differ and the speed queue decides who acts first, so swapping sides controls for the armies without controlling for the positional advantage. It is a useful variance reduction and not a proof of fairness.

## Measured, 2026-08-03

The argument above was written before any of it could be tested. It now has been, and two of its predictions held while a third assumption turned out to be wrong in a way that matters.

### The band is intrinsically narrow

Sampling 90 matchups over the `simple_v1` roster and measuring each with the cloned policy put 8 percent inside the 20 to 80 percent band. Thirty-one were too easy, fifty-two too hard, and the median win rate was 0.05. Constraining the sampler to small totals and to sides within 15 percent of each other, on the theory that variance needs to be large relative to the mean, raised that only to 10 percent.

The reason is visible in a mirror matchup, where the win rate is a step function of the count. Fifty Peasants beat seventy defenders 96.9 percent of the time and beat seventy-one zero percent of the time. Damage rolls average out across fifty creatures, so arithmetic decides the battle and play barely moves it. Small stacks behave better, and five against five sits at 0.79 because one bad exchange is a fifth of the army, but the general picture is that a battle outcome is close to a deterministic function of the matchup.

The consequence for the generator is concrete. Rejection sampling is the wrong mechanism at a 10 percent hit rate. A usable generator has to calibrate each matchup, searching the count that puts a given army pair in band, which the 70 against 71 result shows is a one-step search rather than a hard one. That is a different and more expensive design than sampling and filtering, and it was not anticipated here.

`scenarios.calibrate` implements it. Given an army pair it bisects a scale on the defender's counts until the win rate reaches a target, which is cheap because the win rate is close to monotone in the defender's strength. Seven steps of twelve episodes turned a matchup the cloned policy lost every time into one it wins 58 percent of the time, over nineteen decisions with a reward standard deviation of 1.12. Hand-designed matchups do no better than sampled ones at finding the band: five built deliberately to look balanced measured four at zero and one at one. Calibration is what makes a given pair usable, not judgement about army composition.

### Reward variance is the cheaper filter

Every degenerate matchup measured a reward standard deviation of exactly 0.00, and every in-band one measured between 1.0 and 1.3. That is the same fact the identity predicts, since equal returns across a group make every advantage zero, and it is a better filter in practice than the win rate: it needs no threshold, it is what the gradient actually depends on, and a matchup that produces zero variance can be discarded after a handful of episodes.

### Training across matchups, first attempt and then properly

The first attempt at this said PPO overfits, and it was wrong. It is recorded here with its correction because the error is instructive.

Splitting a pool of five in-band matchups into three for training and two held out, PPO improved the training matchups by 0.267 win rate and appeared to make the held-out ones worse by 0.208. With two held-out matchups the standard error is around 0.14, so that was 1.5 standard errors and was reported as suggestive rather than settled. It should have carried less weight than it did.

Repeating it on a calibrated pool of forty-five, split thirty-three for training and twelve held out, with twenty-four evaluation episodes per matchup:

| | Before | After | Change |
|---|---|---|---|
| Training matchups | $0.473 \pm 0.034$ | $0.785 \pm 0.056$ | $+0.312 \pm 0.066$, 4.7 standard errors |
| Held-out matchups | $0.542 \pm 0.058$ | $0.646 \pm 0.112$ | $+0.104 \pm 0.126$, 0.8 standard errors |

The improvement on trained matchups is unambiguous. The held-out change is positive and indistinguishable from zero, which is a weaker claim than it looks but a much stronger one than the earlier run supported: whatever else is true, the policy is not degrading on matchups it never saw.

Errors here are across matchups rather than across episodes, because the claim is about the pool. Held-out spread is wide, about 0.39 across twelve matchups, so resolving a change of 0.1 to two standard errors needs roughly fifty held-out matchups. That measurement has since been made.

### At the size the previous entry asked for, and the defect it exposed

A calibrated pool of 140, split 90 for training and 50 held out, 40 iterations of 4 groups of 8, leave-one-out advantage under the ratio clip, 24 evaluation episodes per matchup. Run once with a defect and again without it, and only the second is a measurement of the method.

| Configuration | Training gain | Held-out gain |
|---|---|---|
| Groups spanning eight matchups, defective | $+0.234$ | $+0.047$ |
| Groups sharing one matchup, three seeds | $+0.198$ | $+0.247$ |

The defect is in how the pool composed with a group-relative baseline. `MatchupPool` drew a new army pair on every reset and the trainer resets once per episode, so a group of eight was eight different battles. Every mode here compares an episode with the others beside it, which is why GRPO on a language model samples several completions of one prompt, and comparing one completion each of eight prompts measures which prompt was easy. The baseline was subtracting scenario difficulty rather than establishing a reference for play.

That explains the shape of the defective run exactly. Fitting the training matchups still worked, because the useful signal survives averaging over a batch. Transfer did not, because the advantage carried mostly which matchup had been drawn, and that is information with nothing to transfer.

With groups sharing a starting position the held-out win rate after training goes from 0.514 to 0.699 across three seeds, and the gap the earlier version of this section reported is gone. On every seed the held-out gain is at least as large as the training gain.

Which error to use decides what that last sentence means, and the two available differ by a factor of five. Across seeds the held-out gain exceeds the training gain by $+0.049 \pm 0.012$, which is 4.1 standard errors and would read as a reverse gap. Across matchups the same difference is $+0.049 \pm 0.056$, which is 0.9 standard errors and reads as nothing.

The second is the one that bears on generalization. The across-seed error asks whether the measurement repeats on these particular 90 and 50 matchups, and it does, tightly. The across-matchup error asks whether it would hold on other matchups the generator produces, which is the actual question, and by it there is no gap in either direction. Quoting the tighter number would turn a null result into a discovery, using an error bar that answers a question nobody asked.

The calibration holds at scale, which is worth recording separately: the pool targeted 0.5 and both halves measured within 0.05 of it against the checkpoint that calibrated them, on 24 evaluation episodes rather than the 8 used to probe.

Three lessons, and the first two are the ones this page kept having to learn. A two-matchup holdout cannot support a claim in either direction. A quantity too small to resolve may still be large enough to matter, and the difference of two such quantities can be resolvable when neither is, which is how the defective run produced a confident and wrong conclusion at 2.7 standard errors. The third is that a statistically clean result computed from a broken configuration is still broken, and no amount of seeds or standard errors would have caught it; what caught it was a second method disagreeing.

## The generator, built and measured

Everything above says what a generator has to do. `scenarios.build_pool` does it, and the mechanism is calibration rather than filtering.

Sampling a composition and keeping it if it happens to be contested wastes most of its budget, at three percent on a real map and eight to ten on synthetic pairs. Calibrating instead samples a composition and then searches the defender scale that makes that pair contested, which converts a rejection problem into a bisection. Measured hit rate is **38.7 percent**, four to thirteen times better, at roughly three seconds per accepted matchup on the target machine.

A pool records the policy that calibrated it, and this matters more than it looks. Difficulty is a property of the pair *and the policy*, so a pool measured against one checkpoint and reused with a stronger one is quietly a pool of easy matchups. That is the moving-band problem stated earlier; storing the provenance is what turns it from something to remember into something checkable.

What calibration cannot do is rescue a matchup whose win rate is a step function, and those exist: `calibrate` reports `calibrated: false` when no scale inside its range lands in band, rather than returning the least-bad probe. Roughly three in five sampled compositions are like this, which is why the hit rate is 39 percent rather than higher.

## What is decided and what is open

Decided. The acceptance criterion for the generator, in [[../decisions/0005-training-and-reward]], that a scenario carries gradient only when the policy neither always wins nor always loses it. The held-out seed set, fixed in advance and excluded from training, before any headline number is quoted.

Open, and now with the analysis attached. Which of the three mechanisms is used, and in what combination, since they are complementary rather than exclusive. The generator itself, which remains the largest undocumented modeling choice in the project. Whether evaluation adopts paired seeds and both-sides play, which this page recommends and which nothing currently depends on.

Not yet investigated. Whether the difficulty band should be measured against the teacher, which is stable and cheap, or against the current policy, which is the quantity the identity actually refers to but which drifts during training and makes the filter a moving target that could interact badly with the update.

## The samplers, three generations

The generator has been rebuilt twice since the narrow seven-creature sampler, each time on a measurement. The wide-roster sampler drew every `simple_v1` creature by hit-point share and calibrated at 19 to 26 percent depending on regimes and heroes. The value-budget sampler from the owner-supplied guide, [[../research/works/generalized-battle-agent-guide]], prices stacks by the engine's own creature strength, allocates a log-uniform budget by Dirichlet shares, and draws the enemy budget from a mixture concentrated near parity; head to head it reached the band in a third fewer attempts, 31.4 against 23.8 percent at about 1.8 standard errors, with less calibration rescue. It is the recommended sampler for new pools, with the predecessors kept so recorded pools stay reproducible.

## Related

- [[../decisions/0005-training-and-reward]], where the reward and the initial-state distribution are decided and deferred.
- [[rlhf-transfer]], for difficulty filtering and critic-free baselines with their sources.
- [[training-design]], for critic pre-fitting and the hyperparameters these choices touch.
- [[rl-and-the-battle-domain]], for why a win rate is a statement about $\rho_0$.
