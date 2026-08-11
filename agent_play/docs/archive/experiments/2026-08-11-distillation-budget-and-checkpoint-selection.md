---
title: "The distillation budget, and the metric that was choosing checkpoints, 2026-08-11"
type: experiment-log
updated: 2026-08-11
tags: [agent-env, archive, experiment, distillation, training, measurement]
---

# The distillation budget, and the metric that was choosing checkpoints, 2026-08-11

Every distillation arm on record stopped at its final epoch and was saved by the epoch whose holdout agreement peaked. Both facts turn out to be problems, and they are the same problem seen from two sides.

## Why no run on record could evidence its own convergence

The trainer arms its schedule with `CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)`, so the learning rate reaches its floor exactly at whatever budget the run was given. A run annealed that way flattens at its own boundary, and a flat tail is a statement about the schedule rather than about the model. Reading the three arms of the 2026-08-09 band run confirms the shape: on seed 0 the holdout agreement gained $+0.0216$ over epochs 10 to 15, where the rate ran from $2.2 \times 10^{-4}$ down to $1.3 \times 10^{-4}$, and $+0.0025$ over epochs 20 to 24, where it ran from $4.9 \times 10^{-5}$ to $1.1 \times 10^{-5}$. The gain ratio is 8.6 against a rate ratio of 8. The curve decelerated in proportion to the step size, which is an unconverged model taking smaller steps.

The scope of that matters, because it is a property of this configuration rather than of cosine schedules generally. Torch's `CosineAnnealingLR` is itself periodic with period $2 T_{\max}$: stepped past $T_{\max} = 25$ it does not rest at `eta_min` but climbs back to the full $3 \times 10^{-4}$ by epoch 50. What keeps a run on the descending half is that `T_max` is tied to `epochs`, and that coupling is load-bearing rather than incidental.

Under a cyclic schedule the argument would not hold. Raising the epoch count against a fixed `T_max` produces accidental warm restarts, and `CosineAnnealingWarmRestarts` jumps the rate discontinuously back to its maximum every $T_0$, so each cycle end is a genuinely annealed point rather than an artifact of where the run stopped. A flat tail there would carry the meaning this one cannot.

The shorter direction had been measured, at eight epochs on 2026-08-07 with the cosine re-armed to that horizon. The longer direction never had.

## The three selectors, and how far apart they sit

The run was repeated at 100 epochs with the cosine re-armed, three seeds, snapshots every 25 epochs, against the standing 25-epoch configuration as the control. The trainer now records the held-out cross-entropy and both entropy forms beside the agreement it already kept, because agreement is a top-1 match count and cannot see a model growing overconfident about the same answers.

The two offline selectors disagree by roughly fifty epochs, on both seeds where a full run exists:

| seed | held-out loss minimum | agreement maximum | gap | effective actions, first epoch to last |
|---|---|---|---|---|
| 0 | epoch 51, loss 0.4631 | epoch 94, loss 0.4886 | 43 epochs | 4.83 to 1.22 |
| 1 | epoch 44, loss 0.4643 | epoch 98, loss 0.4809 | 54 epochs | 4.88 to 1.24 |

Best-agreement checkpointing, which is what ships, saves a model some fifty epochs past the point where held-out loss says it stopped generalizing better.

## What play says, which is the only column that decides

Each snapshot was played on the five suites that can separate players, network alone, three seeds, seeded per matchup. The battery builds ten, and the other five are left out for two different reasons: four of them are saturated, the real-map suite at 22 matchups of 24 through the Thunk ladder at 2 of 4, and a suite mean is unreadable when its matchups are not contested; the fifth, `stress_wide_only`, is simply outside the standing nine every earlier scoreboard was quoted on, so including it here would have made these arms incomparable to them.

| arm | agreement | held-out loss | effective actions | win rate | reward | strength margin |
|---|---|---|---|---|---|---|
| 25 epochs, standing | 0.8509 | 0.5635 | 1.66 | 0.314 | 0.122 | $-0.192$ |
| 100 epochs at 25 | 0.8696 | 0.5018 | 1.49 | 0.306 | 0.092 | $-0.214$ |
| 100 epochs at 50 | 0.8982 | 0.4660 | 1.30 | 0.299 | 0.085 | $-0.214$ |
| 100 epochs at 75 | 0.9100 | 0.4659 | 1.25 | 0.283 | 0.062 | $-0.221$ |
| 100 epochs at 100 | 0.9144 | 0.4659 | 1.23 | 0.284 | 0.060 | $-0.224$ |

Agreement rises by $0.064$ along that ladder and held-out loss improves by $0.098$, while every column describing play degrades. Paired within seed, the long budget costs $-0.0301$ win rate at a standard error of $0.0112$, $-0.0619$ mean reward at $0.0160$, and $-0.0317$ strength margin at $0.0089$. All three seeds carry the same sign. Two of the three effects clear three standard errors, and the win rate alone would have read as noise at 1.5, which is why the quality columns travel with it.

Across the five budget arms above, fifteen arm-seeds in all, the correlation between held-out agreement and win rate is $-0.436$, and between effective actions in play and win rate is $+0.482$. Over the range explored, the metric the trainer selects on runs against the metric the project is trying to raise.

## The other direction, and whether the schedule was the problem

Two questions the 25-to-100 ladder could not settle. Nothing on this corpus had ever run below 25 epochs, so early stopping was open, and the 2026-08-07 eight-epoch cut had read at par with a $+0.017$ post-reinforcement delta inside one standard error. And a longer single cosine differs from a shorter one in two ways at once, total optimization and how long the rate stays high, which warm restarts separate: SGDR gives the extra steps while still ending each cycle annealed.

| arm | agreement | held-out loss | win rate | reward | strength margin |
|---|---|---|---|---|---|
| 5 epochs | 0.6887 | 1.0663 | 0.314 | 0.091 | $-0.224$ |
| 10 epochs | 0.7595 | 0.8407 | 0.297 | 0.063 | $-0.234$ |
| 15 epochs | 0.7888 | 0.7323 | 0.296 | 0.052 | $-0.244$ |
| 25 epochs, standing | 0.8509 | 0.5635 | 0.314 | 0.122 | $-0.192$ |
| 100 epochs | 0.9144 | 0.4659 | 0.284 | 0.060 | $-0.224$ |
| 100 epochs, restarts every 25 | 0.9183 | 0.4061 | 0.226 | $-0.089$ | $-0.315$ |

Early stopping is now answered and the answer is no. Paired against the standing configuration, five epochs matches it on win rate at $+0.001$ and loses on everything else, $-0.031$ reward at 4.3 standard errors and $-0.031$ strength margin at 20.1. Ten and fifteen epochs lose on all three. The rate alone would have called five epochs a tie, and the quality columns are what separate them, so the 2026-08-07 hint does not survive contact with a fuller metric block. Twenty-five sits at the optimum of everything measured in both directions.

The restart arm settles the mechanism question, by being the worst player of the twelve while holding the best offline numbers of the twelve. Restarts bought more effective optimization, the highest agreement at 0.9183 and the lowest held-out loss at 0.4061, and cost $-0.088$ win rate, $-0.211$ reward and $-0.123$ strength margin against the standing clone at 2.7, 3.5 and 4.0 standard errors. So the damage a long budget does is not the shape of the anneal. It is the fitting.

That also names the mechanism the earlier sections left open. Held-out loss falling while play falls cannot be classical overfitting, because the holdout is drawn from the teacher's own episodes: the model is genuinely generalizing better on the states the teacher visits. It plays worse on the states it visits itself, which is covariate shift rather than overfitting, and it is the failure DAgger exists to address. Across all twelve arms agreement correlates $-0.389$ with play and held-out loss $+0.237$, meaning lower loss goes with worse play. Both offline instruments point the wrong way across the whole design.

## Sharpness is a correlate of the damage, not its cause

The obvious reading of that ladder is that sharpening is what costs the play, since the two move together and the mechanism is easy to tell: a policy choosing among 1.2 of some 29 legal moves has no second option when its first is wrong, and root PUCT scales its exploration term by exactly this prior. The reading is wrong, and a controlled arm is what shows it.

Holding the budget at 25 epochs and paying the student to stay uncertain, through $-\beta H(\pi_\theta(\cdot \mid s))$ added to the loss, moves entropy in the intended direction and moves play the wrong way at every dose:

| arm | effective actions in play | win rate | reward | strength margin |
|---|---|---|---|---|
| 25 epochs, no bonus | 2.73 | 0.314 | 0.122 | $-0.192$ |
| $\beta = 0.05$ | 3.28 | 0.276 | 0.045 | $-0.231$ |
| $\beta = 0.15$ | 4.62 | 0.264 | $-0.009$ | $-0.273$ |
| $\beta = 0.40$ | 12.88 | 0.248 | $-0.049$ | $-0.297$ |

Paired against the control at the same seed, $\beta = 0.05$ is not distinguishable from it at $-0.038$ win rate and 0.9 standard errors, one of its three seeds landing positive. The larger doses are distinguishable and get worse together: $\beta = 0.15$ reads $-0.050$ win rate, $-0.131$ reward and $-0.080$ strength margin at 1.8, 2.9 and 3.8 standard errors, and $\beta = 0.40$ reads $-0.066$, $-0.171$ and $-0.105$ at 2.7, 4.8 and 9.1. The strength margin is the sharpest instrument of the three at every dose, and the win rate the bluntest, which is the third time on this program that the rate alone would have called an effect noise.

The correlation flips sign under intervention. Across the budget ladder, effective actions correlate $+0.482$ with win rate; within the entropy grid, where the budget is fixed and only the bonus moves, they correlate $-0.561$. Entropy travelled with the damage because the budget moved both, so it is not the mediator of what a long budget costs.

## The same arms searched, where the mechanism is real and pays nothing

Every number above is the network playing alone, and the deployed agent does not play that way. Root search is the only mechanism on this project measured above the built-in AI, and PUCT scales its exploration term by the prior, so the prior's spread is an input to the search rather than only a description of the weights. The arms were rerun on the two mirror suites at thirty-two playouts with the side environment's dice made independent of the live battle.

On the first seed the ranking inverted. The heaviest entropy bonus, worst of the three playing alone at 0.248, read 0.653 searched against the standing clone's 0.569 and the collapsed hundred-epoch arm's 0.486. That was worth replicating rather than reporting, and replication across three seeds returns exactly nothing:

| seed | standing clone | $\beta = 0.40$ | difference |
|---|---|---|---|
| 0 | 0.569 | 0.653 | $+0.083$ |
| 1 | 0.611 | 0.542 | $-0.069$ |
| 2 | 0.611 | 0.597 | $-0.014$ |

Paired, the entropy bonus is worth $+0.0000$ win rate at a standard error of $0.0446$ and $+0.0037$ mean reward at $0.0847$. The first seed was noise and the inversion it suggested is withdrawn.

What did replicate is the mechanism, and it replicated cleanly. Search visit entropy runs 0.67, 0.60 and 0.46 for the standing clone against 1.12, 1.14 and 1.09 for the bonused arm, roughly double in every seed. A broad prior really does make PUCT explore more widely, exactly as the exploration term predicts. That breadth simply does not convert into wins, which is the same verdict coverage forcing earned in [[../../decisions/0008-search-configuration|ADR 0008]], where visiting every candidate once lost at every budget and lost more as the budget grew. Two different routes to a wider search, one through the prior and one through the visit rule, both measure neutral to negative at this playout budget.

So the entropy bonus is negative for the network alone and neutral under search, and it is not a lever either way.

The honest configuration also reproduced ADR 0008's dice price independently. The same checkpoint on the same suites reads 0.750 and 0.972 when the side environment shares the live battle's combat stream against 0.528 and 0.611 when it does not, a gap of 0.222 and 0.361 against the $+0.323$ the ablation recorded on this suite.

## What that does and does not license

It does not say agreement is meaningless. Along the budget ladder agreement and sharpness are confounded, so the narrow statement is that raising agreement by training longer costs play, and the entropy that falls alongside it is a symptom rather than the mechanism. What the mechanism is remains open. The shape is the one behavior cloning is known for: a longer fit to the teacher's own state distribution need not help on the states the student itself reaches, and nothing here measured that divergence directly.

Part of the answer is in what the number is agreement *with*. The holdout is `split_by_episode(hard, 0.2, seed)` over the hard corpora only, and both of those are labelled by `AI::BattlePlanner`, through the passive recorder for the diverse set and the planner probe for the DAgger set. The search-labelled rows are all trained on and none of them is held out. So holdout agreement measures similarity to the built-in AI, which is the opponent the program exists to beat, and its optimum is to become that opponent rather than to pass it. Only the search-taught rows carry a signal that could exceed the teacher, and they are excluded from this number by construction.

The ceiling is not yet binding, since the network alone plays well below the engine and early training raises agreement and play together. This compounds the distribution argument rather than replacing it. What it settles is narrower: no amount of agreement can ever license a claim of exceeding the engine.

One reassurance falls out of the same data. At the standing budget the best-agreement epoch is the last epoch on all three seeds, so the selector is inert today and no shipped checkpoint was chosen against play by it. The finding is a constraint on raising the budget, not a defect in what exists.

It does not overturn the supervised plateau recorded in [[../../rl/program-review|program-review]], which is about rounds of search teaching rather than epochs within a round. It does qualify it: some of what a plateau looked like was measured on checkpoints chosen by a criterion that prefers the sharper model.

The general form of this is documented outside the project. Codevilla et al. (ICCV 2019) report that offline prediction error is not necessarily correlated with driving quality, and that two models with identical prediction error can differ dramatically in what they do. This program had already met its own version of it, in the 2026-08-07 sharpness sweep where the arm with the worst agreement was the only one with a positive held-out delta afterwards. That reading is now three measurements old rather than one.

## Where the labels sit, which is why the gap survives every budget

The budget arms all treat the loss as one thing, and it is not. The hard rows carry `nll_loss` against a single recorded action, plain log loss on an argmax, and that action is `AI::BattlePlanner`'s. The soft rows carry cross-entropy against $\bar\pi$, which is KL up to the target's own entropy. On the standing recipe that is 242,570 hard rows at weight 1.0 against 5,143 soft rows at weight 2.0, so the search-taught term is 4.1 percent of the loss mass and the other 95.9 percent instructs the student to imitate the engine.

Asking where those labels sit under the policy that has to learn them (`distillation_support.py`) explains the gap without needing another training run:

| | informative, search overruled | confirming, search agreed |
|---|---|---|
| decisions | 619, 12.0 percent | 4,524, 88.0 percent |
| prior probability on the labeled action | mean 0.0705, median 0.0002 | mean 0.8623, median 0.9786 |
| its rank under the prior | mean 12.4, median 8, p95 41 | 1.00 |
| below the one percent support threshold | 63.3 percent | 0.0 percent |

Three causes compound, and none of them is a bad label. Most of the corpus is redundant, 88 percent of it naming a move the policy already ranks first, which contributes no gradient. The informative remainder is largely off-support: at the median it names an action carrying two parts in ten thousand, ranked eighth of some thirty-three legal moves. And the soft target barely asks for movement anyway, $D_{\mathrm{KL}}(\bar\pi \,\|\, \text{prior})$ having mean 0.0541 but median 0.0003, so only the top few percent of rows request anything at all.

The student behaves exactly as that predicts. On the informative rows its probability on the labeled action moves from the prior's 0.0705 to 0.0872, a gain of 0.0166, while its rank on those same rows drifts from 12.43 to 13.20 and the share below support rises from 63.3 to 64.0 percent. Four percent of the loss mass, pulling toward actions the policy holds almost no mass on, against ninety-six percent pulling toward the engine that does not play them, moves the student essentially nowhere on the only decisions that carried information.

This is also why the 2026-08-09 regret weighting was the largest paired distillation effect on record at $+0.063$ held out. Reweighting the informative rows at equal total mass attacks the redundancy and leaves the support problem and the mass ratio untouched.

## What changed in the code

The trainer's heartbeat carries four more columns per epoch: `holdout_loss`, `holdout_entropy`, `holdout_normalized_entropy` and `holdout_effective_actions`. `convergence_report.py` reads them, so a supervised run's verdict is no longer a single agreement trend.

`soft_distill.py` gained `--checkpoint-every`, which saves a snapshot at a fixed epoch beside the best-agreement checkpoint, so a budget can be judged by play instead of by the criterion that selected it. It gained `--restart-period`, which swaps the single cosine for `CosineAnnealingWarmRestarts` at that $T_0$, and `--entropy-bonus`, adding $-\beta H(\pi_\theta(\cdot \mid s))$ to the loss. The bonus is guarded rather than multiplied by zero, so an unset value leaves the graph and the arithmetic untouched and arms stay comparable. Illegal actions carry probability exactly zero, because the policy masks with a fill of $-10^{8}$, so the bonus cannot leak mass onto moves the environment would reject.

`search_action_detail` now returns the prior's own pick when the simulation budget is zero. It previously fell through to a tie-break over an all-zero visit count and an all-zero value, which returns the lowest legal action index: not the policy, not an error. `search_strength.py` avoided it by branching before the call, and the zero rung of ADR 0008's budget ladder is therefore sound, but any harness passing `--simulations 0` would have measured array order.

`distillation_budget.py` runs this measurement end to end, sweeping budgets and bonuses and playing each arm. The numbers above were produced arm by arm and aggregated through its `--from-reports` mode, which re-reads a finished run without repeating it; the script exists so the procedure survives the scratchpad rather than because it drove this particular run.

## Provenance, and one substitution

The corpus the band run used no longer exists. Its matchup directories survive in the scratchpad with every episode removed, which is the failure the experiment-script convention was written against. The substitute is `data_diverse_planes` plus `data_dagger_planes` counted twice, 242,570 hard decisions against the band run's 247,937, with the identical `regret_corpus` soft set of 5,143 rows. All three load at 634-dimensional `obs_encoding_v3`, the planes layer being a field `load_dir` ignores. The arms here are matched against each other and are not comparable to the 0.512 held-out figure quoted elsewhere.

<!-- verify
exists  agent_play/experiments/distillation_budget.py
exists  agent_play/experiments/distillation_support.py
exists  agent_play/experiments/soft_distill.py
grep    agent_play/experiments/soft_distill.py :: entropy_bonus
grep    agent_play/experiments/soft_distill.py :: checkpoint_every
grep    agent_play/experiments/soft_distill.py :: restart_period
grep    agent_play/experiments/soft_distill.py :: holdout_loss
grep    agent_play/experiments/convergence_report.py :: holdout_effective_actions
grep    python/fheroes2_agent/search.py :: simulations <= 0
-->
