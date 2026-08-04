---
title: "Training runs, 2026-08-03"
type: experiment-log
updated: 2026-08-03
tags: [agent-env, archive, experiment, training]
---

# Training runs, 2026-08-03

Every training and measurement run from the day the learning side was first built, with its configuration and its numbers. This is provenance: the conclusions drawn from it live in [[../../rl/training-design]] and [[../../rl/scenario-distribution]], and nothing here is a decision.

All runs on the Apple M2 target machine, encoding `obs_encoding_v2` unless stated, policy 396,570 parameters, worker built from `src/dist` with asserts live.

## Stage 1, behaviour cloning

| Run | Data | Result |
|---|---|---|
| BC, `obs_encoding_v1` | 45,380 decisions from 2,000 episodes, 400 world seeds per fixture | Held-out agreement **0.8867** at epoch 23, 25 epochs, 13 s |
| BC, `obs_encoding_v2` | same | Held-out agreement **0.8873** at epoch 23, 12.5 s |

Baselines on the same held-out set: always the most common teacher action 0.131, uniform over the legal set 0.079. Split by episode, 1,600 training and 400 held-out episodes.

The two encodings differ by a one-hot over the 41 `simple_v1` creatures, widening an observation from 224 to 634. The difference between them is 0.0006, which is nothing. The training fixtures use three creature types whose stat lines already separate them.

Gate calibration, used to set the threshold in `verify_agent.sh` rather than guessing it: on the gate's own 200-episode dataset, 5 epochs gives 0.457, 10 gives 0.592, 15 gives 0.660.

## Stage 3, reinforcement learning

### Single matchup, critic and GAE

| Matchup | Start | End | Note |
|---|---|---|---|
| 5 Peasants vs 5 Peasants | 0.646 | 0.979 | 25 iterations of 48 episodes, 24.6 s |
| `m1_tiny_melee`, 50 vs 50 | 0.958 | 1.000 | Already near the ceiling |
| `m1_three_stack` | 0.000 | 0.000 | Degenerate, every rollout returned exactly $-1.000$ |

The third ran 15 iterations without moving. Reward variance was exactly zero, so every advantage was zero.

### Critic-free, leave-one-out baseline

| Matchup | Start | End |
|---|---|---|
| 8 VetPikemen, 7 Archers, 20 Peasants vs 9 Pikemen, 5 Rangers, 21 Peasants | 0.583 | 1.000 |

20 iterations, 6 groups of 8, no value network at all.

### Advantage and trust region, four ways

Matchup: 6 Archers and 10 Peasants against 121 Peasants, calibrated. Same cloned checkpoint, same seed, 20 iterations of 4 groups of 8.

| Advantage | Trust region | Start | Last five | Best |
|---|---|---|---|---|
| Leave-one-out | ratio | 0.188 | 0.925 | 1.000 |
| Group-relative, studentized | ratio | 0.188 | 0.925 | 0.969 |
| Group-relative, unstudentized | ratio | 0.188 | 0.925 | 1.000 |
| Leave-one-out | divergence | 0.188 | 0.944 | 1.000 |

At 32 episodes an iteration the standard error on a win rate is about 0.05, so nothing separates these. One seed only.

### Trust-region instrumentation

On the Thunk matchup below, PPO's ratio clip fired on 7 to 14 percent of samples while total-variation distance exceeded its threshold on 22 to 40 percent of the same samples.

### Generalization, properly

Calibrated pool of 45 split 33 for training and 12 held out, 40 iterations of 4 groups of 8, leave-one-out advantage and ratio trust region, 24 evaluation episodes per matchup. Errors are across matchups.

| | Before | After | Change |
|---|---|---|---|
| Training | $0.473 \pm 0.034$ | $0.785 \pm 0.056$ | $+0.312 \pm 0.066$ (4.7 SE) |
| Held out | $0.542 \pm 0.058$ | $0.646 \pm 0.112$ | $+0.104 \pm 0.126$ (0.8 SE) |

76 s total including both evaluations.

### Generalization, at the size the smaller run asked for

Calibrated pool of 140 split 90 for training and 50 held out, everything else as above. 8 minutes including both evaluations.

| | Before | After | Change |
|---|---|---|---|
| Training, 90 | $0.459 \pm 0.021$ | $0.694 \pm 0.036$ | $+0.234 \pm 0.042$ (5.6 SE) |
| Held out, 50 | $0.467 \pm 0.026$ | $0.514 \pm 0.047$ | $+0.047 \pm 0.054$ (0.9 SE) |

The held-out gain is still not separable from zero. The difference between the two gains is: $+0.187 \pm 0.068$, 2.7 standard errors, a ratio near fivefold. Before training the two halves measured within 0.007 of each other, so the split is not confounded with difficulty, and both sat within 0.04 of the 0.5 the pool targeted, which is the calibration holding at 24 evaluation episodes against the 8 used to probe.

Three measurements of the same quantity, in order: $-0.208$ on 2 held-out matchups, $+0.104 \pm 0.126$ on 12, $+0.047 \pm 0.054$ on 50.

### Generalization, first attempt, superseded

Pool of in-band matchups split 5 for training and 2 held out, 30 iterations of 32 episodes. Its apparent held-out regression did not survive a larger pool and should not be cited.

| | Before | After | Change |
|---|---|---|---|
| Training matchups | 0.667 | 0.933 | **+0.267** |
| Held-out matchups | 0.667 | 0.458 | **−0.208** |

At 12 evaluation episodes per matchup the standard error is about 0.14, so the regression is roughly 1.5 standard errors. Suggestive, not settled. The pool was far too small.

## Scenario measurement

| Population | Sampled | In band | Too easy | Too hard |
|---|---|---|---|---|
| Synthetic, unconstrained sampler | 90 | 7 (8%) | 31 | 52 |
| Synthetic, constrained to small and near-equal | 60 | 6 (10%) | 23 | 31 |
| Real map, `Thunk.mx2` hero against neutral stack | 68 | 2 (3%) | 33 | 33 |
| Hand-designed to look balanced | 5 | 0 | 1 | 4 |
| Calibrated by bisection | 31 attempts | 12 (39%) | — | — |

The mirror-matchup step function: 50 Peasants beat 70 defenders 96.9 percent of the time and beat 71 zero percent of the time.

Every degenerate matchup measured a reward standard deviation of exactly 0.00; every in-band one measured between 0.86 and 1.30.

## The Thunk map

Loaded through the engine's own map loader with `--dump-map`. 108 by 108, 13 heroes, 93 neutral stacks.

Ten of thirteen hero armies and 66 of 93 neutral stacks are representable under `simple_v1`. The player's opening fight is not: Catarina starts with 17 Rangers and 6 Cavalries against 12 Genies, and Cavalry is wide while Genies fly.

Most contested matchup found anywhere in this work: Corribus with 3 Crusaders and 2 Paladins against 22 Orc Chiefs, measuring 0.50 with a reward standard deviation of 1.26 over 28 decisions. Training it with a leave-one-out baseline and the divergence trust region, 25 iterations of 4 groups of 8, took it from **0.562 to 1.000** in 23.6 s.

## Stage 2b, the critic pre-fitted on teacher play

Built from the proposal in [[../../rl/training-design#Pre-fitting the critic on teacher play]]. The teacher plays both sides of every recorded battle, so each episode supplies positive targets for the winner's decisions and negative ones for the loser's, and the 2,000 episodes above yield 45,380 of each. Explained variance is $1 - \text{MSE}/\text{Var}$ on the same held-out 400 episodes the cloning result uses.

| Variant | Explained variance | Teacher agreement | Trainable |
|---|---|---|---|
| Cloned, value head at initialization | $-3.061$ | 0.8873 | — |
| Value head only, trunk frozen | $+0.835$ | 0.8873 | 193 |
| End to end | $+0.946$ | 0.7012 | 396,570 |

A negative explained variance means the untrained head is worse than predicting the mean, which it is: it emits near zero while returns average $+0.489$. The end-to-end fit buys $0.111$ more and costs $0.186$ of teacher agreement, because the trunk is shared and nothing in the value objective preserves the features the policy head reads.

### What limits the fit

The frozen value head is 193 parameters over a 192-wide trunk, so it is linear regression on the cloned policy's features. Both the data and the representation bind. One fixed held-out set of 400 episodes scores every cell, and gradient steps are equalized at 6,000 rather than epochs, or the small conditions would quietly get less optimization.

| Trunk | 50 episodes | 200 | 800 | 1,600 |
|---|---|---|---|---|
| Cloned to 0.606 agreement | $+0.129$ | $+0.025$ | $+0.375$ | $+0.489$ |
| Cloned to 0.887 agreement | $+0.278$ | $+0.252$ | $+0.563$ | $+0.841$ |

At 1,600 episodes the two trunks differ by 0.35 on identical data, so no amount of extra data recovers what the representation discarded. The first attempt at this table reported the opposite, more data fitting worse, because each condition was scored on its own held-out split and the batches were never shuffled. The inversion is what exposed the design error.

The gate's own 200-episode dataset reaches only $+0.109$ in 20 epochs, which is both factors binding at once rather than a weakness of the method.

### Does it help reinforcement learning

Paired so both arms share an action-sampling stream, on 6 Archers and 10 Peasants against 121 Peasants, 25 iterations of 32 episodes. Collapse means a run that reached 0.95 at some iteration and finished with a last-five mean below 0.5. The run was extended twice, and the extension changed the answer, so both readings are recorded.

| Sample | Cold critic | Pre-fitted critic | Paired difference | Collapses |
|---|---|---|---|---|
| First 20 seeds | $0.901 \pm 0.039$ | $0.951 \pm 0.006$ | $+0.050 \pm 0.040$ | 2/20 against 0/20, $p = 0.244$ |
| 60 and 35 seeds | $0.923 \pm 0.015$ | $0.938 \pm 0.009$ | $+0.033 \pm 0.027$ | 2/60 against 0/35, $p = 0.396$ |

At twenty seeds the cold arm looked six times as variable and the two collapses looked like a pattern. Forty further cold seeds produced no more collapses at all, so the rate is 2 in 60 rather than 2 in 20, and at that rate the expected number in 35 pre-fitted runs is 1.2. Observing zero is unremarkable.

The conclusion is a negative one and should be read as such. Pre-fitting produces a much better critic and no measurable improvement in what stage 3 achieves on this matchup. Rollout value loss starts at 12.177 against 2.099 and ends at 0.280 against 0.132, so the critic really is informative from the first iteration rather than after roughly ten, and that advantage does not show up in the win rate. Both arms solve the matchup every time, which is the most likely reason: a matchup that every run solves cannot show which run solved it better.

Chasing the two collapses is what turned up the defect below, and the defect explains them better than the critic does.

The 60-seed sweep was killed at pre-fitted seed 34 when a verification gate relinked the worker binary underneath it. The completed runs are reported; nothing was rerun to fill the gap.

## The advantage-normalization collapse

Instrumenting the collapsing seeds showed the mechanism. Advantage normalization divides a batch by its own spread. Once a matchup is solved every episode scores alike, the spread collapses, and the division rescales what remains, which is value-function error, up to unit variance.

Cold critic, seed 1, the two iterations before it fell:

| Iteration | Win rate | Reward spread | Raw advantage spread | Value loss |
|---|---|---|---|---|
| 20 | 1.000 | 0.0000 | 0.0219 | 0.0008 |
| 21 | 1.000 | 0.0000 | 0.0195 | 0.0007 |
| 22 | 0.188 | 0.8921 | 0.7840 | 2.5638 |

A healthy raw spread on this matchup is 0.3 to 1.0, so dividing by 0.02 amplifies about fiftyfold, and four epochs of that drove a 1.000 win rate to 0.031 by iteration 23.

Two repairs were tried. Dropping the batch when every episode scores alike, which is what `train_group` already did for its groups, and flooring the divisor so a degenerate batch produces a small update instead of a huge one.

| Arm and seed | No guard | Drop the batch | Floor at 0.1 |
|---|---|---|---|
| Cold, seed 1 | 0.494 | 0.988 | 0.975 |
| Cold, seed 9 | 0.319 | 0.269 | 0.994 |
| Cold, seed 0, control | 0.931 | 0.931 | 0.938 |
| Pre-fitted, seed 9 | 0.963 | 1.000 | 0.900 |

Dropping fixes one collapse and makes the other worse. The trace explains why: at seed 9 the spread at iterations 17 to 20 is 0.013 to 0.018, small enough to amplify fiftyfold but far above a $10^{-6}$ threshold, so the drop never fires before the collapse. Afterwards the spread is exactly zero because the policy now loses every episode, the drop does fire, and it blocks every remaining update, freezing the run at 0.000. A guard meant to protect a winning policy traps a collapsed one instead.

Flooring fixes both and leaves the control alone, so it is what both trainers now use. The floor sits at 0.1, which is not derived from anything: the terminal reward spans $[-1, 2]$ so its natural scale is about one, and a floor of 0.1 caps amplification near tenfold against that scale. A healthy raw spread on the matchups measured here is 0.3 to 1.0 and a degenerate one 0.02 to 0.07, so the value separates them with room on both sides, which is the whole of its justification.

### Across twenty seeds

| Arm | Last-five win rate | Spread | Worst run | Collapsed |
|---|---|---|---|---|
| Unfloored | $0.903 \pm 0.039$ | 0.174 | 0.319 | 2/20 |
| Floor at 0.1 | $0.961 \pm 0.009$ | 0.039 | 0.856 | 0/20 |

The collapse comparison is again $p = 0.244$, and at a rate near 2 in 60 no affordable number of seeds settles it by counting. The evidence that the floor works is mechanistic rather than statistical: the instrumented trace shows the amplification, and rerunning the two seeds that collapsed turns 0.494 into 0.975 and 0.319 into 0.994 while leaving a seed that never collapsed at 0.931 against 0.938.

### What the reward design was already protecting against

Raising the event rate looked like the way to settle it, since a matchup already solved should sit in the degenerate regime from the first iteration. It does not. On 50 Peasants against 30, which the cloned policy wins every time, the reward spread is 0.166 rather than zero and no iteration of any seed fell below the floor.

The margin-weighted terminal reward is why. It is $\pm 1 + h_T/h_0$, so with the win-loss bit constant the surviving-force term still varies between episodes and keeps the spread alive. That is the property [[../../rl/reward-design]] claims for it, holding in the one case where it matters most.

The collapse therefore needs episodes identical in outcome and in surviving force, not merely a matchup that is always won, which is a much narrower condition and accounts for the low rate better than "solved" does.

## Defects found by running things

Recorded because each was invisible to the tests that existed at the time.

- The relative margin reward, $(h^{\text{own}} - h^{\text{foe}})/(h^{\text{own}} + h^{\text{foe}})$, is 1.0 whenever the loser is wiped out, so a pyrrhic win scored identically to a clean one.
- `calibrate` returned the probe closest to its target, a maximum over noisy estimates and so optimistically biased. A calibration reporting 0.42 measured 0.19 when re-run.
- `total_variation` trusted its caller to have masked the logits; softmax over unmasked logits leaks probability onto illegal actions.
- The creature name table, hand-maintained, silently rejected every creature above id 20.
- `build_worker.sh` relinks without recompiling the agent library, so a source change reported success while the binary was unchanged.
- Byte-parsing the map produced 24,576 Genies, because monster counts are randomised during load rather than stored.
- Advantage normalization divided by an unfloored spread, amplifying value-function error about fiftyfold once a matchup was solved and collapsing 2 of 20 runs. Both trainers carried it; only `train_group` had a partial guard, and that guard fires on the wrong side.
- `build_pool` documented that it records which policy calibrated a pool. It did not, so a pool calibrated for one checkpoint could be reused with another and would silently be a pool of easy matchups. It now records a fingerprint of the weights.
- The proposal for pre-fitting rested on value data being more plentiful than policy data. Both are read from the same 45,380 rows, so the asymmetry does not exist at that stage.

## Related

- [[../../rl/training-design]], the conclusions about method.
- [[../../rl/scenario-distribution]], the conclusions about the distribution.
- [[../../decisions/0005-training-and-reward]], which none of this has yet amended.
