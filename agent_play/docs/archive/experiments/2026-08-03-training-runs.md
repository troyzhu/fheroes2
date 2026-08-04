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

## Defects found by running things

Recorded because each was invisible to the tests that existed at the time.

- The relative margin reward, $(h^{\text{own}} - h^{\text{foe}})/(h^{\text{own}} + h^{\text{foe}})$, is 1.0 whenever the loser is wiped out, so a pyrrhic win scored identically to a clean one.
- `calibrate` returned the probe closest to its target, a maximum over noisy estimates and so optimistically biased. A calibration reporting 0.42 measured 0.19 when re-run.
- `total_variation` trusted its caller to have masked the logits; softmax over unmasked logits leaks probability onto illegal actions.
- The creature name table, hand-maintained, silently rejected every creature above id 20.
- `build_worker.sh` relinks without recompiling the agent library, so a source change reported success while the binary was unchanged.
- Byte-parsing the map produced 24,576 Genies, because monster counts are randomised during load rather than stored.

## Related

- [[../../rl/training-design]], the conclusions about method.
- [[../../rl/scenario-distribution]], the conclusions about the distribution.
- [[../../decisions/0005-training-and-reward]], which none of this has yet amended.
