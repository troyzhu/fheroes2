---
title: "The trust-region flip and the collapse condition, 2026-08-04"
type: experiment-log
updated: 2026-08-04
tags: [agent-env, archive, experiment, training]
---

# The trust-region flip and the collapse condition, 2026-08-04

Two questions left open by [[2026-08-03-training-runs]]: why the divergence trust region's single-matchup gain vanished on the pool with its point estimate reversed, and what distinguishes a matchup where the advantage-normalization collapse can happen from one where it cannot. Method for both: mine the telemetry already on disk before spending compute, register the predictions, then run the cheapest experiment that separates the hypotheses. All new runs use a pinned copy of the worker so a rebuild cannot kill them.

## The flip was a horizon artifact

Three hypotheses were registered before the runs: the fixed threshold blocks so much on a heterogeneous pool that the arm is budget-limited at 30 iterations; the mask specifically harms cross-matchup learning even with more budget; or both readings are noise around a small true effect.

The recorded trajectories already leaned toward the first. Over the last five iterations of the 30-iteration pool runs the divergence arm was climbing at $+0.054$ per five iterations against the ratio arm's $+0.020$, meaning it was cut off mid-climb.

The decisive test is a continuation rather than a re-run. The same five seeds were run to 60 iterations, and because both trainers are deterministic given a seed, the first 30 iterations of the new runs are bit-identical to the recorded ones, verified elementwise across all ten trajectories. The same runs that read $-0.031 \pm 0.037$ at iteration 30 read $+0.034 \pm 0.031$ at iteration 60, the sign of the single-matchup result restored, at the same magnitude that result had.

| Reading | Divergence minus ratio, paired |
|---|---|
| Single matchup, 25 iterations, 10 seeds | $+0.036 \pm 0.014$ |
| Pool, 30 iterations, 5 seeds | $-0.031 \pm 0.037$ |
| Pool, the same runs at 60 iterations | $+0.034 \pm 0.031$ |

Telemetry, now that the scripts keep it: at threshold 0.05 the mask suppresses the gradient of 16.5 percent of samples in the first ten iterations, decaying to 12.8 percent by the last ten, while the ratio arm clips 8.3 falling to 6.6 percent. Twice as much of the gradient is being withheld, on a 90-matchup task where early movement is what the budget buys.

| Threshold, 30 iterations | Shifted fraction | Last-five win rate | Against the ratio clip, paired |
|---|---|---|---|
| 0.05 | 0.146 | $0.584 \pm 0.037$ | $-0.031 \pm 0.037$ |
| 0.10 | 0.079 | $0.561 \pm 0.038$ | $-0.054 \pm 0.019$ |
| 0.20 | 0.020 | $0.660 \pm 0.021$ | $+0.045 \pm 0.013$ |

The blocked fraction is monotone in the threshold and the win rate is not, which at five seeds means the middle row is noise and only the ends carry information. At 0.20 the mask fires on two percent of samples, the arm is close to unconstrained policy gradient, and it beats the clipped baseline at the same budget. The consistent reading across all of it: constraint strength trades early speed, and none of the constraints has yet been needed for stability in this setting at this learning rate.

So the resolution is that nothing here says the divergence trust region hurts on a distribution. The 30-iteration reading caught a slower-starting arm mid-climb, and [[../../research/works/dppo-trust-region]] offers no guidance on carrying the threshold across state distributions, which is the parameter the whole difference rode on. The trainer defaults stay at leave-one-out under the ratio clip: two settings have now each shown a small divergence gain at their own natural budget, and neither separates cleanly enough to move a default.

## The dip census over every recorded run

A dip is a run that reached a win rate of 0.95 and later fell to 0.60 or below, which at 32 episodes an iteration cannot be sampling noise. A terminal collapse is a dip that never recovered. Mining every run with a stored history on the contested matchup, 6 Archers and 10 Peasants against 121 Peasants:

| Arm | Runs | Dipped | Recovered | Terminal |
|---|---|---|---|---|
| Cold critic, unfloored | 20 | 2 | 0 | 2 |
| Pre-fitted critic, unfloored | 20 | 1 | 1 | 0 |
| Cold critic, floored | 20 | 2 | 2 | 0 |
| Safe matchups, all four arms | 80 | 0 | 0 | 0 |

This refines the floor's story in a way the earlier write-up missed. Knock-offs happen under the floor at about the same rate as without it; what changes is that every floored dip recovered and both unfloored dips that went terminal did so. The floor's measured benefit is recovery, not knock-off prevention. Why the floored runs dipped cannot be answered from the record, because the experiment script at the time stored win rates alone, which is the telemetry gap fixed today: both experiment scripts now keep the full per-iteration history, and `train_group` records the raw advantage spread the way `train_ppo` already did.

## Census on three more contested matchups

Twelve seeds per arm, 40 iterations, on three pool matchups spanning the length range, registered prediction: contested matchups dip at a nonzero rate once solved, safe ones never do. Full per-iteration telemetry kept this time, so any dip gets an autopsy.

| Matchup | Calibrated | Unfloored dips | Floored dips | Exposure | Minimum spread |
|---|---|---|---|---|---|
| 7:1 against 7:1, short | 0.38 | 0/12 | 0/12 | 0.69 | 0.0033 |
| 5:2,4:3,1:50 against 4:4,3:4 | 0.50 | 0/12 | 0/12 | 0.41 | 0.0337 |
| 4:3,3:5,7:2 against 2:15, long | 0.56 | 1/12 | 0/12 | 0.40 | 0.0321 |

Exposure is the fraction of an unfloored run's iterations spent with a raw advantage spread below 0.1, meaning the fraction on which the unfloored divisor amplified by more than tenfold. It is the number that reframes the mechanism: these runs spend 40 to 69 percent of their iterations amplifying, and dip almost never. Amplification is the permanent background condition of a solved matchup, not a rare accident, and the knock-off is a rare event within it.

The one dip that did occur carries the signature the prediction asked for. Unfloored, seed 5 of the long matchup, at iteration 28: the iteration before shows a raw advantage spread of 0.0475, a reward spread of 0.0224 and a value loss of 0.0024, which is the same near-degenerate, well-fit-critic state the two matchup-A collapses died in. It fell to 0.469 and recovered fully, finishing at 0.998. The floored twin of the same matchup did not dip at all.

Against the registered prediction: one dip in 36 unfloored contested runs, where a rate uniform with matchup A's 3 in 40 would have produced zero dips with probability 0.06 and one or fewer with probability 0.23. So the phenomenon generalizes, the signature travels with it, and matchup A is still several times more dip-prone per run than any other matchup measured. Adjusting roughly for exposure, its knock-off rate per amplified iteration is near an order of magnitude above the census matchups', which is what the solved-region probe below was built to explain.

The safe matchups rerun with telemetry show the same background more strongly: exposure 0.87 and 0.72 of iterations, spreads reaching $10^{-4}$, and zero dips in six runs on top of the eighty earlier ones recorded without telemetry. The earlier conclusion that the reward's survival term keeps these matchups above the floor rested on the unfloored arm's counter, which counts spreads below its own $10^{-8}$ parameter and says nothing about 0.1; the correction is recorded in [[2026-08-03-training-runs#What the reward design was already protecting against]], and exposure here is computed from the recorded spreads instead.

## The solved-region width probe

Registered prediction, written into the script before the numbers existed: the one matchup that collapses degrades fastest under parameter noise. Method: train one floored run per matchup to plateau, add Gaussian noise scaled per tensor by that tensor's own spread, and measure the win rate as the noise grows, three draws per scale, 24 episodes per draw. A first pass with scales up to 0.10 left every curve flat, so the scales were widened until something moved.

| Matchup | Plateau | 0.10 | 0.20 | 0.30 | 0.50 | Unfloored dips on record |
|---|---|---|---|---|---|---|
| 6 Archers, 10 Peasants against 121 Peasants | 0.931 | 0.94 | 0.88 | 0.79 | 0.51 | 3 of 40, two terminal |
| 7:1 against 7:1 | 0.969 | 0.96 | 0.96 | 0.93 | 0.82 | 0 of 12 |
| 5:2,4:3,1:50 against 4:4,3:4 | 0.994 | 1.00 | 1.00 | 1.00 | 0.94 | 0 of 12 |
| 4:3,3:5,7:2 against 2:15 | 0.988 | 1.00 | 0.99 | 0.99 | 0.96 | 1 of 12, recovered |
| 1:50 against 1:30 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 0 of 23 |
| 2:10,1:20 against 1:60 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 0 of 23 |

The prediction holds at the extremes. The only matchup with terminal collapses is the narrowest by a wide margin, losing half its wins at scale 0.5 where the next-narrowest holds 0.82, and the two matchups completely flat at that scale have never dipped in 46 recorded runs. The middle is not resolved: the second-narrowest produced no dips in twelve seeds and the one that produced the single transient dip barely degrades, so at one dip and six matchups the width predicts the extremes and says nothing about the ordering between them. One training seed per matchup and one noise model, so this is a measurement of a proxy, not a theory of the geometry.

## The mechanism, assembled

Solving a matchup drives episode outcomes toward identical, the raw advantage spread toward zero, and the unfloored normalization then rescales critic residue to unit size, on 40 to 87 percent of post-solve iterations across every matchup measured. Those unit-size noise steps random-walk the policy. Whether the walk ever exits the winning region depends on the region's width: matchups flat under half-scale parameter noise never dipped, and the narrowest matchup measured produced every terminal collapse on record. The floor does not prevent the knock-offs, whose rate is similar floored and unfloored on the vulnerable matchup; it shrinks the steps, and every floored dip recovered while both unfloored terminal collapses did not. Why the floored runs still dip at all is not answerable from the record, because those histories were stored as win rates alone before today's telemetry fix.

## The Thunk opening fight, pinned down

The owner has now corrected this fight three times, and each correction exposed a different defect in how the map data was read. The first identification, Catarina's Genie battle, was the wrong hero. The second, Ta Arg-Majj's Goblins against 1,000 Peasants, was a proximity-pairing artifact, and a full measurement of that matchup, a cliff at 115 Peasants that frontier training moves by zero, stands as a result about that matchup while its "the opening fight is a recruitment problem" conclusion is retracted with it. The third pass started from the owner's description and a clean dump, and everything in the description verifies.

The fight: Corribus, the Blue player's hero at (95,45), attack 13 and defense 12 as dumped, army 1 Crusader, 1 Crusader, 1 Crusader, 2 Paladins, 2 Champions, against the 1,000-Peasant stack two tiles away at (95,43). The owner recalls attack 14 and defense 9, a discrepancy left open, plausibly a map revision or in-game leveling. A neutral stack fights split into `Rand(3..5)` sub-stacks as evenly as possible, `Army::ArrangeForBattle` in `army.cpp`, which is the owner's 334/333/333. Two earlier passes lost pieces of this: the extraction filter silently dropped the Champions because they are wide, and a peasant search matched nothing because the dump prints plural names.

Two of the fight's load-bearing elements are outside the current environment: hero commander stats, which the scenario schema simply does not carry, and Champions, which are wide and outside `simple_v1`. Both are now tracked as work items with this fight as the acceptance test. What can be measured today is the army minus Champions with no commander, and it already reshapes the picture.

| Peasants | 100 | 200 | 300 | 400 | 500 | 700 | 1000 |
|---|---|---|---|---|---|---|---|
| Against one stack | 1.000 | 0.562 | 0.125 | 0.625 | 0.000 | 0.000 | 0.000 |
| Against the three-way split | 1.000 | 1.000 | 0.938 | 0.562 | 0.062 | 0.000 | 0.000 |

The split helps the attacker at every count, opposite to a first guess: three stacks striking for a third each is what single-Crusader bait stacks are for. And unlike the Goblin cliff, this frontier is a slope, which means tactics matter here. Training 30 iterations at the 400 frontier lifts 400 from 0.562 to 0.833 and opens 450 to 0.292 while 500 stays shut at 0.042, so learning moves this frontier where it moved the Goblin one not at all.

The real fight remains open at this point in the log: at 1,000 the commander-less, Champion-less approximation is 0.000, and the factor the missing pieces must supply is what defense 12 against Peasant attack 1 and two more fast bodies are worth. The commander work item was built the same evening, and the next section carries what it measured.

## The commander closes the gap

Hero commander support landed as `agent_commander.h`, a minimal `HeroBase` carrying primary stats and nothing else, attached through `Army::SetCommander` when a scenario asks for one. Absent commanders leave every code path untouched, which the milestone gates' golden digests prove, and the protocol test pins the stat flow exactly: with a 30:20 commander every Peasant on that side observes attack 31 and defense 21 while the other side stays at 1 and 1.

With Corribus's dumped stats attached, the same army against the same splits:

| Peasants | 400 | 500 | 700 | 850 | 1000 |
|---|---|---|---|---|---|
| No commander | 0.562 | 0.062 | 0.000 | — | 0.000 |
| Commander 13:12 | 1.000 | 1.000 | 0.667 | 0.167 | 0.000 |

The frontier moves from roughly 450 to roughly 800 on stats alone, and 850 lands inside the training band. Two curriculum stages from there, 30 iterations each, single seed:

| Stage | Trained at | 700 | 850 | 900 | 950 | 1000 |
|---|---|---|---|---|---|---|
| Clone | — | 0.667 | 0.167 | — | 0.000 | 0.000 |
| 1 | 850 | 1.000 | 1.000 | 0.292 | 0.000 | 0.000 |
| 2 | 950 | 0.292 | 0.625 | 0.583 | 0.542 | **0.167** |

So the actual opening fight, still missing its two Champions, is won about one time in six by a curriculum-trained policy, from exactly zero at every earlier attempt. The fight is a training problem after all, and what it was gated on was the environment's missing commander, not the arithmetic. Two honest limits: one seed, 24 evaluation episodes, so 0.167 carries an error near 0.08; and stage 2 forgets the easier rungs, 700 falling from 1.000 to 0.292, which is the standard argument for training on a mixture of rungs rather than a ladder of single ones. The Champions stay tracked as the wide-creature work item, and closing them should only raise these numbers.

## Capacity, asked and measured

Prompted by the owner asking whether the trust-region result reflected limited model capacity. Cloning at three widths on the same 45,380 decisions, then PPO from each clone on the same 90 training matchups, 40 iterations, three paired seeds, everything else identical.

| Width | Parameters | Cloning agreement | Pool win rate |
|---|---|---|---|
| Half | 139,546 | 0.8699 | $0.615 \pm 0.010$ |
| Deployed | 396,570 | 0.8873 | $0.644 \pm 0.019$ |
| Double | 1,265,434 | 0.9013 | $0.602 \pm 0.013$ |

Two answers. Cloning agreement rises monotonically with width, so the clone is data-limited rather than capacity-saturated, and the recorded reason for sizing the network down from 626k, a widening train-holdout loss gap, does not show up as an agreement cost even at triple that size. Reinforcement learning at this budget points the other way: the double model is worse than the deployed one by 0.042 at about 1.8 standard errors over three seeds, which is what more parameters on the same 32-episode batches should do, and no ceiling is in sight since every width was still climbing at cutoff.

And nothing in the trust-region question was capacity to begin with: the flip reversed within a single model read at two horizons, and a capacity ceiling would bound both arms alike rather than reorder them.

## Wide units, and the complete fight

The last missing piece was the two Champions, excluded because they are wide. The exclusion turned out to be prudence rather than a gap: the melee enumeration already targets both cells of a wide defender, moves come from the engine's own pathfinder, and the open question, melee from a wide attacker, was adjudicated by teacher coverage rather than argued. Over 120 recorded episodes spanning Champions, Cavalry and Paladins on both sides, 1,238 of 1,238 teacher decisions resolved to an enumerated candidate and matched it, 526 of them taken by wide units. The `wide_v1` profile, opt-in and digest-inert when off, admits two-cell walkers and nothing else.

One known limitation, recorded rather than hidden: `obs_encoding_v2` carries the wide flag but not the tail cell, so a policy sees a wide unit's head and knows it is wide without seeing its orientation. The masks are engine-exact regardless; only policy quality is at stake, and the results below say it was not decisive here.

With the full army the fight could finally be posed as the game poses it, and a four-stage curriculum closes it. Stages at 850, 950 and 985 Peasants, then a final stage on the real fight itself, which stage 3 had brought inside the training band. Single seed per stage, 30 iterations each.

| Policy | 850 | 950 | 985 | 1000, the real fight |
|---|---|---|---|---|
| Clone, full army and commander | 0.125 | — | — | 0.000 |
| Stage 2 | 0.83 | 1.00 | — | 0.021 |
| Stage 3, trained at 985 | — | 0.75 | 0.65 | 0.438 |
| Stage 4, trained at 1000 | — | — | — | 0.891, 64 episodes |

The summary across the whole hunt, every configuration measured at the rolled fight, 334 and 333 and 333 Peasants:

| Configuration | Win rate at the real fight |
|---|---|
| No commander, no Champions, clone or curriculum | 0.000 |
| Commander, no Champions, two-stage curriculum | 0.167 |
| Full army and commander, clone | 0.000 |
| Full army, three-stage curriculum | 0.438 |
| Full army, fourth stage on the fight itself | 0.891 |

Everything the owner's correction insisted on was load-bearing: the split, the commander, the Champions. The environment now poses the fight faithfully up to the dumped stats, and the one-in-nine-hundred long shot every earlier pass reported was a measurement of missing features, not of the fight.

## Sources consulted

Bay and Yearick, arXiv 2607.00152, carried the identity behind the identical divergence arms as its Proposition 1 and is written up in [[../../research/works/group-std-identity]]. Its binary-reward bound on the group spread is what separates the LLM literature's difficulty-bias reading of studentization from the unbounded amplification measured here.

Liu et al., arXiv 2503.20783, the Dr. GRPO paper, names the question-level difficulty bias that studentization induces, which is the group-level face of the same division.

Andrychowicz et al., arXiv 2006.05990, the large-scale on-policy study, treats per-minibatch advantage normalization as a minor detail with no strong recommendation. Their benchmark is five dense-reward continuous-control tasks, where the batch spread never degenerates, so the folklore verdict that normalization is harmless is conditional on a regime this project's terminal-reward battles leave exactly when a matchup is solved.

Qi et al., ICML 2026, the DPPO paper in [[../../research/works/dppo-trust-region]], supplies the divergence mask and tunes its threshold per setting, with nothing carrying the choice across distributions.

## Related

- [[2026-08-03-training-runs]], the runs these questions came from.
- [[../../rl/training-design]], which carries the conclusions.
- [[../../rl/rlhf-transfer]], for the estimator side.
