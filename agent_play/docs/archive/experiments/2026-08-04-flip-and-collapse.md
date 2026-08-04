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

## The solved-region width probe

Registered prediction, from the script before the numbers existed: the one matchup that collapses degrades fastest under parameter noise.

PENDING_MARGIN

## Sources consulted

Bay and Yearick, arXiv 2607.00152, carried the identity behind the identical divergence arms as its Proposition 1 and is written up in [[../../research/works/group-std-identity]]. Its binary-reward bound on the group spread is what separates the LLM literature's difficulty-bias reading of studentization from the unbounded amplification measured here.

Liu et al., arXiv 2503.20783, the Dr. GRPO paper, names the question-level difficulty bias that studentization induces, which is the group-level face of the same division.

Andrychowicz et al., arXiv 2006.05990, the large-scale on-policy study, treats per-minibatch advantage normalization as a minor detail with no strong recommendation. Their benchmark is five dense-reward continuous-control tasks, where the batch spread never degenerates, so the folklore verdict that normalization is harmless is conditional on a regime this project's terminal-reward battles leave exactly when a matchup is solved.

Qi et al., ICML 2026, the DPPO paper in [[../../research/works/dppo-trust-region]], supplies the divergence mask and tunes its threshold per setting, with nothing carrying the choice across distributions.

## Related

- [[2026-08-03-training-runs]], the runs these questions came from.
- [[../../rl/training-design]], which carries the conclusions.
- [[../../rl/rlhf-transfer]], for the estimator side.
