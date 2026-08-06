---
title: "The planner probe, the difficulty-weighted reward, and battlefields, 2026-08-05"
type: experiment-log
updated: 2026-08-05
tags: [agent-env, archive, experiment, dagger, reward, seeds]
---

# The planner probe, the difficulty-weighted reward, and battlefields, 2026-08-05

Three owner directives in one message: settle whether the built-in planner can be queried without advancing the arena, and settle it with evidence rather than reading; weight the terminal reward by fight difficulty so easy wins and lopsided losses stop distorting the gradient; and make battles vary over battlefields. This log carries all three, plus the discovery the third one forced. Conclusions live in [[../../rl/training-design]], [[../../rl/reward-design]], and [[../../rl/scenario-distribution]]; run reports are vendored under `files/2026-08-05-run-reports/`.

## The planner query, resolved yes

The reading came first and predicted the answer. `AI::BattlePlanner` contains no `Rand::` call site in either of its translation units, so planning consumes nothing from the combat stream; `analyzeBattleState` takes the arena const and reassigns every member fresh per call, so the singleton's state never carries information between queries; and the pathfinder cache a query warms is the same cache the action-space enumeration has warmed at every decision since Milestone 3, under gates that hash terminal states. The one hazard found was the entry point: public `BattleTurn` also runs the attacking side's turn-limit bookkeeping, whose forced-retreat path asserts on retreat-incapable commanders, so the engine gained a three-line public `queryUnitTurn` forwarding to the private planner instead.

The evidence is `planner_query.py`: a deterministic scripted policy plays each configuration twice, plain and with `--probe-teacher` querying the planner at every controlled decision, and the paired terminal state digests must match. Over 100 paired episodes covering all five fixtures, all three controlled sides, four world seeds each, and 20 budget-pool matchups with commanders and wide units, every pair matched. The probe resolved 4,297 of 4,297 teacher choices into `simple_v1`, which is the DAgger label rate, at about 19 percent wall-time overhead. Verdict: the planner is queryable, DAgger is unblocked, and the mechanism ships as the worker's `--probe-teacher` emitting `teacher_action` per decision. Scope limit: verified for spellbook-less commanders; the spell-planning path never ran and needs its own pass before spellcasting heroes enter.

## Battlefields, and the plumbing that was dead

The owner endorsed evaluating over battlefields, and wiring it exposed a structural surprise: `BattleEnv.reset()` killed and respawned its worker every episode, so the worker only ever played the first scenario of its list and the `seeds` parameter had been dead configuration since the class existed. Every number this project has reported was measured on one obstacle layout per matchup, while [[../../rl/scenario-distribution]] stated the over-seeds doctrine the loop never implemented. The fix is reset continuation: between episodes the worker stays alive and advances through its seed variants, a fresh process restarts the cycle, and a mid-episode abandon still respawns. The environment also now reports `scenario_id` per episode, which is how a caller tells battlefields apart, and `MatchupPool` keeps its environment across same-matchup episodes so the rotation survives grouping.

`battlefield_spread.py` then measured what one layout had hidden: clone v4 on 12 budget-pool matchups, six battlefields each, 24 episodes per battlefield. Per-battlefield win rates spread 0.137 on average against a binomial expectation of 0.086, excess variance worth about 0.11 of win rate from the layout alone, and 2 of 12 matchups exceeded twice binomial. The worst case was matchup 10, at 0.42 on its calibration battlefield and 0.00 on five others, a matchup the pool believes is in the training band and mostly is not. Implications recorded in [[../../rl/scenario-distribution#The battlefield term, measured 2026-08-05]]: calibrate and evaluate with several seeds from now on, and treat the battlefield as a free diversity axis for the generalization problem.

## The difficulty-weighted reward, designed and measured

The owner's direction: reward should scale with fight difficulty, the ratio of opponent to own army strength, so easy victories are not over-rewarded and lopsided losses not over-penalized. The implemented candidate prices both starting armies by engine creature strength, clips the enemy-to-own ratio into $[1/4, 4]$, takes $w$ as its square root, and multiplies wins by $w$ and losses by $1/w$. The design analysis in [[../../rl/reward-design#The difficulty-weighted candidate, owner-directed 2026-08-05]] says where this can and cannot bite: group-standard-deviation normalization cancels a pure per-matchup scaling and a critic absorbs it, so the live effect is the win-loss asymmetry within mixed groups and the cross-matchup reallocation in plain PPO batches.

The measurement ran both arms from clone v4 on the budget pool, 40 training and 20 held-out matchups, 40 PPO iterations, three paired torch seeds, scored on raw unweighted win rate. Clone baseline 0.473 training, 0.478 held-out.

| Arm | Training win rate | Held-out win rate |
|---|---|---|
| Unweighted | 0.602 | 0.403 |
| Difficulty-weighted | 0.599 | 0.396 |

Paired differences, weighted minus unweighted: $-0.003 \pm 0.017$ on training matchups and $-0.007 \pm 0.038$ held out. A clean null at this scale, and consistent with the design analysis rather than surprising: the trained critic absorbs per-matchup scaling into its baseline, evaluation stands outside the weighting entirely, and what survives, the mixed-group asymmetry and cross-matchup reallocation, is evidently below three seeds' resolution. What three seeds cannot rule out is a within-band reallocation, better on hard matchups paid for on easy ones, which per-matchup difficulty bins on a larger run would resolve. The candidate stays implemented, off by default, documented as measured-null-so-far; [[../../decisions/0005-training-and-reward]] still decides adoption.

The run's sharper finding is beside its question. Training rose from 0.473 to about 0.60 in both arms while held-out fell from 0.478 to about 0.40, per-seed held-out numbers swinging from 0.338 to 0.503 for the same arm, so on this pool the training gain did not transfer and may have cost something, echoing the capstone null. The battlefield result above says part of that swing is measurement: held-out evaluation here ran on one battlefield per matchup, whose layout term alone is worth about 0.11 of win rate. The next generalization run should evaluate over seeds before concluding anything about transfer.

## Alongside

The same day added the durable-tracking mechanics this log depends on, vendored run reports and anchor checkpoints under `files/`, and the documentation gate's orphan-script check, all recorded in the commit history rather than here.
