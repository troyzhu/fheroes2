---
title: "The audit, three measurement defects, and where the signal actually is, 2026-08-08"
type: experiment-log
updated: 2026-08-08
tags: [agent-env, archive, experiment, audit, distillation, search]
---

# The audit, three measurement defects, and where the signal actually is, 2026-08-08

The owner asked for an audit of the repository and documentation and for a researched plan to exceed the built-in AI. Fifteen agents ran over the tree and the literature, four auditing structure, facts, currency and writing, five researching levers with an adversarial verifier behind each. Their findings are applied in the commits of this day; this log records the three measurement defects the pass exposed, because each one changed a number that had already been reported, and the one new measurement that changes where the program should aim.

## Three defects, and what each had been distorting

The battery's reward column measured the wrong objective. `scenarios.measure` built its environment without passing `reward_margin`, so every `rw` column since the column existed reported the hit-point margin no matter what the checkpoint had trained on, while the conventions file claimed the column followed the run's configuration. It surfaced only when the built-in AI's own quality columns were computed for the first time and both sides of a comparison had to go on one scale. Under the corrected objective the KL leash's recovery is essentially complete rather than three fifths, so the defect had been understating the day's main result.

The search collector broke visit ties by action index. `max(visits, key=visits.get)` returns the lowest key, and `visits` is keyed in ascending legal-action order, so a tie resolved to array position. On the scaled corpus 1,669 of 15,007 decisions (11.12 percent) were tied at the top and every one of them was labeled by position. Ties now break on the mean rollout value, which is the quantity the search actually measured; relabeling the corpus offline moved 721 labels, one in twenty.

Every self-play round had trained one chair. `learner_side` defaulted to attacker and three rounds of copied scratch drivers never changed it, while two of the three largest remaining gaps to the engine are on the defending side. The driver is now one indexed script with the chair as a flag and per-chair matchup validation, and a related defect fell out of the same reading: `win_rate` scored every episode against one fixed termination string, so under alternating chairs every defender episode read inverted. That was caught while the first alternating round was in flight, and the run was discarded rather than reported on an inverted column.

## The tie-break was load-bearing

Redistilling the relabeled corpus against its hard twin, three seeds, changes the verdict the scale test had reached. Against the pre-relabel deltas, held-out holds at $+0.027$ against $+0.032$, but the Thunk ladder moves from $-0.003$ to $+0.066$ with the reward column agreeing at $+0.166$, and two suites that had been positive turn slightly negative. A coherent move of that size on the flagship suite is above the measured three-seed band, so the audit's own falsifier fires: distillation verdicts computed on the pre-relabel corpus are not safe to quote, and the value-target arm looks better after the fix than before it (`battery_relabel.json`, `relab_distill_s*.json`).

## Where the signal actually is

The distillation program has rested on a statistic that was never measured where it matters. Search agrees with the prior's argmax on about 96 percent of decisions, which reads as "search mostly confirms the policy, so there is little to clone". Every corpus in the record was collected under a win filter, on matchups search wins, which are largely matchups the prior already wins. The decisive positions are the opposite ones: a handful of held-out matchups the policy loses at 0.00 to 0.21 and search wins at 0.50 to 1.00.

`deviation_probe.py` measured agreement on both groups, under both search modes, over 1,219 decisions:

| Search mode | On matchups the policy loses | On matchups it wins | Ratio |
|---|---|---|---|
| Concentrating UCB, the historical statistic's mode | deviation 0.145 | deviation 0.052 | 2.79 |
| Coverage-forced sweep | deviation 0.345 | deviation 0.202 | 1.71 |

The magnitude of each disagreement separates the groups further than its frequency does. Under UCB, a disagreement on a losing matchup is worth $+0.787$ in the reward units search measured, against $+0.152$ on a winning one; two of the four winning matchups produce essentially no disagreement at all (agreement 0.986 and 1.000, value gained 0.005 and 0.000), while every losing matchup produces both frequent and valuable ones.

So the 96 percent figure is a win-filter artifact, and the reading it supported was wrong. There is an action-level signal, it is about three times denser and five times more valuable per instance in exactly the positions that carry the gap to the built-in AI, and no corpus has ever been collected there: `--min-win 0.5` exists to keep out the least-bad line of a lost position, and it has been excluding the only states where the teacher has something to teach. The instruction this yields is specific rather than architectural, which is what makes it worth acting on before any of the modeling levers: collect where the policy loses and search wins, gate on search's counterfactual at the labeled state rather than on the played episode's outcome, and keep the coverage-forced sweep so every candidate carries a real rollout (`deviation_probe.json`).

## What this does not say

It does not say the gap is closable. It says the gap has an action-level component that is measurable and has never been in a training corpus, which is a different and weaker claim. The deviation rates are single-checkpoint and single-suite, the four losing matchups are four, and the two search modes disagree by a factor of two on the same positions, so the absolute numbers are soft even where the contrast is stable. What licenses acting on it is the direction and the size of the value gaps, not the precision.

## Weighting the loss by regret, which is what the finding predicts

If 93.2 percent of a corpus carries no regret, an unweighted loss spends almost all of its gradient teaching the policy choices it already makes. The arm that follows is cheap and paired: the same relabeled corpus, the same soft value targets, the same three seeds, with soft rows weighted by rank-transformed measured regret and the multiplier renormalized to mean one, so the weighted arm carries exactly the soft mass of its unweighted twin and the only difference is where that mass sits.

| Suite | Regret-weighted | Uniform soft | Hard twin |
|---|---|---|---|
| Held-out pool | 0.535 / $+0.64$ | 0.472 / $+0.50$ | 0.433 / $+0.42$ |
| Thunk ladder | 0.767 / $+1.20$ | 0.747 / $+1.12$ | 0.632 / $+0.88$ |
| Mirrors as defender | 0.336 / $+0.17$ | 0.303 / $+0.12$ | 0.275 / $+0.07$ |
| Held-out as defender | 0.266 / $-0.07$ | 0.274 / $-0.05$ | 0.260 / $-0.08$ |
| Commanders | 0.962 / $+1.75$ | 0.958 / $+1.75$ | 0.962 / $+1.76$ |
| Fresh sampled | 0.380 / $+0.14$ | 0.385 / $+0.16$ | 0.376 / $+0.14$ |
| Real maps | 0.560 / $+0.64$ | 0.566 / $+0.64$ | 0.560 / $+0.63$ |

The held-out gain over the unweighted twin is $+0.063$ on rate and $+0.135$ on the trained reward, and it is paired per seed at $+0.079$, $+0.035$ and $+0.075$, every seed the same sign, against a same-checkpoint band of about $\pm 0.03$. The ordering regret-weighted above uniform above hard holds on held-out, on the ladder and on the defender mirror, and the three suites that do not move are the ones already at par with the engine. Against the hard twin the total is $+0.102$ on held-out, which is the largest paired distillation effect this project has measured.

Read carefully, this is not a crossing and not a new best policy: at 0.535 the arm sits where the supervised anchor already sits, inside its re-evaluation band. What it establishes is mechanism. The soft-target program was not weak because search had nothing to teach, it was weak because the corpus was overwhelmingly composed of decisions where search had nothing to teach, and pointing the same gradient at the 6.8 percent that carry regret recovers an effect three times the noise band from data already on disk. The next question is whether collecting more of that 6.8 percent, rather than reweighting what exists, compounds it (`battery_regret.json`, `regw_distill_s*.json`).

## Training both chairs is worse at matched budget

The scoreboard's reading was that the defending chair is not the harder one, only the untrained one, since the engine's own mirror split gives the defender 0.639 and every round so far had trained the attacker alone. The arm that follows is one flag on the wide leashed recipe, three seeds, chair drawn per episode from a matchup set validated as playable from both, against round four's attacker-only trio as the control. All three runs converged, the leash tension included at KL 0.13.

The hypothesis fails, and it fails on its own target suite. Against the attacker-only control the both-chair trio reads mirrors as defender $0.241$ against $0.326$, and the per-seed values do not overlap at all, 0.201, 0.271 and 0.250 against 0.278, 0.333 and 0.368. The Thunk ladder falls 0.094, held-out 0.030, mirrors as attacker 0.042. What moves the intended way is small: held-out as defender $+0.024$, fresh samples and commanders $+0.014$ each. Symmetry is unchanged, the both-chair seed reading excess $-0.221$ against the attacker-only seed's $-0.225$, so the chair split did not even move the side balance it was most likely to move.

The reading that survives is that chair experience is not the binding constraint. Attacker-only gradients already reach defender play, which the round-four numbers had shown and this arm confirms by removing half the attacker experience and losing defender play with it, and at a matched budget the split simply halves the data behind each chair. Recorded as a measured negative: the defender gap is real, and it is not explained by which chair was trained (`battery_round5.json`, `convergence_round5.json`, `symmetry_round5.json`).

## The deployment rule is checkpoint-dependent, which is not what was expected

Every weights-only number in the record was measured with the policy sampling its distribution, while the built-in AI is deterministic and pays no such penalty. The obvious correction is to evaluate greedily, and the battery now takes a deployment rule so the question can be settled on all ten suites rather than on one slice.

The answer refuses to be a single rule. On the supervised anchor, greedy is worse nearly everywhere: held-out $-0.015$, the ladder $-0.042$, the defender mirror $-0.132$, with only commanders gaining. On the regret-weighted distillation, greedy is better nearly everywhere: held-out $+0.069$ to 0.613, mirrors as attacker $+0.146$, commanders $+0.073$. On the leashed reinforcement checkpoint it splits again, the ladder $+0.115$ and commanders $+0.062$ against the defender mirror $-0.097$. The entropy-adaptive nucleus sits between the two on most cells, which is what it was designed to do.

So sharpening the deployment rule is not free reporting recovered; it is a property of how well a particular checkpoint ranks its actions. A policy whose argmax is trustworthy gains from committing to it and a policy whose argmax is no better than its second choice loses the hedging that was covering it. 

The honest form of the rule is per checkpoint and stated: measure both, adopt the better on a validation split, report which was used. The three-seed confirmation then trims the headline, which is why it was flagged as pending. Across seeds the regret-weighted arm reads 0.546 greedy against 0.535 sampled, a gain of $+0.011$ rather than the $+0.069$ its best seed showed, and the 0.613 belongs to that seed rather than to the arm. The uniform and hard arms gain more from greedy, $+0.032$ and $+0.038$, so the sharpening premium is largest where the policy is weakest and the ordering across arms is unchanged, regret-weighted 0.546 above uniform 0.504 above hard 0.471.



Two things survive the trim. Greedy helps every distilled arm and hurts the supervised anchor, so the rule really is checkpoint-dependent rather than universal. And the best weights-only reading this project has, 0.546 on held-out under a stated deployment rule, is still 0.11 short of the engine's 0.660, which is the number that matters and which no amount of deployment bookkeeping closes (`battery_deploy_*.json`, `battery_greedy_3seed.json`).
