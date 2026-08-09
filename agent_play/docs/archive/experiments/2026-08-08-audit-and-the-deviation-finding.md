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
