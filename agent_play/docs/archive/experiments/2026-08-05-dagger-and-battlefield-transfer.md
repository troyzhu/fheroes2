---
title: "DAgger's first round, and transfer re-grounded on battlefields, 2026-08-05"
type: experiment-log
updated: 2026-08-05
tags: [agent-env, archive, experiment, dagger, generalization, seeds]
---

# DAgger's first round, and transfer re-grounded on battlefields, 2026-08-05

The owner left a three-hour autonomous block with the afternoon's three results fresh: the planner probe unblocked DAgger, battlefields were measured at 0.11 of win rate, and the transfer numbers of the difficulty run were suspect because they sat on single layouts. The evening ran the program those findings implied. Everything below evaluates over four battlefields per matchup, which is the protocol the afternoon said any transfer claim needs; reports are vendored under `files/2026-08-05-run-reports/` and the DAgger clone under `files/2026-08-05-checkpoints/`.

## Transfer, re-measured properly

`generalization_battlefields.py` reran pool training from clone v4 with two arms, episodes collected on one battlefield per matchup (the historical behaviour) against four rotated ones, three paired torch seeds, everything evaluated over four battlefields including the baseline.

The baseline moved first: clone v4 reads 0.447 training and 0.429 held-out over battlefields, against 0.473 and 0.478 on the single calibration layout, so the pools' calibration battlefields were systematically easier than their matchups, which is what the spread finding predicted.

| Arm | Training pool | Held-out pool |
|---|---|---|
| Clone v4 baseline | 0.447 | 0.429 |
| PPO, single battlefield | $0.487 \pm 0.009$ | $0.426 \pm 0.025$ |
| PPO, rotated battlefields | $0.505 \pm 0.013$ | $0.391 \pm 0.014$ |

Two conclusions. The afternoon's apparent held-out damage was substantially measurement: under honest evaluation, single-battlefield PPO shows a clean transfer null ($-0.003 \pm 0.025$ against baseline), the familiar shape, not a loss. And the battlefield-diversity hypothesis is defeated at this budget: rotation trains slightly better on-pool (paired $+0.018 \pm 0.008$) and transfers no better, if anything worse (paired $-0.035 \pm 0.023$). The generalization lever is not battlefield variety in the collection stream, at least not at 40 iterations.

## DAgger's first round

`dagger_iteration.py` ran the stage the probe unblocked at noon. Clone v4 played its 40 training matchups over four battlefields, 25 episodes each, 1,000 episodes, and the planner labeled every one of the 22,750 decisions the student actually reached. The clone retrained from scratch on the aggregate, 244,595 teacher decisions plus the 22,750 relabelings, 267,345 samples, reaching 0.8413 episode-split holdout agreement in 85 seconds of training; the previous clone's comparable figure was 0.8606 on teacher data alone, and the small drop is the mixed distribution, not a defect.

What matters is student-reached play, and there the result is the evening's headline.

| Measure | Clone v4 | DAgger clone |
|---|---|---|
| Training pool, over battlefields | 0.447 | $0.541 \pm 0.050$ |
| Held-out pool, over battlefields | 0.429 | $0.487 \pm 0.065$ |
| Thunk ladder 500 / 700 / 850 / 1000 | 1.000 / 0.917 / 0.250 / 0.000 | 1.000 / 1.000 / 0.917 / 0.667 |

Paired per matchup against the baseline on identical matchups and protocol: $+0.094 \pm 0.036$ on the training pool, 2.6 standard errors and real; $+0.058 \pm 0.053$ held out, directionally positive and unresolved at 20 matchups. The Thunk ladder is the independent validation no sampler drew, and one supervised round moved the full 1,000-Peasant fight from never won to 0.667, with 0.917 at the 850 rung where clone v4 read 0.250. The four-stage reinforcement-learning curriculum reached 0.891 on that fight after staged hours; one DAgger round recovers three quarters of that as a side effect while improving the whole pool, where the same afternoon's PPO runs improved nothing held-out.

The mechanism reading: cloning's compounding error concentrates exactly where the student drifts off the teacher's state distribution mid-battle, which is where the 850 and 1,000 rungs are lost, and relabeling student-reached states is aimed at that failure and nothing else. PPO from the same clone attacks the same matchups by reward and transfers none of it; the teacher's answer at the student's mistake transfers.

## The control that decides attribution, and the weighting accident inside it

The gain could still have been data volume rather than relabeling: the aggregate carries 22,750 more decisions than the teacher corpus. The intended control records 1,000 teacher-played episodes on the same 40 matchups, 25 battlefields each, and retrains identically. What actually ran was ten times that: the recording script omitted `--runs`, the worker's default is 10, and the teacher is deterministic per battlefield, so every unique episode was recorded ten times over, 10,000 files of which 1,000 are distinct, about 451,000 decisions of duplicated targeted data dominating two thirds of the aggregate. The accident was caught by arithmetic, a 718,825-sample corpus that no honest accounting produced, and the numbers below are therefore the numbers of a heavily upweighted targeted corpus, kept because upweighting is a legitimate condition once named. The corrected equal-weight arms follow in the next section.

| Measure | Clone v4 | Teacher control | DAgger clone |
|---|---|---|---|
| Training pool | 0.447 | 0.601 | 0.541 |
| Held-out pool | 0.429 | 0.444 | 0.487 |
| Thunk 500 / 700 / 850 / 1000 | 1.000 / 0.917 / 0.250 / 0.000 | 1.000 / 0.917 / 0.458 / 0.000 | 1.000 / 1.000 / 0.917 / 0.667 |

The split verdict is more informative than a clean win would have been. On the training pool, upweighted targeted teacher data is at least as good, 0.601 against 0.541, paired DAgger-minus-control $-0.060 \pm 0.038$; its 0.9232 cloning agreement against DAgger's 0.8413 says the pure-teacher distribution is also simply easier to fit. Off the training distribution the ranking inverts: held-out paired DAgger-minus-control $+0.044 \pm 0.069$, directional only, but the Thunk ladder separates them without ambiguity, the control at 0.458 on the 850 rung and still 0.000 at 1,000, the DAgger clone at 0.917 and 0.667. Volume alone, on the very matchups the student trained from and at ten times the weight, never taught the full fight; relabeling the student's own drift states did.

## The first combination, dominated

Aggregating everything, teacher corpus plus the duplicated control plus the relabelings, produced the best held-out pool number of the day and lost the off-distribution result: train 0.600, held-out 0.546 against the baseline's 0.429, Thunk 1.000 / 0.875 / 0.333 / 0.000. With the duplicated control at two thirds of the corpus and the relabelings at three percent, the mixture behaves control-like where distributions overlap, and the drift-recovery signal the Thunk rungs need is outvoted. What this measures is therefore mixture weighting, not any intrinsic incompatibility, and the corrected arms below hold the weights honest.

## The corrected arms, equal weight

After deduplication to the 1,000 unique control episodes, both arms reran identically: the control as teacher corpus plus unique targeted data (289,743 samples), the combination as that plus the relabelings (312,493 samples).

| Measure | Baseline | Control, equal weight | DAgger alone | Combination, equal weight |
|---|---|---|---|---|
| Training pool | 0.447 | 0.570 | 0.541 | 0.592 |
| Held-out pool | 0.429 | 0.423 | 0.487 | 0.552 |
| Thunk 500 / 700 / 850 / 1000 | 1.000 / 0.917 / 0.250 / 0.000 | 1.000 / 1.000 / 0.833 / 0.250 | 1.000 / 1.000 / 0.917 / 0.667 | 1.000 / 1.000 / 0.917 / 0.083 |

Three revisions to the earlier reading. Deduplication improved the control's off-distribution transfer outright, 0.833 and 0.250 on the hard rungs against the duplicated version's 0.458 and 0.000, so ten-fold duplication had been overfitting the targeted matchups and fair-weight targeted teacher data does teach the hard rungs partially; the attribution softens from "only relabeling teaches the full fight" to "relabeling teaches it far better", 0.667 against 0.250 at the top rung. The combination is the day's best generalist, $+0.145 \pm 0.040$ over baseline on the training pool and $+0.123 \pm 0.063$ held out, the largest held-out gain anything has produced on this pool. And the combination still loses the extreme rung, 0.083 at 1,000 against DAgger-alone's 0.667: even at fair weight, the targeted teacher decisions overlap the relabelings' states two-to-one, and where the teacher's own line and the drift-recovery label disagree, volume wins.

## Round two, from the combination clone

The second round walked the combination clone through the same 40 matchups, 1,000 episodes, 23,428 decisions labeled to the last, and retrained on everything: teacher corpus, unique targeted data, both rounds of relabelings, 335,921 samples with the relabel share at fourteen percent.

| Measure | Combination (parent) | DAgger round 2 |
|---|---|---|
| Training pool | 0.592 | 0.570 |
| Held-out pool | 0.552 | 0.490 |
| Thunk 500 / 700 / 850 / 1000 | 1.000 / 1.000 / 0.917 / 0.083 | 1.000 / 1.000 / 1.000 / 0.542 |

Iteration is not monotone. Round two recovered the extreme regime its parent had lost, a perfect 850 rung and 0.542 at 1,000 against the parent's 0.083, because the parent's own drift states include exactly those horde failures and the fresh relabelings re-taught them. It paid on the pool, $-0.022 \pm 0.014$ training and $-0.063 \pm 0.038$ held out against the parent, so the growing relabel share trades pool generality for drift repair at this mixture. No corpus so far dominates every regime, which is the cleanest possible motivation for the mixture-share sweep the previous section already ranked.

## PPO from the strongest anchor, and what it erodes

The day's last question was whether reinforcement learning still earns anything once the supervised levers have done their work. Three PPO seeds from the combination clone on the same pool, same 40 iterations, everything evaluated over battlefields against the anchor's own vendored evaluation (`ppo_from_strongest.py`): the training pool moved $-0.003 \pm 0.030$, nothing, where the same recipe from clone v4 had gained about $+0.14$; and the held-out pool moved $-0.060 \pm 0.016$, a resolved degradation at 3.7 standard errors. From a weak anchor, PPO converts headroom into on-pool gains that do not transfer; from a strong anchor there is no headroom left at this budget and the same optimization erodes the transfer the supervised pipeline built, which is the in-domain face of the over-optimization pattern [[../../rl/rlhf-transfer]] describes from the language-model literature, arriving here without any proxy reward involved.

## Where this leaves the training program

Every point of transferable progress today came from supervised data design, none from reinforcement learning. The day-final ranking on held-out play over battlefields: the equal-weight combination at 0.552, DAgger round two at 0.490 with the best hard ladder, DAgger round one at 0.487 with the best extreme rung, everything PPO produced at or below 0.492 and paid for out of its anchor. The shape after the corrected arms is that teacher-state data generalizes teacher coverage, student-state relabeling repairs student drift, the two combine into the best pool generalist so far, and mixture weight decides which signal survives where their states overlap, with the extreme-horde rung the visible casualty and iteration non-monotone.

The recorded next steps, in the order the evidence ranks them: sweep the relabeling share as an explicit axis; grow the supervised corpus along both axes it responds to, targeted demonstrations and drift relabelings, before spending anything more on policy-gradient fine-tuning; and when reinforcement learning returns, anchor it, since unanchored PPO measurably spends the clone's generality. The difficulty-weighted reward and battlefield rotation stay available and measured-null at current scales.
