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

## The control that decides attribution

The gain could still have been data volume rather than relabeling: the aggregate carries 22,750 more decisions than the teacher corpus. The control records 1,000 teacher-played episodes on the same 40 matchups and retrains identically, so the corpus grows by the same order from the same matchups with the difference being whose states carry the labels. One design honesty note: the teacher is deterministic per battlefield, so matching a thousand unique episodes meant 25 battlefields per matchup against the DAgger collection's 4, and the control therefore also enjoys denser battlefield coverage.

| Measure | Clone v4 | Teacher control | DAgger clone |
|---|---|---|---|
| Training pool | 0.447 | 0.601 | 0.541 |
| Held-out pool | 0.429 | 0.444 | 0.487 |
| Thunk 500 / 700 / 850 / 1000 | 1.000 / 0.917 / 0.250 / 0.000 | 1.000 / 0.917 / 0.458 / 0.000 | 1.000 / 1.000 / 0.917 / 0.667 |

The split verdict is more informative than a clean win would have been. On the training pool, targeted teacher data is at least as good, 0.601 against 0.541, paired DAgger-minus-control $-0.060 \pm 0.038$; recording more demonstrations on matchups you already train on is a cheap and strong lever, helped here by its battlefield density, and its 0.9232 cloning agreement against DAgger's 0.8413 says the pure-teacher distribution is also simply easier to fit. Off the training distribution the ranking inverts: held-out paired DAgger-minus-control $+0.044 \pm 0.069$, directional only, but the Thunk ladder separates them without ambiguity, the control at 0.458 on the 850 rung and still 0.000 at 1,000, the DAgger clone at 0.917 and 0.667. Volume alone, on the very matchups the student trained from, never taught the full fight; relabeling the student's own drift states did.

## Where this leaves the training program

Stage 2 exists, its first round is measured, and the attribution is clean: teacher-state data generalizes teacher coverage, student-state relabeling fixes student drift, and they are complements rather than substitutes. The recorded next steps, in the order the evidence ranks them: combine the levers, since the control's targeted demonstrations and the DAgger relabelings are independent additions to the same corpus; iterate DAgger from the DAgger clone, since one round returned this much and iteration is the method's whole design; and only then reinforcement learning from the strongest clone, which restarts the PPO question from a better-anchored policy. The difficulty-weighted reward and battlefield rotation stay available and measured-null at current scales.
