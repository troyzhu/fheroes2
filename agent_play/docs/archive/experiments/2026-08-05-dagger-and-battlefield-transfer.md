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

## The relabeling share, swept

The ranked next step ran the same evening: the equal-weight combination corpus with the round-one relabelings duplicated to two and four times their natural weight, everything else identical.

| Relabel share | Training pool | Held-out pool | Thunk 850 | Thunk 1000 |
|---|---|---|---|---|
| $\times 1$ (the combination) | 0.592 | 0.552 | 0.917 | 0.083 |
| $\times 2$ | 0.574 | 0.542 | 1.000 | 0.500 |
| $\times 4$ | 0.589 | 0.496 | 1.000 | 0.583 |

The extreme rung climbs monotonically with the share while the held-out pool holds flat at double weight (paired $-0.010 \pm 0.045$ against the combination) and pays at quadruple ($-0.056 \pm 0.036$). Double weight is the sweet spot at this corpus: `policy_share2` keeps the day's best-tier held-out number and a 1.000 / 1.000 / 1.000 / 0.500 ladder, the best all-round checkpoint produced today. Mixture weight is a real, cheap, monotone-then-costly dial, and one supervised recipe now holds most of both regimes at once.

## Targeted rounds, and where relabeling yield actually comes from

Two targeted collections closed the evening, and the first was a wash whose cause is the finding. A horde-targeted round walked the sweet-spot clone through 24 matchups of the horde supplement's manifest, 600 episodes, and collected only 3,258 labeled decisions, a seventh of a pool round's yield, because the student is already strong there and strong play ends a horde battle in a handful of decisions. Retrained, everything moved within noise: training pool $+0.004 \pm 0.017$ paired against its parent, held-out $-0.050 \pm 0.052$, Thunk 1.000 / 1.000 / 0.917 / 0.333. DAgger's data yield is proportional to how much the student struggles, so targeting a solved regime buys almost nothing to learn from.

The second targeted round aimed at struggle instead: the fifteen training matchups the sweet-spot clone wins least, from 0.00 to 0.50 over battlefields by its own vendored evaluation, forty episodes each. Yield confirmed the hypothesis, 10,022 labels against the horde round's 3,258 from the same episode count. The results did not: the fifteen targeted matchups got worse, 0.208 against 0.258 paired at three standard errors, the extreme rungs regressed (Thunk 0.708 and 0.125), and the one bright cell, a day-best held-out 0.583, sits at 1.1 standard errors over its parent. Relabeling where the student loses taught the teacher's play in fights the teacher also loses, which is imitating the least-bad line of a lost position.

That reading is checkable, and checking it produced the evening's terminal finding. The teacher's own attacker win rates on the 40 training matchups, computed from the control recordings, correlate with the sweet-spot clone's at 0.815, and the intended relabeling band, teacher wins at least half while the student wins at most 0.4, is empty, zero of forty matchups. The student has converged to the teacher's difficulty profile on this pool. What remains unlearnable by imitation is what the teacher cannot do either, which is why round one from the drift-rich weak clone paid enormously and every targeted round from the strong one paid nothing: DAgger exhausted its own target by succeeding.

## The search-taught round, and the report card

The owner set the goal, a sufficiently good agent validated three ways, and the search-as-teacher round ran end to end: 560 episodes played by root-PUCT across the 40 training matchups, hard band deepest, six shards in parallel, 12,485 labels with 508 of 560 collection episodes won. Distilled at double weight into the sweet-spot recipe, the clone posts the best on-pool number of the day, 0.622, and the battery below tells the rest honestly.

`validation_battery.py` measures every checkpoint over battlefields on the three owner-fixed surfaces: matchups sampled fresh from the generator with a seed nothing trained or calibrated on, the held-out pool, three out-of-distribution stress suites, and the Thunk ladder no training ever touched.

| Suite | Clone v4 | share2 | search-taught |
|---|---|---|---|
| Fresh sampled, 24 | 0.399 | 0.394 | 0.391 |
| Held-out pool, 20 | 0.421 | 0.567 | 0.448 |
| Stress, hordes to 3,000 | 0.158 | 0.200 | 0.175 |
| Stress, wide-only armies | 0.444 | 0.389 | 0.431 |
| Stress, commander extremes | 0.979 | 0.969 | 0.958 |
| Thunk ladder | 0.573 | 0.875 | 0.719 |

Three verdicts. `policy_share2` is the agent: it leads every surface that separates policies at all, held-out by 0.12 to 0.15, the Thunk ladder at 1.000 / 1.000 / 0.958 / 0.542 with the full fight now won more often than lost, and the horde stress. The search-taught distillation regressed its parent off-pool, held-out 0.448 against 0.567 and the ladder 0.719 against 0.875, the evening's law once more: labels concentrated on the training matchups buy on-pool play with off-pool generality, and the searched labels, twelve thousand decisions on those same forty matchups, are no exception even at teacher grade. The improvement operator itself remains a different tier, the probe's $+0.79$ stands, so search earns its keep as a wrapper at decision time, half a second per move where quality matters, not yet as distilled weights.

The stress suites name the walls precisely. Commander extremes are no weakness at all, 0.96 to 0.98 everywhere, so the stat encoding generalizes past the sampled range. Wide-only armies sit mid-hard and undifferentiated. Hordes beyond 1,000 are a cliff for every checkpoint, zeros across the Thunk-army rungs at 1,500 through 3,000, the count-extrapolation limit ADR 0006 measured in features now measured in behavior. And the fresh raw distribution reads near 0.40 identically for all three, because uncalibrated draws include matchups no policy wins, which is the band-empty finding restated from the sampler's side.

## The midnight follow-ups: sampling, sides, duels, and a strength margin

Four owner questions closed the night. On repetition per fixed scenario: stochastic learners run many episodes per scenario and the group baselines require it, while the deterministic teacher yields exactly one useful trajectory per battlefield, which the duplication accident already demonstrated the hard way. On the reward: the value-weighted survival margin the design page had ranked next is now implemented owner-directed, engine strength pricing both totals in the terminal record, opt-in as `reward_margin="strength"`, with search rollout scoring as its live consumer so root-PUCT can prefer minimum-value-loss wins; the training measurement stays pending.

On side swapping: everything ever trained controlled the attacker, and the measurement says that mattered. As defender on the same held-out matchups, the agent wins 0.252 against its 0.567 as attacker, and only 0.017 ahead of clone v4's defender play, so the attacker-side gains barely crossed the symmetric encoding. Both-side training and self-play are now mechanically reachable: the capture pipeline routes each decision to the model owning the active side, which recorded the first checkpoint duel tonight.

The duel surfaced a defect worth its own investigation. A both-side recording replays bit-exactly through the protocol channel and diverges through `runEpisode` at decision 17 of 41, while every single-side recording verifies exact through both paths. The repro is vendored (`files/2026-08-05-run-reports/replay_duel.json`), the real-engine rendering of duels waits on it, and the decision-stream digests are the investigation tool.

## Where this leaves the training program

Every point of transferable progress today came from supervised data design, none from reinforcement learning. The day-final ranking on held-out play over battlefields: the equal-weight combination at 0.552, DAgger round two at 0.490 with the best hard ladder, DAgger round one at 0.487 with the best extreme rung, everything PPO produced at or below 0.492 and paid for out of its anchor. The shape after the corrected arms is that teacher-state data generalizes teacher coverage, student-state relabeling repairs student drift, the two combine into the best pool generalist so far, and mixture weight decides which signal survives where their states overlap, with the extreme-horde rung the visible casualty and iteration non-monotone.

The evening's final state supersedes the interim rankings. The supervised program on this pool is complete: the share sweep found its sweet spot, targeted rounds returned nothing further, and the 0.815 teacher-student correlation with an empty relabeling band says imitation has delivered the teacher's profile and cannot deliver more. What remains above the current policy is exactly what the teacher cannot play, which imitation cannot teach.

The owner's review sharpened what the next escalation must be, because a KL anchor alone was the wrong prescription. An anchor to the strongest checkpoint solves only retention, the measured erosion of unanchored PPO, and is if anything anti-exploration; and the matchups both teacher and student lose carry zero outcome variance under the terminal reward, so no exploration knob rescues them directly, the degeneracy [[../../rl/scenario-distribution]] already measured. The mechanism this project has proven for manufacturing signal is the curriculum, the Corribus ladder generalized: recalibrate a fresh pool against the strongest clone so a sometimes-wins band exists for it, interpolating toward the lost matchups, and climb that band with anchored PPO, the anchor guarding retention while the rungs supply gradient. Exploration today is only the masked softmax's full support over legal actions plus a 0.01 entropy bonus, worth widening only if rungs stall.

The same review named two further directions, both now defined in [[../../rl/rl-methods]]. The failure is formally a sampling problem, terminal-only signal with vanishing mass on winning trajectories, which prefix-replay resets and critic-potential shaping also address. And search is an improvement operator, a short tree search over the learned policy and critic serving as the next teacher through the existing relabeling pipeline, recorded as a direction for the owner to weigh against the curriculum route. The difficulty-weighted reward and battlefield rotation stay available and measured-null at current scales.

## The review, taken to the literature and to two measurements

The owner pressed on whether the design answers were literature-grounded, what is wrong with the value model, and whether the UCB variant of search applies. The corpus answered more than expected once actually consulted: [[../../research/works/alphastar-unplugged]] had been vendored since the first sweep and barely cited, and it reports at StarCraft scale exactly the shapes measured here today, one-step improvement against the behavior value as the recipe that works, every multi-step variant failing to beat behavior cloning, search at inference improving while search at training collapses the policy by exploiting value error, and iterated improvement against a frozen value degrading. Five primaries joined the vendored set with digests: [[../../research/works/uct]] and [[../../research/works/alphazero]] for the UCB and PUCT forms, [[../../research/works/one-step-offline-rl]] for the one-step theory, [[../../research/works/bcq-extrapolation]] and [[../../research/works/double-q-overestimation]] for the two value pathologies worth separating.

The value question got its measurement (`critic_calibration.py`). Refitted at v3 on the full corpus, the critic explains 0.302 of held-out return variance on teacher play, so the fixture-era 0.835 was a narrow-era artifact, and on student-played states it explains less than zero, $-0.131$, with a $+0.32$ optimistic bias: worse than predicting the mean, exactly where an improvement operator would consult it. That is extrapolation error measured in place, it plausibly contributed to the PPO erosion through the GAE baseline, and it fixes the search design: rollout returns, never critic leaves. [[../../rl/training-design#The behavior value, measured where it would be spent]] carries the consequences.

The search question got its probe (`search_probe.py`), root-PUCT with the clone prior, rollout scoring, and reset-continuation supplying simulations at milliseconds each: 32 simulations per decision on the Thunk 1,000 fight and the three training matchups the sweet-spot clone loses worst.

| Matchup | Policy alone | Root-PUCT, 32 simulations |
|---|---|---|
| Thunk 1,000 | 0.500 | 1.000 |
| Worst pool matchup (0.08 band) | 0.083 | 1.000 |
| Second-worst (0.00) | 0.000 | 0.750 |
| Third-worst (0.00) | 0.000 | 1.000 |

A mean lift of $+0.79$ across the four, in 340 seconds of wall time, about half a second per searched decision. Two caveats stated plainly: the probe is single-battlefield by construction, since simulations must replay the live episode's battlefield, and twelve episodes per cell carry error near $\pm 0.10$; the aggregate is far beyond both. What it settles is the review's sharpest question. The matchups the policy and mostly the teacher lose are not unwinnable, and the winning lines sit close enough to the clone's support that a prior-guided root search finds them at thirty-two simulations, no deep tree required: off-support in probability, not in reachability. Search is the improvement operator with real headroom here, its labels are teacher-grade at about three hours serial for a full relabeling round or under half an hour across the cores, and the discipline the literature imposes, rollout scoring and one distillation round per search generation, is exactly what the probe already implements.
