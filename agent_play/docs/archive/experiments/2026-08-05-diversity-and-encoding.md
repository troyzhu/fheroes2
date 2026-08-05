---
title: "Diversity, counts, and the encoding, 2026-08-05"
type: experiment-log
updated: 2026-08-05
tags: [agent-env, archive, experiment, encoding, diversity]
---

# Diversity, counts, and the encoding, 2026-08-05

The owner asked whether the current approach holds up against the diversity the environment now reaches, 41 creatures with counts from one to a thousand, and the honest answer was no stated confidence without measurement. This log carries the audit, the runs, and what they decide. Conclusions migrate to [[../../rl/training-design]] and [[../../implementation/observation-design]] once the evidence is in; nothing here is a decision.

## The audit, before touching anything

Four weaknesses were named up front, each testable.

The demonstrations. The clone that anchors calibration, the critic and every reinforcement-learning start was trained on the five Milestone 1 fixtures, which contain three creature types. Thirty-eight of the roster's 41 creatures reach it only as stat lines at inference time. The wide-pool transfer result, $+0.153 \pm 0.056$ held out, says the encoding carries something across creatures; it says nothing about what a clone trained on the full bestiary would carry.

The count scale. `obs_encoding_v2` divides counts and hit points by 100, linearly and uncapped. Over the range the environment now produces this inverts tactical salience: one creature against five differ by 0.04 while nine hundred against a thousand differ by 1.0, and a large stack's hit-point feature reaches the hundreds while every flag sits at one. The in-code comment claiming a count of 1,000 is the schema's maximum is wrong, the cap is 500,000. Logarithmic scaling is the standard repair and the ablation below prices it.

The identity gap. The 41-way one-hot covers `simple_v1` only, so every wide creature encodes with all-zero identity, including the Champions that fought the entire Corribus curriculum. Whether identity one-hots earn their 41 of 63 slot features at all, against pure stat features, has never been ablated.

The architecture. The shared slot encoder concatenates in uid order rather than pooling, so slot arrangement is visible to the trunk. Noted as a separate axis and deliberately not bundled into the encoding ablation, one variable at a time.

## The plan, registered

Record diverse demonstrations first, because every later measurement needs them: sampled `wide_v1` armies across three count regimes, skirmish, battle and elite-against-horde, with commanders on a coin flip per side. Then clone at v2 unchanged, stratified by creature and by count, against the old clone. Then ablate encodings on the same data with the same architecture, including a count-extrapolation split, train on stacks up to 200 and test above, which is the direct test of the owner's question about numbers. Only a decisive ablation justifies an encoding version bump and its decision record.

## Recording at scale, and what it caught

The recorder samples `wide_v1` armies over three regimes with commanders on a coin flip per side. The first full run, 400 matchups and 11,921 episodes, reported 71 failures, and diagnosing them found three distinct defects, none of them in the sampled matchups.

The wide-attacker melee gap. 993 of 191,993 decisions, 0.52 percent, resolved to a canonical action the enumeration never offered, and every one had a wide active unit, 674 attacking in place and 319 on the move. The engine accepts a melee strike when either cell of the attack position is adjacent to the target; the enumeration inverted only the head case. The 1,238-decision coverage run that admitted wide units had simply lacked exposure at that rate. Fixed by proposing the in-place and tail-landing destinations too, with the canonical index following the resolved geometry; single-cell attackers produce exactly the original proposals and the m3 goldens hold.

The stalemate abort. The built-in AI forces the attacking hero to retreat after 50 turns without deaths and asserts a retreat-capable commander exists, which a scenario captain is not and a commander-less army lacks entirely. A latent abort since Milestone 1, unreachable until diverse armies of few high-hit-point stacks danced long enough. The runner now ends such episodes first as a Stalemate truncation, bootstrapped like the round limit.

The verdict blindness. The recorder kept stderr alone while the worker's coverage verdict goes to stdout, so 68 coverage-incomplete runs looked like inexplicable failures with benign diagnostics. The recorder now stores the verdict line per matchup and fails on INCOMPLETE explicitly.

The re-record with all three fixes: 12,000 episodes over 400 matchups, zero failures, every verdict complete.

## The clones, cross-evaluated

Clone v3 trains on the diverse data at the unchanged encoding and architecture, 25 epochs, and each clone is evaluated on both held-out sets, split by episode.

| Clone | Narrow held-out, 8,561 decisions | Diverse held-out, 38,864 |
|---|---|---|
| v2, three creatures | 0.8873 | 0.2868 |
| v3, whole bestiary | 0.3914 | 0.8650 |

The number that matters is 0.2868. The clone that anchored every calibration, every critic fit and every reinforcement-learning start agrees with the teacher on twenty-nine percent of decisions once the battle leaves its three creatures, which quantifies exactly how much the earlier confidence rested on a narrow anchor. The asymmetry runs both ways, v3 at 0.39 on the fixture distribution, so neither dominates and the fixture set is structurally special rather than a subset; a mixed-data clone is the obvious follow-up, measured rather than assumed.

## The ablation, and the split that lied first

Four variants, same data, same architecture, same budget: v2 unchanged, log-scaled counts and hit points, log-scaled with the one-hot extended to `wide_v1` plus a tail cell, and log-scaled with the one-hot removed. The first harness run died on a shape mismatch of its own making, the width patch restored too early, recorded here because the fix explains the machinery.

The episode split came back flat, everything between 0.849 and 0.861, and it was measuring the wrong thing. Twelve thousand episodes come from four hundred matchups, thirty episodes each, and an episode-level split puts siblings of every matchup on both sides. The honest matchup-level split lands near 0.52, not 0.86, which also recontextualizes every earlier agreement number in this project: the fixture-era 0.887 was within-matchup generalization across seeds, never across matchups.

| Split | v2 | v2log | v2log_wid | v2log_noid |
|---|---|---|---|---|
| Episode, leaky | 0.8560 | 0.8490 | 0.8609 | 0.8569 |
| Matchup, honest, seed 0 | 0.5082 | 0.5105 | 0.5017 | 0.5214 |

The first count-extrapolation attempt had two defects of its own: it confounded count with regime, large-count decisions being almost all horde episodes, and the recorder's horde totals capped at 900 hit points, so no stack exceeded 300 and the test set above that line was empty. A horde-only supplement recorded to 3,000 total hit points, stacks to 1,000, fixed the range, 3,000 episodes, zero failures, coverage complete, which also certifies the wide-melee enumeration at full scale.

## The clean extrapolation, and the seed checks

Within the horde regime alone, trained on stacks of at most 300, tested on the supplement:

| Variant | Control, same range | 301 to 600 | Above 600 |
|---|---|---|---|
| v2 | 0.4424 | 0.3573 | 0.2431 |
| v2log | 0.4763 | 0.3905 | 0.3008 |
| v2log_noid | 0.4821 | 0.3913 | 0.3047 |

Two contrasts then went to three training seeds each, because a single-seed ranking had already misled once today. Cross-matchup, in range: v2 $0.5208 \pm 0.0091$ against v2log_noid $0.5256 \pm 0.0027$, a paired $+0.005 \pm 0.007$, nothing, and the first cross-matchup ranking was substantially seed luck. Above 600 creatures: v2 $0.2394 \pm 0.0019$ against $0.3033 \pm 0.0019$, a gap of 24 standard errors, real beyond argument.

So the encoding does not matter where counts stay in range and matters decisively where they leave it, which is the real-map regime. [[../../decisions/0006-encoding-count-scaling]] accepts exactly the log scaling and nothing else; the one-hot's removal, never worse and never decisively better, stays open, and so does the tail cell. Clone v4 retrains at v3 on the diverse data, 0.8606 on the leaky split for continuity with earlier numbers, and the honest cross-matchup figure near 0.52 is the one later work should quote.

What no encoding fixes: the best variant still falls from 0.48 in range to 0.30 at high counts. Count generalization is an open problem of the model, not the features, and the architecture axis noted in the audit, pooling against concatenation, is the natural next suspect.

## Downstream, with the new anchor

Clone v4, diverse data at v3, walks the Corribus ladder better than the old clone before any curriculum: 0.917 at 700 Peasants against 0.833, 0.208 at 850 against 0.125, still 0.000 at the rolled 1,000, so demonstrations carry real distance toward the fight and the last stretch stays reinforcement learning's.

The diverse pool builder, sampling commanders and wide creatures through the shared sampler, calibrated 120 matchups at a 26.0 percent hit rate against v4, better than the narrow roster's 22.5 and the wide-roster-without-heroes 19.3, so calibration gets easier as the distribution gets more like the real game, not harder. The accepted pool holds 99 mixed-army matchups and 21 hordes, 91 of 120 with a commander on at least one side, fingerprint-stamped to v4.

## The owner-supplied guide, and the sampler it improved

Mid-push the owner supplied a design guide for this exact problem, vendored with a digest at [[../../research/works/generalized-battle-agent-guide]]. Its value-budget sampler, counts priced by engine creature strength with Dirichlet stack allocation and a near-one budget-ratio mixture, went head to head against the hit-point sampler: same clone, same target of sixty calibrated matchups.

| Sampler | Band hit rate | Mean $\lvert\log \text{scale}\rvert$ | Attempts |
|---|---|---|---|
| Hit-point shares | 23.8% | 0.902 | 252 |
| Value budget | 31.4% | 0.861 | 191 |

A third fewer attempts per accepted matchup and less calibration rescue, about 1.8 standard errors on the hit rate, so directionally consistent on both metrics and principled in mechanism, strength pricing what hit points miss, rather than statistically overwhelming. `sample_budget_matchup` becomes the recommended sampler for new pools; the older samplers stay for reproducing recorded ones.

The capstone run is a null worth stating plainly. Generalization on this pool, 70 training and 50 held-out matchups, 40 iterations from clone v4: training gain $+0.173 \pm 0.039$, held-out gain $+0.007 \pm 0.046$. The transfer that was separable on the creature-diverse pool, $+0.153$ at 2.7 standard errors, vanishes on the commander-and-horde-diverse one. One seed, and a smaller training split than the earlier run, so the shape is a finding and the magnitude is not settled; what it says either way is that generalization gets harder as the distribution gets more like the real game, and the levers it points at are bigger training pools, longer budgets, and the pooling-against-concatenation architecture axis, not the encoding.
