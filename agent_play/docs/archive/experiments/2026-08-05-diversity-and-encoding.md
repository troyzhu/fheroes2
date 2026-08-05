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

PENDING_ABLATION
