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

PENDING_RESULTS
