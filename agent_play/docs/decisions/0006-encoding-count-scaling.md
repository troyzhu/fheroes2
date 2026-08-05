---
title: "ADR 0006 — Logarithmic count scaling in the observation encoding"
type: adr
status: accepted
updated: 2026-08-05
related_concepts: ["[[0001-observation-profiles]]", "[[../implementation/observation-design]]", "[[../archive/experiments/2026-08-05-diversity-and-encoding]]"]
tags: [adr, agent-env, encoding]
---

# ADR 0006 — Logarithmic count scaling in the observation encoding

`obs_encoding_v3` changes exactly one thing against v2: stack counts and hit points are scaled as $\log(1 + x)$ over the schema-typical range instead of divided by 100. Everything else, the feature list, the creature one-hot, the slot layout, is unchanged. This record carries why, on what evidence, and what was deliberately not changed.

## Table of contents
- [[#The sub-problem]]
- [[#The options and the evidence]]
- [[#What was deliberately not changed]]
- [[#Costs]]

## The sub-problem

The environment now poses battles whose stack counts span one to a thousand, and real-map fights sit at the top of that range, the Thunk opening fight's Peasant stack rolls at 1,000. v2 divided counts by 100 linearly, which inverts tactical salience across that span: one creature against five differ by 0.04 while nine hundred against a thousand differ by 1.0, and a large stack's hit-point feature reaches the hundreds while every flag sits at one. Whether that mattered had never been measured, and the in-code comment justifying the constant was factually wrong about the schema's maximum.

## The options and the evidence

Four variants ran on the same 202,437-decision diverse dataset with the same architecture and budget: v2 unchanged, log-scaled counts, log-scaled with the one-hot extended to wide creatures plus a tail cell, and log-scaled with the one-hot removed.

The first split, by episode, was silently leaky, sibling episodes of one matchup on both sides, and its verdict, everything within a point of everything, measured near-memorization; the honest cross-matchup cloning number is near 0.52, not 0.86. On leak-free splits the picture is two-sided. Where counts stay inside the trained range the encoding does not matter: cross-matchup, three seeds, the log-without-one-hot variant leads v2 by $+0.005 \pm 0.007$, nothing. Where counts leave the trained range it matters decisively: trained on hordes of at most 300 creatures and tested above 600, same regime on both sides, v2 agrees with the teacher on $0.2394 \pm 0.0019$ of decisions and the log encoding on $0.3033 \pm 0.0019$, a gap of 24 standard errors across training seeds and 27 percent relative.

That is the shape a scaling defect should have: invisible in distribution, decisive under extrapolation, in the direction the audit predicted before the runs. The count-extrapolation regime is not exotic, it is the real-map regime, so the change is accepted.

## What was deliberately not changed

The creature one-hot stays, and the registered bar is why. Removing it was never worse and sometimes slightly better on leak-free splits, $+0.004$ at high counts, $+0.005$ cross-matchup, and it shrinks the model, but no single contrast cleared seed noise decisively. The stats-only question stays open with its measurements in the log, and the same goes for a tail-cell feature, whose only test was bundled with a wider one-hot that measured worst of the four cross-matchup.

Even the accepted change is necessary rather than sufficient: the best variant still degrades from 0.48 in range to 0.30 at high counts, so count generalization remains an open problem the encoding alone does not close.

## Costs

Every checkpoint and calibrated pool stamps its encoding version, so nothing mixes silently; the cost is that v2 artifacts are retired rather than reused, the clone was retrained on the diverse demonstrations at v3, and pools rebuild against it. The version stamp machinery from ADR 0003 is what makes this a planned migration rather than a hazard.

## Related

- [[../archive/experiments/2026-08-05-diversity-and-encoding]], the full evidence chain including the leaky split and the seed checks.
- [[0001-observation-profiles]], which this amends in one constant's semantics and nothing else.
- [[../implementation/observation-design]], the primer, updated to v3.
