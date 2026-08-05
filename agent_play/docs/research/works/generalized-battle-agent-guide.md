---
title: "Owner-supplied guide: training a generalized battle agent for fheroes2"
type: report
year: 2026
quality: primary
tags: [reference, design, sampling, abilities, generalization]
local: ["files/generalized-battle-agent-guide.md"]
---

# The generalized-battle-agent guide, mapped against this project

Supplied by the owner on 2026-08-05, reviewed against `master` of the same date, vendored unmodified beside this note. It is a design document for exactly this project's problem, written independently of this tree, which makes the overlap informative in both directions: where it agrees it confirms, and where it goes further it is a to-do list with an argument attached.

## Where it and this project already agree

The engine stays the authority on legality and transitions with the network scoring enumerated candidates, its sections 2 and 8, which is ADR 0002 and the action-space module as built. The staging, cloning then DAgger then population play then reinforcement learning, its section 11, is ADR 0005's. Log-scaled counts, its 6.3, became [[../../decisions/0006-encoding-count-scaling]] the morning the guide arrived, from independent measurement. Scenario-family evaluation splits rather than state-level ones, its section 13, is the matchup-leakage lesson this project paid for empirically the same day. Sampling concentrated on close battles, its 10, is the calibration band in [[../../rl/scenario-distribution]].

## What it adds, adopted now

The value-budget sampler, its section 10 and the piece the owner pointed at. Draw a total army budget log-uniformly, split it over stacks with a Dirichlet draw, price each stack's count by the creature's worth rather than its hit points, and set the enemy budget by a ratio mixture concentrated near one. Two concrete improvements over the sampler this project had: engine strength as the price, `Monster::GetMonsterStrength`, now exported by the capability audit, values what hit points miss, a Ranger costing two thirds more than an Archer at equal hit points; and Dirichlet shares produce uneven stacks where the old sampler produced near-equal ones. Implemented as `sample_budget_matchup` beside the old sampler, with the calibration hit rate as the adoption criterion, measured rather than assumed.

## What it adds, tracked as work items

Structured ability observations, its sections 3 to 5: raw `MonsterAbility` records plus a deterministic semantic adapter in the observation, and engine-computed per-candidate effect summaries, expected damage, retaliation, kill probability, attached to each legal action. This project's encoding carries five booleans and a one-hot; abilities like no-retaliation are invisible to it. The guide's layered design is the right shape and is a phase of its own.

Its 6.3 also corrects ADR 0006 at the margin: keep an absolute count channel beside the log one, because kill thresholds are discontinuities that pure log scaling smooths over. That is a cheap follow-up ablation, not a revision of the decision.

The lexicographic objective, its section 9: win probability first, losses second among near-ties, with two value heads. This lands in [[../../rl/reward-design]] as a documented candidate with the scalarization pathology it avoids.

## Where this project's evidence pushes back

The guide's confidence in identity embeddings as a residual feature meets a measured shrug here: on leak-free splits the one-hot was never decisively better than stats alone and once measured worst. Its ID-masking suggestion is the right hedge and costs nothing. And its curriculum section assumes transfer that this project's capstone null, training gain $+0.17$ against held-out $+0.01$ on the fully diverse pool, shows is not yet free; the guide's own adaptive-sampling and auxiliary-loss machinery is the plausible remedy, untested here.

## Related

- [[../../decisions/0006-encoding-count-scaling]], the convergent count decision.
- [[../../rl/scenario-distribution]], where the sampler comparison lands.
- [[../../archive/experiments/2026-08-05-diversity-and-encoding]], the day's evidence this guide interleaves with.
