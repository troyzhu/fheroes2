---
title: "Teacher coverage and behavior cloning: proving the action space is complete"
type: concept-primer
depth: standard
grounded-in: "fheroes2 agent-env branch (Milestones 2–3, spec §10.6, §21)"
related_concepts: ["[[legal-actions-and-masking]]", "[[battle-turn-dispatch]]"]
tags: [concept, behavior-cloning, evaluation, agent-env]
---

> **What this is.** What "the teacher" means here, why *teacher coverage* is the sharpest
> completeness test available for an action space, and how it feeds the training plan. Nobody has
> to play the game for any of this.

## The one-sentence version

The built-in engine AI plays both sides; we record every decision it makes and check that each
one maps onto a legal action in our canonical space — 100 % coverage means our action space can
express everything a competent player does, and the same recordings become the behavior-cloning
dataset.

## Who the teacher is

The **teacher is `AI::BattlePlanner`**, the game's own tactical AI — not a human, not an LLM. It
already plays every headless battle we run. The passive recorder observes its choices through the
`DecisionController` hook without influencing them.

This is worth stating plainly because it is a common misreading: *demonstration data requires no
human play*. Running the worker generates it.

## Coverage as a completeness proof

For each full-fledged decision we compute two things at the same pre-application state:

1. the set of legal candidates our enumerator produces, and
2. the canonical index of what the teacher actually chose.

Then:

$$\text{coverage} = \frac{\#\{\text{decisions where the teacher's action is in our legal set}\}}{\#\{\text{decisions}\}}$$

**Why this is a strong test.** A missing legal action is invisible to ordinary testing — the
environment runs fine, the policy just never learns a move it was never offered. But the teacher
*does* use those moves, so any gap in our enumeration shows up immediately as a coverage miss.
Coverage below 100 % means one of:

- our enumeration missed a legal action (a bug in the generator), or
- the canonical indexing cannot express that action (a design gap), or
- the creature is outside `simple_v1` (a scenario that should have been rejected).

All three are things you want to know before training, not after.

**Current state:** 116/116 decisions, all five fixtures, 100 %. Minimum candidate count per
decision ≥ 5.

## From coverage to behavior cloning

The same recordings are the BC dataset. The staging the literature supports:

1. **Collect** — passive teacher trajectories (done: `agent_passive_v0` JSONL).
2. **Behavior-clone** — supervised learning of $\pi(a \mid s)$ from teacher decisions. AlphaStar's
   purely supervised stage reached 87 % win rate against the game's Elite bot *before any RL*,
   which is the strongest available evidence that this step is worth doing first.
3. **Correct** (DAgger-style) — roll out the student, ask the teacher what it would have done in
   the states the student actually visits, add those labels. Fixes the distribution shift that
   pure BC suffers.
4. **Reinforce** — masked PPO against a *mixture* of scripted opponents (single-opponent training
   produces agents that lose to simple rushes).

Steps 3–4 have a documented evidence gap at our scale: no verified small-scale BC→RL recipe
exists. Expect iteration.

## Calibration: what "good" looks like

From the only shipped comparable system (vcmi-gym, HoMM3): the first working model reached ~75 %
against the weak scripted bot and ~45 % against the strong one; a much later iteration averaged
~65 % against the strong bot. **Parity with the engine's AI is a multi-iteration goal, not a
first-run outcome.**

## Why it matters here

Coverage is Milestone 3's exit criterion precisely because it converts "we think enumeration is
complete" into a measured number. It also runs continuously — `verify_m3.sh` re-measures it on
every check, so a future refactor that quietly drops an action type fails the gate.

## What this does *not* say

100 % coverage proves our space contains everything *the teacher does*. It does not prove it
contains every legal action in principle — a move no AI ever plays could still be missing. That
residual is bounded by the capability audit (which excludes creatures whose action space we do
not model) rather than by coverage itself.

## See also
- [[legal-actions-and-masking]] — the space coverage is measured against.
- [[battle-turn-dispatch]] — the hook that observes the teacher.
