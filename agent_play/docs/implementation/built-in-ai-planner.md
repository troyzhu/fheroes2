---
title: The built-in AI planner — a primer
aliases:
  - built-in-ai
  - battle-planner
  - the-teacher
tags:
  - agent-env
  - primer
concept: the engine's hand-written battle AI, its value function, and why it is both teacher and benchmark
domain: game AI
grounded_in: "src/fheroes2/ai/ai_battle.cpp, src/fheroes2/battle/battle_troop.cpp; measured in agent_play/docs/archive/experiments/2026-08-06-night-block-search-generations.md"
depth: quick
updated: 2026-08-06
---

# The built-in AI planner — a primer

`AI::BattlePlanner` is the engine's own battle commander. It is this project's teacher, since every demonstration comes from it, and its benchmark, since the measured question is whether a trained policy plays better than it. This primer says what it actually computes, because "the teacher" and "the baseline" are the same hand-written function and its shape explains both what the policies learned and where they stopped.

## Motivation

A learner imitating an expert inherits the expert's structure. Knowing that structure predicts what imitation can and cannot deliver: whatever the planner scores well, a clone can approach; whatever the planner never considers, no amount of imitation will discover. The project's central measured finding, that the student converges to the teacher's difficulty profile and stops ([[../archive/experiments/2026-08-05-dagger-and-battlefield-transfer]]), is a statement about this function.

## The idea in one sentence

It is a deterministic one-ply greedy planner: for the acting unit it enumerates reachable positions and enemy targets, scores each option with a hand-written damage-and-threat value, and takes the best, with no search, no learning and no randomness.

## What it computes

The core quantity is `Unit::evaluateThreatForUnit(defender)`, the value of one unit attacking another, and it is damage adjusted by three hand-written considerations.

It starts as `getPotentialDamage(defender)`, the expected damage of the attack. It is then divided by a distance modifier: flyers, shooters and towers get 1.0, and a melee unit that cannot reach the target this turn is discounted by `1.5 × distance / speed`, so threats that take longer to deliver count for less. Units that strike twice add the value of the second strike, reduced by the retaliation they will absorb between the two, and doubled outright when retaliation cannot happen, which is how shooters and retaliation-ignoring units come out ahead. `optimalAttackValue` then wraps this per candidate position: all-adjacent attackers sum the threat over every enemy the position touches, two-cell attackers add the second cell's target, and life-draining abilities add the hit points they would recover plus an estimate of the extra damage the resurrected stack deals over an assumed five-turn battle.

Around that sits `analyzeBattleState`, which recomputes army aggregates every turn, own and enemy strength, shooter strength, average speed, and sets the mode flags that gate whole branches: defensive tactics, cautious offensive, whether to avoid stacking units, whether retreat is even considered. The planner then picks, for the acting unit only, the position and target maximizing the value above, with `ValueHasImproved` breaking ties on position value and then on the enemy's own threat.

## What it does not do

There is no search: it never simulates the opponent's reply, so the plan is one ply deep and the constants that stand in for the future, the five-turn battle, the average of two attackers, the 1.5 distance factor, are the whole model of what comes next.

There is no learning: `ai_battle.cpp` contains no weights, no network and no training path, so the function is exactly what its authors wrote and never adapts to an opponent.

There is no randomness: the file contains zero `Rand::` call sites, which is why one recording per battlefield is all it can produce, why the DAgger probe could be proven inert by digest, and why its win rates are properties of the matchup rather than of a sampling process.

## Why this matters here

It sets the ceiling. Imitation reproduces a one-ply greedy value function, so the clone's blind spots are the planner's blind spots, and the measured convergence of student to teacher is that inheritance made visible. It also explains where search beats it: root search evaluates candidates by playing the battle out, which is exactly the ply the planner never looks at.

It sets the bar. The planner is a fixed, deterministic opponent, so "better than the built-in AI" is a well-defined finish line rather than a moving target, and [[../archive/experiments/2026-08-06-night-block-search-generations]] measures the current gap on every validation suite.

And it explains a style. The value is damage-shaped, discounted by delay and inflated by unanswered strikes, with no term for preserving one's own force beyond the retaliation it absorbs. A teacher with no preservation term produces demonstrations with no preservation habit, which is visible in the measured win quality: the planner wins the Thunk fight keeping 0.393 of its army strength where the policies, whose reward does carry a survival term, keep 0.44 to 0.47.

## Key terms

**Threat**: potential damage adjusted for delay, retaliation and multi-strike, the planner's atom of value.

**One-ply greedy**: choose the best immediate option under a static value, with no lookahead over replies.

**Mode flags**: the per-turn booleans (defensive tactics, cautious offensive) that switch the planner's policy wholesale from the army aggregates.

## Go deeper

- `src/fheroes2/ai/ai_battle.cpp`, the planner, and `Battle::Unit::evaluateThreatForUnit` in `src/fheroes2/battle/battle_troop.cpp`, the value.
- [[teacher-coverage-and-behavior-cloning]], how its decisions become a dataset.
- [[../archive/experiments/2026-08-06-night-block-search-generations]], the suite-by-suite gap between it and the trained policies.
