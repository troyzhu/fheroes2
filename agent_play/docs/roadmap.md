---
title: Scope and roadmap
type: roadmap
updated: 2026-07-30
related_concepts: ["[[rl-and-the-battle-domain]]", "[[overview]]"]
tags: [agent-env, roadmap, scope]
---

# Scope and roadmap

Everything built so far concerns battles only, and deliberately so. This note records what the wider goal is, why the battle came first, what the later phases involve, and which questions have to be answered with evidence before any of them starts. It exists so that the current narrow scope is not mistaken for the intended end state.

## Where the project is aimed

The end state is an agent that plays fheroes2, which means two quite different problems joined together.

A battle is a small, turn-based, nearly fully observed tactical problem on 99 hex cells with at most ten stacks. That is the problem the current environment solves, and [[rl-and-the-battle-domain]] describes it in full.

The adventure map is everything outside a battle. A hero moves across a large map under fog of war, picks up resources and artifacts, visits locations, recruits creatures, manages towns, and decides which battles to start in the first place. It is a strategy problem rather than a tactical one, and almost every property that makes the battle tractable is absent from it.

An agent that plays the game needs both, and the battle agent is a component the adventure agent invokes rather than a smaller version of it.

## Why the battle came first

Three reasons, in order of weight.

The battle is the part where a substrate can be made trustworthy. It is deterministic under a seed, it runs at roughly 4,600 episodes per second, it has a competent built-in teacher, and its legal action set can be enumerated exactly. Those four properties made it possible to build an environment whose correctness is demonstrable rather than assumed, which is what Milestones 1 through 3 did.

The battle is also where a mistake is cheapest. Getting the action space or the observation schema wrong in a self-contained tactical problem costs a rewrite of one module. Getting it wrong across the whole game costs the project.

Finally, the only comparable shipped system, vcmi-gym for Heroes III, is battle-only. There is a precedent to learn from at battle scope and none at adventure scope, which is itself informative about relative difficulty.

## Phase 1b, wider battles

The next expansion stays inside battles and lifts the `simple_v1` restrictions. Wide two-cell units, flying movement, two-cell and all-adjacent attacks, area shots, and eventually heroes with spells and castle sieges.

The blocking work is the capability audit and the action indexing. The current canonical space keys melee actions on a target cell and a direction, which presumes a single-cell attacker. A two-cell unit has no single head cell for that purpose, so the indexing needs revisiting before wide units enter. Spells multiply the action space by targets and parameters, which is the point where a flat masked space stops being obviously right and a factorized or pointer head earns consideration.

## Phase 2, the navigation and management agent

This is the part that is intended but not yet designed. Recording it now so the design is not improvised later.

The adventure agent covers hero movement across the map, resource and artifact collection, visiting locations, recruiting creatures, town building and management, and choosing which battles to fight. The battle agent becomes a subroutine it calls.

### Why it is a different problem, not a bigger one

Every simplifying property of the battle disappears.

Observability inverts. The battle is effectively fully observed. The adventure map has genuine fog of war, so the problem becomes partially observed in the real sense, and the recurrence, belief states, and memory architectures that battles let us skip become mandatory. This is the single largest change.

The horizon lengthens by orders of magnitude. A battle is 5 to 40 decisions. A campaign map is thousands of turns of decisions whose consequences arrive much later, which is a credit-assignment problem the battle never poses.

The action space stops being enumerable in the same way. Movement targets, building choices, recruitment quantities, and trade decisions produce a large and structurally heterogeneous space, closer to StarCraft's than to the current 793 slots.

The reward stops being obvious. A battle has a natural terminal signal in winning. A campaign has resource curves, map control, army growth, and eventual victory, on wildly different timescales. See [[decisions/0005-training-and-reward]].

Determinism gets harder. Map generation, random map events, and creature growth all draw randomness across a long horizon, so the seeding discipline that works for one battle needs extending.

The engine seam is unknown. Milestones 1 through 3 depended on finding a clean hook inside `Arena::UnitTurn`. Whether the adventure-map turn loop offers an equivalent seam, and whether its AI can be intercepted the same way, has not been investigated. `AI::Planner` and the kingdom-level planners in `src/fheroes2/ai/` are the place to look.

### Where it sits among comparable environments

On the axes in [[rl-and-the-battle-domain]], the adventure map moves from the fheroes2-battle row to somewhere near StarCraft II and the Lux AI competitions. Turn-based rather than real-time, which helps, but fogged, long-horizon, hierarchical, and with a heterogeneous action space, which does not.

The natural structural answer is hierarchy, with a strategic policy choosing where to go and what to build, and the existing battle policy invoked when a fight starts. That is an assumption rather than a conclusion. Hierarchical reinforcement learning has a mixed record, and a flat policy over a well-designed action abstraction sometimes beats an explicit hierarchy.

### The research this needs before it starts

The claim that hierarchy is right, and the choice of algorithm, observation encoding, and reward for a long-horizon fogged strategy game, all need the same treatment the battle environment got: a credibility-gated literature sweep with adversarial verification, not a design improvised from intuition. The questions worth putting to it:

1. What has actually been tried for turn-based 4X and strategy games, and what shipped? Freeciv, Civilization, Polytopia, Lux AI, and any Heroes-family adventure-map work.
2. Does hierarchical reinforcement learning outperform a flat policy over temporally extended actions in this class of game, and under what conditions?
3. How do published systems handle fog of war at this scale, meaning recurrence against belief states against learned world models?
4. How is reward shaped for long-horizon strategy without teaching the wrong objective?
5. How is a tactical sub-policy composed with a strategic one, and is the battle policy trained jointly or frozen?
6. What does the fheroes2 adventure-map AI expose that could serve as a teacher, in the way `AI::BattlePlanner` does for battles?

Until that sweep is run and consolidated, no adventure-map design decision should be recorded as settled.

## Sequencing

The battle environment is finished as a substrate only when Milestones 4 through 6 are done, meaning the protocol, the Python client, and the hardening and benchmark work. Training a battle policy comes after that, and it is where the reward and algorithm choices in [[decisions/0005-training-and-reward]] first meet reality.

Phase 1b and Phase 2 both wait on that, for the same reason the battle came first. A second environment built before the first one has trained anything would be a second untested substrate.

## Related

- [[overview]], current state and the milestone table.
- [[rl-and-the-battle-domain]], the battle domain and the axes this note compares against.
- [[decisions/0005-training-and-reward]], the training and reward design, including what is deliberately still open.
- [[research/findings]], the evidence base the battle decisions rest on.
