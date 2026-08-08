---
title: "The learning side"
type: moc
updated: 2026-08-03
tags: [agent-env, rl, index]
---

# The learning side

Everything about training a policy. The environment itself is documented under `implementation/`, and the two are kept apart because they change for different reasons: the environment by milestone, this side by evidence. Much of this side is now implemented under `python/fheroes2_agent/` and gated by `verify_agent.sh`; each page says which of its own content exists, and [[../implementation/inventory]] is the per-component list.

The documents, in the order they are worth reading. The code column names where each page's subject is implemented, so a claim can be inspected rather than trusted; pages without a column entry are conceptual.

| Document | What it answers | Read when | Code |
|---|---|---|---|
| [[rl-and-the-battle-domain]] | What reinforcement learning is, what a Heroes battle is, and how this problem compares with the environments the project borrows from | First, if either half is unfamiliar | — |
| [[rl-methods]] | Every technique this documentation names, derived from the objective through to PPO, with a verdict on each alternative | To look one up, or to follow the chain once | `python/fheroes2_agent/objectives.py` |
| [[training-design]] | How a policy is actually fitted. Architecture, the loss at each stage, hyperparameter tables, and the alternatives at every choice | Before training anything | `python/fheroes2_agent/{policy,train_bc,train_critic,train_ppo,train_rloo,train_group}.py` |
| [[reward-design]] | The reward space: the implemented candidate, both families, shaping, and how a change would be chosen | Before touching what the agent is asked to optimize | reward assembly in `python/fheroes2_agent/env.py` |
| [[scenario-distribution]] | Why the army matchup dominates outcomes, and what training and evaluation do about it | Before defining a scenario generator or reporting a win rate | `python/fheroes2_agent/scenarios.py` |
| [[the-policy-network]] | The exact tensor path from worker JSON to masked logits, the slot lifecycle, and why pooling against concatenation is the live architectural question | Reading `policy.py`, or judging any architecture arm | `python/fheroes2_agent/policy.py`, `encoding.py` |
| [[off-support-and-offline-improvement]] | The off-support problem stated once, four remedy families with vendored primaries, and the order this project tries them | Choosing any offline improvement step | five works notes fetched 2026-08-06 |
| [[value-estimation-lab]] | Every value estimator fitted here, what each measured, and the literature concept each measurement grounds | Studying value methods, or before fitting the next estimator | probes vendored under the archive's 2026-08-06 run reports |
| [[program-review]] | The commissioned master review: every approach's verdict, the promising lines ranked, and the remaining experiments, each grounded in a measurement or a works note | Orienting on the whole program, or choosing the next experiment | written 2026-08-07 after the compaction |
| [[rlhf-transfer]] | What the language-model reinforcement-learning literature contributes here, and what it does not | When a technique from that world looks applicable | group baselines and trust regions in `python/fheroes2_agent/objectives.py` |

## What is ours and what is precedent

Unlabeled prose in this tree describes this project. Evidence from any other system, game, or paper always names its source and links the note under `research/works/` at first use, so provenance is visible at the mention and the full context is one link away. A sentence about another game that could be misread as a statement about fheroes2 is a defect; fix it or report it.

## How they fit together

[[rl-and-the-battle-domain]] fixes the vocabulary and places the problem. [[rl-methods]] is the reference the others link into rather than restating, so a technique is defined once. [[training-design]] applies those techniques to this network and this teacher, and [[reward-design]] carries the one choice that decides what is being optimized. [[scenario-distribution]] concerns the data the whole thing trains on, which turns out to dominate both the gradient and any reported number. [[rlhf-transfer]] is a survey of one adjacent literature, kept separate because most of it does not apply and saying so is the useful part. [[value-estimation-lab]] is the owner-requested study record of the value thread, every estimator beside the concept its failure or success demonstrates. [[the-policy-network]] walks the architecture end to end, owner-requested so the pooling question reads from a full picture rather than a fragment. [[off-support-and-offline-improvement]] is the owner-requested survey of the off-support problem's remedies, each family grounded in a vendored primary.

[[program-review]] assembles all of it into the master review the owner commissioned, verdict by verdict with the promising lines ranked.

Symbols are fixed in [[../overview#Notation]], which also records which topics this tree carries itself and which it assumes from the owner's existing study notes.

## What binds this material

Decisions live in `decisions/`, not here. [[../decisions/0005-training-and-reward]] is the one that governs everything above, and it states the staging, the reward candidates, and what is deliberately still open. Where a document here recommends something, that is a recommendation until a record accepts it.

## Related

- [[../overview]], the problem, the notation, the current state.
- [[../roadmap]], what comes after battles.
- [[../implementation/README]], the environment these methods would train against.
- [[../research/findings]], the evidence base.

<!-- verify
# Invalidators for the implemented-status sentence at the top of this file.
exists  python/fheroes2_agent/train_ppo.py
exists  agent_play/verify_agent.sh
-->

