---
title: "The learning side"
type: moc
updated: 2026-08-03
tags: [agent-env, rl, index]
---

# The learning side

Everything about training a policy. The environment itself is documented under `implementation/`, and the two are kept apart because the environment is built and verified while nothing here is implemented yet.

Five documents, in the order they are worth reading.

| Document | What it answers | Read when |
|---|---|---|
| [[rl-and-the-battle-domain]] | What reinforcement learning is, what a Heroes battle is, and how this problem compares with the environments the project borrows from | First, if either half is unfamiliar |
| [[rl-methods]] | Every technique this documentation names, derived from the objective through to PPO, with a verdict on each alternative | To look one up, or to follow the chain once |
| [[training-design]] | How a policy is actually fitted. Architecture, the loss at each stage, hyperparameter tables, and the alternatives at every choice | Before training anything |
| [[scenario-distribution]] | Why the army matchup dominates outcomes, and what training and evaluation do about it | Before defining a scenario generator or reporting a win rate |
| [[rlhf-transfer]] | What the language-model reinforcement-learning literature contributes here, and what it does not | When a technique from that world looks applicable |

## How they fit together

[[rl-and-the-battle-domain]] fixes the vocabulary and places the problem. [[rl-methods]] is the reference the other three link into rather than restating, so a technique is defined once. [[training-design]] applies those techniques to this network and this teacher. [[scenario-distribution]] concerns the data the whole thing trains on, which turns out to dominate both the gradient and any reported number. [[rlhf-transfer]] is a survey of one adjacent literature, kept separate because most of it does not apply and saying so is the useful part.

Symbols are fixed in [[../overview#Notation]], which also records which topics this tree carries itself and which it assumes from the owner's existing study notes.

## What binds this material

Decisions live in `decisions/`, not here. [[../decisions/0005-training-and-reward]] is the one that governs everything above, and it states the staging, the reward candidates, and what is deliberately still open. Where a document here recommends something, that is a recommendation until a record accepts it.

## Related

- [[../overview]], the problem, the notation, the current state.
- [[../roadmap]], what comes after battles.
- [[../implementation/README]], the environment these methods would train against.
- [[../research/findings]], the evidence base.
