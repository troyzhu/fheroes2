---
title: Teacher coverage and behavior cloning — a primer
aliases:
  - teacher-coverage
  - behavior-cloning
tags:
  - agent-env
  - primer
concept: measuring action-space completeness against a scripted teacher
domain: imitation learning and RL evaluation
grounded_in: "agent_play/docs/references/ (AlphaStar, Gym-µRTS, vcmi-gym); this repo's Milestones 2 and 3"
depth: quick
updated: 2026-07-30
---

# Teacher coverage and behavior cloning — a primer

The engine's own tactical AI plays every battle we run, and recording it serves two purposes at once. Checking that each of its decisions maps onto a legal action in our canonical space measures whether that space is complete, and the same recordings become the dataset for behavior cloning.

## Motivation

An action space can be incomplete in a way that ordinary testing never reveals. If the enumerator silently omits a legal move, nothing crashes: episodes run, gates pass, and the policy simply never learns a move it was never offered. The defect surfaces much later as a capability ceiling that looks like a training problem.

Exhaustively proving completeness is impractical, because it would mean deriving the legal set independently of the engine, which is the duplication the whole design avoids. What is available instead is a competent player whose choices can be checked against our set.

## The idea in one sentence

Record what the built-in AI does at every decision and measure the fraction of its choices our action space can express.

## Intuition

This is recall over a candidate set, the metric a retrieval system reports as recall@k. A retriever is useless if the gold document never appears among the candidates, however good the reranker is. Here the built-in AI's move is the gold label and our enumeration is the candidate set, so coverage asks whether the right answer was on the menu at all.

The choice of teacher matters for the same reason. A random player would exercise few interesting actions, so its coverage would prove little, while a competent one probes the parts of the space that competent play actually uses.

## How it works

The teacher is `AI::BattlePlanner`, the game's own tactical AI, which already plays both sides of every headless battle. The passive recorder observes its choices through the decision hook without influencing them.

Call $\mathcal{D}$ the set of recorded decisions: one element per full-fledged teacher turn, so a 116-decision recording gives $|\mathcal{D}| = 116$. At each decision $d \in \mathcal{D}$, captured at the same pre-application state the teacher saw, the enumerator produces the legal candidate set $\mathcal{A}^{\text{legal}}_d$ and the recorder resolves the teacher's actual choice to its canonical index $a^{\text{teacher}}_d$. Coverage is the fraction of decisions whose teacher action lands inside the enumerated legal set:

$$\text{coverage} = \frac{\left|\{\, d \in \mathcal{D} : a^{\text{teacher}}_d \in \mathcal{A}^{\text{legal}}_d \,\}\right|}{|\mathcal{D}|}$$ A value below 1 has three possible causes. the enumeration missed a legal action, the canonical indexing cannot express it, or the creature lies outside the `simple_v1` allowlist and the scenario should have been rejected. All three are worth knowing before training rather than after.

The current measurement is 116 of 116 decisions across all five fixtures, with a minimum candidate count of 5 per decision. Zero failures in 116 correlated trials puts the 95% Clopper-Pearson lower bound at 0.974, so the honest reading is that the miss rate is under roughly 2.6% rather than exactly zero.

**From coverage to cloning.** The same recordings are the imitation dataset, and the staging the evidence supports runs in four steps. Collect passive teacher trajectories, which Milestone 2 completed. Behavior-clone $\pi(a \mid o)$ from those decisions, conditioning on what the student will actually receive rather than on the full state, the step [[../research/works/alphastar|AlphaStar]]'s supervised stage carried to an 87% win rate against that game's strongest built-in opponent before any reinforcement learning. The analogy is weaker than it looks. That stage cloned roughly a million human games and was then measured against a different agent, whereas here the demonstrator and the opponent are the same scripted AI, so cloning it can at best approach it.

The fixture gate's corpus is 116 decisions; the recorded training corpora have long since outgrown it, 45,380 samples from the first 2,000-episode recording and hundreds of thousands across the later diverse and relabeled sets, with the cloning ladder, DAgger, and search distillation all run on them. Correct the distribution shift with DAgger, rolling out the student, having the teacher label the states the student actually visited, and training on the union of all data gathered so far. The bound it buys is linear rather than quadratic in the horizon (Ross, Gordon and Bagnell, 2011). Its precondition, that the teacher be answerable at an arbitrary student-visited state, was settled on 2026-08-05: the public `queryUnitTurn` probe answers exactly that way, digest-proven inert, and the first DAgger round has run and measured.

The third and fourth steps carry a documented evidence gap at this scale, because no verified small-scale transition recipe exists.

## Comparison with alternatives

| Method | What it proves | Cost | Blind spot | When preferred |
|---|---|---|---|---|
| Teacher coverage (ours) | the space expresses competent play | free, the AI already plays | actions no AI ever chooses | Any environment with a scripted opponent |
| Unit tests on the enumerator | specific cases behave | cheap | the cases you thought of | Regression pinning, alongside coverage |
| Random-action fuzzing | no crash on valid input | cheap | rarely reaches interesting states | Robustness, not completeness |
| Independent legality derivation | genuine completeness | high, and duplicates battle rules | its own bugs | Never here, by design |
| Human play traces | expresses human strategy | needs a human | small samples | Late-stage evaluation |

Coverage is preferred because it is continuous and free. It re-measures on every gate run, so a refactor that quietly drops an action type fails immediately.

## When to use it

Measure coverage on every verification run and treat any value below 100% as a defect in the enumerator or the scenario filter, not as a tolerance to accept.

## Key terms

- Teacher: the built-in `AI::BattlePlanner`, the source of demonstrations.
- Coverage: fraction of teacher decisions expressible as a legal canonical action.
- Behavior cloning: supervised learning of the policy from recorded teacher decisions.
- DAgger: iterative correction that labels states the student visits, fixing distribution shift.

## Why it came up here

Coverage is Milestone 3's exit criterion because it converts a belief about enumeration into a measured number, and because it keeps measuring afterward.

Calibration for what follows comes from [[../research/works/vcmi-gym|vcmi-gym]], the one shipped comparable system, whose numbers are self-reported: its first working model reached roughly 75% against the weak scripted bot and 45% against the strong one, and a much later iteration averaged about 65% against the strong bot. Parity with the engine's AI is a multi-iteration goal.

## What this does not say

Full coverage proves our space contains everything the teacher does, not everything legal in principle. A move no AI ever plays could still be missing, and that residual is bounded by the capability audit, which excludes creatures whose action space we do not model, rather than by coverage itself.

## Go deeper

- [[legal-actions-and-masking]] — the space coverage is measured against.
- [[battle-turn-dispatch]] — the hook that observes the teacher.
- `agent_play/docs/references/summary.md` — the training-staging evidence and its gaps.
