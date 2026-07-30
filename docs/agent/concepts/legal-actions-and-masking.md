---
title: "Legal actions and masking: the fixed action space and why it must be masked"
type: concept-primer
depth: standard
grounded-in: "fheroes2 agent-env branch (Milestone 3, ADR 0002)"
related_concepts: ["[[battle-turn-dispatch]]", "[[teacher-coverage-and-behavior-cloning]]", "[[observation-design]]"]
tags: [concept, action-space, masking, rl-theory, agent-env]
---

> **What this is.** Why a policy over battle actions needs a *fixed* action space plus a
> per-state legality mask, how our `Discrete(793)` indexing is laid out, and why the mask and the
> candidate list are two views of a single enumeration. This is the concept the project's
> largest risk was about.

## The one-sentence version

The policy always outputs a distribution over the same 793 action slots; before sampling, the
illegal slots are set to $-\infty$ so they receive exactly zero probability *and* zero gradient —
which is provably still a correct policy gradient, and empirically the difference between an
agent that learns and one that does not.

## The problem

In a battle state, a stack can move to some cells, attack some enemies from some directions, or
skip. Which of those are legal changes every single decision. Two naive approaches both fail:

- **Variable-length action lists** ("here are today's 14 options, pick one") break every standard
  RL implementation, which assumes a fixed-size output layer, and make action identity unstable
  across states.
- **Penalising illegal actions** with negative reward is the approach the literature most
  clearly refutes: as the illegal space grows, penalty-trained agents collapse while masked ones
  stay flat in time-to-solve.

## The mechanism: masking

Let the policy produce logits $\ell \in \mathbb{R}^{793}$ and let $m \in \{0,1\}^{793}$ be the
legality mask for the current state. Replace the logits before the softmax:

$$\ell'_i = \begin{cases} \ell_i & m_i = 1 \\ -10^{8} & m_i = 0 \end{cases}, \qquad
\pi(a\mid s) = \operatorname{softmax}(\ell')_a$$

Two consequences, both load-bearing:

1. $\pi(a \mid s) \approx 0$ for every illegal $a$ — the agent cannot emit an invalid command.
2. $\partial \mathcal{L} / \partial \ell_i = 0$ for masked entries — no gradient is wasted
   learning "don't do that here".

**The theory.** Masking is a *state-dependent differentiable transform of the logits*, so the
masked update is the policy gradient of the masked policy — not a heuristic hack (Huang &
Ontañón, FLAIRS 2022). Apply the mask at **both** sampling and gradient time; sample-only masking
destabilises PPO's KL.

**The evidence.** In full-game microRTS, PPO with no mask scored a **0.0** cumulative win rate;
masking only the action *type* (the PySC2/SMAC style) reached 0.32; full per-component masking
reached 0.82–0.91. Mask everything, not just the head.

## Our layout: `Discrete(793)`

The index is a pure function of board geometry and the action taxonomy — **stable across states,
episodes, and machines** (ADR 0002):

| Range | Meaning | Size |
|---|---|---|
| `0` | SKIP | 1 |
| `1 … 99` | MOVE to head cell $c$ → index $1 + c$ | 99 |
| `100 … 198` | RANGED attack on the enemy whose head cell is $c$ → $100 + c$ | 99 |
| `199 … 792` | MELEE onto target cell $t$ from direction $d$ → $199 + 6t + d$ | 594 |

$d \in \{0..5\}$ enumerates the six hex directions in engine enum order. The melee index is keyed
by *(target cell, direction)* rather than by defender id because that pair uniquely determines
the attacking cell — the engine derives it by stepping backwards from the target along the
reflected direction, and we index the same way it validates.

## One enumeration, two views

The subtle design point. For each decision we enumerate candidates **once**, validating every
probe through `battle_action_validation` — the exact functions `Arena::ApplyAction*` executes
with — and emit:

- the **boolean mask** (for the learner), and
- the **candidate list** with semantic metadata and engine-ready command parameters (for the
  protocol, teacher matching, and debugging).

The invariant `mask[i] == 1 ⟺ a candidate with index i exists` is asserted at runtime. Because
both come from one validated enumeration, the learner's mask can never disagree with what the
engine would accept — the failure mode that would otherwise surface as a crash mid-training.

> [!warning]- Never test legality by applying a candidate
> The tempting shortcut — "apply it and see if the engine complains" — mutates the arena and
> consumes randomness, corrupting the very episode you are enumerating for. Enumeration must be
> non-mutating by construction, which is why the validators were extracted rather than re-derived.

## Why it matters here

This closed the project's #1 risk. The engine has no public "list legal actions" API, and its
validation lived in anonymous-namespace lambdas — so the choice was *extract them* or *re-derive
battle rules and get them subtly wrong*. We extracted, verbatim, and the engine now runs through
the extracted code. The proof it worked: **100 % of built-in-AI decisions map onto legal
canonical actions** (116/116 across all fixtures).

## What this does *not* say

The space covers `simple_v1` only: single-cell, walking creatures. Wide units, flyers, two-cell
and all-adjacent attacks, and area shots are excluded by the capability audit and would each add
targeting semantics this indexing does not express.

## See also
- [[teacher-coverage-and-behavior-cloning]] — how we prove the space is complete.
- [[battle-turn-dispatch]] — where enumeration is called from.
