---
title: Legal actions and masking — a primer
aliases:
  - action-masking
  - legal-actions
  - invalid-action-masking
tags:
  - agent-env
  - primer
concept: legal-action masking over a fixed discrete action space
domain: policy-gradient RL
grounded-in: "docs/agent/references/ (Huang & Ontañón FLAIRS 2022; Gym-µRTS; vcmi-gym); this repo's Milestone 3"
depth: quick
updated: 2026-07-30
---

# Legal actions and masking — a primer

A battle policy must choose among moves, attacks, and skipping, where the legal set changes every decision. This primer explains why the answer is a fixed action space plus a per-state mask rather than a variable list of options, how our 793-slot space is laid out, and why the mask and the candidate list come from a single enumeration. Masking is the result the project's largest risk turned on, and it is the difference between an agent that learns and one that never does.

## Motivation

In any given state a stack can reach some cells, strike some enemies from some directions, or skip. That set is state-dependent, which breaks the assumption most RL implementations make, that the policy's output layer has a fixed width.

Two obvious workarounds both fail. Emitting a variable-length list of options ("here are today's fourteen choices") gives an action whose meaning shifts between states, so the network cannot learn a stable mapping, and no standard implementation consumes it. Letting the policy propose anything and penalizing illegal choices with negative reward fails empirically: as the illegal space grows, penalty-trained agents collapse while masked agents hold roughly constant time-to-solve.

## The idea in one sentence

Keep one fixed set of action slots for every state, and before sampling set the logits of the currently illegal slots to a large negative number so they receive no probability and no gradient.

## Intuition

This is the attention mask an ML engineer already uses. Padded positions in a transformer batch stay in the tensor and keep their shape; the mask adds $-\infty$ to their scores so softmax assigns them zero weight, and the backward pass sends them no gradient. Legal-action masking is the same operation applied to a policy head, with legality standing in for padding.

The mental picture is a piano with 793 keys where the illegal keys are physically locked at each moment. The keyboard never changes shape, so the player's muscle memory transfers between states, which is exactly the property a variable-length list destroys.

## How it works

Let the policy produce logits $\ell \in \mathbb{R}^{793}$ and let $m \in \{0,1\}^{793}$ be the legality mask for the current state. Replace the logits before the softmax:

$$\ell'_i = \begin{cases} \ell_i & m_i = 1 \\ -10^{8} & m_i = 0 \end{cases}, \qquad
\pi(a \mid s) = \operatorname{softmax}(\ell')_a$$

Two consequences follow, and both matter. Illegal actions get $\pi(a \mid s) \approx 0$, so the agent cannot emit an invalid command. Masked entries also get $\partial\mathcal{L}/\partial\ell_i = 0$, so no capacity is spent learning which actions were unavailable in states already past.

The theory says this is principled rather than a hack. Masking is a state-dependent differentiable transform of the logits, so the masked update is the policy gradient of the masked policy. Apply the mask at both sampling time and gradient time; masking only at sampling time leaves the ratio in the PPO objective inconsistent and the KL term diverges.

The evidence says it is decisive. In full-game microRTS, PPO with no mask reached a cumulative win rate of 0.0, masking only the action type reached 0.32, and full per-component masking reached 0.82 to 0.91.

**Our layout.** The index is a pure function of board geometry and the action taxonomy, so it is stable across states, episodes, and machines.

| Range | Meaning | Slots |
|---|---|---|
| 0 | SKIP | 1 |
| 1 to 99 | MOVE to head cell $c$, index $1 + c$ | 99 |
| 100 to 198 | RANGED attack on the enemy whose head cell is $c$, index $100 + c$ | 99 |
| 199 to 792 | MELEE onto target cell $t$ from direction $d$, index $199 + 6t + d$ | 594 |

Melee is keyed by target cell and direction rather than by defender id because that pair already determines the attacking cell. The engine derives it by stepping backward from the target along the reflected direction, so indexing the same way keeps our enumeration and its validation aligned by construction.

**One enumeration, two views.** Each decision enumerates candidates once, validating every probe through `battle_action_validation`, the functions `Arena::ApplyAction*` itself executes. That one pass emits the boolean mask for the learner and the candidate list carrying semantic metadata and engine-ready command parameters for the protocol, teacher matching, and debugging. The invariant `mask[i] == 1` exactly when a candidate with index `i` exists is asserted at runtime.

> [!warning]- Why legality is never tested by applying a candidate Applying a candidate to see whether the engine complains mutates the arena and consumes combat randomness, corrupting the episode being enumerated. Enumeration has to be non-mutating by construction, which is why the validators were extracted from the engine rather than re-derived beside it.

## Comparison with alternatives

| Approach | Output shape | Tooling support | Evidence | When preferred |
|---|---|---|---|---|
| Fixed space plus mask (ours) | one softmax of fixed width | CleanRL `CategoricalMasked`, sb3-contrib `MaskablePPO` | 0.82–0.91 win rate in microRTS ablations | Discrete spaces up to roughly $10^3$–$10^4$ slots |
| Variable-length candidate list | width changes per state | none standard | no verified codebase consumes one | Never as the learner interface; useful as metadata |
| Factorized (composed) heads | several independent softmaxes | supported, more wiring | reduces $10^7$ joint actions to about 300 logits | When the flat space would be combinatorially large |
| Pointer network over candidates | attention over a candidate set | bespoke | AlphaStar's supervised ablation credits it | Very large or genuinely unbounded candidate sets |
| Penalize illegal actions | unconstrained | trivial | collapses on larger maps | Never for legality |

At 793 slots the flat masked space is small enough that factorization buys nothing. If the space later grows by spells crossed with targets and parameters, factorize into masked components rather than flattening the cross product. The candidate list we keep alongside the mask is what makes an eventual pointer head a drop-in rather than a rewrite.

## When to use it

Use a fixed masked space whenever the environment can compute legality cheaply and the action count stays in the thousands. Both hold here, since the engine already computes legality to execute commands.

## Key terms

- Legal mask: `uint8[793]`, one entry per action slot, 1 when that action is currently legal.
- Candidate: a legal action carrying its canonical index, semantic metadata, and command parameters.
- Canonical action index: position in the fixed space, stable across states and machines.
- Factorized action: an action split into independent components, each with its own softmax.

## Why it came up here

This closed the project's top risk. The engine exposes no "list legal actions" API and kept its validation in anonymous-namespace lambdas, so the choice was to extract those rules or to re-derive battle legality and get it subtly wrong. We extracted them verbatim and the engine now runs through the extracted code. The measured result is that 100 % of built-in-AI decisions map onto legal canonical actions, 116 of 116 across all fixtures.

## What this does not say

The space covers the `simple_v1` profile only, meaning single-cell walking creatures. Wide units, flyers, two-cell and all-adjacent attacks, and area shots are excluded by the capability audit, and each would add targeting semantics this indexing cannot express.

## Go deeper

- [[teacher-coverage-and-behavior-cloning]] — how completeness of this space is measured.
- [[battle-turn-dispatch]] — where enumeration is called from.
- `docs/agent/references/notes/ref-invalid-action-masking.md` — the theory and canonical code.
- `docs/agent/decisions/0002-action-space.md` — the decision record for this layout.
