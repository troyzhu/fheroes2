---
title: "The policy network, end to end"
type: reference
updated: 2026-08-06
related_concepts: ["[[../implementation/observation-design]]", "[[training-design]]", "[[../decisions/0006-encoding-count-scaling]]", "[[value-estimation-lab]]"]
tags: [agent-env, architecture, network, reference]
---

# The policy network, end to end

The owner asked for the full picture in one place, because the pooling-against-concatenation question is meaningless without it. This page walks the exact tensor path from the worker's JSON to the 793 logits, then shows where the slot lifecycle creates the problem that pooling would remove. The implementation is `python/fheroes2_agent/policy.py` consuming `encoding.py`; widths here are quoted from code, and [[training-design#As built, 2026-08-05|training-design]] carries the capacity measurements behind them.

For the runtime view of this same network, one real decision followed from engine JSON through the forward pass to the chosen action, read [[../implementation/inference-walkthrough]] beside this page.

## From battle to tensor

A battle state arrives as up to ten living stacks plus four battle-level scalars. The encoder lays them out as ten fixed slots of 63 features each, 630 numbers, then the 4 globals, 634 in total (`obs_encoding_v3`).

Each slot's 63 features are: presence and side flags, the active-turn flag, counts (current, initial, fraction), hit points (stack total and the damaged top creature), the six stat fields (attack, defense, speed, shots, morale, luck), position (row, column, cell), four ability flags (wide, flying, archer, hand-fighting), and a 41-way creature one-hot. An empty slot is all zeros with the presence flag off.

Slots are filled in the engine's unit order, own side and enemy side mixed, living stacks only. That last clause is the load-bearing one and the section below returns to it.

## The network, layer by layer

```mermaid
flowchart TD
    slots["10 unit slots<br/>each 63 features"] -- "shared weights, applied per slot" --> enc["slot encoder<br/>63 → 96 → 96"]
    globals["4 globals"] --> genc["globals encoder<br/>4 → 32"]
    enc -- "10 × 96, slot order" --> cat["concatenate<br/>960 + 32 = 992"]
    genc --> cat
    cat --> trunk["trunk<br/>992 → 192 → 192"]
    trunk --> pi["policy head<br/>192 → 793 logits"]
    trunk --> v["value head<br/>192 → 1"]
    pi --> mask["mask illegal to −10⁸,<br/>softmax over the rest"]
    planes["(7, 9, 11) planes,<br/>optional"] -.-> pconv["plane conv 7→32→32,<br/>flatten, 3168 → 128"]
    pconv -. "planes arm only" .-> cat
```

The same architecture as PyTorch itself prints it, which is the ground truth the diagram summarizes, captured verbatim from `print(BattlePolicy())`:

```text
BattlePolicy(
  (slot_encoder): Sequential(
    (0): Linear(in_features=63, out_features=96, bias=True)
    (1): ReLU()
    (2): Linear(in_features=96, out_features=96, bias=True)
    (3): ReLU()
  )
  (global_encoder): Sequential(
    (0): Linear(in_features=4, out_features=32, bias=True)
    (1): ReLU()
  )
  (trunk): Sequential(
    (0): Linear(in_features=992, out_features=192, bias=True)
    (1): ReLU()
    (2): Linear(in_features=192, out_features=192, bias=True)
    (3): ReLU()
  )
  (policy_head): Linear(in_features=192, out_features=793, bias=True)
  (value_head): Linear(in_features=192, out_features=1, bias=True)
)
```

A planes-built checkpoint adds two modules and widens the trunk's first layer to 1120. The (7, 9, 11) input is not pixels: 9 by 11 is the battlefield grid itself, one entry per hex cell, and the seven channels are semantic quantities written at each cell (per-side occupancy, count fraction, log-scaled hit points, speed, shooter, obstacle), image-shaped so a convolution can read the spatial structure while every value stays an exact engine fact, the SC2 feature-layer idea [[../decisions/0004-spatial-observation-modality]] commits to and the reason rendered pixels are rejected permanently:

```text
  (plane_conv): Sequential(
    (0): Conv2d(7, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
  )
  (plane_fc): Sequential(
    (0): Linear(in_features=3168, out_features=128, bias=True)
    (1): ReLU()
  )
```

One two-layer perceptron, 63 to 96 to 96, embeds every slot with the same weights, which is what makes a stack's meaning come from its features rather than from which slot happens to hold it, at the embedding stage. The ten 96-wide embeddings are then concatenated in slot order into a 960-wide block, the encoded globals append 32 more, and a two-layer 192-wide trunk feeds both heads. The trunk is the shared body of a multi-head network, what some papers call the backbone or torso: everything before it is specific to one input kind, everything after it is specific to one output, and both heads read the same 192-wide representation because the features that pick an action and the features that judge a position overlap heavily.

396,570 parameters in total. The policy head's 793 logits are masked to the legal actions ([[../implementation/legal-actions-and-masking]]) and softmaxed. The `Sequential` printout above is regenerated in one line, `python3 -c "import sys; sys.path.insert(0, 'python'); from fheroes2_agent.policy import BattlePolicy; print(BattlePolicy())"`, so a drifted diagram is always one command from being caught.

## Where the slot lifecycle bites

The concatenation step undoes half of what the shared encoder bought. Because slots hold living stacks in engine order, a death shifts every later stack down one slot, and concatenation wires each slot to a fixed range of trunk inputs.

Consider three own stacks A, B, C in slots 0, 1, 2, with C a Champion whose embedding feeds trunk inputs 192 through 287. B dies. Now C sits in slot 1 and the same Champion, same cell, same counts, feeds inputs 96 through 191 instead. To the trunk these are different input coordinates, so every fact it has learned about "the Champion pattern at slot 2" must be relearned at slot 1, and at slot 0, and for every arrangement deaths can produce. The trunk must spend capacity learning that ten copies of the same pattern mean the same thing, a permutation invariance the architecture could simply grant.

## What pooling would change

```mermaid
flowchart LR
    e0["embed 96"] --> pool["order-insensitive pool\nsum, mean, or attention\n→ 96 (or a few heads)"]
    e1["embed 96"] --> pool
    e9["embed 96"] --> pool
    pool --> trunk2["trunk"]
```

A pooled aggregation, summing or averaging the ten embeddings, or attending over them, produces the same output under any slot permutation, so a death changes nothing but the dead stack's absence. The invariance arrives by construction instead of by training data.

The cost is what concatenation was chosen for at ten slots: a concatenated trunk can, in principle, represent relations between specific stacks ("my slot-2 archer versus their slot-7 flyer") that a plain sum collapses; attention pooling recovers that expressiveness at the price of more machinery. [[training-design#The policy network|The original design]] chose concatenation as the simplest thing that could work and named mean or attention pooling the fallback, so this was always a deferred experiment rather than a settled conviction.

## Why it is the live suspect

Three measured symptoms fit the diagnosis. Held-out transfer on the fully diverse pool collapsed to $+0.007 \pm 0.046$ where the training split gained $+0.173 \pm 0.039$. The count-extrapolation ablation lost a third of its agreement above the training range under every encoding, so the encoding was not the axis. And the supervised plateau of 2026-08-06, five label arms all trading axes at a fixed network, is what capacity spent on relearnable invariances would look like. None of the three is proof, which is exactly why pooling-against-concatenation is the recorded experiment rather than an adopted change: same corpus, same battery, one axis moved.

## The planes arm, built after this page's first draft

`BattlePolicy(planes=True)` now exists: two 3-by-3 convolutions at 32 channels over the (7, 9, 11) tensor of [[../implementation/observation-design#The planes, built across 2026-08-06 and 07|encode_planes]], no downsampling, squeezed to 128 and concatenated into the trunk beside the slots and globals, about 441k extra parameters, with `load_policy` inferring the arm from the state dict. The three-seed capacity-controlled ablation found its fidelity gain real (agreement 0.897 against the width control's 0.875) and its play gains modest; [[../archive/experiments/2026-08-07-overnight-champion-mixture]] carries the champion-mixture measurement. The pooling question above is unchanged by any of this, since the slot lifecycle bites the entity half either way.

## The experiment, when it runs

Swap the concatenation for mean pooling first (cheapest, most invariant), then attention if the mean collapses needed distinctions. Everything else stays fixed: corpus, losses, battery, multi-seed gate per the experiments README conventions. The battery columns that should move if the diagnosis is right are held-out transfer and count extrapolation; the ladder should hold. A trade, ladder down for transfer up, would say stack-specific relations were load-bearing after all, and attention is the next arm.
