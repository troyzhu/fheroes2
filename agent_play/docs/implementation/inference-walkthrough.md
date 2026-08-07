---
title: Inference, one decision at a time — the authoritative walkthrough
aliases:
  - inference
  - inference-walkthrough
  - example-play
tags:
  - agent-env
  - primer
concept: what a trained policy receives, computes, and returns at play time, shown on one real battle
domain: RL environment design
grounded_in: "python/fheroes2_agent/{env,encoding,policy,render}.py, src/agent_worker/main.cpp; every number from files/2026-08-07-run-reports/inference_walkthrough.json"
depth: quick
updated: 2026-08-07
---

# Inference, one decision at a time — the authoritative walkthrough

The owner asked for one place that states the inference process cleanly: given a trained policy, what exactly goes in, what exactly happens, what comes out, shown on an example play. This page is that reference. Every number below comes from one real recorded episode, `files/2026-08-07-run-reports/inference_walkthrough.json` in the archive, so nothing here is illustrative fiction, and the closing section reproduces it in one command.

## The loop in five steps

Inference is a conversation between two processes. The worker (`fheroes2_agent_worker --protocol`) runs the real engine and blocks whenever a controlled stack must act; the Python side (`BattleEnv` in `env.py`) turns that into the familiar reset-and-step loop.

1. **The engine reaches a controlled decision** and the worker emits one JSON line: the observation (below), the legal-action list, and then it blocks on stdin.
2. **Python encodes** the observation into a flat `float32[634]` vector (`encode_observation`) and the legal list into a `bool[793]` mask (`encode_mask`).
3. **The network runs once**: `logits, value = model(observation, mask)`, one forward pass, no search, no history, the current state only.
4. **Illegal actions are impossible by construction**: the mask sets their logits to $-10^8$ before the softmax, and the action is sampled from the surviving distribution (or argmax when playing greedily).
5. **The canonical index goes back** on one stdin line; the worker translates it into the engine command (`applyCanonicalAction`) and the battle advances to the next controlled decision or the terminal record.

A planes-built policy changes only step 2 and 3: the worker is launched with `--planes`, the observation gains a 99-cell `obstacles` array, `encode_planes` rasterizes a third input of shape `(7, 9, 11)`, and the forward pass takes it as a third argument. Nothing else moves.

## The input, exactly

The policy's input is the pair (observation vector, legality mask), plus the planes tensor when the checkpoint was built with the conv arm. The vector is $10 \times 63 + 4 = 634$ floats: ten fixed unit slots of 63 features each, then four globals. Slots hold living stacks only, both sides mixed, sorted by engine uid; unused slots are all zero with `present` off. `FEATURE_NAMES` in `encoding.py` is the single authoritative layout, and the 63 decompose as 22 named features plus a 41-way creature one-hot.

| Index | Feature | Exact transform |
|---|---|---|
| 0 | `present` | 1 for a real stack, 0 for padding |
| 1 | `is_own_side` | 1 when the slot is on the acting stack's side, so the encoding is side-symmetric |
| 2 | `is_attacker` | absolute side, kept because starting positions differ |
| 3 | `is_active` | 1 on exactly the stack whose turn it is |
| 4, 5 | `count`, `initial_count` | $\log(1+n)/\log(1+1000)$, the measured count scaling of [[../decisions/0006-encoding-count-scaling]] |
| 6 | `count_fraction` | survivors over initial, linear |
| 7, 8 | `hit_points`, `top_hit_points` | $\log(1+h)/\log(1+50000)$; stack total, and the damaged top creature |
| 9, 10, 11 | `attack`, `defense`, `speed` | divided by 10 |
| 12 | `shots` | divided by 20 |
| 13, 14 | `morale`, `luck` | divided by 3 |
| 15, 16, 17 | `row`, `column`, `cell` | row/8, column/10, cell/98 |
| 18 to 21 | `is_wide`, `is_flying`, `is_archer`, `is_hand_fighting` | flags |
| 22 to 62 | `is_monster_<id>` | one-hot creature identity over the 41-creature allowlist |

The four globals are `round` (divided by 20), `active_is_attacker`, and `own_stacks`, `enemy_stacks` (living stacks on each side, divided by the ten-slot capacity). Commander stats appear nowhere separately, because the engine folds them into every unit's effective attack and defense before the observation is captured. What the policy cannot see is the board itself, no obstacle or terrain features, unless the planes tensor is supplied; [[observation-design#What the board does not say]] carries that boundary.

## The network, exactly

The default `BattlePolicy` is 396,570 parameters: a shared two-layer perceptron $63 \to 96 \to 96$ embeds every slot with the same weights, padding slots are re-zeroed after encoding, the ten embeddings concatenate with a $4 \to 32$ globals encoding into a 992-wide vector, a two-layer $992 \to 192 \to 192$ trunk produces the representation, and two linear heads emit the 793 action logits and a scalar value estimate. [[../rl/the-policy-network]] walks every layer with diagrams and the pooling question; checkpoints self-describe, so `load_policy` reconstructs optional arms (ability features, the planes convolution) from the state dict alone.

## One real decision, under the microscope

The vendored episode plays 4 Veteran Pikemen and 8 Archers against 5 Rogues and 10 Zombies, `policy_gen1` at torch seed 7. At decision 2, round 2, the Pikemen hold cell 12 with 33 legal actions. Two of the four live slots, exactly as the network received them:

| Feature | Slot 0, 4 Veteran Pikemen (acting) | Slot 1, 8 Archers |
|---|---|---|
| `present` / `is_own_side` / `is_active` | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 0.0 |
| `count` / `initial_count` / `count_fraction` | 0.233 / 0.233 / 1.0 | 0.318 / 0.318 / 1.0 |
| `hit_points` / `top_hit_points` | 0.4061 / 0.2814 | 0.4061 / 0.2216 |
| `attack` / `defense` / `speed` | 0.5 / 0.9 / 0.5 | 0.5 / 0.3 / 0.2 |
| `shots` | 0.0 | 0.55 |
| `row` / `column` / `cell` | 0.125 / 0.1 / 0.1224 | 0.25 / 0.0 / 0.2245 |
| one-hot | `is_monster_5` | `is_monster_2` |

Globals: `round` 0.10, `active_is_attacker` 1.0, `own_stacks` 0.2, `enemy_stacks` 0.1, which says round 2, two friendly stacks, and only one enemy stack still alive. The forward pass also returns a value estimate, here $-1.459$ on the trained-on return scale; it is visibly miscalibrated for a fight the policy is winning, an honest reminder that this head was fitted long before the champion distillations and [[../rl/value-estimation-lab]] carries exactly why it is not trusted. After masking, the distribution over the 33 legal actions concentrates like this:

| Rank | Action | Probability |
|---|---|---|
| 1 | MOVE to cell 23 | 0.453 |
| 2 | MOVE to cell 68 | 0.374 |
| 3 | MOVE to cell 67 | 0.072 |
| 4 | MOVE to cell 1 | 0.027 |
| 5 | MOVE to cell 27 | 0.022 |

Sampling drew the mode, MOVE to cell 23, and the index went back to the worker as one line. Two candidate advances, cells 23 and 68, hold 83 percent of the mass between them, so this is a real fork the sample resolves. Everything above is a few hundred floats and one forward pass away from the JSON the engine emitted; there is no other state, no memory of earlier decisions, and no lookahead unless a search wrapper is explicitly driving ([[../rl/rl-methods#Search as an improvement operator]]).

## The example play, end to end

The whole episode, as the play-by-play the JSON records:

| Decision | Round | Acting stack | Chose | p |
|---|---|---|---|---|
| 0 | 1 | 4 Veteran Pikemen at cell 0 | MOVE to cell 12 | 0.101 |
| 1 | 1 | 8 Archers at cell 22 | SHOOT the stack on cell 28 | 0.532 |
| 2 | 2 | 4 Veteran Pikemen at cell 12 | MOVE to cell 23 | 0.453 |
| 3 | 2 | 8 Archers at cell 22 | SHOOT the stack on cell 18 | 1.000 |
| 4 | 3 | 4 Veteran Pikemen at cell 23 | MELEE cell 18 from the top-right | 0.083 |
| 5, 7 | 3, 4 | 8 Archers | SHOOT the stack on cell 18 | 1.000 both |
| 6, 8 | 3, 4 | 4 Veteran Pikemen | MELEE cell 18, top-right then right | 0.362, 0.054 |

Victory in 9 decisions, terminal reward 1.744, which reads as $+1$ for the win plus 0.744 of the starting force surviving ([[../rl/reward-design]]). The opening is the policy's signature shape: the shooter fires from standing while the melee stack closes, then both converge on the survivor. Note decision 3's probability of 1.000, a state so deep in the training distribution that the distribution collapsed; decision 4's 0.083 is the opposite, a genuinely uncertain choice where sampling matters.

## Reproduce it

```bash
W=work  # any scratch directory holding a worker copy and checkpoint
python3 - <<'EOF'
# the exact generator of the vendored JSON, seed and all
# see files/2026-08-07-run-reports/inference_walkthrough.json for the recorded output
EOF
```

The generating script is vendored beside the JSON. Determinism note: the engine side reproduces exactly (same seed, same battle); the policy side samples, so `torch.manual_seed(7)` is part of the record.

## Go deeper

- [[../rl/the-policy-network]], every layer with diagrams and the live architectural question.
- [[observation-design]], both observation axes and what the board does not say.
- [[legal-actions-and-masking]], the 793-slot layout and the index arithmetic.
- [[battle-turn-dispatch]], where in the engine the decision hook sits.
- `python/fheroes2_agent/watch.py`, a human-readable live printout of the same loop.
