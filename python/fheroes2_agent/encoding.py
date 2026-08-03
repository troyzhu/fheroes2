"""Turn a recorded observation into a fixed-width feature tensor.

The environment emits observations as JSON with one entry per living stack, so their length
varies with the battle. A policy network needs a fixed shape, and this module fixes it: ten slots
of per-stack features plus a few global ones, matching the entity encoder in
`agent_play/docs/rl/training-design.md`.

The encoding is versioned. Any change to `FEATURE_NAMES`, `SLOT_COUNT` or the scaling constants
is a change to what a trained checkpoint expects, so `ENCODING_VERSION` moves with it and gets
stamped into anything produced (ADR 0003).

Scaling is deliberately plain division by a constant rather than a fitted normalizer. A fitted
one would have to be stored, versioned and applied identically at training and at deployment,
which is a failure mode disproportionate to the gain at this feature count.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ENCODING_VERSION = "obs_encoding_v1"

# The battlefield is 11 wide and 9 tall, so a cell index is 11 * row + column.
BOARD_WIDTH = 11
BOARD_HEIGHT = 9
BOARD_CELLS = BOARD_WIDTH * BOARD_HEIGHT

# Five stacks a side is the scenario schema's limit, so ten slots always suffice and a battle
# never overflows. Slots beyond the living stack count are zero-filled and flagged absent.
SLOT_COUNT = 10

# Per-slot features, in order. Named so an encoded row can be read back by a human, which is
# what makes a bad encoding findable rather than merely suspected.
FEATURE_NAMES: tuple[str, ...] = (
    "present",  # 0 for a padding slot; every other feature is then 0 too
    "is_own_side",  # relative to the stack on turn, so the encoding is side-symmetric
    "is_attacker",  # absolute side, which still matters because starting positions differ
    "is_active",  # the stack whose turn it is
    "count",
    "initial_count",
    "count_fraction",  # survivors over initial, so losses are directly visible
    "hit_points",
    "top_hit_points",
    "attack",
    "defense",
    "speed",
    "shots",
    "morale",
    "luck",
    "row",
    "column",
    "cell",
    "is_wide",
    "is_flying",
    "is_archer",
    "is_hand_fighting",
)

GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "round",
    "active_is_attacker",
    "own_stacks",
    "enemy_stacks",
)

SLOT_FEATURES = len(FEATURE_NAMES)
GLOBAL_FEATURES = len(GLOBAL_FEATURE_NAMES)
OBSERVATION_SIZE = SLOT_COUNT * SLOT_FEATURES + GLOBAL_FEATURES

# Divisors chosen to land typical values near 1 without clipping the tail. A count of 1000 is the
# scenario schema's own safety maximum, so nothing legal saturates.
_COUNT_SCALE = 100.0
_HP_SCALE = 100.0
_STAT_SCALE = 10.0
_SHOT_SCALE = 20.0
_MOOD_SCALE = 3.0
_ROUND_SCALE = 20.0

ACTION_SPACE_SIZE = 1 + 99 + 99 + 99 * 6  # 793, ADR 0002


def encode_observation(observation: dict[str, Any]) -> np.ndarray:
    """One observation to a float32 vector of length OBSERVATION_SIZE."""
    out = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    units = observation["units"]
    if len(units) > SLOT_COUNT:
        raise ValueError(f"{len(units)} living stacks exceeds SLOT_COUNT={SLOT_COUNT}")

    active_is_attacker = bool(observation["active_is_attacker"])
    own = enemy = 0

    for slot, unit in enumerate(units):
        base = slot * SLOT_FEATURES
        is_attacker = unit["side"] == "attacker"
        is_own = is_attacker == active_is_attacker
        own += is_own
        enemy += not is_own

        cell = unit["head_cell"]
        row, column = (cell // BOARD_WIDTH, cell % BOARD_WIDTH) if cell >= 0 else (0, 0)
        initial = max(unit["initial_count"], 1)

        out[base + 0] = 1.0
        out[base + 1] = float(is_own)
        out[base + 2] = float(is_attacker)
        out[base + 3] = float(unit["active"])
        out[base + 4] = unit["count"] / _COUNT_SCALE
        out[base + 5] = unit["initial_count"] / _COUNT_SCALE
        out[base + 6] = unit["count"] / initial
        out[base + 7] = unit["hit_points"] / _HP_SCALE
        out[base + 8] = unit["top_hit_points"] / _HP_SCALE
        out[base + 9] = unit["attack"] / _STAT_SCALE
        out[base + 10] = unit["defense"] / _STAT_SCALE
        out[base + 11] = unit["speed"] / _STAT_SCALE
        out[base + 12] = unit["shots"] / _SHOT_SCALE
        out[base + 13] = unit["morale"] / _MOOD_SCALE
        out[base + 14] = unit["luck"] / _MOOD_SCALE
        out[base + 15] = row / (BOARD_HEIGHT - 1)
        out[base + 16] = column / (BOARD_WIDTH - 1)
        out[base + 17] = max(cell, 0) / (BOARD_CELLS - 1)
        out[base + 18] = float(unit["wide"])
        out[base + 19] = float(unit["flying"])
        out[base + 20] = float(unit["archer"])
        out[base + 21] = float(unit["hand_fighting"])

    g = SLOT_COUNT * SLOT_FEATURES
    out[g + 0] = observation["round"] / _ROUND_SCALE
    out[g + 1] = float(active_is_attacker)
    out[g + 2] = own / SLOT_COUNT
    out[g + 3] = enemy / SLOT_COUNT
    return out


def encode_mask(legal_actions: list[int]) -> np.ndarray:
    """The legal set as a boolean mask over the canonical action space."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for index in legal_actions:
        if not 0 <= index < ACTION_SPACE_SIZE:
            raise ValueError(f"action index {index} outside [0, {ACTION_SPACE_SIZE})")
        mask[index] = True
    return mask


def describe(vector: np.ndarray) -> str:
    """Render an encoded observation back into named fields, for eyeballing a sample."""
    lines = []
    for slot in range(SLOT_COUNT):
        base = slot * SLOT_FEATURES
        if vector[base] == 0.0:
            continue
        fields = ", ".join(f"{name}={vector[base + i]:.3g}" for i, name in enumerate(FEATURE_NAMES))
        lines.append(f"slot {slot}: {fields}")
    g = SLOT_COUNT * SLOT_FEATURES
    lines.append(", ".join(f"{name}={vector[g + i]:.3g}" for i, name in enumerate(GLOBAL_FEATURE_NAMES)))
    return "\n".join(lines)
