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

ENCODING_VERSION = "obs_encoding_v3"

# The 41 monsters the simple_v1 allowlist supports, from the generated capability audit. Creature
# identity is one-hot rather than a scalar id, because ids are labels and their magnitudes mean
# nothing: encoding Swordsman as 6 and Peasant as 1 would tell the network that a Swordsman is six
# Peasants. vcmi-gym, the only shipped comparable system, encodes categories the same way with an
# explicit NULL for empty slots, which the `present` flag serves here.
#
# v1 omitted identity entirely, leaving the policy to infer creature type from attack, defense,
# speed and the ability flags. Those carry most of it, which is why v1 trained at all, but they do
# not separate creatures that share a stat line and differ in something unmodelled.
SIMPLE_V1_MONSTERS: tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19, 22, 23, 24, 25, 26,
    27, 33, 34, 39, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 58, 63, 64, 65, 66,
)
MONSTER_SLOT = {monster_id: index for index, monster_id in enumerate(SIMPLE_V1_MONSTERS)}

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
) + tuple(f"is_monster_{monster_id}" for monster_id in SIMPLE_V1_MONSTERS)

GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "round",
    "active_is_attacker",
    "own_stacks",
    "enemy_stacks",
)

SLOT_FEATURES = len(FEATURE_NAMES)
GLOBAL_FEATURES = len(GLOBAL_FEATURE_NAMES)
OBSERVATION_SIZE = SLOT_COUNT * SLOT_FEATURES + GLOBAL_FEATURES

# Counts and hit points are log-scaled, and that is a measured decision rather than taste. v2
# divided them by 100 linearly, which across the range the environment now produces inverts
# tactical salience: one creature against five differ by 0.04 while nine hundred against a
# thousand differ by 1.0. Trained on stacks of at most 300 and tested above 600, the linear
# encoding agreed with the teacher on 0.239 of decisions and the log encoding on 0.303, a gap of
# 24 standard errors across training seeds, while on counts inside the trained range the two are
# indistinguishable. The denominators put the schema cap near one; log1p keeps zero at zero.
# The stat divisors stay linear, since attack and defense span one order of magnitude, not three.
import math as _math

_LOG_COUNT_SCALE = _math.log1p(1000.0)
_LOG_HP_SCALE = _math.log1p(50000.0)
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
        out[base + 4] = np.log1p(unit["count"]) / _LOG_COUNT_SCALE
        out[base + 5] = np.log1p(unit["initial_count"]) / _LOG_COUNT_SCALE
        out[base + 6] = unit["count"] / initial
        out[base + 7] = np.log1p(unit["hit_points"]) / _LOG_HP_SCALE
        out[base + 8] = np.log1p(unit["top_hit_points"]) / _LOG_HP_SCALE
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
        # One-hot creature identity. An id outside the allowlist leaves every slot zero rather
        # than raising, because the capability gate should have rejected the scenario earlier and
        # a training run should not die on a stale allowlist.
        identity = MONSTER_SLOT.get(unit["monster_id"])
        if identity is not None:
            out[base + 22 + identity] = 1.0

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


# --- planes_v1, ADR 0004's spatial modality ---------------------------------------------------
#
# The engine emits only what the entity list cannot carry, the obstacle layer; every other
# committed channel is rasterized here from the same units the slot encoding reads, so the two
# modalities share one source of truth. Layout is channels-first for a convolution,
# (channel, row, column) over the engine's own 11-wide row-offset hex indexing, cell = row * 11
# + column, exactly Battle::Board's.

BOARD_WIDTH = 11
BOARD_HEIGHT = 9
PLANE_CHANNELS = (
    "attacker_occupancy",
    "defender_occupancy",
    "count_fraction",
    "hit_points",
    "speed",
    "shooter",
    "obstacle",
)


def encode_planes(observation: dict) -> np.ndarray:
    """The planes_v1 tensor for one observation, shape (7, 9, 11).

    Unit channels write at head and tail cells. Hit points scale by log1p against 1000 per ADR
    0006's measured convention, counts as the surviving fraction, speed against the game's cap
    of 10. The obstacle channel needs an observation recorded with the worker's --planes flag;
    without one it stays zero, which callers must treat as "unknown", not "open ground".
    """
    planes = np.zeros((len(PLANE_CHANNELS), BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    for unit in observation["units"]:
        cells = [unit["head_cell"]]
        if unit.get("tail_cell", -1) is not None and unit.get("tail_cell", -1) >= 0:
            cells.append(unit["tail_cell"])
        for cell in cells:
            if cell < 0 or cell >= BOARD_WIDTH * BOARD_HEIGHT:
                continue
            row, column = divmod(cell, BOARD_WIDTH)
            planes[0 if unit["side"] == "attacker" else 1, row, column] = 1.0
            planes[2, row, column] = unit["count"] / max(unit["initial_count"], 1)
            planes[3, row, column] = np.log1p(unit["hit_points"]) / np.log1p(1000.0)
            planes[4, row, column] = unit["speed"] / 10.0
            planes[5, row, column] = 1.0 if (unit["archer"] and unit["shots"] > 0) else 0.0
    for index, blocked in enumerate(observation.get("obstacles", ())):
        row, column = divmod(index, BOARD_WIDTH)
        planes[6, row, column] = float(blocked)
    return planes
