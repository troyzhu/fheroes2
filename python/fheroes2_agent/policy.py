"""The battle policy network.

Architecture follows `agent_play/docs/rl/training-design.md`: a per-slot multilayer perceptron
with shared weights, concatenated, joined with the global features, through a shared trunk to two
heads. The policy head emits one logit per canonical action; the value head emits a scalar and is
unused during cloning but is what the critic pre-fitting and PPO stages need, so it exists from
the start rather than being bolted on later.

The masking is the load-bearing detail. Illegal logits are replaced by a large negative constant
before the softmax, so the distribution normalizes over the legal set alone. Written as minus
infinity in the mathematics and as -1e8 in code, because a literal infinity produces NaN for a
fully masked row and those propagate silently through a batch.
"""

from __future__ import annotations

import json
import math
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import ACTION_SPACE_SIZE, GLOBAL_FEATURES, SIMPLE_V1_MONSTERS, SLOT_COUNT, SLOT_FEATURES

# Large enough that exp() underflows to zero in float32, small enough not to overflow in float16.
MASK_FILL = -1e8

# The creature one-hot occupies the tail of each slot's features; everything before it is the
# named scalar block. Derived here so a FEATURE_NAMES change breaks this loudly.
ONE_HOT_START = SLOT_FEATURES - len(SIMPLE_V1_MONSTERS)


def ability_feature_table() -> torch.Tensor:
    """A fixed per-creature ability matrix, rows aligned with the one-hot, from the capability
    audit's layer-1 records. Multiplying the observation's existing one-hot by this table gives
    each slot its creature's ability profile as input features, an architectural inductive bias
    that changes no observation bytes and therefore needs no encoding version."""
    path = pathlib.Path(__file__).parent / "data" / "monster_capabilities_v1.json"
    records = {r["monster_id"]: r for r in json.loads(path.read_text())}
    rows = []
    for monster_id in SIMPLE_V1_MONSTERS:
        r = records[monster_id]
        rows.append([
            float(r["is_wide"]), float(r["is_flying"]), float(r["is_archer"]),
            float(r["has_double_shooting"]), float(r["has_double_melee_attack"]),
            float(r["is_double_cell_attack"]), float(r["has_area_or_multi_target_attack"]),
            math.log1p(float(r["hit_points"])) / math.log1p(1000.0),
            math.log1p(float(r["strength"])) / math.log1p(1000.0),
            len(r.get("abilities", [])) / 8.0,
            len(r.get("weaknesses", [])) / 8.0,
        ])
    return torch.tensor(rows, dtype=torch.float32)


ABILITY_FEATURES = ability_feature_table().shape[1]


def load_policy(state_dict: dict) -> "BattlePolicy":
    """Construct the architecture a checkpoint's state dict describes and load it. The ability
    table ships as a buffer, so its presence is the self-describing marker; widths are read off
    the slot encoder's first layer."""
    ability = "ability_table" in state_dict
    planes = "plane_conv.0.weight" in state_dict
    slot_hidden = state_dict["slot_encoder.0.weight"].shape[0]
    trunk_hidden = state_dict["trunk.0.weight"].shape[0]
    global_hidden = state_dict["global_encoder.0.weight"].shape[0]
    # The trunk's input width tells concatenation (SLOT_COUNT * slot_hidden + ...) apart from
    # mean pooling (slot_hidden + ...), so the pooling choice is self-describing too.
    trunk_in = state_dict["trunk.0.weight"].shape[1]
    plane_width = 128 if planes else 0
    pooling = "concat" if trunk_in >= SLOT_COUNT * slot_hidden + global_hidden + plane_width else "mean"
    model = BattlePolicy(slot_hidden=slot_hidden, trunk_hidden=trunk_hidden,
                         global_hidden=global_hidden, ability_features=ability, planes=planes,
                         pooling=pooling)
    model.load_state_dict(state_dict)
    return model


class BattlePolicy(nn.Module):
    # Sized against the data rather than by habit. The concatenation pool sends
    # SLOT_COUNT * slot_hidden into the trunk, so slot_hidden drives most of the parameter count.
    # At 128 and 256 the model held 626k parameters against roughly 37k training decisions, which
    # is the memorization regime training-design.md warns about; 96 and 192 give 396,570 at the
    # current 634-wide encoding.
    def __init__(self, slot_hidden: int = 96, trunk_hidden: int = 192, global_hidden: int = 32,
                 ability_features: bool = False, planes: bool = False,
                 pooling: str = "concat") -> None:
        super().__init__()
        # Optional architectural inductive bias: each slot's input is extended by its creature's
        # fixed ability profile, computed inside the model from the one-hot the observation
        # already carries. The table is a non-trainable buffer, so checkpoints stay
        # self-describing through their state dict.
        self.ability_features = ability_features
        slot_in = SLOT_FEATURES + (ABILITY_FEATURES if ability_features else 0)
        if ability_features:
            self.register_buffer("ability_table", ability_feature_table())
        # Shared across slots, which is what enforces that a stack's meaning comes from its
        # fields rather than from the slot it happens to occupy.
        self.slot_encoder = nn.Sequential(
            nn.Linear(slot_in, slot_hidden),
            nn.ReLU(),
            nn.Linear(slot_hidden, slot_hidden),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(nn.Linear(GLOBAL_FEATURES, global_hidden), nn.ReLU())
        # The planes_v1 fusion arm of ADR 0004: a small convolution over the (7, 9, 11) tensor,
        # no downsampling at this board size, squeezed to a fixed width so the trunk grows by a
        # bounded amount. Absent unless requested, and load_policy infers it from the state dict.
        self.planes = planes
        plane_width = 0
        if planes:
            from .encoding import BOARD_HEIGHT, BOARD_WIDTH, PLANE_CHANNELS

            self.plane_conv = nn.Sequential(
                nn.Conv2d(len(PLANE_CHANNELS), 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            plane_width = 128
            self.plane_fc = nn.Sequential(nn.Linear(32 * BOARD_HEIGHT * BOARD_WIDTH, plane_width), nn.ReLU())
        # Mean pooling replaces the ordered concatenation with a present-masked average of the
        # slot embeddings, granting permutation invariance by construction; the trunk narrows
        # accordingly. the-policy-network.md carries the design argument on both sides.
        if pooling not in ("concat", "mean"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling
        slot_block = SLOT_COUNT * slot_hidden if pooling == "concat" else slot_hidden
        self.trunk = nn.Sequential(
            nn.Linear(slot_block + global_hidden + plane_width, trunk_hidden),
            nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(trunk_hidden, ACTION_SPACE_SIZE)
        self.value_head = nn.Linear(trunk_hidden, 1)

    def features(self, observations: torch.Tensor, planes: torch.Tensor | None = None) -> torch.Tensor:
        batch = observations.shape[0]
        slots = observations[:, : SLOT_COUNT * SLOT_FEATURES].view(batch, SLOT_COUNT, SLOT_FEATURES)
        globals_ = observations[:, SLOT_COUNT * SLOT_FEATURES :]

        if self.ability_features:
            one_hot = slots[:, :, ONE_HOT_START:]
            slots = torch.cat([slots, one_hot @ self.ability_table], dim=-1)

        encoded = self.slot_encoder(slots)
        # An absent slot is all zeros on input but not on output, since the encoder has biases.
        # Zeroing it here keeps padding from contributing, which matters because battles end with
        # far fewer stacks than they start with.
        present = slots[:, :, :1]
        encoded = encoded * present

        if self.pooling == "mean":
            present_counts = present.sum(dim=1).clamp(min=1.0)
            pooled = encoded.sum(dim=1) / present_counts
            slot_block = pooled
        else:
            slot_block = encoded.flatten(1)
        parts = [slot_block, self.global_encoder(globals_)]
        if self.planes:
            if planes is None:
                raise ValueError("this policy was built with planes=True and needs the tensor per sample")
            parts.append(self.plane_fc(self.plane_conv(planes).flatten(1)))
        return self.trunk(torch.cat(parts, dim=1))

    def forward(self, observations: torch.Tensor, masks: torch.Tensor,
                planes: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns masked logits and the value estimate."""
        hidden = self.features(observations, planes)
        logits = self.policy_head(hidden)
        logits = logits.masked_fill(~masks, MASK_FILL)
        return logits, self.value_head(hidden).squeeze(-1)

    @torch.no_grad()
    def act(self, observations: torch.Tensor, masks: torch.Tensor, greedy: bool = False) -> torch.Tensor:
        logits, _ = self.forward(observations, masks)
        if greedy:
            return logits.argmax(dim=-1)
        return torch.distributions.Categorical(logits=logits).sample()


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over already-masked logits.

    The minimum is not zero but the conditional entropy of the teacher's own policy given the
    observation, so residual loss is not by itself evidence of underfitting.
    """
    return F.cross_entropy(logits, targets)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
