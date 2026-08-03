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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import ACTION_SPACE_SIZE, GLOBAL_FEATURES, SLOT_COUNT, SLOT_FEATURES

# Large enough that exp() underflows to zero in float32, small enough not to overflow in float16.
MASK_FILL = -1e8


class BattlePolicy(nn.Module):
    # Sized against the data rather than by habit. The concatenation pool sends
    # SLOT_COUNT * slot_hidden into the trunk, so slot_hidden drives most of the parameter count.
    # At 128 and 256 the model held 626k parameters against roughly 37k training decisions, which
    # is the memorization regime training-design.md warns about; 96 and 192 give 393k.
    def __init__(self, slot_hidden: int = 96, trunk_hidden: int = 192, global_hidden: int = 32) -> None:
        super().__init__()
        # Shared across slots, which is what enforces that a stack's meaning comes from its
        # fields rather than from the slot it happens to occupy.
        self.slot_encoder = nn.Sequential(
            nn.Linear(SLOT_FEATURES, slot_hidden),
            nn.ReLU(),
            nn.Linear(slot_hidden, slot_hidden),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(nn.Linear(GLOBAL_FEATURES, global_hidden), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(SLOT_COUNT * slot_hidden + global_hidden, trunk_hidden),
            nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(trunk_hidden, ACTION_SPACE_SIZE)
        self.value_head = nn.Linear(trunk_hidden, 1)

    def features(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        slots = observations[:, : SLOT_COUNT * SLOT_FEATURES].view(batch, SLOT_COUNT, SLOT_FEATURES)
        globals_ = observations[:, SLOT_COUNT * SLOT_FEATURES :]

        encoded = self.slot_encoder(slots)
        # An absent slot is all zeros on input but not on output, since the encoder has biases.
        # Zeroing it here keeps padding from contributing, which matters because battles end with
        # far fewer stacks than they start with.
        present = slots[:, :, :1]
        encoded = encoded * present

        joined = torch.cat([encoded.flatten(1), self.global_encoder(globals_)], dim=1)
        return self.trunk(joined)

    def forward(self, observations: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns masked logits and the value estimate."""
        hidden = self.features(observations)
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
