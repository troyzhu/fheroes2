"""Advantage estimators and trust regions, as swappable pieces.

Four methods appear across `agent_play/docs/rl/`, and they differ along two independent axes
rather than being four separate algorithms. Separating the axes is what makes them comparable: a
run that changes both at once cannot attribute its result to either.

Advantage, meaning what the baseline is:

- `gae`   a learned critic with generalized advantage estimation, per rl-methods.md
- `loo`   leave-one-out over a group, exactly unbiased, per rlhf-transfer.md
- `grpo`  the group mean including the sample, studentized, with the O(1/K) bias that carries
- `drgrpo` the group mean without the studentization, which the Dr. GRPO variant drops

Trust region, meaning what bounds the step:

- `ratio`      PPO's clip on the sampled importance ratio
- `divergence` DPPO's mask on a computed divergence, per research/works/dppo-trust-region.md

The second is worth having here for a reason specific to this project. DPPO spends most of its
methodology approximating the divergence, because summing over a 10^5-token vocabulary at every
position is prohibitive. This action space is 793 slots with 5 to 30 legal after masking, so the
exact total-variation distance over the legal set is a handful of operations and the
approximations are unnecessary.
"""

from __future__ import annotations

import numpy as np
import torch

ADVANTAGE_MODES = ("gae", "loo", "grpo", "drgrpo")
TRUST_REGIONS = ("ratio", "divergence")


def group_advantages(returns: np.ndarray, mode: str) -> np.ndarray:
    """Turn one group's returns into advantages.

    All three group modes answer the same question, how good was this episode compared with the
    others from the same start, and differ only in whether the episode counts toward its own
    baseline and whether the result is rescaled.
    """
    k = len(returns)
    if k < 2:
        raise ValueError("a group baseline needs at least two episodes")

    if mode == "loo":
        # Excluding the sample keeps the baseline independent of its own actions, which is what
        # makes this exactly unbiased.
        return returns - (returns.sum() - returns) / (k - 1)

    if mode in ("grpo", "drgrpo"):
        centred = returns - returns.mean()
        if mode == "drgrpo":
            return centred
        # Studentizing rescales each group by its own noisy spread, so a group that happens to be
        # homogeneous has its advantages inflated. Dr. GRPO drops it for exactly that reason;
        # both are kept here so the difference can be measured rather than argued.
        return centred / (returns.std() + 1e-8)

    raise ValueError(f"unknown group advantage mode {mode!r}")


def normalize_advantages(advantages, floor: float = 0.1):
    """Centre and rescale a batch, dividing by a spread never allowed below `floor`.

    Normalization exists so the step size does not depend on the reward scale. The hazard is at
    the other end. Once a matchup is solved every episode scores alike, the spread collapses
    toward zero, and dividing by it rescales what is left, which is value-function error, up to
    unit variance. Measured on a calibrated matchup: a raw spread of 0.02 against a healthy 0.3 to
    1.0, so the amplification reaches fiftyfold, and four epochs of it drove a policy from a 1.000
    win rate to 0.000 in two iterations.

    Flooring beats dropping the batch. Dropping is symmetric in the wrong way, since it fires when
    every episode loses too, which is when a collapsed run most needs a gradient, and it freezes
    the run instead. A floor keeps the sign and the ranking of every advantage and makes a
    degenerate batch produce a small update rather than a huge one or none.

    Accepts a numpy array or a torch tensor, since the two trainers hold advantages in different
    forms at the point they normalize.
    """
    spread = float(advantages.std())
    return (advantages - advantages.mean()) / max(spread, floor)


def total_variation(new_logits: torch.Tensor, old_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Exact total-variation distance between two masked policies, per row.

    D_TV(p, q) = 0.5 * sum_a |p(a) - q(a)|, summed over the legal set alone. Illegal actions carry
    zero probability under both, so they contribute nothing and are excluded rather than relied on
    to cancel.
    """
    # Mask before the softmax rather than trusting the caller to have done it. Softmax over
    # unmasked logits puts mass on illegal actions, which changes the legal probabilities and
    # silently produces a wrong divergence. The policy's forward already masks, so this is
    # normally a no-op, and it is what makes the function correct on its own terms.
    from .policy import MASK_FILL

    p = torch.softmax(old_logits.masked_fill(~mask, MASK_FILL), dim=-1)
    q = torch.softmax(new_logits.masked_fill(~mask, MASK_FILL), dim=-1)
    return 0.5 * ((p - q).abs() * mask).sum(dim=-1)


def surrogate(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    *,
    trust_region: str,
    clip: float = 0.2,
    divergence: torch.Tensor | None = None,
    threshold: float = 0.05,
) -> torch.Tensor:
    """The per-sample objective, before averaging. Higher is better."""
    if trust_region == "ratio":
        return torch.min(ratio * advantages, torch.clamp(ratio, 1 - clip, 1 + clip) * advantages)

    if trust_region == "divergence":
        if divergence is None:
            raise ValueError("the divergence trust region needs a divergence")
        # DPPO equation (12): block an update only when it is already moving away from the
        # trusted region and the distribution has genuinely shifted. Updates pulling the ratio
        # back toward one are never blocked, which is the asymmetry PPO's clip also has.
        moving_away = ((advantages > 0) & (ratio > 1)) | ((advantages < 0) & (ratio < 1))
        blocked = moving_away & (divergence > threshold)
        # A blocked term is detached rather than zeroed, so it contributes its value to the
        # objective but no gradient, which is what "the clip removes the incentive" means.
        live = ratio * advantages
        return torch.where(blocked, live.detach(), live)

    raise ValueError(f"unknown trust region {trust_region!r}")


def clip_fraction(ratio: torch.Tensor, clip: float = 0.2) -> float:
    """Diagnostic: how much of the batch sits outside the clip window."""
    return float(((ratio < 1 - clip) | (ratio > 1 + clip)).float().mean())
