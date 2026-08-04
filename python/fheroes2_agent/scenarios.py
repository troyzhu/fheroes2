"""Sample battle matchups, and measure how hard each one is.

`decisions/0005-training-and-reward.md` calls the initial-state distribution the largest
undocumented modelling choice in the project, and gives it one acceptance criterion: a scenario
carries gradient only when the policy neither always wins nor always loses it. This module is the
generator plus the measurement that criterion needs.

Difficulty is measured, never asserted from army sizes. Measuring it turned up a constraint the
design had not anticipated: in a mirror matchup the win rate is a step function of the count,
because damage rolls average out across many creatures and arithmetic decides the battle. A band
therefore needs either small stacks, where one exchange is a large fraction of the army, or mixed
creature types, where positioning decides. The sampler below leans on both.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from .env import BattleEnv
from .policy import BattlePolicy

# Monster ids from the simple_v1 allowlist: single-cell, walking or shooting, ordinary targeting.
# Values are (id, hit points, is shooter), used only to keep sampled armies roughly comparable.
ROSTER = [
    (1, 1, False),   # Peasant
    (2, 10, True),   # Archer
    (3, 10, True),   # Ranger
    (4, 15, False),  # Pikeman
    (5, 20, False),  # Veteran Pikeman
    (6, 25, False),  # Swordsman
    (7, 30, False),  # Master Swordsman
]


@dataclass(frozen=True)
class Matchup:
    attacker: str
    defender: str

    def label(self) -> str:
        return f"{self.attacker} vs {self.defender}"


def _side(rng: random.Random, total_hit_points: float, max_stacks: int) -> str:
    """One side worth roughly `total_hit_points`, spread over up to `max_stacks` stacks."""
    stacks = rng.randint(1, max_stacks)
    share = total_hit_points / stacks
    parts = []
    for _ in range(stacks):
        monster, hp, _ = rng.choice(ROSTER)
        count = max(1, min(500, int(round(share / hp))))
        parts.append(f"{monster}:{count}")
    return ",".join(parts)


def sample_matchups(n: int, seed: int = 0, max_stacks: int = 3) -> list[Matchup]:
    """Sample matchups likely to carry gradient.

    Two constraints, both learned by measuring rather than assumed. Total strength is kept small,
    because a large mirror resolves deterministically and its win rate is a step function of the
    count; small stacks leave one exchange worth a large fraction of the army. And the two sides
    are held close in strength, because a matchup decided by arithmetic teaches nothing whatever
    the reward shape.

    Naive sampling over the whole roster put 8 percent of matchups inside the band. Constraining
    both raises that substantially, which matters because the survey that filters them costs a
    dozen episodes each.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        # Small enough that damage variance still moves the outcome.
        strength = rng.choice([15, 20, 25, 30, 40])
        # Within 15 percent, so neither side is arithmetically decided.
        ratio = rng.uniform(0.85, 1.15)
        out.append(Matchup(_side(rng, strength, max_stacks), _side(rng, strength * ratio, max_stacks)))
    return out


@torch.no_grad()
def measure(model: BattlePolicy, worker: str, matchup: Matchup, episodes: int = 16, side: str = "attacker") -> dict:
    """Win rate and mean reward for one matchup, which is what makes difficulty a number."""
    env = BattleEnv(worker, side=side, attacker=matchup.attacker, defender=matchup.defender)
    wins, rewards, lengths = [], [], []
    try:
        for _ in range(episodes):
            observation, mask = env.reset()
            steps = 0
            while True:
                logits, _ = model(torch.from_numpy(observation).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0))
                action = int(torch.distributions.Categorical(logits=logits).sample())
                step = env.step(action)
                steps += 1
                if step.done:
                    target = "victory" if side == "attacker" else "defeat"
                    wins.append(step.info["termination"] == target)
                    rewards.append(step.reward)
                    lengths.append(steps)
                    break
                observation, mask = step.observation, step.mask
    finally:
        env.close()

    win_rate = float(np.mean(wins))
    return {
        "matchup": matchup,
        "win_rate": win_rate,
        "mean_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "in_band": 0.2 <= win_rate <= 0.8,
    }


def survey(model: BattlePolicy, worker: str, matchups: list[Matchup], episodes: int = 16) -> list[dict]:
    return [measure(model, worker, m, episodes) for m in matchups]


def band_report(results: list[dict]) -> str:
    rates = np.array([r["win_rate"] for r in results])
    in_band = sum(r["in_band"] for r in results)
    return (
        f"{len(results)} matchups: {in_band} in band ({in_band / len(results):.0%}), "
        f"{int((rates > 0.8).sum())} too easy, {int((rates < 0.2).sum())} too hard. "
        f"Win rate mean {rates.mean():.3f}, median {np.median(rates):.3f}"
    )


def scale_army(spec: str, factor: float) -> str:
    """Multiply every stack's count, keeping at least one creature per stack."""
    parts = []
    for item in spec.split(","):
        monster_id, _, count = item.partition(":")
        parts.append(f"{monster_id}:{max(1, int(round(int(count) * factor)))}")
    return ",".join(parts)


def calibrate(model: BattlePolicy, worker: str, attacker: str, defender: str, target: float = 0.5,
              episodes: int = 12, steps: int = 7, side: str = "attacker") -> dict:
    """Find the defender scale that puts a matchup at the target win rate.

    This is the mechanism a usable generator needs. Rejection sampling puts about one matchup in
    ten inside the band, because a battle outcome is close to a deterministic function of the
    army pair, so filtering a stream of samples wastes most of the budget. Calibrating instead
    searches the scale that makes a given pair contested, and the search is short because the
    win rate is close to monotone in the defender's strength.

    Returns the best matchup found and the scale that produced it.
    """
    low, high = 0.25, 3.0
    best = None
    for _ in range(steps):
        mid = (low + high) / 2
        candidate = Matchup(attacker, scale_army(defender, mid))
        result = measure(model, worker, candidate, episodes, side)
        result["scale"] = mid
        if best is None or abs(result["win_rate"] - target) < abs(best["win_rate"] - target):
            best = result
        if result["win_rate"] > target:
            # Winning too often, so the defender needs to be stronger.
            low = mid
        else:
            high = mid

    # Re-measure the chosen point. The search keeps the probe closest to the target, which is a
    # maximum over several noisy estimates and therefore optimistically biased toward the target:
    # a probe that landed near 0.5 by luck is exactly the one selected. Measuring again on fresh
    # episodes gives the number to quote. Observed once at a factor of two, where a calibration
    # reporting 0.42 measured 0.19 when re-run.
    confirmed = measure(model, worker, best["matchup"], episodes * 2, side)
    confirmed["scale"] = best["scale"]
    confirmed["search_win_rate"] = best["win_rate"]
    return confirmed
