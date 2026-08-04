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


def policy_fingerprint(model: BattlePolicy) -> str:
    """A short, stable hash of a policy's weights.

    Difficulty is policy-relative, so a calibrated pool is only valid for the checkpoint that
    measured it. Recording a fingerprint alongside the pool is what lets a later run detect that
    it is about to reuse someone else's calibration, which otherwise fails silently as a pool of
    matchups that are merely easy.

    Hashed from the weights rather than from a file path, because the path says nothing about
    whether the file still holds the same tensors.
    """
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]

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
    # Wide, because the band can sit well outside a naive guess: 17 Rangers beat 54 Skeletons
    # every time, so a range topping out at 3x never reached the crossover. Bisection costs a
    # logarithm, so widening it is nearly free.
    low, high = 0.1, 20.0
    best = None
    for _ in range(steps):
        mid = (low + high) / 2
        candidate = Matchup(attacker, scale_army(defender, mid))
        result = measure(model, worker, candidate, episodes, side)
        result["scale"] = mid
        # Strictly better only, so a tie keeps the earlier probe; with a step-function matchup
        # every probe ties at distance 0.5 and the result is reported as out of band below rather
        # than dressed up as a calibration.
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
    # A matchup whose win rate is a step function has no scale inside the band, and saying so is
    # more useful than returning the least-bad probe as though it were calibrated.
    confirmed["calibrated"] = bool(confirmed["in_band"])
    confirmed["search_range"] = (low, high)
    return confirmed


# --------------------------------------------------------------------------------------------
# The generator. Sampling and filtering wastes most of its budget, because a battle outcome is
# close to a deterministic function of the army pair, so this calibrates instead: pick a
# composition, then search the defender scale that makes it contested.


def sample_composition(rng: random.Random, max_stacks: int = 3, strength: float | None = None) -> str:
    """One side's composition, without regard to whether it is balanced against anything."""
    stacks = rng.randint(1, max_stacks)
    if strength is None:
        strength = rng.choice([20, 40, 60, 100, 150])
    share = strength / stacks
    parts = []
    for _ in range(stacks):
        monster, hp, _ = rng.choice(ROSTER)
        parts.append(f"{monster}:{max(1, min(500, int(round(share / hp))))}")
    return ",".join(parts)


def build_pool(
    model: BattlePolicy,
    worker: str,
    target_size: int,
    seed: int = 0,
    target: float = 0.5,
    episodes: int = 10,
    max_attempts: int | None = None,
    side: str = "attacker",
    progress: bool = False,
) -> dict:
    """Calibrate compositions until `target_size` of them sit inside the band.

    The pool records which policy calibrated it. Difficulty is policy-relative, so a pool is only
    valid for the checkpoint it was measured against, and a pool used with a stronger policy is
    quietly a pool of easy matchups. That is the moving-band problem
    `agent_play/docs/rl/scenario-distribution.md` describes, made explicit rather than left to be
    rediscovered.
    """
    rng = random.Random(seed)
    attempts = max_attempts if max_attempts is not None else target_size * 6
    pool, tried, calibrated = [], 0, 0

    while len(pool) < target_size and tried < attempts:
        tried += 1
        attacker = sample_composition(rng)
        defender = sample_composition(rng)
        try:
            result = calibrate(model, worker, attacker, defender, target=target, episodes=episodes, side=side)
        except Exception:
            # A composition the scenario schema rejects is not a calibration failure; skip it.
            continue
        if not result["calibrated"]:
            continue
        calibrated += 1
        pool.append({
            "attacker": result["matchup"].attacker,
            "defender": result["matchup"].defender,
            "win_rate": result["win_rate"],
            "reward_std": result["reward_std"],
            "mean_length": result["mean_length"],
            "scale": result["scale"],
        })
        if progress:
            print(f"  {len(pool):3d}/{target_size}  win {result['win_rate']:.2f}  std {result['reward_std']:.2f}  "
                  f"{result['mean_length']:.0f} dec  (tried {tried})")

    return {
        "matchups": pool,
        "attempts": tried,
        "hit_rate": len(pool) / tried if tried else 0.0,
        "target": target,
        "episodes_per_probe": episodes,
        "side": side,
        "seed": seed,
        "policy_fingerprint": policy_fingerprint(model),
    }


def pool_matchups(pool: dict) -> list[Matchup]:
    return [Matchup(m["attacker"], m["defender"]) for m in pool["matchups"]]
