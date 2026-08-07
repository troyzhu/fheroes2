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
    """A short, stable hash of the weights that decide actions.

    Difficulty is policy-relative, so a calibrated pool is only valid for the checkpoint that
    measured it. Recording a fingerprint alongside the pool is what lets a later run detect that
    it is about to reuse someone else's calibration, which otherwise fails silently as a pool of
    matchups that are merely easy.

    The value head is excluded, and that is the whole point of hashing by name rather than hashing
    the file. `measure` samples from the logits, so the value head never touches which action is
    taken, and a checkpoint whose critic has been fitted plays exactly the same battles as the one
    it was fitted from. Hashing everything would reject that checkpoint for a difference that
    cannot affect difficulty.

    Hashed from the weights rather than from a file path, because the path says nothing about
    whether the file still holds the same tensors.
    """
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("value_head"):
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]

def _load_roster() -> list[tuple[int, int, bool]]:
    """The sampling roster, derived from the engine's own capability audit.

    Every `simple_v1`-supported creature, as (id, hit points, is shooter). This was a hand-listed
    seven-creature table once, the whole Knight line and nothing else, which quietly narrowed
    every synthetic pool to one faction's basic troops. Hand-maintained creature tables are the
    defect class the audit exists to prevent, and the name table made the same mistake before it.
    """
    import json
    import pathlib

    path = pathlib.Path(__file__).parent / "data" / "monster_capabilities_v1.json"
    records = json.loads(path.read_text())
    roster = [(r["monster_id"], int(r["hit_points"]), bool(r["is_archer"]))
              for r in records if r["simple_v1_supported"]]
    roster.sort()
    if len(roster) < 30:
        raise RuntimeError(f"capability audit lists only {len(roster)} simple_v1 creatures; regenerate it")
    return roster


ROSTER = _load_roster()


@dataclass(frozen=True)
class Matchup:
    attacker: str
    defender: str
    # Optional hero commanders, "attack:defense". None preserves the commander-less behaviour
    # every earlier measurement used.
    attacker_hero: str | None = None
    defender_hero: str | None = None
    # Admits wide (two-cell) walkers, the wide_v1 profile.
    allow_wide: bool = False

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


def load_wide_roster() -> list[tuple[int, int, bool]]:
    """Every wide_v1 creature as (id, hit points, is shooter), from the capability audit."""
    import json
    import pathlib

    path = pathlib.Path(__file__).parent / "data" / "monster_capabilities_v1.json"
    records = json.loads(path.read_text())
    roster = [(r["monster_id"], int(r["hit_points"]), bool(r["is_archer"]))
              for r in records if r["wide_v1_supported"]]
    roster.sort()
    return roster


def _side_from(rng: random.Random, roster, total_hit_points: float, max_stacks: int) -> str:
    stacks = rng.randint(1, max_stacks)
    share = total_hit_points / stacks
    parts = []
    for _ in range(stacks):
        monster, hp, _ = rng.choice(roster)
        parts.append(f"{monster}:{max(1, min(500, int(round(share / hp))))}")
    return ",".join(parts)


def load_valued_roster() -> list[tuple[int, float]]:
    """Every wide_v1 creature as (id, engine strength), from the capability audit.

    Strength is Monster::GetMonsterStrength at base stats, the engine's own scalar worth of one
    creature. It prices what hit points miss: a Ranger and an Archer share ten hit points and
    differ by two thirds in strength, because the double shot is worth something.
    """
    import json
    import pathlib

    path = pathlib.Path(__file__).parent / "data" / "monster_capabilities_v1.json"
    records = json.loads(path.read_text())
    roster = [(r["monster_id"], float(r["strength"])) for r in records if r["wide_v1_supported"]]
    roster.sort()
    return roster


def sample_budget_matchup(rng: random.Random, budget_range: tuple[float, float] = (15.0, 900.0),
                          max_stacks: int = 5, alpha: float = 1.0,
                          close_weight: float = 0.8, sigma_close: float = 0.12,
                          sigma_wide: float = 0.6) -> Matchup:
    """One matchup by army-value budget, the owner-supplied guide's sampling scheme.

    Each side draws a total budget log-uniformly, splits it over its stacks by a Dirichlet draw,
    and prices each stack's count by the creature's engine strength rather than its hit points.
    The enemy budget is the attacker's times a ratio drawn from a mixture concentrated near one,
    so most battles are close and some are lopsided on purpose. Uneven Dirichlet shares are the
    point: the hit-point sampler produced near-equal stacks, and real armies are not.
    """
    import math

    roster = load_valued_roster()

    def side(budget: float) -> str:
        stacks = rng.randint(1, max_stacks)
        weights = [rng.gammavariate(alpha, 1.0) for _ in range(stacks)]
        total = sum(weights)
        parts = []
        for w in weights:
            monster, strength = rng.choice(roster)
            count = max(1, min(500, int(round(budget * (w / total) / max(strength, 0.1)))))
            parts.append(f"{monster}:{count}")
        return ",".join(parts)

    low, high = budget_range
    budget = math.exp(rng.uniform(math.log(low), math.log(high)))
    sigma = sigma_close if rng.random() < close_weight else sigma_wide
    ratio = math.exp(rng.gauss(0.0, sigma))

    heroes = {}
    for key in ("attacker_hero", "defender_hero"):
        if rng.random() < 0.5:
            heroes[key] = f"{rng.randint(0, 25)}:{rng.randint(0, 20)}"
    return Matchup(side(budget), side(budget * ratio), allow_wide=True, **heroes)


def sample_diverse_matchup(rng: random.Random, horde_total_range: tuple[int, int] = (60, 900),
                           horde_only: bool = False) -> Matchup:
    """One matchup over the whole wide_v1 bestiary, with commanders and count regimes.

    Three regimes, each an archetype the narrow sampler lacked: the proven small-stack skirmish,
    a five-stack battle, and elite-against-horde, the Thunk opening fight's shape, with the horde
    split into three near-equal stacks the way the engine splits a neutral stack. Commanders land
    on a coin flip per side with map-hero-like stats, because real maps always have one.
    """
    roster = load_wide_roster()
    cheap = [entry for entry in roster if entry[1] <= 5]
    regime = "horde" if horde_only else rng.choices(("skirmish", "battle", "horde"), weights=(4, 4, 2))[0]
    if regime == "skirmish":
        strength = rng.choice([15, 20, 25, 30, 40])
        attacker = _side_from(rng, roster, strength, 3)
        defender = _side_from(rng, roster, strength * rng.uniform(0.85, 1.15), 3)
    elif regime == "battle":
        strength = rng.choice([60, 90, 120, 150])
        attacker = _side_from(rng, roster, strength, 5)
        defender = _side_from(rng, roster, strength * rng.uniform(0.85, 1.15), 5)
    else:
        attacker = _side_from(rng, roster, rng.choice([80, 120, 160]), 4)
        monster, hp, _ = rng.choice(cheap)
        total = rng.randint(*horde_total_range) // max(hp, 1)
        a = total // 3 + (1 if total % 3 else 0)
        defender = f"{monster}:{a},{monster}:{total // 3},{monster}:{max(1, total - a - total // 3)}"
    heroes = {}
    for side in ("attacker_hero", "defender_hero"):
        if rng.random() < 0.5:
            heroes[side] = f"{rng.randint(0, 25)}:{rng.randint(0, 20)}"
    return Matchup(attacker, defender, allow_wide=True, **heroes)


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
def measure(model: BattlePolicy, worker: str, matchup: Matchup, episodes: int = 16, side: str = "attacker",
            seeds: int = 1) -> dict:
    """Win rate and mean reward for one matchup, which is what makes difficulty a number.

    With seeds above one the episodes rotate over that many battlefield variants, so the number
    is about the matchup rather than about one obstacle layout."""
    # A planes-built policy declares itself on the model; the env then asks the worker for the
    # obstacle layer and the loop feeds the tensor beside the flat vector.
    wants_planes = bool(getattr(model, "planes", False))
    env = BattleEnv(worker, side=side, attacker=matchup.attacker, defender=matchup.defender,
                    attacker_hero=matchup.attacker_hero, defender_hero=matchup.defender_hero,
                    allow_wide=matchup.allow_wide, seeds=seeds, planes=wants_planes)
    wins, rewards, lengths, survival, damage, margins = [], [], [], [], [], []
    try:
        for _ in range(episodes):
            observation, mask = env.reset()
            steps = 0
            while True:
                plane_arg = (torch.from_numpy(env.last_planes).unsqueeze(0),) if wants_planes else ()
                logits, _ = model(torch.from_numpy(observation).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0), *plane_arg)
                action = int(torch.distributions.Categorical(logits=logits).sample())
                step = env.step(action)
                steps += 1
                if step.done:
                    target = "victory" if side == "attacker" else "defeat"
                    won = step.info["termination"] == target
                    wins.append(won)
                    rewards.append(step.reward)
                    lengths.append(steps)
                    own = step.info["attacker" if side == "attacker" else "defender"]
                    foe = step.info["defender" if side == "attacker" else "attacker"]
                    own_initial = float(own.get("initial_strength", 0.0))
                    foe_initial = float(foe.get("initial_strength", 0.0))
                    own_kept = float(own.get("strength", 0.0)) / own_initial if own_initial > 0 else 0.0
                    foe_kept = float(foe.get("strength", 0.0)) / foe_initial if foe_initial > 0 else 0.0
                    # The unconditional margin is selection-free: own strength kept minus the
                    # enemy's, defined for every episode, so it cannot flatter by conditioning.
                    margins.append(own_kept - foe_kept)
                    if won:
                        # Win quality: engine creature strength kept, near 1 for a bloodless win.
                        survival.append(own_kept)
                    else:
                        # Loss quality, the owner's counterpart metric: how much of the enemy a
                        # losing fight destroyed, near 0 for a rout, near 1 for a near-win.
                        damage.append(1.0 - foe_kept)
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
        "surviving_strength": float(np.mean(survival)) if survival else None,
        "loss_damage": float(np.mean(damage)) if damage else None,
        "strength_margin": float(np.mean(margins)),
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
              episodes: int = 12, steps: int = 7, side: str = "attacker",
              attacker_hero: str | None = None, defender_hero: str | None = None,
              allow_wide: bool = False) -> dict:
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
        candidate = Matchup(attacker, scale_army(defender, mid),
                            attacker_hero=attacker_hero, defender_hero=defender_hero, allow_wide=allow_wide)
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
    sampler=None,
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
        if sampler is not None:
            drawn = sampler(rng)
            attacker, defender = drawn.attacker, drawn.defender
            extras = {"attacker_hero": drawn.attacker_hero, "defender_hero": drawn.defender_hero,
                      "allow_wide": drawn.allow_wide}
        else:
            attacker = sample_composition(rng)
            defender = sample_composition(rng)
            extras = {}
        try:
            result = calibrate(model, worker, attacker, defender, target=target, episodes=episodes, side=side,
                               **extras)
        except Exception:
            # A composition the scenario schema rejects is not a calibration failure; skip it.
            continue
        if not result["calibrated"]:
            continue
        calibrated += 1
        entry = {
            "attacker": result["matchup"].attacker,
            "defender": result["matchup"].defender,
            "win_rate": result["win_rate"],
            "reward_std": result["reward_std"],
            "mean_length": result["mean_length"],
            "scale": result["scale"],
        }
        # Heroes and the wide flag are part of the matchup's identity when a sampler supplied
        # them, and a pool that dropped them would rebuild different battles than it measured.
        matchup = result["matchup"]
        if matchup.attacker_hero or matchup.defender_hero or matchup.allow_wide:
            entry["attacker_hero"] = matchup.attacker_hero
            entry["defender_hero"] = matchup.defender_hero
            entry["allow_wide"] = matchup.allow_wide
        pool.append(entry)
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
    return [Matchup(m["attacker"], m["defender"], attacker_hero=m.get("attacker_hero"),
                    defender_hero=m.get("defender_hero"), allow_wide=bool(m.get("allow_wide", False)))
            for m in pool["matchups"]]
