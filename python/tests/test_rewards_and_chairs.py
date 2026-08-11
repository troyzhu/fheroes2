"""Unit tests for the reward, chair and self-play surfaces changed on 2026-08-07 and 08-08.

The code review of 2026-08-09 found these untested and highest-risk: the two-sided reward and its
stalemate resolution decide what every reinforcement run optimizes, and the chair-aware win rate
is the convergence report's primary series. All are pure functions or need only a fake env.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fheroes2_agent.env import (reward_from_record, terminal_reward_balanced,
                                terminal_reward_two_sided, _side_won)
from fheroes2_agent.train_ppo import win_rate

CHECKS = []


def check(name, condition):
    CHECKS.append((name, bool(condition)))


def record(termination, own_strength, own_initial, foe_strength, foe_initial):
    return {"termination": termination,
            "attacker": {"strength": own_strength, "initial_strength": own_initial},
            "defender": {"strength": foe_strength, "initial_strength": foe_initial}}


def commanded_record(termination, own, own_initial, own_c, own_initial_c, foe, foe_initial, foe_c, foe_initial_c):
    return {"termination": termination,
            "attacker": {"strength": own, "initial_strength": own_initial,
                         "strength_commanded": own_c, "initial_strength_commanded": own_initial_c},
            "defender": {"strength": foe, "initial_strength": foe_initial,
                         "strength_commanded": foe_c, "initial_strength_commanded": foe_initial_c}}


# The win branch prices own strength kept and never reads the foe.
clean = record("victory", 100.0, 100.0, 0.0, 50.0)
pyrrhic = record("victory", 10.0, 100.0, 0.0, 50.0)
check("clean win is 2.0", abs(terminal_reward_two_sided(clean, "attacker") - 2.0) < 1e-9)
check("pyrrhic win is 1.1", abs(terminal_reward_two_sided(pyrrhic, "attacker") - 1.1) < 1e-9)
check("win branch ignores the foe",
      terminal_reward_two_sided(record("victory", 100.0, 100.0, 40.0, 50.0), "attacker")
      == terminal_reward_two_sided(clean, "attacker"))

# The loss branch prices enemy strength destroyed, so a rout is -1 and a near-win approaches 0.
rout = record("defeat", 0.0, 100.0, 50.0, 50.0)
near = record("defeat", 0.0, 100.0, 2.0, 50.0)
check("rout is -1.0", abs(terminal_reward_two_sided(rout, "attacker") + 1.0) < 1e-9)
check("near-win loss approaches zero", -0.05 < terminal_reward_two_sided(near, "attacker") < 0.0)
check("every loss is at or below every win",
      terminal_reward_two_sided(near, "attacker") < terminal_reward_two_sided(pyrrhic, "attacker"))

# Stalemate: the defender takes the graded win branch, the forfeiting attacker reads -1.0 flat.
stale = record("stalemate", 80.0, 100.0, 90.0, 100.0)
check("stalemate is a defender win", _side_won(stale, "defender") and not _side_won(stale, "attacker"))
check("stalled attacker gets no survival credit",
      abs(terminal_reward_two_sided(stale, "attacker") + 1.0) < 1e-9)
check("stalling defender is paid for what it kept",
      abs(terminal_reward_two_sided(stale, "defender") - (1.0 + 0.9)) < 1e-9)

# Zero initial strength must not divide by zero.
check("zero initial strength is finite",
      abs(terminal_reward_two_sided(record("victory", 0.0, 0.0, 0.0, 0.0), "attacker") - 1.0) < 1e-9)

# The chair-aware win rate: without per-episode chairs it uses the run's side, with them it
# scores each episode from the seat it was played in.
outcomes = [{"termination": "victory"}, {"termination": "defeat"}]
check("attacker-only run scores victories", win_rate(outcomes, "attacker") == 0.5)
check("defender-only run scores defeats", win_rate(outcomes, "defender") == 0.5)
check("mixed chairs score per episode",
      win_rate(outcomes, "attacker", ["attacker", "defender"]) == 1.0)
check("mixed chairs score losses per episode",
      win_rate(outcomes, "attacker", ["defender", "attacker"]) == 0.0)
check("a chair list of the wrong length falls back to the run side",
      win_rate(outcomes, "attacker", ["attacker"]) == 0.5)
check("no outcomes is zero, not an error", win_rate([], "attacker") == 0.0)

# The commander-aware pricing, added 2026-08-09. Identical armies price identically at base
# stats and differently once the commander is counted, which is the whole point: the base
# valuation called an army led by a hero equal to one that is not.
led = commanded_record("victory", 200.0, 400.0, 360.0, 720.0, 0.0, 400.0, 0.0, 400.0)
check("base pricing keeps half", abs(terminal_reward_two_sided(led, "attacker") - 1.5) < 1e-9)
check("commanded pricing keeps the same fraction of a larger base",
      abs(terminal_reward_two_sided(led, "attacker", commanded=True) - 1.5) < 1e-9)
lopsided = commanded_record("defeat", 0.0, 400.0, 0.0, 720.0, 200.0, 400.0, 360.0, 720.0)
check("commanded loss prices the enemy's commanded remainder",
      abs(terminal_reward_two_sided(lopsided, "attacker", commanded=True) + 0.5) < 1e-9)
check("a record without commanded fields falls back to the base pricing",
      terminal_reward_two_sided(record("victory", 50.0, 100.0, 0.0, 100.0), "attacker", commanded=True)
      == terminal_reward_two_sided(record("victory", 50.0, 100.0, 0.0, 100.0), "attacker"))

# The balanced margin, added 2026-08-09. Its whole claim is that it equals the two-sided form with
# the flat win bonus removed, so the checks pin that identity, the zero-sum property the identity
# buys, and the stalemate case where the two deliberately part company.
check("balanced clean win is the fraction kept",
      abs(terminal_reward_balanced(clean, "attacker") - 1.0) < 1e-9)
check("balanced pyrrhic win is 0.1", abs(terminal_reward_balanced(pyrrhic, "attacker") - 0.1) < 1e-9)
for name, r in (("clean", clean), ("pyrrhic", pyrrhic),
                ("rout", record("defeat", 0.0, 100.0, 45.0, 50.0)),
                ("near-win loss", record("defeat", 0.0, 100.0, 2.0, 50.0))):
    for side in ("attacker", "defender"):
        two = terminal_reward_two_sided(r, side)
        check(f"balanced equals two-sided minus the win bonus, {name} as {side}",
              abs(terminal_reward_balanced(r, side) - (two - (1.0 if _side_won(r, side) else 0.0))) < 1e-9)
    check(f"balanced is zero sum on a decided battle, {name}",
          abs(terminal_reward_balanced(r, "attacker") + terminal_reward_balanced(r, "defender")) < 1e-9)
check("balanced still ranks every win above every loss on decided battles",
      min(terminal_reward_balanced(r, "attacker") for r in (clean, pyrrhic))
      > max(terminal_reward_balanced(record("defeat", 0.0, 100.0, f, 50.0), "attacker") for f in (1.0, 45.0)))

# The two unfinished terminations are where a bare margin would read the outcome off material and
# get it wrong, so these pin that the branch is decided by `_side_won` first. Both fixtures put the
# attacker well ahead on strength, which is precisely when the exploit would pay.
stalled_ahead = record("stalemate", 70.0, 100.0, 6.0, 50.0)
check("balanced stalemate still costs the attacker a flat -1",
      abs(terminal_reward_balanced(stalled_ahead, "attacker") + 1.0) < 1e-9)
check("balanced stalemate pays the outlasting defender its own strength kept",
      abs(terminal_reward_balanced(stalled_ahead, "defender") - 0.12) < 1e-9)
truncated = record("round_limit", 70.0, 100.0, 6.0, 50.0)
for side in ("attacker", "defender"):
    check(f"balanced scores a round-limit truncation a loss for the {side}, as _side_won does",
          terminal_reward_balanced(truncated, side) < 0.0)
check("balanced never lets the material leader profit from a truncation",
      terminal_reward_balanced(truncated, "attacker") < 0.0
      and terminal_reward_two_sided(truncated, "attacker") < 0.0)
# The identity has to survive the unfinished terminations too, not only the decided ones.
for name, r in (("stalemate", stalled_ahead), ("round limit", truncated)):
    for side in ("attacker", "defender"):
        two = terminal_reward_two_sided(r, side)
        check(f"balanced equals two-sided minus the win bonus, {name} as {side}",
              abs(terminal_reward_balanced(r, side) - (two - (1.0 if _side_won(r, side) else 0.0))) < 1e-9)
check("balanced commanded pricing reads the commanded fields",
      abs(terminal_reward_balanced(led, "attacker", commanded=True) - 0.5) < 1e-9)

# The dispatch is the guard against a new margin silently training the old objective.
for margin, fn in (("balanced", terminal_reward_balanced), ("two_sided", terminal_reward_two_sided)):
    check(f"reward_from_record routes {margin} to its own function",
          reward_from_record(clean, "attacker", margin) == fn(clean, "attacker"))
check("reward_from_record routes the commanded balanced variant",
      reward_from_record(led, "attacker", "balanced_commanded")
      == terminal_reward_balanced(led, "attacker", commanded=True))
try:
    reward_from_record(clean, "attacker", "hit_points")
    check("reward_from_record refuses a margin it cannot compute", False)
except ValueError:
    check("reward_from_record refuses a margin it cannot compute", True)

failed = [name for name, ok in CHECKS if not ok]
for name, ok in CHECKS:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
sys.exit(1 if failed else 0)
