"""Unit tests for the RL pieces: reward shape, GAE, truncation, and the env contract."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fheroes2_agent import train_ppo  # noqa: E402
from fheroes2_agent.env import terminal_reward  # noqa: E402

passed = failed = 0


def check(condition, name, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
        passed += 1
    else:
        print(f"  FAIL  {name}" + (f"  [{detail}]" if detail else ""))
        failed += 1


def rec(term, hp):
    return {"termination": term, "attacker": {"hit_points": hp}, "defender": {"hit_points": 0}}


# --- the reward must order outcomes the way ADR 0005 intends
clean = terminal_reward(rec("victory", 50), "attacker", 50.0)
pyrrhic = terminal_reward(rec("victory", 5), "attacker", 50.0)
cheap = terminal_reward(rec("defeat", 30), "attacker", 50.0)
rout = terminal_reward(rec("defeat", 0), "attacker", 50.0)
check(clean > pyrrhic > 0 > cheap > rout, "outcomes order clean win > pyrrhic > cheap loss > rout",
      f"{clean:.2f} {pyrrhic:.2f} {cheap:.2f} {rout:.2f}")
check(abs(clean - 2.0) < 1e-9 and abs(rout + 1.0) < 1e-9, "the reward is bounded by +2 and -1")
# The bug this replaced: relative margin is 1.0 whenever the loser is wiped out, so a pyrrhic win
# scored the same as a clean one and the term carried no information.
check(pyrrhic < clean, "a pyrrhic win scores strictly below a clean one")
# Losing cheaply must beat being routed, which is what keeps a hopeless matchup informative.
check(cheap > rout, "losing cheaply beats being routed")
check(terminal_reward({"termination": "defeat", "defender": {"hit_points": 40}}, "defender", 40.0) == 2.0,
      "the defender's win is scored from the defender's point of view")

# --- GAE: a terminated episode bootstraps nothing, a truncated one does
rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
values = np.array([0.5, 0.5, 0.5], dtype=np.float32)
dones = np.array([False, False, True])

adv_term, ret_term = train_ppo.compute_gae(rewards, values, dones, np.array([False, False, False]))
adv_trunc, ret_trunc = train_ppo.compute_gae(rewards, values, dones, np.array([False, False, True]))
check(adv_term[-1] < adv_trunc[-1], "a truncated final step has a higher advantage than a terminated one",
      f"{adv_term[-1]:.3f} vs {adv_trunc[-1]:.3f}")
# This is the direction that matters: treating truncation as termination throws away the future
# and biases every value estimate downward.
check(abs(adv_term[-1] - (1.0 - 0.5)) < 1e-5, "a terminated step's advantage is reward minus value")
check(np.allclose(ret_term, adv_term + values), "returns are advantages plus values")

# Reward earned only at the end must still credit earlier steps through the trace.
check(adv_term[0] > 0, "a terminal reward propagates back to the first step", f"{adv_term[0]:.3f}")
# Episode boundaries must not leak: a second episode's reward cannot reach the first.
two = train_ppo.compute_gae(
    np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    np.zeros(4, dtype=np.float32),
    np.array([False, True, False, True]),
    np.zeros(4, dtype=bool),
)[0]
check(abs(two[1] - 1.0) < 1e-5 and abs(two[3] - 1.0) < 1e-5, "each episode's terminal reward stays in its own episode")

# --- win rate is computed from the controlled side's point of view
outcomes = [{"termination": "victory"}, {"termination": "defeat"}]
check(train_ppo.win_rate(outcomes, "attacker") == 0.5, "attacker win rate counts victories")
check(train_ppo.win_rate(outcomes, "defender") == 0.5, "defender win rate counts the attacker's defeats")
check(train_ppo.win_rate([], "attacker") == 0.0, "an empty batch has a defined win rate")

# --- army specs: names, ids, and the calibration helper
from fheroes2_agent.render import parse_army, describe_army  # noqa: E402
from fheroes2_agent.scenarios import scale_army  # noqa: E402

check(parse_army("pikeman:20,archer:10") == "4:20,2:10", "creature names resolve to ids")
check(parse_army("4:20,2:10") == "4:20,2:10", "raw ids pass through unchanged")
check(parse_army("PIKEMAN:5") == "4:5", "name matching ignores case")
check(describe_army("4:20,2:10") == "20 Pikeman, 10 Archer", "ids render back into names")
try:
    parse_army("dragon:5")
    check(False, "an unknown creature name raises")
except ValueError:
    check(True, "an unknown creature name raises")

check(scale_army("4:20,2:10", 0.5) == "4:10,2:5", "scaling an army multiplies every stack")
check(scale_army("4:1,2:1", 0.1) == "4:1,2:1", "scaling never empties a stack")

# --- advantage normalization, and the floor that stops a solved matchup destroying its policy
normalize = train_ppo.normalize_advantages

healthy = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
out = normalize(healthy, floor=0.1)
check(abs(out.mean()) < 1e-6, "normalized advantages are centred")
check(abs(out.std() - 1.0) < 1e-5, "a healthy batch is rescaled to unit spread, as PPO expects")

# A batch whose spread is far below the floor must not be blown up to unit variance. This is the
# property that matters: it is the amplification, not the small numbers, that destroys a policy.
tiny = np.array([-0.02, -0.01, 0.0, 0.01, 0.02], dtype=np.float32)
out = normalize(tiny, floor=0.1)
check(out.std() < 0.2, "a degenerate batch stays small instead of being rescaled to unit spread",
      f"std {out.std():.3f}")
check(float(np.max(np.abs(out))) < float(np.max(np.abs(normalize(tiny, floor=1e-8)))),
      "the floor strictly shrinks what an unfloored divisor would produce")

# The ordering has to survive, or the floor would be discarding the signal it is protecting.
check(np.all(np.diff(normalize(tiny, floor=0.1)) > 0), "flooring preserves the ranking of advantages")
check(np.all(np.sign(normalize(tiny, floor=0.1)) == np.sign(tiny - tiny.mean())),
      "flooring preserves the sign of every advantage")

# An all-equal batch has nothing to say, and must say nothing rather than dividing by zero.
flat = np.zeros(8, dtype=np.float32)
out = normalize(flat, floor=0.1)
check(np.all(np.isfinite(out)) and float(np.abs(out).max()) == 0.0,
      "an all-equal batch yields exactly zero advantages, with no division by zero")

# Difficulty weighting: wins scale by the tempered strength ratio, losses by its inverse, so a
# hard fight pays more for winning and forgives losing, and an easy fight does the opposite.
from fheroes2_agent.env import apply_difficulty, difficulty_weight  # noqa: E402

def _obs(own_count, enemy_count, monster=1):
    return {"units": [
        {"side": "attacker", "monster_id": monster, "count": own_count},
        {"side": "defender", "monster_id": monster, "count": enemy_count},
    ]}

check(abs(difficulty_weight(_obs(10, 10), "attacker") - 1.0) < 1e-9, "equal armies weigh 1")
check(abs(difficulty_weight(_obs(10, 40), "attacker") - 2.0) < 1e-9,
      "a 4x stronger opponent weighs 2 at the default exponent of one half")
check(abs(difficulty_weight(_obs(40, 10), "attacker") - 0.5) < 1e-9, "a 4x weaker opponent weighs one half")
check(abs(difficulty_weight(_obs(1, 1000), "attacker") - 2.0) < 1e-9,
      "the ratio is capped, so a horde cannot dominate a batch")
check(abs(difficulty_weight(_obs(10, 40), "defender") - 0.5) < 1e-9, "the ratio is taken from the given side")
check(abs(apply_difficulty(1.5, 2.0) - 3.0) < 1e-9, "a hard win is amplified")
check(abs(apply_difficulty(-1.0, 2.0) - (-0.5)) < 1e-9, "a hard loss is forgiven")
check(abs(apply_difficulty(1.5, 0.5) - 0.75) < 1e-9, "an easy win is damped")
check(abs(apply_difficulty(-1.0, 0.5) - (-2.0)) < 1e-9, "an easy loss is punished harder")

print(f"{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
