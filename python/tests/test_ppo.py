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

print(f"{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
