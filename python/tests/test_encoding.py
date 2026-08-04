"""Unit tests for the observation encoding and dataset loader."""
import json
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fheroes2_agent import dataset, encoding  # noqa: E402

passed = failed = 0


def check(condition, name):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}")
        failed += 1


def stack(uid, side, cell, count=50, active=False, **kw):
    unit = dict(uid=uid, monster_id=1, side=side, active=active, count=count, initial_count=count,
                hit_points=count, top_hit_points=1, attack=1, defense=1, speed=3, shots=0,
                morale=0, luck=0, head_cell=cell, tail_cell=-1, wide=False, flying=False,
                archer=False, hand_fighting=False)
    unit.update(kw)
    return unit


def obs(units, active_is_attacker=True, rnd=1):
    return dict(schema="observation_full_v1", engine_decision_index=1, round=rnd,
                active_uid=units[0]["uid"], active_is_attacker=active_is_attacker, units=units)


# --- shape and padding
v = encoding.encode_observation(obs([stack(1, "attacker", 0, active=True)]))
check(v.shape == (encoding.OBSERVATION_SIZE,), "encoded observation has the declared width")
check(v.dtype == np.float32, "encoding is float32")
check(v[encoding.SLOT_FEATURES] == 0.0, "an unused slot is flagged absent")
check(np.all(v[encoding.SLOT_FEATURES:encoding.SLOT_FEATURES * 2] == 0.0), "an unused slot is entirely zero")

# --- board geometry: cell 34 is row 3, column 1
v = encoding.encode_observation(obs([stack(1, "attacker", 34, active=True)]))
row_i, col_i = encoding.FEATURE_NAMES.index("row"), encoding.FEATURE_NAMES.index("column")
check(abs(v[row_i] - 3 / 8) < 1e-6, "cell 34 encodes as row 3")
check(abs(v[col_i] - 1 / 10) < 1e-6, "cell 34 encodes as column 1")

# --- side symmetry: own/enemy is relative to whoever is on turn
own_i = encoding.FEATURE_NAMES.index("is_own_side")
a = encoding.encode_observation(obs([stack(1, "attacker", 0, active=True)], active_is_attacker=True))
d = encoding.encode_observation(obs([stack(1, "attacker", 0, active=True)], active_is_attacker=False))
check(a[own_i] == 1.0 and d[own_i] == 0.0, "own-side flag follows whose turn it is")

# --- losses are visible
frac_i = encoding.FEATURE_NAMES.index("count_fraction")
v = encoding.encode_observation(obs([stack(1, "attacker", 0, count=25, initial_count=50, active=True)]))
check(abs(v[frac_i] - 0.5) < 1e-6, "half a stack lost encodes as count_fraction 0.5")

# --- distinct states must not collide
s1 = encoding.encode_observation(obs([stack(1, "attacker", 0, active=True)]))
s2 = encoding.encode_observation(obs([stack(1, "attacker", 1, active=True)]))
check(not np.array_equal(s1, s2), "a moved stack changes the encoding")

# --- overflow is an error, not silent truncation
try:
    encoding.encode_observation(obs([stack(i, "attacker", i, active=(i == 0)) for i in range(11)]))
    check(False, "more stacks than slots raises")
except ValueError:
    check(True, "more stacks than slots raises")

# --- masks
m = encoding.encode_mask([0, 3, 411])
check(m.shape == (793,) and m.dtype == bool, "mask is 793 booleans")
check(m.sum() == 3 and m[0] and m[3] and m[411], "mask marks exactly the legal indices")
try:
    encoding.encode_mask([793])
    check(False, "an out-of-range action index raises")
except ValueError:
    check(True, "an out-of-range action index raises")

# --- dataset loading, including the illegal-label guard
with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp)
    def write(name, action, legal):
        rec = dict(record="decision", engine_decision_index=1, unit_uid=1, actions=[],
                   observation=obs([stack(1, "attacker", 0, active=True)]),
                   legal_actions=legal, teacher_resolved=True, teacher_matched=True,
                   teacher_action=action)
        (root / name).write_text(json.dumps(rec) + "\n")

    write("ep0.jsonl", 3, [0, 3, 5])
    write("ep1.jsonl", 5, [0, 3, 5])
    s = dataset.load_dir(root)
    check(len(s) == 2, "one sample per decision record")
    check(s.observations.shape == (2, encoding.OBSERVATION_SIZE), "observations stack correctly")
    check(list(np.unique(s.episode_ids)) == [0, 1], "episode ids are assigned per file")

    tr, ho = dataset.split_by_episode(s, holdout_fraction=0.5, seed=0)
    check(len(tr) + len(ho) == len(s), "split covers every sample")
    check(not (set(tr.episode_ids.tolist()) & set(ho.episode_ids.tolist())), "no episode spans the split")

    write("ep2.jsonl", 7, [0, 3, 5])  # 7 is not legal
    try:
        dataset.load_dir(root)
        check(False, "a teacher action outside its mask is rejected")
    except ValueError:
        check(True, "a teacher action outside its mask is rejected")


# --- discounted returns, the targets a pre-fitted critic regresses on
def decision(is_attacker, hp_attacker=100, hp_defender=100):
    units = [stack(1, "attacker", 0, count=hp_attacker, active=is_attacker),
             stack(2, "defender", 50, count=hp_defender, active=not is_attacker)]
    return dict(record="decision", teacher_resolved=True, teacher_action=0, legal_actions=[0],
                observation=obs(units, active_is_attacker=is_attacker))


def terminal(termination, hp_attacker, hp_defender):
    return dict(record="terminal", termination=termination,
                attacker=dict(hit_points=hp_attacker), defender=dict(hit_points=hp_defender))


# Attacker wins with 60 of its 100 hit points left; defender is wiped out.
episode = [decision(True), decision(False), decision(True), terminal("victory", 60, 0)]
r = dataset.episode_returns(episode, gamma=1.0)
check(len(r) == 3, "one return per decision carrying an observation")
check(abs(r[2] - 1.6) < 1e-6, "the winner's last decision earns win plus surviving fraction")
check(abs(r[1] - (-1.0)) < 1e-6, "the loser's decision earns the loser's reward, in the same episode")
check(r[0] > 0 and r[1] < 0, "sign follows whichever side was on turn")

# Discounting is by decisions still to come, so magnitude grows toward the end.
d = dataset.episode_returns(episode, gamma=0.5)
check(abs(d[2] - 1.6) < 1e-6, "the final decision is undiscounted")
check(abs(d[0] - 1.6 * 0.25) < 1e-6, "an earlier decision is discounted by the steps remaining")
check(abs(d[0]) < abs(d[2]), "discounting shrinks returns further from the terminal")

# The survival fraction is read from the first observation, before any damage.
hurt = [decision(True, hp_attacker=100), decision(True, hp_attacker=40), terminal("victory", 40, 0)]
check(abs(dataset.episode_returns(hurt, gamma=1.0)[0] - 1.4) < 1e-6,
      "survival is measured against the starting force, not the current one")

# A rout scores -1 exactly, which is the degenerate case scenario-distribution.md warns about.
routed = [decision(True), terminal("defeat", 0, 90)]
check(abs(dataset.episode_returns(routed, gamma=1.0)[0] + 1.0) < 1e-6, "a rout scores exactly -1")

# An episode recorded without --audit-coverage has no observations and so no returns, and the
# loader must drop them rather than mis-align them against the decisions it did encode.
with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp)
    (root / "full.jsonl").write_text("\n".join(json.dumps(x) for x in episode) + "\n")
    s = dataset.load_dir(root)
    check(s.returns is not None and len(s.returns) == len(s), "returns line up row for row with actions")
    check(np.isfinite(s.returns).all(), "a complete episode yields finite returns")

    bare = [dict(record="decision", teacher_resolved=True, teacher_action=0, legal_actions=[0],
                 observation=obs([stack(1, "attacker", 0, active=True)]))]
    (root / "noterminal.jsonl").write_text("\n".join(json.dumps(x) for x in bare) + "\n")
    s = dataset.load_dir(root)
    check(len(s) == 4, "an episode without a terminal record still yields its samples")
    check(np.isnan(s.returns).sum() == 1, "its returns are dropped rather than guessed")

print(f"{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
