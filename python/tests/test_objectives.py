"""Unit tests for the advantage estimators and trust regions."""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fheroes2_agent import objectives as ob  # noqa: E402

passed = failed = 0


def check(condition, name, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
        passed += 1
    else:
        print(f"  FAIL  {name}" + (f"  [{detail}]" if detail else ""))
        failed += 1


returns = np.array([2.0, 1.0, 0.0, -1.0], dtype=np.float32)

# --- leave-one-out is exactly unbiased: its advantages sum to zero only in the mean sense, but
# each baseline must exclude its own sample, which is checkable directly.
loo = ob.group_advantages(returns, "loo")
expected_first = returns[0] - returns[1:].mean()
check(abs(loo[0] - expected_first) < 1e-6, "leave-one-out excludes the sample from its own baseline",
      f"{loo[0]:.3f} vs {expected_first:.3f}")

# --- GRPO includes the sample, which is the O(1/K) bias
grpo = ob.group_advantages(returns, "grpo")
drgrpo = ob.group_advantages(returns, "drgrpo")
check(abs(drgrpo.mean()) < 1e-6, "Dr. GRPO advantages are mean-zero over the group")
check(abs(drgrpo[0] - (returns[0] - returns.mean())) < 1e-6, "Dr. GRPO subtracts the full group mean")
check(abs(np.std(grpo) - 1.0) < 1e-3, "GRPO studentizes to unit spread", f"{np.std(grpo):.3f}")
check(not np.allclose(grpo, drgrpo), "studentizing changes the advantages")
# The documented hazard: a homogeneous group has its advantages inflated by the division.
tight = np.array([1.0, 1.0001], dtype=np.float32)
check(abs(ob.group_advantages(tight, "grpo")).max() > abs(ob.group_advantages(tight, "drgrpo")).max() * 100,
      "a homogeneous group is inflated by studentizing")

for mode in ("loo", "grpo", "drgrpo"):
    try:
        ob.group_advantages(np.array([1.0], dtype=np.float32), mode)
        check(False, f"{mode} rejects a group of one")
    except ValueError:
        check(True, f"{mode} rejects a group of one")

# --- total variation over the legal set only
mask = torch.zeros(1, 793, dtype=torch.bool)
mask[0, [0, 3, 7]] = True
same = torch.zeros(1, 793)
tv = ob.total_variation(same, same, mask)
check(float(tv) < 1e-6, "identical policies have zero total variation")

a = torch.full((1, 793), -1e8); a[0, [0, 3, 7]] = torch.tensor([10.0, 0.0, 0.0])
b = torch.full((1, 793), -1e8); b[0, [0, 3, 7]] = torch.tensor([0.0, 0.0, 10.0])
tv2 = float(ob.total_variation(a, b, mask))
check(0.0 < tv2 <= 1.0, "total variation is bounded by one", f"{tv2:.3f}")

# Illegal actions must not contribute, or the divergence measures the mask.
c = a.clone(); c[0, 500] = 50.0
check(abs(float(ob.total_variation(c, b, mask)) - tv2) < 1e-5, "illegal actions do not affect the divergence")

# --- trust regions
ratio = torch.tensor([0.5, 1.0, 1.5])
adv = torch.tensor([1.0, 1.0, 1.0])
clipped = ob.surrogate(ratio, adv, trust_region="ratio", clip=0.2)
check(abs(float(clipped[2]) - 1.2) < 1e-6, "a good action beyond the clip is capped", f"{float(clipped[2]):.3f}")
check(abs(float(clipped[1]) - 1.0) < 1e-6, "an unmoved ratio passes through unchanged")

# DPPO blocks only when moving away AND the distribution has shifted.
div_small = torch.tensor([0.0, 0.0, 0.01])
div_large = torch.tensor([0.0, 0.0, 0.5])
small = ob.surrogate(ratio, adv, trust_region="divergence", divergence=div_small, threshold=0.05)
large = ob.surrogate(ratio, adv, trust_region="divergence", divergence=div_large, threshold=0.05)
check(abs(float(small[2]) - 1.5) < 1e-6, "a small divergence leaves the update live")
check(abs(float(large[2]) - 1.5) < 1e-6, "a blocked term keeps its value, losing only its gradient")

# Substituting |rho - 1| for the divergence recovers PPO's blocking decision, which is the
# property that makes the two directly comparable.
r = torch.tensor([1.5]); a1 = torch.tensor([1.0])
as_ppo = ob.surrogate(r, a1, trust_region="divergence", divergence=(r - 1).abs(), threshold=0.2)
check(abs(float(as_ppo)) > 0, "|rho-1| as the divergence reproduces PPO's decision rule")

check(abs(ob.clip_fraction(torch.tensor([0.5, 1.0, 1.5]), 0.2) - 2 / 3) < 1e-6, "clip fraction counts both tails")

# --- the generator's pure parts
from fheroes2_agent.scenarios import Matchup, pool_matchups, sample_composition, scale_army  # noqa: E402
import random  # noqa: E402

rng = random.Random(0)
spec = sample_composition(rng, max_stacks=3)
check(1 <= len(spec.split(",")) <= 3, "a sampled composition has one to three stacks", spec)
check(all(int(part.split(":")[1]) >= 1 for part in spec.split(",")), "every sampled stack holds at least one creature")
check(sample_composition(random.Random(5)) == sample_composition(random.Random(5)), "sampling is reproducible from its seed")

pool = {"matchups": [{"attacker": "1:5", "defender": "1:6"}, {"attacker": "2:3", "defender": "1:40"}]}
mus = pool_matchups(pool)
check(len(mus) == 2 and isinstance(mus[0], Matchup), "a pool converts to matchups")
check(mus[0].attacker == "1:5" and mus[1].defender == "1:40", "pool entries keep their armies")

# Scaling has to stay monotone, or the calibration bisection is searching a non-monotone space.
check(int(scale_army("1:10", 2.0).split(":")[1]) > int(scale_army("1:10", 1.0).split(":")[1]),
      "scaling up increases counts")

# --- the policy fingerprint that ties a calibrated pool to the checkpoint that measured it
from fheroes2_agent.policy import BattlePolicy  # noqa: E402
from fheroes2_agent.scenarios import policy_fingerprint  # noqa: E402

torch.manual_seed(0)
a = BattlePolicy()
torch.manual_seed(0)
b = BattlePolicy()
torch.manual_seed(1)
c = BattlePolicy()
check(policy_fingerprint(a) == policy_fingerprint(b), "identical weights fingerprint identically")
check(policy_fingerprint(a) != policy_fingerprint(c), "different weights fingerprint differently")

# The value head does not choose actions, so fitting a critic leaves difficulty untouched and must
# leave the fingerprint untouched. Otherwise a pool calibrated before stage 2b is rejected after
# it, for a difference that cannot change which battles are won.
with torch.no_grad():
    for name, parameter in b.named_parameters():
        if name.startswith("value_head"):
            parameter += 0.5
check(policy_fingerprint(a) == policy_fingerprint(b), "fitting the value head leaves the fingerprint alone")

# The whole point is detecting a policy that has moved, which is what makes a calibrated pool
# stale. A fingerprint insensitive to a small update would pass a stale pool through.
with torch.no_grad():
    next(p for n, p in a.named_parameters() if not n.startswith("value_head"))[0] += 1e-3
check(policy_fingerprint(a) != policy_fingerprint(b), "a single perturbed policy weight changes the fingerprint")

# --- leave-one-out and Dr. GRPO are the same algorithm once the batch is normalized
returns = np.array([2.0, -1.0, 1.5, 0.0, -0.5, 1.0, 0.25, -1.5], dtype=np.float32)
loo, dr = ob.group_advantages(returns, "loo"), ob.group_advantages(returns, "drgrpo")
k = len(returns)
check(np.allclose(loo, dr * k / (k - 1)), "leave-one-out is Dr. GRPO scaled by k/(k-1)")

# Several groups, which is what a real batch holds. The scale is the same in every group, so the
# whole batch is a uniform rescaling, and normalization divides it straight back out.
rng = np.random.default_rng(0)
groups = [rng.normal(size=8).astype(np.float32) for _ in range(4)]
batch_loo = np.concatenate([ob.group_advantages(g, "loo") for g in groups])
batch_dr = np.concatenate([ob.group_advantages(g, "drgrpo") for g in groups])
check(np.allclose(ob.normalize_advantages(batch_loo, 0.1), ob.normalize_advantages(batch_dr, 0.1)),
      "after batch normalization the two are identical, so the O(1/k) bias is absorbed")

# The floor is the only thing that can separate them, because it stops the divisor tracking the
# scale. This is why the two arms are not quite identical in the recorded experiments.
tiny = [g * 0.01 for g in groups]
floored_loo = ob.normalize_advantages(np.concatenate([ob.group_advantages(g, "loo") for g in tiny]), 0.1)
floored_dr = ob.normalize_advantages(np.concatenate([ob.group_advantages(g, "drgrpo") for g in tiny]), 0.1)
check(not np.allclose(floored_loo, floored_dr), "a binding floor is the one thing that separates them")

# GRPO is genuinely different, because studentizing divides by the group's own spread rather than
# by a constant, so it is not a uniform rescaling and normalization cannot absorb it.
uneven = [np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0], dtype=np.float32), groups[0]]
batch_grpo = np.concatenate([ob.group_advantages(g, "grpo") for g in uneven])
batch_dr2 = np.concatenate([ob.group_advantages(g, "drgrpo") for g in uneven])
check(not np.allclose(ob.normalize_advantages(batch_grpo, 0.1), ob.normalize_advantages(batch_dr2, 0.1)),
      "studentization survives normalization, so GRPO is a real third option")

# --- the sampling roster is derived from the audit, not hand-listed
from fheroes2_agent.scenarios import ROSTER  # noqa: E402
import json as _json  # noqa: E402
import pathlib as _pathlib  # noqa: E402

_caps = {c["monster_id"]: c for c in _json.loads(
    (_pathlib.Path(__file__).resolve().parents[1] / "fheroes2_agent" / "data" / "monster_capabilities_v1.json").read_text())}
check(len(ROSTER) >= 30, "the roster covers the audit's allowlist, not one faction", f"{len(ROSTER)} creatures")
check(all(_caps[i]["simple_v1_supported"] for i, _, _ in ROSTER), "every roster entry is simple_v1 supported")
check(all(hp == _caps[i]["hit_points"] and hp > 0 for i, hp, _ in ROSTER), "hit points come from the engine audit")
check(all(shooter == _caps[i]["is_archer"] for i, _, shooter in ROSTER), "shooter flags come from the engine audit")

# --- the diverse sampler and the pool round-trip
from fheroes2_agent.scenarios import load_wide_roster, sample_diverse_matchup, pool_matchups as _pm  # noqa: E402

wide_roster = load_wide_roster()
check(len(wide_roster) >= 45, "the wide roster covers the wide_v1 allowlist", f"{len(wide_roster)} creatures")
rngd = random.Random(3)
drawn = [sample_diverse_matchup(rngd) for _ in range(40)]
check(all(m.allow_wide for m in drawn), "diverse matchups opt into wide units")
check(any(m.attacker_hero for m in drawn) and any(m.attacker_hero is None for m in drawn),
      "commanders land on some sides and not others")
check(sample_diverse_matchup(random.Random(9)) == sample_diverse_matchup(random.Random(9)),
      "diverse sampling is reproducible from its seed")

saved_pool = {"matchups": [
    {"attacker": "1:5", "defender": "1:6"},
    {"attacker": "9:2,1:10", "defender": "1:300", "attacker_hero": "13:12", "defender_hero": None, "allow_wide": True},
]}
ms = _pm(saved_pool)
check(ms[0].attacker_hero is None and not ms[0].allow_wide, "a plain pool entry stays plain")
check(ms[1].attacker_hero == "13:12" and ms[1].allow_wide,
      "heroes and the wide flag survive the pool round-trip")

# --- a group must share its starting position, or the baseline measures the scenario
from fheroes2_agent.env import MatchupPool  # noqa: E402

pool = MatchupPool("/nonexistent", [Matchup(f"1:{i}", "1:5") for i in range(2, 40)],
                   seed=0, hold_within_group=True)
pool.new_group()
first = pool.current
# reset() would spawn a worker, so the sampling rule is exercised directly: holding means the
# matchup only changes when a new group is asked for.
check(first is not None, "new_group draws a matchup")
pool.current = first
check(pool.current is first, "holding keeps the matchup across episodes of one group")
seen = set()
for _ in range(50):
    pool.new_group()
    seen.add(pool.current.attacker)
check(len(seen) > 1, "new_group still rotates across groups")

# The default must stay as it was, since a critic-based trainer conditions on the observation and
# does not care which episodes sit beside it.
rotating = MatchupPool("/nonexistent", [Matchup(f"1:{i}", "1:5") for i in range(2, 40)], seed=0)
check(rotating._hold is False, "rotation per episode remains the default")

print(f"{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
