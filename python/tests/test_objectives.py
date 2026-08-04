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
# The whole point is detecting a policy that has moved, which is what makes a calibrated pool
# stale. A fingerprint insensitive to a small update would pass a stale pool through.
with torch.no_grad():
    next(iter(a.parameters()))[0] += 1e-3
check(policy_fingerprint(a) != policy_fingerprint(b), "a single perturbed weight changes the fingerprint")

print(f"{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
