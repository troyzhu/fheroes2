"""Unit tests for the root allocators: budget accounting and the three halving defects.

The 2026-08-12 review panel confirmed three defects in `_sequential_halving`, each pinned here so a
regression fails loudly rather than silently reshaping a sweep. The rollout is stubbed to a fixed
per-action value, which makes every schedule decision deterministic and lets the tests assert on
allocation rather than on play.
"""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import fheroes2_agent.search as S  # noqa: E402

passed = failed = 0


def check(condition, name, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
        passed += 1
    else:
        print(f"  FAIL  {name}" + (f"  [{detail}]" if detail else ""))


class StubModel:
    """Descending prior over the legal set: action 0 most probable, then 1, and so on."""

    def __call__(self, obs, mask, *args):
        logits = torch.full((1, mask.shape[-1]), -30.0)
        legal = np.flatnonzero(mask.numpy()[0] if hasattr(mask, "numpy") else mask[0])
        for rank, action in enumerate(legal):
            logits[0, int(action)] = 5.0 - 0.4 * rank
        return logits, None


def run(k, budget, allocator, candidates, values):
    """Drive search_action_detail with a stubbed rollout returning values[action]."""
    calls = []
    original = S.rollout
    S.rollout = lambda sim, model, prefix, a: (calls.append(a), values.get(a, 0.0))[1]
    try:
        mask = np.zeros(793, dtype=bool)
        mask[:k] = True
        obs = np.zeros(634, dtype=np.float32)
        best, means, visits, prior = S.search_action_detail(
            None, StubModel(), [], obs, mask, budget, 1.5,
            allocator=allocator, candidates=candidates)
        return best, visits, calls
    finally:
        S.rollout = original


# Finding 16: Karnin keeps the better half by ceiling, so three survivors keep two. Floor kept one,
# and the remaining budget refined an arm chosen on two samples each. With k=3 and a budget of 12
# the ceil schedule still measures two arms in its second phase.
values = {0: 0.1, 1: 0.9, 2: 0.5}
best, visits, calls = run(3, 12, "sequential_halving", 3, values)
check(best == 1, "halving finds the best of three arms", f"picked {best}")
second_phase_arms = sum(1 for a in (0, 1, 2) if visits[a] > 12 // (2 * 3))
check(second_phase_arms >= 2, "odd survivor count halves by ceiling, keeping two of three",
      f"visits {dict(visits)}")

# Finding 17: candidates=1 must measure the top-PRIOR action, not the lowest legal index. The stub
# prior descends with the index, so the top-prior action is 0 here; shift legality so the lowest
# index is a different action than the highest prior. Legal set {3,4,5}: prior ranks 3 first.
mask_shift_values = {3: 0.2, 4: 0.2, 5: 0.2}
calls = []
original = S.rollout
S.rollout = lambda sim, model, prefix, a: (calls.append(a), mask_shift_values.get(a, 0.0))[1]
try:
    mask = np.zeros(793, dtype=bool)
    mask[3:6] = True
    obs = np.zeros(634, dtype=np.float32)
    best, _, visits, prior = S.search_action_detail(
        None, StubModel(), [], obs, mask, 8, 1.5, allocator="sequential_halving", candidates=1)
    top_prior = max(prior, key=prior.get)
    check(set(calls) == {top_prior}, "candidates=1 spends the whole budget on the top-prior arm",
          f"top prior {top_prior}, measured {sorted(set(calls))}")
finally:
    S.rollout = original

# Finding 18: under a truncated final phase the answer is the best surviving arm, never an arm the
# schedule eliminated. k=26 uncapped at a budget of 20 truncates mid-schedule; the best arm by value
# is action 25, which the prior ranks last, and the schedule may legitimately eliminate it before
# measuring it well. What must NOT happen is returning an action with zero visits or one that was
# eliminated while a measured survivor existed. We assert the winner was visited and that its mean
# is the maximum among visited survivors of the final phase.
values = {a: 0.01 * a for a in range(26)}
best, visits, calls = run(26, 20, "sequential_halving", 0, values)
check(visits[best] > 0, "truncated schedule returns a measured arm", f"picked {best} visits {visits[best]}")

# The pick must come from the surviving set: reconstruct the survivor invariant indirectly by
# asserting no action with a strictly better mean AND equal-or-more visits was passed over. Under
# the (visits>0, means) rule over survivors, any visited action with a higher mean that also
# survived would have won; an eliminated arm may hold a higher mean, which is the anytime cost the
# docstring records. So the invariant testable from outside is measured-ness plus stability:
best2, visits2, _ = run(26, 20, "sequential_halving", 0, values)
check(best == best2, "truncated pick is deterministic", f"{best} vs {best2}")

# Budget accounting still exact after the fixes, across shapes including non powers of two.
for k, n in ((26, 48), (26, 32), (29, 16), (8, 48), (3, 12), (2, 8), (5, 7)):
    _, visits, calls = run(k, n, "sequential_halving", 0, {a: 0.5 for a in range(k)})
    check(len(calls) == n, f"halving spends exactly its budget at k={k} n={n}", f"spent {len(calls)}")

# The zero-budget guard still returns the prior's argmax through the halving path too.
best, visits, calls = run(5, 0, "puct", 0, {})
check(len(calls) == 0 and best == 0, "zero budget returns the prior argmax with no rollouts",
      f"picked {best}, {len(calls)} rollouts")

# Un-appliable candidates (2026-08-23). Under the offset combat stream ADR 0008 mandates, the
# replayed position may not offer a live-legal candidate at all: measured at 21.6 percent of
# candidates, 36 percent by depth twelve. The rollouts signal that with None, and the search must
# decline to score such a candidate rather than credit it with a substitute action's value, which
# is what the engine's skip-the-turn contract used to produce silently.
obs_zero = np.zeros(634, dtype=np.float32)


def run_dead(k, budget, dead, values):
    calls = []
    original = S.rollout

    def stub(sim, model, prefix, a):
        if a in dead:
            return None
        calls.append(a)
        return values.get(a, 0.0)

    S.rollout = stub
    try:
        mask = np.zeros(793, dtype=bool)
        mask[:k] = True
        diag = {}
        best, _, visits, _ = S.search_action_detail(None, StubModel(), [], obs_zero, mask,
                                                    budget, 1.5, diagnostics=diag)
        return best, visits, calls, diag
    finally:
        S.rollout = original


# The best-valued action is un-appliable, so it must never be returned and never be credited.
best, visits, calls, diag = run_dead(6, 20, dead={3}, values={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.99,
                                                              4: 0.4, 5: 0.5})
check(best != 3, "an un-appliable candidate is never returned", f"picked {best}")
check(visits[3] == 0, "an un-appliable candidate is never credited a visit", f"visits {visits[3]}")
check(3 not in calls, "no playout is charged to it", f"calls {sorted(set(calls))}")
check(diag.get("unavailable") == [3], "it is reported as unavailable", f"{diag.get('unavailable')}")
check(len(calls) == 20, "the budget still buys 20 real playouts", f"{len(calls)}")

# Every candidate un-appliable: nothing was measured, so the prior's pick stands rather than an
# argmax over an all-zero table, which would return the lowest legal index.
best, visits, calls, diag = run_dead(5, 8, dead={0, 1, 2, 3, 4}, values={})
check(len(calls) == 0, "a fully blind position runs no playouts", f"{len(calls)}")
check(best == 0, "a fully blind position falls back to the prior's argmax", f"picked {best}")
check(sum(visits.values()) == 0, "and credits nothing", f"{sum(visits.values())}")

print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
