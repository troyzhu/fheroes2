"""Unit tests for the policy network and its masking."""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fheroes2_agent import encoding, policy  # noqa: E402

passed = failed = 0


def check(condition, name):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}")
        failed += 1


torch.manual_seed(0)
model = policy.BattlePolicy()
B = 4
obs = torch.randn(B, encoding.OBSERVATION_SIZE)
mask = torch.zeros(B, encoding.ACTION_SPACE_SIZE, dtype=torch.bool)
mask[:, [0, 3, 411]] = True

logits, value = model(obs, mask)
check(logits.shape == (B, encoding.ACTION_SPACE_SIZE), "policy head emits one logit per canonical action")
check(value.shape == (B,), "value head emits one scalar per row")

# The whole point of masking: illegal actions get no probability and no gradient.
probs = torch.softmax(logits, dim=-1)
check(float(probs[~mask].detach().max()) < 1e-30, "illegal actions receive effectively zero probability")
check(abs(float(probs[mask].detach().sum()) - B) < 1e-4, "probability mass sums to one over the legal set")

loss = policy.masked_cross_entropy(logits, torch.zeros(B, dtype=torch.long))
loss.backward()
grad = model.policy_head.weight.grad
legal_rows = torch.zeros(encoding.ACTION_SPACE_SIZE, dtype=torch.bool)
legal_rows[[0, 3, 411]] = True
check(float(grad[~legal_rows].abs().max()) == 0.0, "masked actions receive exactly zero gradient")
check(float(grad[legal_rows].abs().max()) > 0.0, "legal actions do receive gradient")

# Sampling and greedy selection must never leave the legal set.
model.zero_grad()
sampled = model.act(obs, mask)
greedy = model.act(obs, mask, greedy=True)
check(bool(mask.gather(1, sampled.unsqueeze(1)).all()), "sampled actions are always legal")
check(bool(mask.gather(1, greedy.unsqueeze(1)).all()), "greedy actions are always legal")

# Padding slots must not contribute, or a battle's stack count would leak into the features.
a = torch.zeros(1, encoding.OBSERVATION_SIZE)
a[0, 0] = 1.0  # one present slot
b = a.clone()
b[0, encoding.SLOT_FEATURES + 1] = 5.0  # garbage in an absent slot's non-present features
check(torch.allclose(model.features(a), model.features(b)), "an absent slot contributes nothing")

# A single legal action leaves no choice, so the loss is zero regardless of weights.
single = torch.zeros(1, encoding.ACTION_SPACE_SIZE, dtype=torch.bool)
single[0, 7] = True
lg, _ = model(torch.randn(1, encoding.OBSERVATION_SIZE), single)
check(float(policy.masked_cross_entropy(lg, torch.tensor([7]))) < 1e-6, "one legal action gives zero loss")

n = policy.parameter_count(model)
check(200_000 <= n <= 600_000, f"parameter count {n:,} is in the designed range")

print(f"{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
