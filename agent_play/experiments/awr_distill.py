#!/usr/bin/env python3
"""Advantage-weighted distillation: the off-support survey's first arm.

One improvement step that never queries an unseen action (Peng et al., AWR; the Unplugged
recipe's shape): each teacher decision's cloning loss is weighted by exp(A/beta), where
A = outcome - V(o) and V is a value checkpoint fitted on outcomes. Decisions that beat the
value's expectation are amplified, decisions that underperformed it are attenuated, and no
action outside the data ever enters the loss.

Trains the weighted arm and its unweighted twin from one seed on identical roots, so the
comparison is paired; judge both with validation_battery.py afterwards, multi-seed before any
adoption per the conventions.

Usage:
    ./awr_distill.py --roots DIR [DIR ...] --value VALUE.pt --out awr.pt --plain-out plain.pt
                     [--beta 1.0] [--epochs 25] [--seed 0] [--report awr_distill.json]

The value checkpoint is a dedicated value network state dict (the lab's ValueNet shape:
slot encoder 63-96-96, globals 4-32, trunk 992-192-192, scalar head).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.dataset import load_dir, split_by_episode  # noqa: E402
from fheroes2_agent.encoding import ENCODING_VERSION, GLOBAL_FEATURES, SLOT_COUNT, SLOT_FEATURES  # noqa: E402
from fheroes2_agent.policy import BattlePolicy  # noqa: E402


class DedicatedValue(nn.Module):
    """The lab's dedicated value network, redeclared here so the checkpoint loads without
    importing scratch modules."""

    def __init__(self, slot_hidden: int = 96, trunk: int = 192):
        super().__init__()
        self.slot = nn.Sequential(nn.Linear(SLOT_FEATURES, slot_hidden), nn.ReLU(),
                                  nn.Linear(slot_hidden, slot_hidden), nn.ReLU())
        self.glob = nn.Sequential(nn.Linear(GLOBAL_FEATURES, 32), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(SLOT_COUNT * slot_hidden + 32, trunk), nn.ReLU(),
                                   nn.Linear(trunk, trunk), nn.ReLU())
        self.head = nn.Linear(trunk, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        slots = x[:, : SLOT_COUNT * SLOT_FEATURES].view(b, SLOT_COUNT, SLOT_FEATURES)
        embedded = self.slot(slots) * slots[:, :, :1]
        joined = torch.cat([embedded.flatten(1), self.glob(x[:, SLOT_COUNT * SLOT_FEATURES:])], dim=1)
        return torch.tanh(self.head(self.trunk(joined))).squeeze(-1)


def advantage_weights(value: DedicatedValue, observations: np.ndarray, returns: np.ndarray,
                      beta: float) -> np.ndarray:
    """exp(clip(A/beta)) with A = outcome - V(o), normalized to mean one so the learning rate
    keeps its meaning. The outcome is the sign of the recorded return, matching the value's
    own +/-1 training target rather than the discounted margin."""
    outcomes = np.where(returns > 0, 1.0, -1.0).astype(np.float32)
    estimates = []
    with torch.no_grad():
        for start in range(0, len(observations), 4096):
            estimates.append(value(torch.from_numpy(observations[start:start + 4096])).numpy())
    advantage = outcomes - np.concatenate(estimates)
    weights = np.exp(np.clip(advantage / beta, -3.0, 3.0)).astype(np.float32)
    return weights / weights.mean()


def train_arm(samples, value: DedicatedValue | None, beta: float, epochs: int, seed: int, out: str) -> dict:
    torch.manual_seed(seed)
    train_s, holdout_s = split_by_episode(samples, 0.2, seed)
    model = BattlePolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    obs = torch.from_numpy(train_s.observations)
    masks = torch.from_numpy(train_s.masks)
    actions = torch.from_numpy(train_s.actions)
    # Weights are computed on the training split's own rows, so no cross-split index bookkeeping
    # can drift: each row's weight depends only on its observation and outcome.
    w = None
    if value is not None:
        w = torch.from_numpy(advantage_weights(value, train_s.observations, train_s.returns, beta))
    hobs, hmasks, hactions = (torch.from_numpy(holdout_s.observations), torch.from_numpy(holdout_s.masks),
                              torch.from_numpy(holdout_s.actions))
    best = {"agreement": -1.0}
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(actions))
        for start in range(0, len(actions), 256):
            batch = perm[start:start + 256]
            logits, _ = model(obs[batch], masks[batch])
            losses = torch.nn.functional.cross_entropy(logits, actions[batch], reduction="none")
            loss = (losses * w[batch]).mean() if w is not None else losses.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedule.step()
        model.eval()
        with torch.no_grad():
            agree = hits = 0
            for start in range(0, len(hactions), 4096):
                sl = slice(start, start + 4096)
                logits, _ = model(hobs[sl], hmasks[sl])
                agree += int((logits.argmax(-1) == hactions[sl]).sum())
                hits += len(hactions[sl])
        agreement = agree / hits
        if agreement > best["agreement"]:
            best = {"epoch": epoch, "agreement": agreement}
            torch.save({"state_dict": model.state_dict(), "encoding_version": ENCODING_VERSION}, out)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--value", required=True)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--plain-out", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    started = time.time()
    samples = load_dir(list(args.roots))
    value = DedicatedValue()
    value.load_state_dict(torch.load(args.value, map_location="cpu", weights_only=True)["state_dict"])
    value.eval()
    weights = advantage_weights(value, samples.observations, samples.returns, args.beta)
    kept = float((weights > 1.0).mean())
    print(f"{len(samples.actions)} decisions; advantage weights: mean 1.0 by construction, "
          f"{kept:.1%} amplified, spread [{weights.min():.2f}, {weights.max():.2f}]", flush=True)

    weighted = train_arm(samples, value, args.beta, args.epochs, args.seed, args.out)
    print(f"weighted arm: best agreement {weighted['agreement']:.4f} at epoch {weighted['epoch']}", flush=True)
    plain = train_arm(samples, None, args.beta, args.epochs, args.seed, args.plain_out)
    print(f"plain arm:    best agreement {plain['agreement']:.4f} at epoch {plain['epoch']}", flush=True)

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"roots": args.roots, "beta": args.beta, "seed": args.seed, "weighted": weighted, "plain": plain,
             "weight_spread": [float(weights.min()), float(weights.max())],
             "seconds": round(time.time() - started, 1)}, indent=2))
    print(f"total {round(time.time() - started)}s")


if __name__ == "__main__":
    main()
