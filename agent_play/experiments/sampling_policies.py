#!/usr/bin/env python3
"""Deployment-time sampling schemes: nucleus and entropy-adaptive nucleus, the owner's proposal.

Training-time sampling wants entropy, exploration is the point; deployment wants reliability,
and the walkthrough recorded the hazard, a rank-four action drawn at probability 0.083 in the
middle of a winning fight. Nucleus (top-p) sampling keeps the smallest action set whose
probability mass reaches p and renormalizes, trimming the tail a softmax never zeroes.

The owner's refinement matters in this domain: whether a distribution is "confident" must be
judged against how many actions were even legal, so the adaptive variant conditions on the
normalized entropy H over log K, which is 1 at uniform-over-legal and 0 at deterministic, and
tightens the nucleus only when the policy itself claims confidence, leaving genuine forks wide.

Every variant is a logits transform wrapped around the same checkpoint, evaluated by the same
harness on held-out matchups and the hard Thunk rungs, so the comparison isolates the sampler.

Usage:
    ./sampling_policies.py WORKER CHECKPOINT [--episodes 8] [--seeds 4]
                           [--report sampling_policies.json]
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

from fheroes2_agent.policy import load_policy  # noqa: E402
from fheroes2_agent.scenarios import Matchup, measure  # noqa: E402


class Sampler(nn.Module):
    """Reshape the policy's masked distribution; `measure` then samples the reshaped one."""

    def __init__(self, model, scheme: str, p: float = 0.8):
        super().__init__()
        self.model = model
        self.scheme = scheme
        self.p = p
        self.planes = getattr(model, "planes", False)

    def forward(self, observation, mask, planes=None):
        args = (planes,) if self.planes else ()
        logits, value = self.model(observation, mask, *args)
        if self.scheme == "full":
            return logits, value
        if self.scheme == "greedy":
            hard = torch.full_like(logits, -1e9)
            hard.scatter_(1, logits.argmax(-1, keepdim=True), 0.0)
            return hard, value
        if self.scheme == "temperature":
            # Rescale before the softmax: T>1 softens, T<1 sharpens. Added 2026-08-18 because the
            # distillation arms differ enormously in prior sharpness (normalized entropy 0.281 to
            # 0.315 at 9.4t), and temperature is the only deployment rule that moves along exactly
            # that axis, so it separates "this policy plays better" from "this policy is at a good
            # sharpness for greedy play". Illegal entries stay at MASK_FILL under any positive T.
            return logits / max(self.p, 1e-3), value
        probs = torch.softmax(logits, dim=-1)
        if self.scheme == "adaptive":
            legal = mask.sum(-1, keepdim=True).clamp(min=2)
            masked = probs.clamp_min(1e-12)
            entropy = -(masked * masked.log()).sum(-1, keepdim=True)
            normalized = (entropy / legal.float().log()).clamp(0.0, 1.0)
            # Confident state, small normalized entropy, tight nucleus; genuine fork, wide one.
            p_row = 0.5 + 0.5 * normalized
        else:
            p_row = torch.full((probs.shape[0], 1), self.p, device=probs.device)
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(-1)
        keep_sorted = cumulative - sorted_probs < p_row
        keep = torch.zeros_like(keep_sorted)
        keep.scatter_(1, order, keep_sorted)
        trimmed = torch.where(keep, logits, torch.full_like(logits, -1e9))
        return trimmed, value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    started = time.time()
    base = load_policy(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    base.eval()
    pool = json.loads((pathlib.Path(__file__).resolve().parents[1] / "docs" / "archive" / "experiments"
                       / "files" / "2026-08-05-run-reports" / "pool_value.json").read_text())["matchups"][40:50]
    held_set = [Matchup(e["attacker"], e["defender"], attacker_hero=e.get("attacker_hero"),
                        defender_hero=e.get("defender_hero"), allow_wide=bool(e.get("allow_wide")))
                for e in pool]
    rung_850 = Matchup("11:1,11:1,11:1,10:2,9:2", "1:284,1:283,1:283", attacker_hero="13:12", allow_wide=True)
    rung_1000 = Matchup("11:1,11:1,11:1,10:2,9:2", "1:334,1:333,1:333", attacker_hero="13:12", allow_wide=True)

    schemes = [("full", None), ("greedy", None), ("top_p_0.9", 0.9), ("top_p_0.7", 0.7),
               ("top_p_0.5", 0.5), ("adaptive", None)]
    torch.manual_seed(0)
    report = {}
    for name, p in schemes:
        scheme = "top_p" if name.startswith("top_p") else name
        agent = Sampler(base, scheme, p or 0.8)
        agent.eval()
        held = [measure(agent, args.worker, m, episodes=args.episodes, seeds=args.seeds)["win_rate"]
                for m in held_set]
        r850 = measure(agent, args.worker, rung_850, episodes=12, seeds=4)["win_rate"]
        r1000 = measure(agent, args.worker, rung_1000, episodes=12, seeds=4)["win_rate"]
        report[name] = {"held_out10": round(float(np.mean(held)), 4),
                        "held_se": round(float(np.std(held, ddof=1) / np.sqrt(len(held))), 4),
                        "thunk_850": r850, "thunk_1000": r1000}
        print(f"{name:10s} held-out {report[name]['held_out10']:.3f}±{report[name]['held_se']:.3f}  "
              f"850 {r850:.2f}  1000 {r1000:.2f}", flush=True)

    print(f"\ntotal {round(time.time() - started)}s")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
