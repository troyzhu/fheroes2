#!/usr/bin/env python3
"""Fidelity beyond exact match: top-k accuracy and the probability given to the teacher's move.

Top-1 agreement is binary and blunt, the owner's point: a policy whose second choice is always
the teacher's move scores zero, same as one that never comes close. This report evaluates a
checkpoint against a recorded corpus's holdout split with the diagnostic trio the literature
uses: top-1, top-3 and top-5 accuracy (is the teacher's action among the k highest-probability
legal actions), the mean probability the policy assigns to the teacher's action, and the mean
entropy of the policy's legal-action distribution, which separates confidently-wrong from
undecided.

Usage:
    ./fidelity_report.py CHECKPOINT --roots DIR [DIR ...] [--planes] [--seed 0]
                         [--report fidelity.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fheroes2_agent.policy import load_policy  # noqa: E402
from planes_ablation import load_planes_corpus, split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0, help="must match the training seed for a true holdout")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    started = time.time()
    observations, masks, actions, planes, episodes = load_planes_corpus(args.roots)
    _, hold = split(episodes, 0.2, args.seed)
    model = load_policy(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()
    uses_planes = bool(getattr(model, "planes", False))

    hits = {1: 0, 3: 0, 5: 0}
    teacher_probability, entropies, total = [], [], 0
    with torch.no_grad():
        for start in range(0, len(hold), 4096):
            batch = torch.from_numpy(hold[start:start + 4096])
            plane_arg = (torch.from_numpy(planes[batch]).float(),) if uses_planes else ()
            logits, _ = model(torch.from_numpy(observations[batch.numpy()]),
                              torch.from_numpy(masks[batch.numpy()]), *plane_arg)
            probs = torch.softmax(logits, dim=-1)
            target = torch.from_numpy(actions[batch.numpy()])
            ranks = logits.argsort(dim=-1, descending=True)
            for k in hits:
                hits[k] += int((ranks[:, :k] == target.unsqueeze(1)).any(dim=1).sum())
            teacher_probability.extend(probs.gather(1, target.unsqueeze(1)).squeeze(1).tolist())
            masked = probs.clamp_min(1e-12)
            entropies.extend((-(masked * masked.log()).sum(dim=-1)).tolist())
            total += len(batch)

    result = {
        "checkpoint": args.checkpoint, "roots": args.roots, "seed": args.seed, "holdout_decisions": total,
        "top1": round(hits[1] / total, 4), "top3": round(hits[3] / total, 4), "top5": round(hits[5] / total, 4),
        "mean_teacher_probability": round(float(np.mean(teacher_probability)), 4),
        "median_teacher_probability": round(float(np.median(teacher_probability)), 4),
        "mean_entropy_nats": round(float(np.mean(entropies)), 4),
        "seconds": round(time.time() - started, 1),
    }
    print(f"top-1 {result['top1']}  top-3 {result['top3']}  top-5 {result['top5']}  "
          f"p(teacher) mean {result['mean_teacher_probability']} median {result['median_teacher_probability']}  "
          f"entropy {result['mean_entropy_nats']} nats  ({total} decisions, {result['seconds']}s)")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
