#!/usr/bin/env python3
"""Does PPO still earn anything once the supervised pipeline has done its work?

From clone v4, pool PPO gained about +0.14 on-pool and transferred none of it. This asks the
same question from the day's strongest supervised checkpoint: three PPO seeds from the given
anchor on the budget pool's training matchups, everything evaluated over four battlefields, and
the anchor's own vendored evaluation as the paired baseline. A gain says reinforcement learning
still adds on top of a strong clone; a null or a loss says the supervised levers already banked
this pool's headroom at this budget.

Usage:
    ./ppo_from_strongest.py WORKER ANCHOR_CHECKPOINT ANCHOR_REPORT [--iterations 40] [--seeds 3]
                            [--report ppo_from_strongest.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.env import MatchupPool  # noqa: E402
from fheroes2_agent.policy import load_policy, BattlePolicy  # noqa: E402
from fheroes2_agent.scenarios import Matchup, measure  # noqa: E402
from fheroes2_agent import train_ppo  # noqa: E402

POOL = pathlib.Path(__file__).resolve().parents[2] / "agent_play" / "docs" / "archive" / "experiments" / "files" \
    / "2026-08-05-run-reports" / "pool_value.json"


def as_matchup(entry: dict) -> Matchup:
    return Matchup(entry["attacker"], entry["defender"], attacker_hero=entry.get("attacker_hero"),
                   defender_hero=entry.get("defender_hero"), allow_wide=bool(entry.get("allow_wide")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint", help="the supervised anchor PPO starts from")
    parser.add_argument("anchor_report", help="a dagger_iteration report holding the anchor's evals over battlefields")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--eval-seeds", type=int, default=4)
    parser.add_argument("--reward-margin", default="hit_points", choices=("hit_points", "strength", "two_sided"))
    parser.add_argument("--reward-weighting", default="none", choices=("none", "difficulty"))
    parser.add_argument("--value-warmup", type=int, default=0)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    entries = json.loads(POOL.read_text())["matchups"]
    train_set = [as_matchup(e) for e in entries[:40]]
    held_set = [as_matchup(e) for e in entries[40:60]]
    anchor = json.loads(pathlib.Path(args.anchor_report).read_text())["evals"]

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="ppo_strong_"))
    runs = []
    for seed in range(args.seeds):
        pool = MatchupPool(args.worker, train_set, seed=seed,
                           reward_margin=args.reward_margin, reward_weighting=args.reward_weighting)
        out = workdir / f"s{seed}.pt"
        train_ppo.train(args.worker, checkpoint=args.checkpoint, iterations=args.iterations,
                        seed=seed, env=pool, quiet=True, out=str(out),
                        value_warmup_iters=args.value_warmup)
        model = load_policy(torch.load(out, map_location="cpu", weights_only=True)["state_dict"])
        model.eval()
        run = {"seed": seed,
               "train": [measure(model, args.worker, m, episodes=args.eval_episodes, seeds=args.eval_seeds)["win_rate"]
                         for m in train_set],
               "held": [measure(model, args.worker, m, episodes=args.eval_episodes, seeds=args.eval_seeds)["win_rate"]
                        for m in held_set]}
        runs.append(run)
        print(f"seed {seed}: train {np.mean(run['train']):.3f}, held {np.mean(run['held']):.3f}", flush=True)

    for split in ("train", "held"):
        base = np.array(anchor[split])
        per_seed = np.array([float(np.mean(np.array(r[split]) - base)) for r in runs])
        print(f"{split}: ppo-minus-anchor per-seed {per_seed.mean():+.3f} "
              f"+/- {per_seed.std(ddof=1) / np.sqrt(len(per_seed)):.3f}")

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"runs": runs, "anchor": anchor, "anchor_checkpoint": args.checkpoint,
             "checkpoints": str(workdir)}, indent=2))


if __name__ == "__main__":
    main()
