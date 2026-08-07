#!/usr/bin/env python3
"""Does difficulty-weighting the terminal reward change what training learns?

The candidate scales wins by the tempered opponent-to-own strength ratio and losses by its
inverse (fheroes2_agent.env.apply_difficulty), so easy victories pay less and lopsided losses
cost less. Two things could make it a no-op, and this measures instead of assuming: a critic
absorbs per-matchup difficulty into its baseline, and evaluation win rate is unweighted either
way. Both arms start from the same clone, train PPO on the same matchup split with paired torch
seeds, and are scored on raw win rate over training and held-out matchups.

Usage:
    ./difficulty_reward.py WORKER CHECKPOINT [--iterations 40] [--train-matchups 40]
                           [--held-matchups 20] [--seeds 3] [--report difficulty_reward.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import time

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


def evaluate(model: BattlePolicy, worker: str, matchups: list[Matchup], episodes: int) -> float:
    rates = [measure(model, worker, m, episodes=episodes)["win_rate"] for m in matchups]
    return float(np.mean(rates))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--train-matchups", type=int, default=40)
    parser.add_argument("--held-matchups", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    entries = json.loads(POOL.read_text())["matchups"]
    if len(entries) < args.train_matchups + args.held_matchups:
        raise SystemExit(f"pool has {len(entries)} matchups, need {args.train_matchups + args.held_matchups}")
    train_set = [as_matchup(e) for e in entries[: args.train_matchups]]
    held_set = [as_matchup(e) for e in entries[args.train_matchups: args.train_matchups + args.held_matchups]]

    started = time.time()
    base_model = load_policy(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    base_model.eval()
    baseline = {"train": evaluate(base_model, args.worker, train_set, args.eval_episodes),
                "held": evaluate(base_model, args.worker, held_set, args.eval_episodes)}
    print(f"clone baseline: train {baseline['train']:.3f}, held-out {baseline['held']:.3f}", flush=True)

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="difficulty_reward_"))
    runs = []
    for seed in range(args.seeds):
        for arm in ("none", "difficulty"):
            pool = MatchupPool(args.worker, train_set, seed=seed, reward_weighting=arm)
            out_path = workdir / f"{arm}_s{seed}.pt"
            history = train_ppo.train(args.worker, checkpoint=args.checkpoint, iterations=args.iterations,
                                      seed=seed, env=pool, quiet=True, out=str(out_path))
            model = load_policy(torch.load(out_path, map_location="cpu", weights_only=True)["state_dict"])
            model.eval()
            run = {"arm": arm, "seed": seed,
                   "train": evaluate(model, args.worker, train_set, args.eval_episodes),
                   "held": evaluate(model, args.worker, held_set, args.eval_episodes),
                   "history": history}
            runs.append(run)
            print(f"seed {seed} arm {arm:10s}: train {run['train']:.3f}, held-out {run['held']:.3f}", flush=True)

    for split in ("train", "held"):
        plain = np.array([r[split] for r in runs if r["arm"] == "none"])
        weighted = np.array([r[split] for r in runs if r["arm"] == "difficulty"])
        diff = weighted - plain
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
        print(f"{split}: none {plain.mean():.3f}, difficulty {weighted.mean():.3f}, "
              f"paired diff {diff.mean():+.3f} +/- {se:.3f}")

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"baseline": baseline, "runs": runs, "iterations": args.iterations,
             "train_matchups": args.train_matchups, "held_matchups": args.held_matchups,
             "seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
