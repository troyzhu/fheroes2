#!/usr/bin/env python3
"""Does training over battlefields transfer better, judged by evaluation over battlefields?

Two of today's findings meet here. The battlefield-spread run put the obstacle layout's own
term at about 0.11 of win rate, and the difficulty-reward run reported a held-out drop that was
measured on one layout per matchup, so the drop is suspect. This experiment settles both: two
arms from the same clone, one collecting every training episode on the matchup's first
battlefield (the historical behaviour) and one rotating four battlefields, with the clone
baseline and every trained model evaluated over four battlefields. If the drop was measurement,
the seed-spread evaluation says so; if battlefield diversity is a real lever on transfer, the
arm difference says so.

Usage:
    ./generalization_battlefields.py WORKER CHECKPOINT [--iterations 40] [--seeds 3]
                                     [--train-matchups 40] [--held-matchups 20]
                                     [--report generalization_battlefields.json]
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
from fheroes2_agent.policy import BattlePolicy  # noqa: E402
from fheroes2_agent.scenarios import Matchup, measure  # noqa: E402
from fheroes2_agent import train_ppo  # noqa: E402

POOL = pathlib.Path(__file__).resolve().parents[2] / "agent_play" / "docs" / "archive" / "experiments" / "files" \
    / "2026-08-05-run-reports" / "pool_value.json"
EVAL_SEEDS = 4


def as_matchup(entry: dict) -> Matchup:
    return Matchup(entry["attacker"], entry["defender"], attacker_hero=entry.get("attacker_hero"),
                   defender_hero=entry.get("defender_hero"), allow_wide=bool(entry.get("allow_wide")))


def evaluate(model: BattlePolicy, worker: str, matchups: list[Matchup], episodes: int) -> list[float]:
    """Per-matchup win rates, each measured over EVAL_SEEDS battlefields."""
    return [measure(model, worker, m, episodes=episodes, seeds=EVAL_SEEDS)["win_rate"] for m in matchups]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=3, help="paired torch seeds per arm")
    parser.add_argument("--train-battlefields", type=int, default=4)
    parser.add_argument("--train-matchups", type=int, default=40)
    parser.add_argument("--held-matchups", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=24, help="episodes per matchup, spread over battlefields")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    entries = json.loads(POOL.read_text())["matchups"]
    train_set = [as_matchup(e) for e in entries[: args.train_matchups]]
    held_set = [as_matchup(e) for e in entries[args.train_matchups: args.train_matchups + args.held_matchups]]

    started = time.time()
    base_model = BattlePolicy()
    base_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    base_model.eval()
    baseline = {"train": evaluate(base_model, args.worker, train_set, args.eval_episodes),
                "held": evaluate(base_model, args.worker, held_set, args.eval_episodes)}
    print(f"clone baseline over {EVAL_SEEDS} battlefields: "
          f"train {np.mean(baseline['train']):.3f}, held-out {np.mean(baseline['held']):.3f}", flush=True)

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="gen_battlefields_"))
    runs = []
    for seed in range(args.seeds):
        for arm, train_seeds in (("single", 1), ("rotated", args.train_battlefields)):
            pool = MatchupPool(args.worker, train_set, seed=seed, seeds=train_seeds)
            out_path = workdir / f"{arm}_s{seed}.pt"
            history = train_ppo.train(args.worker, checkpoint=args.checkpoint, iterations=args.iterations,
                                      seed=seed, env=pool, quiet=True, out=str(out_path))
            model = BattlePolicy()
            model.load_state_dict(torch.load(out_path, map_location="cpu", weights_only=True)["state_dict"])
            model.eval()
            run = {"arm": arm, "seed": seed,
                   "train": evaluate(model, args.worker, train_set, args.eval_episodes),
                   "held": evaluate(model, args.worker, held_set, args.eval_episodes),
                   "history": history, "checkpoint": str(out_path)}
            runs.append(run)
            print(f"seed {seed} arm {arm:8s}: train {np.mean(run['train']):.3f}, "
                  f"held-out {np.mean(run['held']):.3f}", flush=True)

    for split in ("train", "held"):
        base = float(np.mean(baseline[split]))
        for arm in ("single", "rotated"):
            arm_rates = np.array([np.mean(r[split]) for r in runs if r["arm"] == arm])
            print(f"{split}: {arm} {arm_rates.mean():.3f} +/- {arm_rates.std(ddof=1) / np.sqrt(len(arm_rates)):.3f} "
                  f"(baseline {base:.3f}, gain {arm_rates.mean() - base:+.3f})")
        single = np.array([np.mean(r[split]) for r in runs if r["arm"] == "single"])
        rotated = np.array([np.mean(r[split]) for r in runs if r["arm"] == "rotated"])
        diff = rotated - single
        print(f"{split}: rotated minus single, paired {diff.mean():+.3f} +/- {diff.std(ddof=1) / np.sqrt(len(diff)):.3f}")

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"baseline": baseline, "runs": runs, "eval_seeds": EVAL_SEEDS,
             "train_battlefields": args.train_battlefields, "iterations": args.iterations,
             "seconds": round(time.time() - started, 1)}, indent=2))
    print(f"total {round(time.time() - started)}s; checkpoints under {workdir}")


if __name__ == "__main__":
    main()
