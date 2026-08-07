#!/usr/bin/env python3
"""Does group-relative training generalize to matchups it never saw, or only fit the ones it did?

The first attempt at this used five training matchups and two held out, reported a held-out
regression of -0.208, and did not survive a larger pool. That is the failure mode this script
exists to prevent: at twelve evaluation episodes per matchup the standard error on a win rate is
about 0.14, so a pool that small cannot separate a real effect from noise, and reporting one
number per arm hides that entirely.

Sizing. Resolving a change of 0.1 to two standard errors needs roughly fifty held-out matchups,
which is where the default split comes from rather than from taste.

The pool is calibrated against a specific checkpoint, because difficulty is policy-relative. A
pool built for a weaker policy is quietly a pool of easy matchups once the policy improves, which
`agent_play/docs/rl/scenario-distribution.md` calls the moving-band problem. This script therefore
refuses a pool whose recorded checkpoint is not the one being trained.

Usage:
    ./generalization.py POOL.json CHECKPOINT.pt WORKER [--holdout 50] [--iterations 40]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.env import MatchupPool  # noqa: E402
from fheroes2_agent.policy import load_policy, BattlePolicy  # noqa: E402
from fheroes2_agent.scenarios import measure, policy_fingerprint, pool_matchups  # noqa: E402
from fheroes2_agent.train_group import train  # noqa: E402


def evaluate(model: BattlePolicy, worker: str, matchups: list, episodes: int) -> dict:
    """Mean win rate over a set of matchups, with the error taken across matchups.

    Across matchups rather than across episodes, deliberately. Episodes inside one matchup share
    an army pair, so pooling them understates the spread of what the policy will meet next.
    """
    rates = [measure(model, worker, m, episodes=episodes)["win_rate"] for m in matchups]
    se = statistics.stdev(rates) / len(rates) ** 0.5 if len(rates) > 1 else 0.0
    return {"mean": statistics.mean(rates), "stderr": se, "n": len(rates), "per_matchup": rates}


def load(checkpoint: str) -> BattlePolicy:
    model = load_policy(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pool", help="JSON written by scenarios.build_pool")
    parser.add_argument("checkpoint", help="the cloned policy the pool was calibrated against")
    parser.add_argument("worker")
    parser.add_argument("--holdout", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--advantage", default="loo")
    parser.add_argument("--trust-region", default="ratio")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    pool = json.loads(pathlib.Path(args.pool).read_text())
    matchups = pool_matchups(pool)
    if len(matchups) <= args.holdout:
        raise SystemExit(f"pool of {len(matchups)} cannot spare {args.holdout} for holdout")

    model = load(args.checkpoint)
    stamped, actual = pool.get("policy_fingerprint"), policy_fingerprint(model)
    if stamped is None:
        print(f"warning: pool carries no policy fingerprint, so its calibration cannot be verified "
              f"against {args.checkpoint}")
    elif stamped != actual:
        raise SystemExit(f"pool was calibrated against policy {stamped}, not {actual}; difficulty is "
                         f"policy-relative, so this pool would be a pool of easy matchups")

    # Split by position rather than at random. The pool is generated from a seeded sampler, so its
    # order is already arbitrary, and a fixed split makes the experiment reproducible.
    held, trained = matchups[: args.holdout], matchups[args.holdout :]
    print(f"{len(matchups)} calibrated matchups: {len(trained)} for training, {len(held)} held out", flush=True)

    started = time.time()
    before = {"training": evaluate(model, args.worker, trained, args.eval_episodes),
              "held_out": evaluate(model, args.worker, held, args.eval_episodes)}
    print(f"before: training {before['training']['mean']:.3f} +- {before['training']['stderr']:.3f}, "
          f"held out {before['held_out']['mean']:.3f} +- {before['held_out']['stderr']:.3f} "
          f"({time.time() - started:.0f}s)", flush=True)

    # train() returns a report and closes the environment it was given; the refined weights come
    # back only through --out, so the destination is chosen here rather than left implicit.
    refined = pathlib.Path(args.report).with_suffix(".pt") if args.report else pathlib.Path("refined.pt")
    result = train(args.worker, checkpoint=args.checkpoint, advantage=args.advantage,
                   trust_region=args.trust_region, iterations=args.iterations,
                   groups_per_iter=args.groups, group_size=args.group_size,
                   seed=args.seed,
                   # Held within a group, or the leave-one-out baseline compares an episode with
                   # episodes of other army pairs and measures scenario difficulty instead of play.
                   env=MatchupPool(args.worker, trained, seed=args.seed, hold_within_group=True),
                   out=str(refined), quiet=True)
    print(f"trained: {result['initial_win_rate']:.3f} -> {result['final_win_rate']:.3f} on the rotating pool, "
          f"{result['seconds']}s", flush=True)
    trained_model = load(str(refined))

    after = {"training": evaluate(trained_model, args.worker, trained, args.eval_episodes),
             "held_out": evaluate(trained_model, args.worker, held, args.eval_episodes)}

    print()
    print(f"{'':10s} {'before':>16s} {'after':>16s} {'change':>18s}")
    for key, label in (("training", "training"), ("held_out", "held out")):
        b, a = before[key], after[key]
        delta = a["mean"] - b["mean"]
        se = (a["stderr"] ** 2 + b["stderr"] ** 2) ** 0.5
        print(f"{label:10s} {b['mean']:.3f} +- {b['stderr']:.3f} {a['mean']:.3f} +- {a['stderr']:.3f}"
              f"   {delta:+.3f} +- {se:.3f} ({abs(delta) / se if se else 0:.1f} SE)")

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"pool": args.pool, "checkpoint": args.checkpoint, "split": {"training": len(trained), "held_out": len(held)},
             "iterations": args.iterations, "eval_episodes": args.eval_episodes, "seed": args.seed,
             "advantage": args.advantage, "trust_region": args.trust_region,
             "before": before, "after": after, "training_history": result["history"],
             "seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
