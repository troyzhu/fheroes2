#!/usr/bin/env python3
"""Does flooring the advantage-normalization divisor prevent converged runs from collapsing?

Advantage normalization divides a batch by its own spread. Once a matchup is solved every episode
scores alike, the spread collapses toward zero, and the division rescales what is left, which is
value-function error, up to unit variance. Measured amplification on a calibrated matchup is about
fiftyfold, and four epochs of it drove a policy from a 1.000 win rate to 0.000 in two iterations.

This compares an unfloored divisor against a floored one on the same seeds. Everything else is
held fixed, including the checkpoint, so the difference is attributable to the floor.

Collapse is defined rather than eyeballed: a run that reached a win rate of 0.95 or better at some
iteration and finished with a last-five mean below 0.5. That is a run which solved the matchup and
then lost it again, which is a different failure from never solving it.

Usage:
    ./advantage_floor.py CHECKPOINT WORKER --attacker 2:6,1:10 --defender 1:121
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.train_ppo import train  # noqa: E402

SOLVED = 0.95
COLLAPSED = 0.5
UNFLOORED = 1e-8  # what the divisor was before, an epsilon guarding division rather than scale


def summarize(rows: list[dict], arm: str) -> dict:
    a = [r for r in rows if r["arm"] == arm]
    finals = [r["final5"] for r in a]
    collapses = [r for r in a if r["best"] >= SOLVED and r["final5"] < COLLAPSED]
    return {
        "arm": arm,
        "seeds": len(a),
        "mean_final5": statistics.mean(finals),
        "stderr": statistics.stdev(finals) / len(finals) ** 0.5 if len(finals) > 1 else 0.0,
        "stdev": statistics.stdev(finals) if len(finals) > 1 else 0.0,
        "worst": min(finals),
        "solved": sum(1 for r in a if r["best"] >= SOLVED),
        "collapsed": len(collapses),
        "collapsed_seeds": [r["seed"] for r in collapses],
        "mean_floored_iterations": statistics.mean(r["floored"] for r in a),
    }


def fisher_one_sided(a: int, b: int, n: int) -> float:
    """One-sided Fisher exact on collapses, a of n against b of n."""
    import math

    total = 0.0
    for i in range(a, min(n, a + b) + 1):
        j = a + b - i
        if 0 <= j <= n:
            total += math.comb(n, i) * math.comb(n, j) / math.comb(2 * n, a + b)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("worker")
    parser.add_argument("--attacker", required=True)
    parser.add_argument("--defender", required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--floor", type=float, default=0.1)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    started = time.time()
    rows = []
    for arm, floor in [("unfloored", UNFLOORED), (f"floor {args.floor}", args.floor)]:
        for seed in range(args.seeds):
            r = train(args.worker, checkpoint=args.checkpoint, attacker=args.attacker,
                      defender=args.defender, iterations=args.iterations,
                      episodes_per_iter=args.episodes, seed=seed,
                      advantage_std_floor=floor, quiet=True)
            wins = [h["win_rate"] for h in r["history"]]
            rows.append({"arm": arm, "seed": seed, "final5": statistics.mean(wins[-5:]),
                         "best": max(wins), "floored": r["floored_iterations"], "history": wins})
            print(f"  {arm:12s} seed {seed:2d}  {statistics.mean(wins[-5:]):.3f} "
                  f"(best {max(wins):.3f}, {r['floored_iterations']:2d} below floor)", flush=True)

    arms = [summarize(rows, a) for a in ("unfloored", f"floor {args.floor}")]
    print(f"\n  {'arm':12s} {'last-five':>16s} {'spread':>9s} {'worst':>8s} {'collapsed':>12s}")
    for s in arms:
        print(f"  {s['arm']:12s} {s['mean_final5']:.3f} +- {s['stderr']:.3f}  {s['stdev']:>8.3f}"
              f" {s['worst']:>8.3f} {s['collapsed']:>6d}/{s['seeds']:<5d}")

    a, b, n = arms[0]["collapsed"], arms[1]["collapsed"], args.seeds
    p = fisher_one_sided(a, b, n)
    print(f"\n  collapses {a}/{n} unfloored against {b}/{n} floored, one-sided Fisher exact p = {p:.4f}")
    print(f"  unfloored collapsed on seeds {arms[0]['collapsed_seeds']}")
    print(f"  {time.time() - started:.0f}s total")

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"arms": arms, "runs": rows, "fisher_p": p, "floor": args.floor,
             "matchup": {"attacker": args.attacker, "defender": args.defender},
             "iterations": args.iterations, "episodes_per_iteration": args.episodes,
             "seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
