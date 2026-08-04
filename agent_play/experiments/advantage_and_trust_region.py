#!/usr/bin/env python3
"""Do the advantage estimator and the trust region matter, and can this setup tell?

Four arms were compared once at a single seed and finished within 0.02 of each other, which is
inside the noise at 32 episodes an iteration and therefore says nothing. This runs the same
comparison across seeds so the question can be answered rather than restated.

The two axes are independent and are held separate deliberately, per `python/fheroes2_agent/
objectives.py`. Advantage is what the baseline is: leave-one-out excludes the sample and is exactly
unbiased, GRPO includes it and studentizes by the group's own spread, Dr. GRPO includes it and does
not. Trust region is what bounds the step: PPO's clip on the sampled ratio, or DPPO's mask on the
exact total-variation distance, which is affordable here because the legal set is 5 to 30 actions
rather than a language model's vocabulary.

The specific claim worth testing is Dr. GRPO's. Studentizing divides each group by its own noisy
spread, so a group that happens to be homogeneous has its advantages inflated, and that is the same
amplification the batch-level floor was added to prevent one level up. If the argument is right,
GRPO should be the more variable of the two across seeds.

One matchup answers whether the arms differ at all. A pool answers whether that survives a
distribution, which is what changing a default would need, and is the harder question.

Usage:
    ./advantage_and_trust_region.py CHECKPOINT WORKER --attacker 2:6,1:10 --defender 1:121
    ./advantage_and_trust_region.py CHECKPOINT WORKER --pool pool140.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.env import MatchupPool  # noqa: E402
from fheroes2_agent.scenarios import pool_matchups  # noqa: E402
from fheroes2_agent.train_group import train  # noqa: E402

ARMS = [
    ("loo", "ratio"),
    ("grpo", "ratio"),
    ("drgrpo", "ratio"),
    ("loo", "divergence"),
    ("drgrpo", "divergence"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("worker")
    parser.add_argument("--attacker", default=None)
    parser.add_argument("--defender", default=None)
    # A single matchup answers whether the arms differ; a pool answers whether that survives a
    # distribution, which is what a change of default would need. Groups are held within a matchup
    # here, or the baseline compares an episode with episodes of other army pairs.
    parser.add_argument("--pool", default=None, help="calibrated pool JSON, instead of one matchup")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    if not args.pool and not (args.attacker and args.defender):
        raise SystemExit("give either --pool or both --attacker and --defender")
    matchups = pool_matchups(json.loads(pathlib.Path(args.pool).read_text())) if args.pool else None
    if matchups:
        print(f"rotating over {len(matchups)} calibrated matchups, held within each group", flush=True)

    started = time.time()
    rows = []
    for advantage, trust in ARMS:
        for seed in range(args.seeds):
            env = MatchupPool(args.worker, matchups, seed=seed, hold_within_group=True) if matchups else None
            r = train(args.worker, checkpoint=args.checkpoint, attacker=args.attacker,
                      defender=args.defender, advantage=advantage, trust_region=trust,
                      iterations=args.iterations, groups_per_iter=args.groups,
                      group_size=args.group_size, seed=seed, quiet=True, env=env)
            wins = [h["win_rate"] for h in r["history"]]
            rows.append({"advantage": advantage, "trust_region": trust, "seed": seed,
                         "initial": r["initial_win_rate"], "final5": r["final_win_rate"],
                         "best": r["best_win_rate"], "history": wins,
                         "degenerate_groups": sum(h.get("degenerate_groups", 0) for h in r["history"])})
            print(f"  {advantage:7s} {trust:11s} seed {seed:2d}  {r['initial_win_rate']:.3f} -> "
                  f"{r['final_win_rate']:.3f}", flush=True)

    print(f"\n  {'advantage':10s} {'trust region':12s} {'last-five':>16s} {'spread':>9s} {'worst':>8s}")
    summary = []
    for advantage, trust in ARMS:
        a = [r["final5"] for r in rows if r["advantage"] == advantage and r["trust_region"] == trust]
        s = {"advantage": advantage, "trust_region": trust, "n": len(a),
             "mean": statistics.mean(a), "stdev": statistics.stdev(a) if len(a) > 1 else 0.0,
             "stderr": statistics.stdev(a) / len(a) ** 0.5 if len(a) > 1 else 0.0, "worst": min(a)}
        summary.append(s)
        print(f"  {advantage:10s} {trust:12s} {s['mean']:.3f} +- {s['stderr']:.3f}  "
              f"{s['stdev']:>8.3f} {s['worst']:>8.3f}")

    # Paired against leave-one-out with the ratio clip, which is what the trainers default to.
    base = {r["seed"]: r["final5"] for r in rows if r["advantage"] == "loo" and r["trust_region"] == "ratio"}
    print(f"\n  paired against loo + ratio, over {len(base)} shared seeds")
    for advantage, trust in ARMS[1:]:
        arm = {r["seed"]: r["final5"] for r in rows if r["advantage"] == advantage and r["trust_region"] == trust}
        d = [arm[s] - base[s] for s in sorted(base) if s in arm]
        se = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0
        print(f"  {advantage:10s} {trust:12s} {statistics.mean(d):+.3f} +- {se:.3f}"
              f" ({abs(statistics.mean(d)) / se if se else 0:.1f} SE)")

    print(f"\n  {time.time() - started:.0f}s total")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"summary": summary, "runs": rows,
             "matchup": {"attacker": args.attacker, "defender": args.defender, "pool": args.pool},
             "iterations": args.iterations, "groups": args.groups, "group_size": args.group_size,
             "seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
