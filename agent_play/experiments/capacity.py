#!/usr/bin/env python3
"""Is the policy network's size a binding constraint, for cloning or for reinforcement learning?

The deployed policy is 396,570 parameters, sized down from 626k when behavior cloning on 37k
decisions overfit. That was a statement about cloning at that data size, not about reinforcement
learning, where rollouts manufacture unlimited data, and the question of whether pool performance
is capacity-limited has never been measured.

Design: clone at each width on the same data, then run PPO from each clone on the same fixed pool
of matchups with paired seeds, everything else identical. If the wide model's cloning agreement
falls, the data bound binds at that stage; if its pool win rate rises, the deployed size is leaving
reinforcement-learning performance on the table.

The pool is used as a fixed task set. Its calibration is relative to the deployed-size clone, and a
different clone shifts what is in band, but a comparison of two models on the same fixed matchups
is fair as a comparison even where the band has drifted.

Usage:
    ./capacity.py DATA_DIR POOL.json WORKER
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.env import MatchupPool  # noqa: E402
from fheroes2_agent.scenarios import pool_matchups  # noqa: E402
from fheroes2_agent.train_bc import train as train_bc  # noqa: E402
from fheroes2_agent.train_ppo import train as train_ppo  # noqa: E402

SIZES = [
    ("deployed", {}),
    ("half", {"slot_hidden": 48, "trunk_hidden": 96, "global_hidden": 16}),
    ("double", {"slot_hidden": 192, "trunk_hidden": 384, "global_hidden": 64}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir")
    parser.add_argument("pool")
    parser.add_argument("worker")
    parser.add_argument("--holdout", type=int, default=50, help="pool entries excluded, matching the split elsewhere")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    matchups = pool_matchups(json.loads(pathlib.Path(args.pool).read_text()))[args.holdout :]
    work = pathlib.Path(tempfile.mkdtemp())
    started = time.time()
    results = []

    for name, kwargs in SIZES:
        checkpoint = work / f"bc_{name}.pt"
        bc = train_bc(args.data_dir, epochs=args.epochs, out=str(checkpoint), model_kwargs=kwargs)
        agreement = bc["best"]["agreement"]

        finals = []
        for seed in range(args.seeds):
            r = train_ppo(args.worker, checkpoint=str(checkpoint), iterations=args.iterations,
                          episodes_per_iter=32, seed=seed, quiet=True, model_kwargs=kwargs,
                          env=MatchupPool(args.worker, matchups, seed=seed))
            finals.append(statistics.mean(h["win_rate"] for h in r["history"][-5:]))

        results.append({"size": name, "model_kwargs": kwargs, "parameters": bc.get("parameters"),
                        "bc_agreement": agreement, "pool_final5": finals,
                        "pool_mean": statistics.mean(finals),
                        "pool_stderr": statistics.stdev(finals) / len(finals) ** 0.5 if len(finals) > 1 else 0.0})
        print(f"  {name:9s} agreement {agreement:.4f}   pool {statistics.mean(finals):.3f} "
              f"+- {results[-1]['pool_stderr']:.3f}  ({', '.join(f'{f:.3f}' for f in finals)})", flush=True)

    print(f"\n  {time.time() - started:.0f}s total")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"sizes": results, "epochs": args.epochs, "iterations": args.iterations,
             "training_matchups": len(matchups), "seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
