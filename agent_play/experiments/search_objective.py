#!/usr/bin/env python3
"""Which objective should search maximize, and does it have to be the training reward?

Found 2026-08-09 while reconciling two search numbers that would not agree: a per-matchup rate of
0.00 from the battery's loop and repeated wins from the capture loop on the same armies, the same
battlefield and the same checkpoint. The loops differ in one place. `rollout()` returns the side
environment's terminal reward, so whatever `reward_margin` that environment was built with is the
quantity root search maximizes; the battery passes `two_sided` and `capture_replay.py` never passed
one and inherited the `hit_points` default. On the cell that exposed it the gap was 0.00 against
0.90, which is far too large to leave as a configuration accident.

The four margins split by their loss branch rather than by their win branch. `hit_points` and
`strength` grade a lost battle by how much of one's own force survived; `two_sided` and `balanced`
grade it by how much of the enemy's was destroyed. A rollout that ends in defeat is the common case
deep in a search tree, so that branch is most of the signal search sees, and the two families
therefore rank candidate moves differently. Whether the difference survives beyond one cell is what
this measures: every margin plays the same matchups on the same battlefields from the same
checkpoint, so the objective is the only thing that varies.

The result does not bind the training reward. What search maximizes and what a policy is trained on
are separate choices, and this exists to keep them separate on purpose rather than by accident.

Usage:
    ./search_objective.py WORKER CHECKPOINT [--margins two_sided hit_points ...]
                          [--episodes 6] [--battlefields 4] [--simulations 16]
                          [--report search_objective.json]
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

from fheroes2_agent.env import REWARD_MARGINS, BattleEnv, ScenarioRejected, _side_won  # noqa: E402
from fheroes2_agent.policy import load_policy  # noqa: E402
from fheroes2_agent.search import search_action  # noqa: E402

# The battery's mirror suite: both sides field the identical army under identical 10/10 commanders,
# so no margin can win by preferring the side the sampler happened to favour.
MIRRORS = ["9:2,11:2,6:12,1:30", "62:3,30:6,15:10", "13:3,48:12,12:20",
           "10:4,7:8", "28:3,40:8,2:15", "51:4,50:4,12:16"]


def play(worker: str, model, army: str, margin: str, side: str, offset: int,
         episodes: int, simulations: int) -> dict:
    """One matchup on one battlefield under one search objective."""
    kwargs = dict(side=side, attacker=army, defender=army, attacker_hero="10:10",
                  defender_hero="10:10", allow_wide=True, seeds=1, seed_offset=offset,
                  reward_margin=margin)
    try:
        env, sim = BattleEnv(worker, **kwargs), BattleEnv(worker, **kwargs)
    except ScenarioRejected:
        return {"wins": [], "lengths": []}
    wins, lengths = [], []
    try:
        for _ in range(episodes):
            observation, mask = env.reset()
            prefix: list[int] = []
            while True:
                action = search_action(sim, model, prefix, observation, mask, simulations, 1.5, live=env)
                prefix.append(action)
                step = env.step(action)
                if step.done:
                    wins.append(bool(_side_won(step.info, side)))
                    lengths.append(len(prefix))
                    break
                observation, mask = step.observation, step.mask
    finally:
        env.close()
        sim.close()
    return {"wins": wins, "lengths": lengths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--margins", nargs="+", default=["two_sided", "balanced", "hit_points", "strength"],
                        choices=REWARD_MARGINS)
    parser.add_argument("--episodes", type=int, default=6, help="per matchup per battlefield")
    parser.add_argument("--battlefields", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--sides", nargs="+", default=["attacker"], choices=("attacker", "defender"))
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = load_policy(state["state_dict"])
    model.eval()

    started = time.time()
    report = {"checkpoint": pathlib.Path(args.checkpoint).name, "episodes": args.episodes,
              "battlefields": args.battlefields, "simulations": args.simulations,
              "armies": MIRRORS, "sides": args.sides, "by_margin": {}}
    for margin in args.margins:
        cells = {}
        for side in args.sides:
            for army in MIRRORS:
                for offset in range(args.battlefields):
                    # The seed is reset per cell so every margin meets the same rollout noise, which
                    # is what makes the comparison paired rather than four independent samples.
                    torch.manual_seed(1000 + offset)
                    result = play(args.worker, model, army, margin, side, offset,
                                  args.episodes, args.simulations)
                    cells[f"{side}|{army}|{offset}"] = result
        rates = [float(np.mean(c["wins"])) for c in cells.values() if c["wins"]]
        report["by_margin"][margin] = {"cells": cells, "mean_rate": float(np.mean(rates)),
                                       "n_cells": len(rates)}
        print(f"search scored by {margin:20s} -> {np.mean(rates):.3f} over {len(rates)} cells "
              f"({time.time() - started:.0f}s elapsed)", flush=True)

    # Paired by cell, which is the comparison that matters: the same armies on the same battlefield
    # under the same rollout noise, differing only in what the rollout return measured.
    base = args.margins[0]
    print(f"\npaired against {base}, cell by cell:")
    for margin in args.margins[1:]:
        keys = [k for k in report["by_margin"][base]["cells"]
                if report["by_margin"][base]["cells"][k]["wins"]
                and report["by_margin"][margin]["cells"].get(k, {}).get("wins")]
        deltas = [float(np.mean(report["by_margin"][margin]["cells"][k]["wins"]))
                  - float(np.mean(report["by_margin"][base]["cells"][k]["wins"])) for k in keys]
        d = np.array(deltas)
        sem = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"  {margin:20s} {d.mean():+.3f}  SE {sem:.3f}  "
              f"({int((d > 0).sum())} up / {int((d < 0).sum())} down / {int((d == 0).sum())} tied)")

    report["seconds"] = round(time.time() - started, 1)
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=1))
    print("SEARCH OBJECTIVE COMPLETE")


if __name__ == "__main__":
    main()
