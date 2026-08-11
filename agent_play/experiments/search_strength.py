#!/usr/bin/env python3
"""How to make root search stronger once it can no longer read the live dice.

The 2026-08-10 leakage ablation removed most of what the corrected search appeared to gain, so the
question is what actually buys strength. Two measurements point at the same suspicion. Decisions
offer about 27 legal moves while the policy puts real weight on about 2.3 of them, and PUCT left
alone visits roughly two candidates per state, so a simulation budget is spent refining a two-way
comparison rather than examining the position. Raising the budget therefore may not help nearly as
much as spending it differently.

Three knobs, measured against each other at a fixed honest configuration:

  simulations       how many playouts per decision. The published ladder measured this with the
                    live dice and needs redoing before any of its rungs are quoted.
  c_puct            the exploration weight. Low values follow the prior and visit almost nothing
                    else; high values spread visits and estimate each candidate worse.
  coverage forcing  visit every candidate once, widest prior first, before PUCT takes over. Built
                    for the soft-target collector, which needs support everywhere, and never
                    measured as a playing rule. With more candidates than budget it degenerates
                    into a prior-ordered sweep with no refinement at all, so it should help at
                    large budgets and hurt at small ones, and that prediction is the point.

Every cell is the mirror suite from both chairs, identical armies and commanders, with the side
environment pinned to the live battlefield and its dice made independent, and the policy seed reset
per cell so all arms meet the same sampling noise. Reports the full column set, not the win rate
alone.

Usage:
    ./search_strength.py WORKER CHECKPOINT [--simulations 8 16 32] [--c-puct 0.5 1.5 4.0]
                         [--coverage both] [--episodes 8] [--report R.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fheroes2_agent.env import REWARD_MARGINS, BattleEnv, _side_won  # noqa: E402
from fheroes2_agent.policy import load_policy  # noqa: E402
from fheroes2_agent.search import policy_action, search_action_detail  # noqa: E402

ARMIES = ["9:2,11:2,6:12,1:30", "62:3,30:6,15:10", "13:3,48:12,12:20",
          "10:4,7:8", "28:3,40:8,2:15", "51:4,50:4,12:16"]
#: Far from any battlefield index, so the perturbed stream cannot coincide with a real one.
COMBAT_OFFSET = 987631


def run_cell(worker, model, army, side, offset, episodes, simulations, c_puct, forced, margin):
    base = dict(side=side, attacker=army, defender=army, attacker_hero="10:10",
                defender_hero="10:10", allow_wide=True, reward_margin=margin)
    env = BattleEnv(worker, seeds=1, seed_offset=offset, **base)
    sim = (BattleEnv(worker, seeds=1, seed_offset=offset, combat_seed_offset=COMBAT_OFFSET, **base)
           if simulations else None)
    rows = []
    try:
        for _ in range(episodes):
            observation, mask = env.reset()
            prefix: list[int] = []
            overruled = 0
            while True:
                if simulations:
                    action, means, visits, prior = search_action_detail(
                        sim, model, prefix, observation, mask, simulations, c_puct,
                        live=env, coverage_forced=forced)
                    # How many distinct candidates the budget actually touched, which is the thing
                    # the knobs are supposed to move.
                    touched = sum(1 for v in visits.values() if v)
                    overruled += int(action != max(prior, key=prior.get))
                else:
                    action = policy_action(model, observation, mask, env=env)
                    touched = 1
                prefix.append(action)
                step = env.step(action)
                if step.done:
                    record = step.info
                    own = record["attacker" if side == "attacker" else "defender"]
                    foe = record["defender" if side == "attacker" else "attacker"]
                    frac = lambda e: e["strength"] / e["initial_strength"] if e["initial_strength"] else 0.0  # noqa: E731
                    rows.append({"won": bool(_side_won(record, side)), "reward": step.reward,
                                 "decisions": len(prefix), "kept": frac(own),
                                 "destroyed": 1.0 - frac(foe), "margin": frac(own) - frac(foe),
                                 "candidates_touched": touched, "overruled": overruled / max(len(prefix), 1)})
                    break
                observation, mask = step.observation, step.mask
    finally:
        env.close()
        if sim is not None:
            sim.close()
    return rows


def summarize(rows):
    if not rows:
        return {}
    won = [r for r in rows if r["won"]]
    lost = [r for r in rows if not r["won"]]
    mean = lambda xs, k: float(np.mean([x[k] for x in xs])) if xs else float("nan")  # noqa: E731
    return {"n": len(rows), "win_rate": len(won) / len(rows), "mean_reward": mean(rows, "reward"),
            "win_quality": mean(won, "kept"), "loss_quality": mean(lost, "destroyed"),
            "margin": mean(rows, "margin"), "decisions": mean(rows, "decisions"),
            "candidates_touched": mean(rows, "candidates_touched"), "overruled": mean(rows, "overruled")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--simulations", type=int, nargs="+", default=[0, 8, 16, 32])
    parser.add_argument("--c-puct", type=float, nargs="+", default=[1.5])
    parser.add_argument("--coverage", default="off", choices=("off", "on", "both"))
    parser.add_argument("--episodes", type=int, default=8, help="per army per side, over the battlefields")
    parser.add_argument("--battlefields", type=int, default=4)
    parser.add_argument("--sides", nargs="+", default=["attacker", "defender"])
    parser.add_argument("--margin", default="hit_points", choices=REWARD_MARGINS)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    model = load_policy(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()
    forcings = {"off": [False], "on": [True], "both": [False, True]}[args.coverage]

    started = time.time()
    print(f"{'sims':>5s}{'c_puct':>8s}{'forced':>8s} | {'rate':>6s}{'wq':>6s}{'lq':>6s}{'mg':>7s}"
          f"{'rw':>7s}{'dec':>6s}{'cands':>7s}{'ovr':>6s}   n")
    results = {}
    for sims, c_puct, forced in itertools.product(args.simulations, args.c_puct, forcings):
        # The knobs only exist above zero simulations; skip the redundant repeats of policy-only play.
        if sims == 0 and (c_puct != args.c_puct[0] or forced):
            continue
        rows = []
        for side in args.sides:
            for army in ARMIES:
                for offset in range(args.battlefields):
                    torch.manual_seed(9000 + offset)
                    rows += run_cell(args.worker, model, army, side, offset,
                                     max(args.episodes // args.battlefields, 1),
                                     sims, c_puct, forced, args.margin)
        s = summarize(rows)
        key = f"{sims}|{c_puct}|{int(forced)}"
        results[key] = s
        print(f"{sims:5d}{c_puct:8.1f}{str(forced):>8s} | {s['win_rate']:6.3f}{s['win_quality']:6.2f}"
              f"{s['loss_quality']:6.2f}{s['margin']:+7.2f}{s['mean_reward']:+7.2f}{s['decisions']:6.1f}"
              f"{s['candidates_touched']:7.1f}{s['overruled']:6.2f}   {s['n']}"
              f"   ({time.time() - started:.0f}s)", flush=True)

    report = {"checkpoint": pathlib.Path(args.checkpoint).name, "episodes": args.episodes,
              "battlefields": args.battlefields, "margin": args.margin, "sides": args.sides,
              "combat_seed_offset": COMBAT_OFFSET, "armies": ARMIES, "cells": results,
              "seconds": round(time.time() - started, 1)}
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=1))
    print("SEARCH STRENGTH COMPLETE")


if __name__ == "__main__":
    main()
