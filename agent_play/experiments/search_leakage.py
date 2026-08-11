#!/usr/bin/env python3
"""Is the corrected search planning, or is it reading the live battle's dice?

The 2026-08-10 result, the searching agent at or above the built-in AI on all nine suites, rests on
pinning the search side environment to the live battle's world seed. That was a correctness fix:
`rollout` replays the action prefix into the side environment and the replay only reproduces the
live position when both are on the same battlefield. But the fix buys more than terrain.
`agent_battle_runner.cpp` computes `combatSeed` from the tile index, the map seed and the two
armies, all of which the world seed fixes, so a pinned side environment inherits the live battle's
exact random stream as well. Candidate values are then drawn under the dice that will actually be
rolled rather than under the distribution they are drawn from, which is oracle access rather than
planning, and a headline resting on it would not be a fair comparison against the engine.

Three arms separate the two effects. Every arm plays the same matchups on the same battlefields with
the same policy and the same simulation budget, and only the side environment's seeding changes.

  shared      the side environment matches the live world seed, so terrain and dice both match.
              This is what the corrected harness does today.
  terrain     the side environment matches the live world seed but carries a combat-seed offset, so
              the battlefield still matches and the dice are independent. This is what an honest
              perfect-dynamics model with unknown randomness looks like.
  desynced    the side environment is left on variant zero while the live episode rotates, which is
              the defect fixed on 2026-08-09, kept here as the floor.

If `terrain` tracks `shared`, the gain is the battlefield and the result stands. If `terrain`
collapses toward `desynced`, the result is an artifact of common random numbers and every search
figure needs restating. The middle arm is the whole point; the other two are its bounds.

Reports the full column set rather than the win rate alone, per the standing measurement contract.

Usage:
    ./search_leakage.py WORKER CHECKPOINT [--armies A B ...] [--episodes 8] [--battlefields 4]
                        [--simulations 16] [--margin hit_points] [--report R.json]
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

from fheroes2_agent.env import REWARD_MARGINS, BattleEnv, _side_won  # noqa: E402
from fheroes2_agent.policy import load_policy  # noqa: E402
from fheroes2_agent.search import search_action  # noqa: E402

# The battery's mirror suite: identical armies and commanders on both sides, so no arm can win by
# preferring the side the sampler favoured.
ARMIES = ["9:2,11:2,6:12,1:30", "62:3,30:6,15:10", "13:3,48:12,12:20",
          "10:4,7:8", "28:3,40:8,2:15", "51:4,50:4,12:16"]
ARMS = ("shared", "terrain", "desynced")
#: Chosen to be far from any battlefield index so a perturbed stream cannot coincide with a real one.
COMBAT_OFFSET = 987631


def episode(env, sim, model, simulations, side):
    observation, mask = env.reset()
    prefix: list[int] = []
    while True:
        action = search_action(sim, model, prefix, observation, mask, simulations, 1.5, live=env)
        prefix.append(action)
        step = env.step(action)
        if step.done:
            record = step.info
            own = record["attacker" if side == "attacker" else "defender"]
            foe = record["defender" if side == "attacker" else "attacker"]
            kept = own["strength"] / own["initial_strength"] if own["initial_strength"] else 0.0
            destroyed = 1.0 - (foe["strength"] / foe["initial_strength"] if foe["initial_strength"] else 0.0)
            won = bool(_side_won(record, side))
            return {"won": won, "reward": step.reward, "decisions": len(prefix),
                    "rounds": record.get("rounds"), "kept": kept, "destroyed": destroyed,
                    "margin": kept - (foe["strength"] / foe["initial_strength"] if foe["initial_strength"] else 0.0)}
        observation, mask = step.observation, step.mask


def play(worker, model, army, arm, side, episodes, battlefields, simulations, margin):
    """Every arm sees the same battlefields; only the side environment's seeding differs."""
    base = dict(side=side, attacker=army, defender=army, attacker_hero="10:10",
                defender_hero="10:10", allow_wide=True, reward_margin=margin)
    out = []
    per_battlefield = max(episodes // battlefields, 1)
    for offset in range(battlefields):
        env = BattleEnv(worker, seeds=1, seed_offset=offset, **base)
        if arm == "shared":
            sim = BattleEnv(worker, seeds=1, seed_offset=offset, **base)
        elif arm == "terrain":
            sim = BattleEnv(worker, seeds=1, seed_offset=offset, combat_seed_offset=COMBAT_OFFSET, **base)
        else:
            sim = BattleEnv(worker, seeds=1, seed_offset=0, **base)
        try:
            for _ in range(per_battlefield):
                out.append(episode(env, sim, model, simulations, side))
        finally:
            env.close()
            sim.close()
    return out


def summarize(rows):
    if not rows:
        return {}
    won = [r for r in rows if r["won"]]
    lost = [r for r in rows if not r["won"]]
    mean = lambda xs, k: float(np.mean([x[k] for x in xs])) if xs else float("nan")  # noqa: E731
    return {"n": len(rows), "win_rate": len(won) / len(rows), "mean_reward": mean(rows, "reward"),
            "win_quality": mean(won, "kept"), "loss_quality": mean(lost, "destroyed"),
            "margin": mean(rows, "margin"), "decisions": mean(rows, "decisions")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--armies", nargs="+", default=ARMIES)
    parser.add_argument("--sides", nargs="+", default=["attacker", "defender"], choices=("attacker", "defender"))
    parser.add_argument("--episodes", type=int, default=8, help="per army per side, spread over the battlefields")
    parser.add_argument("--battlefields", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--margin", default="hit_points", choices=REWARD_MARGINS)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    model = load_policy(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()

    started = time.time()
    cells: dict[str, dict[str, list]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for side in args.sides:
            for army in args.armies:
                # Reset per cell so every arm meets identical policy sampling noise, which is what
                # makes this paired rather than three independent samples.
                torch.manual_seed(4242)
                cells[arm][f"{side}|{army}"] = play(args.worker, model, army, arm, side, args.episodes,
                                                    args.battlefields, args.simulations, args.margin)
        flat = [r for rows in cells[arm].values() for r in rows]
        s = summarize(flat)
        print(f"{arm:10s} rate {s['win_rate']:.3f}  wq {s['win_quality']:.2f}  lq {s['loss_quality']:.2f}  "
              f"mg {s['margin']:+.2f}  rw {s['mean_reward']:+.2f}  dec {s['decisions']:.1f}  "
              f"n {s['n']}  ({time.time() - started:.0f}s)", flush=True)

    print("\npaired by cell against the shared-dice arm:")
    report_cells = {}
    for arm in ARMS[1:]:
        keys = [k for k in cells["shared"] if cells["shared"][k] and cells[arm].get(k)]
        deltas = np.array([np.mean([r["won"] for r in cells[arm][k]])
                           - np.mean([r["won"] for r in cells["shared"][k]]) for k in keys])
        sem = deltas.std(ddof=1) / np.sqrt(len(deltas)) if len(deltas) > 1 else float("nan")
        print(f"  {arm:10s} {deltas.mean():+.3f}  SE {sem:.3f}  "
              f"({int((deltas > 0).sum())} up / {int((deltas < 0).sum())} down / {int((deltas == 0).sum())} tied)")
        report_cells[arm] = {"mean_delta": float(deltas.mean()), "sem": float(sem)}

    report = {"checkpoint": pathlib.Path(args.checkpoint).name, "episodes": args.episodes,
              "battlefields": args.battlefields, "simulations": args.simulations, "margin": args.margin,
              "combat_seed_offset": COMBAT_OFFSET, "armies": args.armies, "sides": args.sides,
              "summary": {arm: summarize([r for rows in cells[arm].values() for r in rows]) for arm in ARMS},
              "paired_vs_shared": report_cells,
              "cells": {arm: {k: summarize(v) for k, v in cells[arm].items()} for arm in ARMS},
              "seconds": round(time.time() - started, 1)}
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=1))
    print("SEARCH LEAKAGE COMPLETE")


if __name__ == "__main__":
    main()
