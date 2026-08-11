#!/usr/bin/env python3
"""Does the search side environment have to be on the same battlefield as the real battle?

Found 2026-08-09. `rollout` in `search_probe.py` replays the action prefix in a side environment
and relies on that replay reproducing the live state exactly, which is only true when both are on
the same world seed, since the obstacle layout and the combat seed both derive from it. Every
search harness built its side environment with the live environment's `seeds`, and a reset while a
decision is pending forces a respawn back to the first scenario, so the side environment sat on
variant zero while the live episode rotated over four. Three episodes in four were therefore
searched against terrain the battle was not being fought on, and the prefix guarantee the whole
method rests on did not hold.

An earlier single-cell check reported this as costing nothing, on six episodes of one matchup under
one objective. That was underpowered and the conclusion was withdrawn. This varies the matchup and
the objective and holds everything else fixed, including the number of battlefields each arm sees.

Usage:
    ./search_sync.py WORKER CHECKPOINT [--armies A B ...] [--margins hit_points two_sided]
                     [--episodes 8] [--simulations 16] [--report R.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fheroes2_agent.env import REWARD_MARGINS, BattleEnv, _side_won  # noqa: E402
from fheroes2_agent.policy import load_policy  # noqa: E402
from fheroes2_agent.search import search_action  # noqa: E402

ARMIES = ["10:4,7:8", "62:3,30:6,15:10", "13:3,48:12,12:20"]


def play(worker, model, army, margin, synced, episodes, simulations, battlefields=4):
    """Both arms see the same battlefields; only whether the side environment follows differs."""
    kwargs = dict(side="attacker", attacker=army, defender=army, attacker_hero="10:10",
                  defender_hero="10:10", allow_wide=True, reward_margin=margin)
    wins = []
    if synced:
        for offset in range(battlefields):
            env = BattleEnv(worker, seeds=1, seed_offset=offset, **kwargs)
            sim = BattleEnv(worker, seeds=1, seed_offset=offset, **kwargs)
            try:
                wins += _episodes(env, sim, model, episodes // battlefields, simulations)
            finally:
                env.close()
                sim.close()
    else:
        env = BattleEnv(worker, seeds=battlefields, **kwargs)
        sim = BattleEnv(worker, seeds=battlefields, **kwargs)
        try:
            wins += _episodes(env, sim, model, episodes, simulations)
        finally:
            env.close()
            sim.close()
    return float(np.mean(wins)) if wins else float("nan")


def _episodes(env, sim, model, count, simulations):
    out = []
    for _ in range(count):
        observation, mask = env.reset()
        prefix: list[int] = []
        while True:
            action = search_action(sim, model, prefix, observation, mask, simulations, 1.5, live=env)
            prefix.append(action)
            step = env.step(action)
            if step.done:
                out.append(bool(_side_won(step.info, "attacker")))
                break
            observation, mask = step.observation, step.mask
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--armies", nargs="+", default=ARMIES)
    parser.add_argument("--margins", nargs="+", default=["hit_points", "two_sided"], choices=REWARD_MARGINS)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    model = load_policy(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()

    rows, deltas = [], []
    print(f"{'army':24s}{'objective':12s}{'synced':>9s}{'desynced':>10s}{'cost':>8s}")
    for army in args.armies:
        for margin in args.margins:
            torch.manual_seed(7)
            synced = play(args.worker, model, army, margin, True, args.episodes, args.simulations)
            torch.manual_seed(7)
            desynced = play(args.worker, model, army, margin, False, args.episodes, args.simulations)
            rows.append({"army": army, "margin": margin, "synced": synced, "desynced": desynced})
            deltas.append(desynced - synced)
            print(f"{army:24s}{margin:12s}{synced:9.2f}{desynced:10.2f}{desynced - synced:+8.2f}")

    d = np.array(deltas)
    print(f"\ndesyncing the side environment costs {d.mean():+.3f} on average over {len(d)} cells, "
          f"{int((d < 0).sum())} negative and {int((d > 0).sum())} positive")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"episodes": args.episodes, "simulations": args.simulations, "cells": rows,
             "mean_cost_of_desync": float(d.mean())}, indent=1))


if __name__ == "__main__":
    main()
