#!/usr/bin/env python3
"""The searching agent on the battery's own suites, with the quality columns.

Root search over the cloned prior is the only mechanism here measured above the built-in AI, but
that was only ever measured on the held-out pool. This runs it on any subset of the battery's
suites, from either chair, reporting the same columns policies report, so the agent regime can be
read against the engine on the same scoreboard rather than on one number.

The question it exists to answer: the 2026-08-08 scoreboard leaves three suites at par and the
gap concentrated in the held-out pool and the two mirror chairs. If search crosses the engine on
the mirror chairs too, the distillation target is well defined everywhere and the remaining work
is transfer into weights. If search does not, the mirror deficit is not a policy-quality problem
at all, and no amount of distillation will move it.

Search costs roughly half a second per decision, so budget episodes accordingly; the defaults
here are deliberately smaller than the battery's, and the report records them.

Usage:
    ./search_agent_battery.py WORKER CHECKPOINT [--suites held_out_pool mirrors_defender ...]
                              [--episodes 8] [--simulations 32] [--report R.json]
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

from fheroes2_agent.env import BattleEnv, ScenarioRejected  # noqa: E402
from fheroes2_agent.policy import load_policy  # noqa: E402
from search_probe import policy_action, search_action  # noqa: E402
from validation_battery import SUITE_SIDE, build_suites  # noqa: E402

DEFAULT_SUITES = ("held_out_pool", "mirrors_attacker", "mirrors_defender", "held_out_as_defender")


def play_episodes(worker: str, matchup, side: str, model, episodes: int, simulations: int,
                  c_puct: float, seeds: int, searched: bool) -> dict:
    """One matchup, measured with the same columns `scenarios.measure` reports for policies."""
    kwargs = dict(side=side, attacker=matchup.attacker, defender=matchup.defender,
                  attacker_hero=matchup.attacker_hero, defender_hero=matchup.defender_hero,
                  allow_wide=matchup.allow_wide, seeds=seeds, reward_margin="two_sided")
    env = BattleEnv(worker, **kwargs)
    sim = BattleEnv(worker, **kwargs) if searched else None
    wins, rewards, lengths, survival, damage, margins = [], [], [], [], [], []
    won_termination = "victory" if side == "attacker" else "defeat"
    try:
        for _ in range(episodes):
            observation, mask = env.reset()
            prefix, steps = [], 0
            while True:
                if searched:
                    action = search_action(sim, model, prefix, observation, mask, simulations, c_puct, live=env)
                else:
                    action = policy_action(model, observation, mask, env=env)
                prefix.append(action)
                step = env.step(action)
                steps += 1
                if step.done:
                    record = step.info
                    won = record["termination"] == won_termination
                    own = record["attacker" if side == "attacker" else "defender"]
                    foe = record["defender" if side == "attacker" else "attacker"]
                    own_initial = float(own.get("initial_strength", 0.0))
                    foe_initial = float(foe.get("initial_strength", 0.0))
                    own_kept = float(own.get("strength", 0.0)) / own_initial if own_initial > 0 else 0.0
                    foe_kept = float(foe.get("strength", 0.0)) / foe_initial if foe_initial > 0 else 0.0
                    wins.append(won)
                    rewards.append(step.reward)
                    lengths.append(steps)
                    margins.append(own_kept - foe_kept)
                    (survival if won else damage).append(own_kept if won else 1.0 - foe_kept)
                    break
                observation, mask = step.observation, step.mask
    finally:
        env.close()
        if sim is not None:
            sim.close()
    return {"win_rate": float(np.mean(wins)), "mean_reward": float(np.mean(rewards)),
            "mean_length": float(np.mean(lengths)), "strength_margin": float(np.mean(margins)),
            "surviving_strength": float(np.mean(survival)) if survival else None,
            "loss_damage": float(np.mean(damage)) if damage else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--fresh", type=int, default=24)
    parser.add_argument("--baseline", action="store_true",
                        help="also measure the same policy without search, the paired control")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    model = load_policy(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()
    suites = build_suites(args.fresh)
    started = time.time()
    report = {"checkpoint": args.checkpoint, "episodes": args.episodes, "eval_seeds": args.seeds,
              "simulations": args.simulations, "reward_margin": "two_sided", "arms": {}}

    arms = [("search", True)] + ([("policy", False)] if args.baseline else [])
    for arm, searched in arms:
        report["arms"][arm] = {}
        for suite in args.suites:
            side = SUITE_SIDE.get(suite, "attacker")
            measured = []
            for m in suites[suite]:
                try:
                    measured.append(play_episodes(args.worker, m, side, model, args.episodes,
                                                  args.simulations, args.c_puct, args.seeds, searched))
                except ScenarioRejected as error:
                    print(f"  {suite}: matchup rejected ({str(error)[:70]})", flush=True)
            if not measured:
                continue
            def column(key):
                vals = [d[key] for d in measured if isinstance(d.get(key), (int, float))]
                return float(np.mean(vals)) if vals else float("nan")
            report["arms"][arm][suite] = {"per_matchup": measured, "win_rate": column("win_rate"),
                                          "mean_reward": column("mean_reward"),
                                          "surviving_strength": column("surviving_strength"),
                                          "loss_damage": column("loss_damage"),
                                          "strength_margin": column("strength_margin"),
                                          "mean_length": column("mean_length")}
            r = report["arms"][arm][suite]
            print(f"{arm:7s} {suite:22s} rate {r['win_rate']:.3f}  wq {r['surviving_strength']:.2f} "
                  f"lq {r['loss_damage']:.2f} mg {r['strength_margin']:+.2f} rw {r['mean_reward']:+.2f} "
                  f"len {r['mean_length']:.0f}", flush=True)

    report["seconds"] = round(time.time() - started, 1)
    print(f"\ntotal {report['seconds']:.0f}s")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
