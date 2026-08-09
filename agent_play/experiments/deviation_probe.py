#!/usr/bin/env python3
"""Does search disagree with the policy where the policy loses, or only where it already wins?

The distillation program rests on a statistic that was never measured where it matters. Search
agrees with the prior's argmax on about 96 percent of decisions, which is usually read as "search
mostly confirms the policy, so there is little to clone". That number comes from win-filtered
collection: every corpus in the record was gathered on matchups search wins, which are largely
matchups the prior already wins. The decisive positions are the opposite ones, a handful of
held-out matchups the policy loses badly and search wins outright, and no corpus has ever measured
agreement there.

This probe measures it directly, on matchups split by whether the policy wins them, with
coverage-forced search so every candidate is priced by a real rollout. The reading is binary and
it decides where the program goes. If deviation on losing matchups is as low as on winning ones,
action choice does not explain the gap: the deficit is in value or in representation, and no
better labeling of an argmax the policy already makes will transfer it. If deviation is much
higher, there is an action-level signal that no corpus has ever contained, and collection should
be aimed at these positions rather than at the ones search wins comfortably.

Usage:
    ./deviation_probe.py WORKER CHECKPOINT --losing 16 9 11 --winning 5 7 13
                         [--episodes 4] [--simulations 48] [--report R.json]
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
from search_probe import search_action_detail  # noqa: E402
from validation_battery import build_suites  # noqa: E402


def probe_matchup(worker: str, matchup, model, episodes: int, simulations: int, c_puct: float,
                  seeds: int, coverage_forced: bool = True) -> dict:
    """Per-decision agreement between coverage-forced search and the policy's greedy argmax."""
    kwargs = dict(side="attacker", attacker=matchup.attacker, defender=matchup.defender,
                  attacker_hero=matchup.attacker_hero, defender_hero=matchup.defender_hero,
                  allow_wide=matchup.allow_wide, seeds=seeds, reward_margin="two_sided")
    env = BattleEnv(worker, **kwargs)
    sim = BattleEnv(worker, **kwargs)
    agree = decisions = wins = played = 0
    value_gaps = []
    try:
        for _ in range(episodes):
            observation, mask = env.reset()
            prefix = []
            while True:
                action, means, visits, prior = search_action_detail(
                    sim, model, prefix, observation, mask, simulations, c_puct, live=env,
                    coverage_forced=coverage_forced)
                greedy = int(np.argmax(np.where(mask, prior_array(prior, mask), -np.inf)))
                decisions += 1
                agree += int(action == greedy)
                if action != greedy and greedy in means:
                    # What the disagreement is worth, in the reward units search measured.
                    value_gaps.append(means[action] - means[greedy])
                # The searched action is played, so the states visited are search's own.
                prefix.append(action)
                step = env.step(action)
                if step.done:
                    wins += step.info["termination"] == "victory"
                    played += 1
                    break
                observation, mask = step.observation, step.mask
    finally:
        env.close()
        sim.close()
    return {"decisions": decisions, "agreement": agree / decisions if decisions else float("nan"),
            "search_win_rate": wins / played if played else float("nan"),
            "mean_value_gap_on_disagreement": float(np.mean(value_gaps)) if value_gaps else 0.0,
            "disagreements": len(value_gaps)}


def prior_array(prior: dict, mask) -> np.ndarray:
    out = np.zeros(len(mask), dtype=np.float64)
    for action, probability in prior.items():
        out[action] = probability
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--suite", default="held_out_pool")
    parser.add_argument("--losing", type=int, nargs="+", required=True, help="matchup indices the policy loses")
    parser.add_argument("--winning", type=int, nargs="+", required=True, help="matchup indices it wins")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--modes", nargs="+", default=["forced"], choices=("forced", "ucb"),
                        help="forced sweeps every candidate once; ucb is the concentrating search the\n"
                             "historical agreement statistic was measured under")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    model = load_policy(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()
    matchups = build_suites(24)[args.suite]
    started = time.time()
    report = {"checkpoint": args.checkpoint, "suite": args.suite, "simulations": args.simulations,
              "episodes": args.episodes, "modes": args.modes, "groups": {}}

    for mode in args.modes:
      forced = mode == "forced"
      for group, indices in (("losing", args.losing), ("winning", args.winning)):
        rows = []
        for index in indices:
            try:
                row = probe_matchup(args.worker, matchups[index], model, args.episodes,
                                    args.simulations, args.c_puct, args.seeds, coverage_forced=forced)
            except ScenarioRejected as error:
                print(f"  matchup {index} rejected ({str(error)[:60]})", flush=True)
                continue
            row["index"] = index
            rows.append(row)
            print(f"[{mode}] {group:8s} matchup {index:2d}: agreement {row['agreement']:.3f} over "
                  f"{row['decisions']:4d} decisions, search wins {row['search_win_rate']:.2f}, "
                  f"mean value gained per disagreement {row['mean_value_gap_on_disagreement']:+.3f}",
                  flush=True)
        if rows:
            total = sum(r["decisions"] for r in rows)
            weighted = sum(r["agreement"] * r["decisions"] for r in rows) / total
            report["groups"][f"{mode}:{group}"] = {"per_matchup": rows, "decisions": total,
                                       "agreement": weighted,
                                       "deviation_rate": 1.0 - weighted}
            print(f"== [{mode}] {group}: agreement {weighted:.4f}, deviation {1 - weighted:.4f} "
                  f"over {total} decisions", flush=True)

    for mode in args.modes:
        lose, win = report["groups"].get(f"{mode}:losing"), report["groups"].get(f"{mode}:winning")
        if lose and win:
            ratio = lose["deviation_rate"] / max(win["deviation_rate"], 1e-9)
            report[f"deviation_ratio_{mode}"] = ratio
            print(f"[{mode}] deviation ratio losing/winning: {ratio:.2f}")
    report["seconds"] = round(time.time() - started, 1)
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
