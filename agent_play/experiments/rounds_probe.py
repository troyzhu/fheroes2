#!/usr/bin/env python3
"""Round-count histograms and stalemate incidence per policy, on both matchup sets.

The owner's double-check of the round semantics: one engine round is one `arena.Turns()` call,
every unit on both sides acting if it can, and the forty is not a cap but a sliding window,
forty consecutive rounds without a death tripping the `stalemate` termination while `maxRounds`
truncates at 100 as `round_limit`. This probe plays each checkpoint over the training and
held-out matchup slices of the standard pool, capturing the terminal record's exact `rounds`
and `termination` per episode, and reports the histogram, the mean and maximum, and every
termination count, so "does anything ever stall" is a measured number per policy rather than
an impression.

Usage:
    ./rounds_probe.py WORKER CHECKPOINT [CHECKPOINT ...] [--episodes 4] [--report R.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.env import BattleEnv  # noqa: E402
from fheroes2_agent.policy import load_policy  # noqa: E402

POOL_FILE = pathlib.Path(__file__).resolve().parents[1] / "docs" / "archive" / "experiments" / "files" \
    / "2026-08-05-run-reports" / "pool_value.json"
BUCKETS = ((5, "1-5"), (10, "6-10"), (20, "11-20"), (40, "21-40"), (99, "41-99"), (10**9, "100"))


def kwargs_of(entry: dict) -> dict:
    return dict(attacker=entry["attacker"], defender=entry["defender"],
                attacker_hero=entry.get("attacker_hero"), defender_hero=entry.get("defender_hero"),
                allow_wide=bool(entry.get("allow_wide")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--episodes", type=int, default=4, help="per matchup, over 3 world seeds")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    pool = json.loads(POOL_FILE.read_text())["matchups"]
    sets = {"training12": [kwargs_of(e) for e in pool[:12]],
            "held_out10": [kwargs_of(e) for e in pool[40:50]]}

    report = {}
    for path in args.checkpoints:
        name = pathlib.Path(path).stem
        model = load_policy(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
        model.eval()
        report[name] = {}
        for set_name, matchups in sets.items():
            rounds_list: list[int] = []
            terminations: collections.Counter = collections.Counter()
            for kw in matchups:
                env = BattleEnv(args.worker, side="attacker", seeds=3, **kw)
                try:
                    for _ in range(args.episodes):
                        obs, mask = env.reset()
                        while True:
                            with torch.no_grad():
                                logits, _ = model(torch.from_numpy(obs).unsqueeze(0),
                                                  torch.from_numpy(mask).unsqueeze(0))
                            step = env.step(int(torch.distributions.Categorical(logits=logits).sample()))
                            if step.done:
                                rounds_list.append(int(step.info["rounds"]))
                                terminations[step.info["termination"]] += 1
                                break
                            obs, mask = step.observation, step.mask
                finally:
                    env.close()
            histogram: collections.Counter = collections.Counter()
            for r in rounds_list:
                histogram[next(label for limit, label in BUCKETS if r <= limit)] += 1
            report[name][set_name] = {
                "episodes": len(rounds_list), "terminations": dict(terminations),
                "rounds_histogram": dict(histogram), "rounds_max": max(rounds_list),
                "rounds_mean": round(sum(rounds_list) / len(rounds_list), 2)}
            print(f"{name:20s} {set_name:11s} mean {report[name][set_name]['rounds_mean']:6.2f} "
                  f"max {report[name][set_name]['rounds_max']:3d} terms {dict(terminations)}", flush=True)

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
