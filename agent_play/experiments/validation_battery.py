#!/usr/bin/env python3
"""The agent's report card: fresh samples, out-of-distribution stress, and the Thunk ladder.

Three validation surfaces the owner fixed. Freshly sampled matchups come straight from the
value-budget generator with a seed no training or calibration ever used, uncalibrated on
purpose, so the number is about the raw distribution rather than a band any clone selected.
Stress suites probe out of distribution: horde counts beyond anything recorded, armies of only
wide creatures, commander stat extremes, and Thunk rungs beyond the rolled fight. The Thunk
ladder itself stays the standing validation no sampler drew and no training touched.

Every matchup is measured over battlefields for every checkpoint given, so the table reads as
paired columns.

Usage:
    ./validation_battery.py WORKER CHECKPOINT [CHECKPOINT ...] [--episodes 24] [--eval-seeds 4]
                            [--fresh 24] [--report validation_battery.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.policy import BattlePolicy  # noqa: E402
from fheroes2_agent.scenarios import Matchup, measure, sample_budget_matchup  # noqa: E402

POOL = pathlib.Path(__file__).resolve().parents[2] / "agent_play" / "docs" / "archive" / "experiments" / "files" \
    / "2026-08-05-run-reports" / "pool_value.json"

THUNK_ARMY = "11:1,11:1,11:1,10:2,9:2"
THUNK_HERO = "13:12"


def thunk_split(total: int) -> str:
    first = total // 3 + (1 if total % 3 else 0)
    return f"1:{first},1:{total // 3},1:{total - first - total // 3}"


def build_suites(fresh_count: int) -> dict[str, list[Matchup]]:
    suites: dict[str, list[Matchup]] = {}

    # 1. Fresh samples: the generator's raw distribution, seed never used anywhere else.
    rng = random.Random(20260805)
    suites["fresh_sampled"] = [sample_budget_matchup(rng) for _ in range(fresh_count)]

    # 2. Held-out pool, the split every result today reported.
    entries = json.loads(POOL.read_text())["matchups"]
    suites["held_out_pool"] = [Matchup(e["attacker"], e["defender"], attacker_hero=e.get("attacker_hero"),
                                       defender_hero=e.get("defender_hero"), allow_wide=bool(e.get("allow_wide")))
                               for e in entries[40:60]]

    # 3. Stress: hordes beyond every recorded count (training and its supplement stop at 1,000).
    suites["stress_hordes"] = [
        Matchup(THUNK_ARMY, thunk_split(total), attacker_hero=THUNK_HERO, allow_wide=True)
        for total in (1500, 2000, 3000)
    ] + [
        Matchup("9:4,10:6,6:12", thunk_split(total), attacker_hero="10:10", allow_wide=True)
        for total in (1500, 2500)
    ]

    # 4. Stress: armies of only two-cell creatures (the whole wide_v1 roster: Cavalry, Champion,
    # Wolf, Unicorn, Centaur, Boar, Nomad, Medusa), a composition the samplers rarely draw.
    wide_armies = [("9:3,28:4,62:3", "8:6,59:5,15:8"), ("15:12,30:9,40:7", "9:2,62:4,28:3"), ("8:8,9:2", "40:10,30:8,15:6")]
    suites["stress_wide_only"] = [Matchup(a, d, attacker_hero="8:8", defender_hero="8:8", allow_wide=True)
                                  for a, d in wide_armies]

    # 5. Stress: commander extremes on one mid pool matchup, stats far outside the sampled range.
    base = entries[45]
    suites["stress_commanders"] = [
        Matchup(base["attacker"], base["defender"], attacker_hero=hero, defender_hero=base.get("defender_hero"),
                allow_wide=bool(base.get("allow_wide")))
        for hero in ("0:0", "30:30", "99:0", "0:99")
    ]

    # 6. The standing ladder, untouched by every training set, plus rungs beyond the real fight.
    suites["thunk_ladder"] = [Matchup(THUNK_ARMY, thunk_split(total), attacker_hero=THUNK_HERO, allow_wide=True)
                              for total in (500, 700, 850, 1000)]
    return suites


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--eval-seeds", type=int, default=4)
    parser.add_argument("--fresh", type=int, default=24)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    suites = build_suites(args.fresh)
    started = time.time()
    report = {"suites": {name: {"matchups": len(ms)} for name, ms in suites.items()},
              "episodes": args.episodes, "eval_seeds": args.eval_seeds, "results": {}}

    for path in args.checkpoints:
        name = pathlib.Path(path).name
        model = BattlePolicy()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
        model.eval()
        report["results"][name] = {}
        for suite, matchups in suites.items():
            rates = [measure(model, args.worker, m, episodes=args.episodes, seeds=args.eval_seeds)["win_rate"]
                     for m in matchups]
            report["results"][name][suite] = rates
            print(f"{name:24s} {suite:18s} mean {np.mean(rates):.3f}  " +
                  " ".join(f"{r:.2f}" for r in rates[:8]) + (" ..." if len(rates) > 8 else ""), flush=True)

    print(f"\ntotal {round(time.time() - started)}s")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
