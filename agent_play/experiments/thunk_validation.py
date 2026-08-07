#!/usr/bin/env python3
"""The Thunk opening fight as a standing validation ladder.

One real map fight, dump-verified: Corribus, attack 13 defense 12, with 1 Crusader, 1 Crusader,
1 Crusader, 2 Paladins and 2 Champions, against the neutral Peasant stack at counts up to its
rolled 1,000, split the way the engine splits a neutral stack. It stays a validation point rather
than a training target because it is a configuration no sampler drew: it has exposed the missing
commander support, the wide-creature exclusion, and the wide-attacker melee gap, each of which
every synthetic pool of its day had missed.

Run it against any checkpoint after any change to the encoding, the demonstrations, or the
samplers, and compare ladders across checkpoints rather than trusting any single rung.

Usage:
    ./thunk_validation.py WORKER CHECKPOINT [CHECKPOINT ...] [--episodes 24]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.policy import load_policy, BattlePolicy  # noqa: E402
from fheroes2_agent.scenarios import Matchup, measure  # noqa: E402

ARMY = "11:1,11:1,11:1,10:2,9:2"
HERO = "13:12"
LADDER = (500, 700, 850, 1000)


def split(total: int) -> str:
    first = total // 3 + (1 if total % 3 else 0)
    return f"1:{first},1:{total // 3},1:{total - first - total // 3}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    started = time.time()
    results = []
    print(f"{'checkpoint':32s} " + "  ".join(f"{n:>6d}" for n in LADDER))
    for path in args.checkpoints:
        model = load_policy(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
        model.eval()
        row = []
        for n in LADDER:
            r = measure(model, args.worker, Matchup(ARMY, split(n), attacker_hero=HERO, allow_wide=True),
                        episodes=args.episodes)
            row.append(r["win_rate"])
        name = pathlib.Path(path).name
        results.append({"checkpoint": name, "ladder": dict(zip(map(str, LADDER), row))})
        print(f"{name:32s} " + "  ".join(f"{w:6.3f}" for w in row), flush=True)

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"army": ARMY, "hero": HERO, "ladder": LADDER, "episodes": args.episodes,
             "results": results, "seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
