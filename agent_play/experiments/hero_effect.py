#!/usr/bin/env python3
"""How much does a hero move a battle the army budget calls balanced?

The owner asked on 2026-08-09 whether army-strength pricing accounts for hero stats. It does not:
`agent_capabilities.cpp` records `Monster::GetMonsterStrength()` at base stats, and the budget
sampler in `scenarios.py` matches side budgets from that table and only then assigns a hero to
each side independently on a coin flip, with attack drawn from 0 to 25 and defense from 0 to 20.
A matchup the budget calls even can therefore carry a large commander on one side and none on the
other, and nothing in the pricing knows.

This measures the size of that hole with the engine playing both sides, so the number isolates the
hero rather than any policy's response to it: identical armies, identical battlefields, one side's
commander varied. Whatever it costs, it is paid by every consumer of the pricing, which is the
calibrated difficulty pools, the contested-band screening, and the difficulty weighting on the
reward.

Usage:
    ./hero_effect.py WORKER [--army "62:3,30:6,15:10"] [--episodes 24] [--report R.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from builtin_ai_baseline import ai_win_rate  # noqa: E402
from fheroes2_agent.scenarios import Matchup  # noqa: E402

# The sampler's own range, plus the midpoint, so the table covers what it can actually draw.
LADDER = [None, "0:0", "5:5", "10:10", "15:15", "20:20", "25:20"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("--army", default="62:3,30:6,15:10", help="used for both sides, so the armies are identical")
    parser.add_argument("--defender-hero", default=None, help="held fixed while the attacker's varies")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    rows = []
    for hero in LADDER:
        matchup = Matchup(args.army, args.army, attacker_hero=hero,
                          defender_hero=args.defender_hero, allow_wide=True)
        measured = ai_win_rate(args.worker, matchup, "attacker", args.episodes)
        rows.append({"attacker_hero": hero, "defender_hero": args.defender_hero,
                     "attacker_win_rate": measured["win_rate"],
                     "mean_reward": measured["mean_reward"],
                     "strength_margin": measured["strength_margin"],
                     "mean_rounds": measured["mean_rounds"]})
        print(f"attacker hero {str(hero):>6s} vs {str(args.defender_hero):>6s}: "
              f"attacker wins {measured['win_rate']:.3f}  margin {measured['strength_margin']:+.3f}  "
              f"reward {measured['mean_reward']:+.3f}  rounds {measured['mean_rounds']:.1f}", flush=True)

    rates = [r["attacker_win_rate"] for r in rows]
    swing = max(rates) - min(rates)
    print(f"\nidentical armies, engine on both sides: the commander alone swings the attacker's "
          f"win rate by {swing:.3f} across the sampler's own range")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"army": args.army, "episodes": args.episodes, "ladder": rows, "swing": swing}, indent=1))


if __name__ == "__main__":
    main()
