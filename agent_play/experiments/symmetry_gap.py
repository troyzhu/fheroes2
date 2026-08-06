#!/usr/bin/env python3
"""Does the same army played from the other chair win as often? The owner's symmetry test.

Hold the agent's army fixed and swap which side it commands: first as attacker against the
opponent army, then as defender while the built-in AI attacks with that same opponent army. A
policy with no side bias, in a game with no side bias, would post the same win rate twice. Any
difference is the two mixed together, so the built-in AI runs the identical pair and supplies the
game's own asymmetry as the reference: the agent's gap is only its own to the extent it exceeds
the engine's.

This differs from the mirror suite in the battery, which gives both sides the same army and asks
which chair wins; here the army travels with the agent and only the chair changes.

Usage:
    ./symmetry_gap.py WORKER CHECKPOINT [CHECKPOINT ...] [--matchups 10] [--episodes 24]
                      [--eval-seeds 4] [--report symmetry_gap.json]
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

from builtin_ai_baseline import ai_win_rate  # noqa: E402
from fheroes2_agent.policy import load_policy  # noqa: E402
from fheroes2_agent.scenarios import Matchup, measure  # noqa: E402

POOL = pathlib.Path(__file__).resolve().parents[2] / "agent_play" / "docs" / "archive" / "experiments" / "files" \
    / "2026-08-05-run-reports" / "pool_value.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--matchups", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--eval-seeds", type=int, default=4)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    entries = json.loads(POOL.read_text())["matchups"][40:40 + args.matchups]
    started = time.time()
    report = {"matchups": entries, "episodes": args.episodes, "eval_seeds": args.eval_seeds, "results": {}}

    # The engine's own asymmetry: the same army as attacker, then as defender, both sides played
    # by the built-in AI, so the difference is the game's rather than any policy's.
    ai_pairs = []
    for e in entries:
        forward = Matchup(e["attacker"], e["defender"], attacker_hero=e.get("attacker_hero"),
                          defender_hero=e.get("defender_hero"), allow_wide=bool(e.get("allow_wide")))
        swapped = Matchup(e["defender"], e["attacker"], attacker_hero=e.get("defender_hero"),
                          defender_hero=e.get("attacker_hero"), allow_wide=bool(e.get("allow_wide")))
        as_attacker = ai_win_rate(args.worker, forward, "attacker", args.episodes)
        as_defender = ai_win_rate(args.worker, swapped, "defender", args.episodes)
        ai_pairs.append((as_attacker, as_defender))
    ai_gap = np.array([a - d for a, d in ai_pairs])
    report["results"]["builtin_ai"] = {"as_attacker": [a for a, _ in ai_pairs], "as_defender": [d for _, d in ai_pairs],
                                       "gap_mean": float(ai_gap.mean()),
                                       "gap_se": float(ai_gap.std(ddof=1) / np.sqrt(len(ai_gap)))}
    print(f"built-in AI      as attacker {np.mean([a for a, _ in ai_pairs]):.3f} | as defender "
          f"{np.mean([d for _, d in ai_pairs]):.3f} | gap {ai_gap.mean():+.3f} +/- "
          f"{ai_gap.std(ddof=1) / np.sqrt(len(ai_gap)):.3f}", flush=True)

    for path in args.checkpoints:
        name = pathlib.Path(path).name
        model = load_policy(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
        model.eval()
        pairs = []
        for e in entries:
            forward = Matchup(e["attacker"], e["defender"], attacker_hero=e.get("attacker_hero"),
                              defender_hero=e.get("defender_hero"), allow_wide=bool(e.get("allow_wide")))
            swapped = Matchup(e["defender"], e["attacker"], attacker_hero=e.get("defender_hero"),
                              defender_hero=e.get("attacker_hero"), allow_wide=bool(e.get("allow_wide")))
            as_attacker = measure(model, args.worker, forward, episodes=args.episodes,
                                  seeds=args.eval_seeds, side="attacker")["win_rate"]
            as_defender = measure(model, args.worker, swapped, episodes=args.episodes,
                                  seeds=args.eval_seeds, side="defender")["win_rate"]
            pairs.append((as_attacker, as_defender))
        gap = np.array([a - d for a, d in pairs])
        report["results"][name] = {"as_attacker": [a for a, _ in pairs], "as_defender": [d for _, d in pairs],
                                   "gap_mean": float(gap.mean()), "gap_se": float(gap.std(ddof=1) / np.sqrt(len(gap)))}
        print(f"{name:16s} as attacker {np.mean([a for a, _ in pairs]):.3f} | as defender "
              f"{np.mean([d for _, d in pairs]):.3f} | gap {gap.mean():+.3f} +/- "
              f"{gap.std(ddof=1) / np.sqrt(len(gap)):.3f}", flush=True)
        excess = gap - ai_gap
        print(f"{'':16s} excess over the engine's own asymmetry {excess.mean():+.3f} +/- "
              f"{excess.std(ddof=1) / np.sqrt(len(excess)):.3f}", flush=True)
        report["results"][name]["excess_over_ai"] = float(excess.mean())

    print(f"\ntotal {round(time.time() - started)}s")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
