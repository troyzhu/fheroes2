#!/usr/bin/env python3
"""Can a policy stall a battle forever by flying away, and what does a stall pay?

The owner's cavity: a fast flyer that simply avoids confrontation could in principle loop a
battle indefinitely, and nothing had demonstrated what actually happens. This script commands
the most evasive policy expressible, always move to the reachable cell farthest from every
enemy and never attack, with fast Rogues defending against slow Zombies by default. Flyers were unavailable until
flying_v1 opened on 2026-08-10, and that absence was itself the original finding: the flying
version of this exploit could not be fielded at all. It can now, so `--allow-flying` with an
evading Sprite is the case this script exists to check, and it reports how the episode ends.

Two protections should catch it. The runner stops after forty consecutive rounds without a
death (`stalemate`), a margin under the engine AI's own fifty-turn breaker, which would force
the attacker to retreat and thereby lose. And the reward scores that termination the way the
engine would have resolved it: defender wins, attacker loses (`_side_won` in env.py), so
evasion is exactly as valuable as the real game makes it and no more.

Usage:
    ./evasion_stalemate.py WORKER [--episodes 3] [--report evasion_stalemate.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.env import BattleEnv, terminal_reward_strength  # noqa: E402

BOARD_WIDTH = 11


def cell_position(cell: int) -> tuple[int, int]:
    return cell // BOARD_WIDTH, cell % BOARD_WIDTH


def most_evasive_action(raw_observation: dict, mask: np.ndarray) -> int:
    """The farthest legal move from every enemy, and never an attack."""
    own_attacker = bool(raw_observation["active_is_attacker"])
    enemies = [cell_position(u["head_cell"]) for u in raw_observation["units"]
               if (u["side"] == "attacker") != own_attacker]
    best, best_distance = 0, -1.0
    for action in range(1, 100):
        if not mask[action]:
            continue
        row, col = cell_position(action - 1)
        distance = min((row - er) ** 2 + (col - ec) ** 2 for er, ec in enemies) if enemies else 0.0
        if distance > best_distance:
            best, best_distance = action, distance
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--attacker", default="49:20")
    parser.add_argument("--defender", default="58:8", help="the evading side")
    parser.add_argument("--allow-flying", action="store_true",
                        help="flying_v1, opened 2026-08-10. The flying version of this exploit could "
                             "not be fielded before it, which the docstring records as its own finding")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    env = BattleEnv(args.worker, attacker=args.attacker, defender=args.defender, side="defender",
                    seeds=args.episodes, allow_flying=args.allow_flying)
    episodes = []
    started = time.time()
    try:
        for _ in range(args.episodes):
            _, mask = env.reset()
            decisions = 0
            while True:
                action = most_evasive_action(env._pending["observation"], mask)
                step = env.step(action)
                decisions += 1
                if step.done:
                    record = step.info
                    episodes.append({
                        "termination": record["termination"],
                        "rounds": record.get("rounds"),
                        "decisions": decisions,
                        "defender_reward": terminal_reward_strength(record, "defender"),
                        "attacker_reward": terminal_reward_strength(record, "attacker"),
                        "defender_survival": record["defender"]["strength"] / max(record["defender"]["initial_strength"], 1e-9),
                    })
                    break
                mask = step.mask
    finally:
        env.close()

    for index, episode in enumerate(episodes):
        print(f"episode {index}: {episode['termination']} after {episode['rounds']} rounds, "
              f"{episode['decisions']} evasive decisions, defender survival {episode['defender_survival']:.2f}, "
              f"rewards defender {episode['defender_reward']:+.2f} / attacker {episode['attacker_reward']:+.2f}", flush=True)
    terminations = {e["termination"] for e in episodes}
    print(f"\nall episodes terminated ({', '.join(sorted(terminations))}); none looped. total {round(time.time() - started)}s")

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps({"episodes": episodes}, indent=2))


if __name__ == "__main__":
    main()
