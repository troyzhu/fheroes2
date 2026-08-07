#!/usr/bin/env python3
"""Real opening fights harvested from shipped maps: the validation suite beyond Thunk.

The owner's point stands since the Thunk extraction: the game ships full of real fights, and a
validation story resting on one of them plus generator samples is thinner than it needs to be.
This script harvests them the way Thunk was pinned down, `--dump-map` over every map under
`devdata/maps/`, pairing each starting hero with neutral monster stacks within reach, filtering
to matchups the environment can express, and splitting the neutral stack three ways evenly per
the engine's own `Army::ArrangeForBattle` convention (`Rand(3..5)`; three is the canonical
validation form, exactly as the Thunk ladder uses).

Each kept fight is evaluated for the checkpoint and, with --ai-baseline, for the built-in AI on
the same fights, over battlefield variants, so the suite lands with its rule-based column
attached from day one.

Usage:
    ./real_map_fights.py WORKER CHECKPOINT [--maps-dir devdata/maps] [--distance 4]
                         [--max-fights 20] [--episodes 8] [--seeds 4] [--ai-baseline]
                         [--manifest real_map_fights.json] [--report real_map_eval.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from builtin_ai_baseline import ai_win_rate  # noqa: E402
from fheroes2_agent.encoding import SIMPLE_V1_MONSTERS  # noqa: E402
from fheroes2_agent.policy import load_policy  # noqa: E402
from fheroes2_agent.scenarios import Matchup, measure  # noqa: E402

ALLOW = set(SIMPLE_V1_MONSTERS)


def dump(worker: str, map_path: pathlib.Path) -> list[dict]:
    run = subprocess.run([worker, "--dump-map", str(map_path)], capture_output=True, text=True)
    records = []
    for line in run.stdout.splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def split_three(count: int) -> str:
    """The neutral stack as the engine would arrange it: three sub-stacks, as even as possible."""
    base, extra = divmod(count, 3)
    parts = [base + (1 if i < extra else 0) for i in range(3)]
    return ",".join(f"{{}}:{p}" for p in parts)


def harvest(worker: str, maps_dir: pathlib.Path, distance: int) -> list[dict]:
    fights = []
    for map_path in sorted(maps_dir.glob("*.[mM][pP]2")):
        records = dump(worker, map_path)
        heroes = [r for r in records if r.get("record") == "hero"]
        monsters = [r for r in records if r.get("record") == "monster"]
        for hero in heroes:
            army = hero.get("army", [])
            if not 1 <= len(army) <= 5:
                continue
            if any(u["monster_id"] not in ALLOW for u in army):
                continue
            for monster in monsters:
                if abs(monster["x"] - hero["x"]) > distance or abs(monster["y"] - hero["y"]) > distance:
                    continue
                if monster["monster_id"] not in ALLOW or not 3 <= monster["count"] <= 1000:
                    continue
                attacker = ",".join(f"{u['monster_id']}:{u['count']}" for u in army)
                defender = split_three(monster["count"]).format(*([monster["monster_id"]] * 3))
                fights.append({
                    "map": map_path.name, "hero": hero["name"],
                    "hero_stats": f"{hero['attack']}:{hero['defense']}",
                    "monster": f"{monster['count']} {monster['name']}",
                    "tiles_away": max(abs(monster["x"] - hero["x"]), abs(monster["y"] - hero["y"])),
                    "attacker": attacker, "defender": defender,
                    "attacker_hero": f"{hero['attack']}:{hero['defense']}",
                    "allow_wide": True,
                })
    return fights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--maps-dir", default="devdata/maps")
    parser.add_argument("--distance", type=int, default=4)
    parser.add_argument("--max-fights", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--ai-baseline", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    started = time.time()
    fights = harvest(args.worker, pathlib.Path(args.maps_dir), args.distance)
    # The first sampling pass taught two lessons now baked in. A hero paired with every stack in
    # reach yields mostly fights the map designed to be avoided, hopeless for policy and
    # rule-based AI alike, so each hero keeps only its nearest stack, the actual opening fight.
    # And flat sampling let one monster-dense map dominate the suite, so selection goes
    # round-robin across maps instead.
    nearest = {}
    for fight in fights:
        key = (fight["map"], fight["hero"], fight["attacker"])
        if key not in nearest or fight["tiles_away"] < nearest[key]["tiles_away"]:
            nearest[key] = fight
    seen = set()
    per_map: dict[str, list] = {}
    for fight in nearest.values():
        dedup = (fight["attacker"], fight["defender"], fight["attacker_hero"])
        if dedup in seen:
            continue
        seen.add(dedup)
        per_map.setdefault(fight["map"], []).append(fight)
    rng = np.random.default_rng(args.sample_seed)
    for fights_of_map in per_map.values():
        rng.shuffle(fights_of_map)
    chosen = []
    while len(chosen) < args.max_fights and any(per_map.values()):
        for name in sorted(per_map):
            if per_map[name] and len(chosen) < args.max_fights:
                chosen.append(per_map[name].pop())
    print(f"{len(fights)} candidates, {len(seen)} unique opening fights across {len(per_map)} maps, "
          f"{len(chosen)} evaluated", flush=True)
    if args.manifest:
        pathlib.Path(args.manifest).write_text(json.dumps(
            {"maps_dir": args.maps_dir, "distance": args.distance, "sample_seed": args.sample_seed,
             "candidates": len(fights), "unique": len(seen), "fights": chosen}, indent=1))

    model = load_policy(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()
    rows = []
    for fight in chosen:
        matchup = Matchup(fight["attacker"], fight["defender"], attacker_hero=fight["attacker_hero"],
                          allow_wide=True)
        policy_rate = measure(model, args.worker, matchup, episodes=args.episodes, seeds=args.seeds)["win_rate"]
        row = {**fight, "policy": policy_rate}
        if args.ai_baseline:
            row["builtin_ai"] = ai_win_rate(args.worker, matchup, "attacker", args.episodes * args.seeds)
        rows.append(row)
        ai_txt = f"  AI {row['builtin_ai']:.2f}" if args.ai_baseline else ""
        print(f"{fight['map']:14s} {fight['hero']:10s} vs {fight['monster']:24s} policy {policy_rate:.2f}{ai_txt}", flush=True)

    policy_mean = float(np.mean([r["policy"] for r in rows]))
    line = f"\nreal-map suite: policy {policy_mean:.3f}"
    if args.ai_baseline:
        line += f", built-in AI {float(np.mean([r['builtin_ai'] for r in rows])):.3f}"
    print(line + f" over {len(rows)} fights, {round(time.time() - started)}s")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps({"fights": rows, "policy_mean": policy_mean}, indent=1))


if __name__ == "__main__":
    main()
