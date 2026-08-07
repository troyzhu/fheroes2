#!/usr/bin/env python3
"""Search plays and labels: the collection half of one search-taught distillation round.

Root-PUCT with the clone prior and rollout scoring chooses every controlled action, the episode
follows the searched action, and each decision is written in the dataset schema with the
searched choice as the label. This is the AlphaZero improvement step assembled from this
project's parts, held to the one-round discipline the offline literature imposes; the probe
measured the operator at +0.79 win rate on the matchups the policy loses.

Collection runs at one battlefield per matchup, deliberately: simulations replay the live
episode's action prefix, and the battlefield rotation lives in the worker's scenario cycle,
which a side environment cannot yet synchronize to. Battlefield variety stays the teacher
corpus's job; these labels contribute decision quality.

Shards partition the matchup list so cores collect in parallel:
    ./search_teacher.py WORKER CHECKPOINT --out-dir DIR --shard 0 --shards 6 [...]

Fresh-distribution mode (--sample N --sample-seed S) draws matchups straight from the
value-budget generator instead of the fixed pool, plays them from the given --side, and keeps
only matchups where search itself wins at least half its episodes, because the credit
measurement and the struggle-round both showed that labels from fights nothing wins are poison.
A manifest of kept and dropped matchups is written beside the episodes.

Usage:
    ./search_teacher.py WORKER CHECKPOINT --out-dir DIR [--shard 0 --shards 1]
                        [--simulations 32] [--hard-episodes 24] [--easy-episodes 8]
                        [--sample N --sample-seed S --side attacker|defender --episodes 8]
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

from fheroes2_agent.env import BattleEnv  # noqa: E402
from fheroes2_agent.policy import BattlePolicy  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from search_probe import policy_action, search_action, search_action_detail  # noqa: E402

POOL = pathlib.Path(__file__).resolve().parents[2] / "agent_play" / "docs" / "archive" / "experiments" / "files" \
    / "2026-08-05-run-reports" / "pool_value.json"
SHARE2_EVALS = POOL.parent / "dagger_share2.json"


def collect_matchup(worker: str, model: BattlePolicy, entry: dict, out_dir: pathlib.Path,
                    episodes: int, simulations: int, c_puct: float, side: str = "attacker",
                    min_win_fraction: float = 0.0, seed_offset: int = 0,
                    record_candidates: bool = False, planes: bool = False) -> tuple[int, int]:
    """Returns (decisions, wins) actually written; (0, wins) when the win filter drops the
    matchup, since labels from fights search cannot win teach the least-bad line of a lost
    position, which the credit measurement showed is exactly the poison. A nonzero seed offset
    moves the matchup onto another battlefield variant, applied identically to the live and the
    search environments so simulations still replay the battlefield being played."""
    kwargs = dict(attacker=entry["attacker"], defender=entry["defender"],
                  attacker_hero=entry.get("attacker_hero"), defender_hero=entry.get("defender_hero"),
                  allow_wide=bool(entry.get("allow_wide")), side=side, seed_offset=seed_offset, planes=planes)
    env = BattleEnv(worker, **kwargs)
    sim = BattleEnv(worker, **kwargs)
    decisions = 0
    wins = 0
    episodes_out = []
    won_termination = "victory" if side == "attacker" else "defeat"
    try:
        for episode in range(episodes):
            observation, mask = env.reset()
            prefix: list[int] = []
            records = []
            while True:
                if record_candidates:
                    action, means, visits, prior = search_action_detail(
                        sim, model, prefix, observation, mask, simulations, c_puct, live=env)
                else:
                    action = search_action(sim, model, prefix, observation, mask, simulations, c_puct, live=env)
                raw = env._pending
                record = {"record": "decision", "observation": raw["observation"],
                          "legal_actions": raw["legal_actions"], "teacher_resolved": True,
                          "teacher_action": int(action)}
                if record_candidates:
                    # Per-candidate search measurements, keyed by canonical action index. Extra
                    # fields are invisible to the standard dataset loader, so these episodes
                    # remain valid hard-label corpora as well.
                    record["search_values"] = {str(a): round(means[a], 4) for a in means}
                    record["search_visits"] = {str(a): visits[a] for a in visits}
                    record["prior"] = {str(a): round(prior[a], 6) for a in prior}
                records.append(record)
                decisions += 1
                prefix.append(action)
                step = env.step(action)
                if step.done:
                    records.append(step.info)
                    wins += step.info["termination"] == won_termination
                    break
                observation, mask = step.observation, step.mask
            episodes_out.append(records)
    finally:
        env.close()
        sim.close()
    if episodes and wins / episodes < min_win_fraction:
        return 0, wins
    out_dir.mkdir(parents=True, exist_ok=True)
    for episode, records in enumerate(episodes_out):
        (out_dir / f"episode_{episode:04d}.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return decisions, wins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--hard-episodes", type=int, default=24,
                        help="episodes for the 15 matchups the prior loses worst, where labels differ most")
    parser.add_argument("--easy-episodes", type=int, default=8)
    parser.add_argument("--sample", type=int, default=0,
                        help="fresh-distribution mode: draw this many matchups from the value-budget generator")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--side", default="attacker", choices=("attacker", "defender"))
    parser.add_argument("--episodes", type=int, default=8, help="episodes per sampled matchup")
    parser.add_argument("--min-win", type=float, default=0.5,
                        help="fresh mode: drop matchups where search wins less than this fraction")
    parser.add_argument("--planes", action="store_true",
                        help="collect with the planes_v1 obstacle layer on every observation")
    parser.add_argument("--record-candidates", action="store_true",
                        help="write per-candidate search values, visits, and the prior onto every "
                             "decision record, the soft-distillation dataset")
    parser.add_argument("--vary-battlefields", action="store_true",
                        help="fresh mode: place each matchup on its own battlefield variant via --seed-offset, "
                             "applied identically to live and search environments (needs a worker built after "
                             "2026-08-06)")
    args = parser.parse_args()

    model = BattlePolicy()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()

    out_root = pathlib.Path(args.out_dir)
    started = time.time()
    total_decisions = 0
    total_wins = 0
    total_eps = 0

    if args.sample:
        import random

        from fheroes2_agent.scenarios import sample_budget_matchup

        rng = random.Random(args.sample_seed * 1000 + args.shard)
        manifest = []
        for index in range(args.sample):
            m = sample_budget_matchup(rng)
            entry = {"attacker": m.attacker, "defender": m.defender, "attacker_hero": m.attacker_hero,
                     "defender_hero": m.defender_hero, "allow_wide": m.allow_wide}
            try:
                decisions, wins = collect_matchup(args.worker, model, entry,
                                                  out_root / f"shard{args.shard}_matchup_{index:03d}",
                                                  args.episodes, args.simulations, args.c_puct,
                                                  side=args.side, min_win_fraction=args.min_win,
                                                  seed_offset=(args.shard * 100 + index) % 16 if args.vary_battlefields else 0,
                                                  record_candidates=args.record_candidates, planes=args.planes)
            except Exception as error:  # a rejected scenario is data, not a crash
                manifest.append(entry | {"kept": False, "error": str(error)[:120]})
                continue
            kept = decisions > 0
            manifest.append(entry | {"kept": kept, "wins": wins, "episodes": args.episodes, "side": args.side})
            total_decisions += decisions
            total_wins += wins
            total_eps += args.episodes if kept else 0
            if (index + 1) % 5 == 0:
                print(f"shard {args.shard} ({args.side}): {index + 1}/{args.sample} sampled, "
                      f"{total_decisions} labels kept", flush=True)
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / f"shard{args.shard}_manifest.json").write_text(json.dumps(
            {"matchups": manifest, "side": args.side, "sample_seed": args.sample_seed}, indent=2))
    else:
        entries = json.loads(POOL.read_text())["matchups"][:40]
        rates = json.loads(SHARE2_EVALS.read_text())["evals"]["train"]
        hard = set(np.argsort(rates)[:15].tolist())
        for index, entry in enumerate(entries):
            if index % args.shards != args.shard:
                continue
            episodes = args.hard_episodes if index in hard else args.easy_episodes
            decisions, wins = collect_matchup(args.worker, model, entry, out_root / f"matchup_{index:03d}",
                                              episodes, args.simulations, args.c_puct,
                                              record_candidates=args.record_candidates)
            total_decisions += decisions
            total_wins += wins
            total_eps += episodes
            print(f"shard {args.shard}: matchup {index} done, {decisions} labeled, {wins}/{episodes} won", flush=True)

    print(f"shard {args.shard}: {total_eps} episodes kept, {total_decisions} labels, "
          f"{total_wins} wins, {round(time.time() - started)}s")


if __name__ == "__main__":
    main()
