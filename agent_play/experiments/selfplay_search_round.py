#!/usr/bin/env python3
"""One expert-iteration round in TRUE self-play: search labels both chairs of policy-vs-policy games.

The owner's curriculum, stated 2026-08-13: extensive initial training and self-distillation run
against the built-in AI, and once the model is in good range the round should be played against
itself. Every search-taught corpus before this script was collected against the engine, opponent
in the live game and opponent model inside the playouts alike. Here both chairs of the live game
are driven by search over the policy, every playout models the other chair with the policy
(`rollout_self_play`), and every decision of both chairs becomes a label, so one episode yields
roughly twice the decisions of a one-chair collection.

Labels are scored by a record-only margin (`strength` by default), because a both-sides
environment's step reward is unperspectived and `hit_points` has no record-only form. The combat
offset keeps the label honest exactly as in the AI-opponent collector: nonzero, so search cannot
see the live game's rolls.

Usage:
    ./selfplay_search_round.py WORKER CHECKPOINT --out-dir DIR [--matchups N] [--episodes 2]
                               [--simulations 32] [--sample-seed 97] [--report R.json]
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

from fheroes2_agent.env import REWARD_MARGINS, BattleEnv, reward_from_record  # noqa: E402
from fheroes2_agent.policy import load_policy  # noqa: E402
from fheroes2_agent.search import search_action_detail  # noqa: E402
from fheroes2_agent.suites import POOL  # noqa: E402

SEARCH_OFFSET = 987631


def collect_matchup(worker, model, entry, out_dir, episodes, simulations, c_puct, margin,
                    index) -> tuple[int, dict]:
    kwargs = dict(attacker=entry["attacker"], defender=entry["defender"],
                  attacker_hero=entry.get("attacker_hero"), defender_hero=entry.get("defender_hero"),
                  allow_wide=bool(entry.get("allow_wide")), side="both", reward_margin=margin)
    live = BattleEnv(worker, **kwargs)
    sim = BattleEnv(worker, combat_seed_offset=SEARCH_OFFSET, **kwargs)
    decisions, outcomes = 0, {"victory": 0, "defeat": 0, "stalemate": 0}
    try:
        for episode in range(episodes):
            torch.manual_seed(1009 * index + episode)
            observation, mask = live.reset()
            prefix: list[int] = []
            rows = []
            while True:
                acting = "attacker" if live._pending["observation"]["active_is_attacker"] else "defender"
                action, means, visits, prior = search_action_detail(
                    sim, model, prefix, observation, mask, simulations, c_puct,
                    rollout_opponent="policy", agent_side=acting, full_prefix=True)
                rows.append({"record": "decision", "side": acting,
                             "observation": live._pending["observation"],
                             "legal_actions": [int(a) for a in np.flatnonzero(mask)],
                             "teacher_action": int(action),
                             "search_values": {str(a): float(v) for a, v in means.items()},
                             "search_visits": {str(a): int(v) for a, v in visits.items()},
                             "prior": {str(a): float(p) for a, p in prior.items()}})
                prefix.append(action)
                step = live.step(action)
                if step.done:
                    outcomes[step.info["termination"]] = outcomes.get(step.info["termination"], 0) + 1
                    for row in rows:
                        row["episode_reward"] = reward_from_record(step.info, row["side"], margin)
                    path = out_dir / f"matchup_{index:03d}" / f"episode_{episode:04d}.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w") as f:
                        for row in rows:
                            f.write(json.dumps(row) + "\n")
                        f.write(json.dumps({"record": "terminal", **step.info}) + "\n")
                    decisions += len(rows)
                    break
                observation, mask = step.observation, step.mask
    finally:
        live.close()
        sim.close()
    return decisions, outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--matchups", type=int, default=24)
    parser.add_argument("--episodes", type=int, default=2, help="per matchup; both chairs label every episode")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--margin", default="strength",
                        choices=[m for m in REWARD_MARGINS if m != "hit_points"])
    parser.add_argument("--sample-seed", type=int, default=97)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    model = load_policy(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.sample_seed)
    pool = json.loads(POOL.read_text())["matchups"]
    entries = [pool[i] for i in rng.permutation(len(pool))[:args.matchups]]

    started = time.time()
    total, all_outcomes = 0, {}
    for index, entry in enumerate(entries):
        n, outcomes = collect_matchup(args.worker, model, entry, out, args.episodes,
                                      args.simulations, args.c_puct, args.margin, index)
        total += n
        for k, v in outcomes.items():
            all_outcomes[k] = all_outcomes.get(k, 0) + v
        print(f"  matchup {index + 1}/{len(entries)}: {n} decisions, outcomes so far {all_outcomes} "
              f"({time.time() - started:.0f}s)", flush=True)

    manifest = {"checkpoint": pathlib.Path(args.checkpoint).name, "matchups": len(entries),
                "episodes_per_matchup": args.episodes, "simulations": args.simulations,
                "margin": args.margin, "search_combat_offset": SEARCH_OFFSET,
                "rollout_opponent": "policy", "sides": "both",
                "decisions": total, "outcomes": all_outcomes,
                "seconds": round(time.time() - started, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(manifest, indent=1))
    print(f"SELF-PLAY COLLECTION COMPLETE: {total} decisions over {len(entries)} matchups")


if __name__ == "__main__":
    main()
