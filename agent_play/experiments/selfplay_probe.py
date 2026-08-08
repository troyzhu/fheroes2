#!/usr/bin/env python3
"""The first self-play training probe: PPO where the opponent is a policy pool, not the AI.

The owner's next stage. The learner trains by PPO against opponents sampled per episode from a
checkpoint shelf plus the built-in AI as an anchor, the league-lite structure that guards
against overfitting the latest self. Everything hardened this week rides along: the value
warmup, the normalized-entropy floor (expected to matter here where it was null against the
fixed AI), the owner's strength-priced two-sided reward with both-chair stall semantics, and
heartbeats to the dashboard.

Judged three ways afterward: the standard battery (with its built-in-AI columns), a duel table
against every pool member (did the learner actually gain on the opponents it trained against),
and the symmetry gauge's concern is respected by training on mirror-heavy matchups.

Usage:
    ./selfplay_probe.py WORKER LEARNER_CKPT --pool CKPT [CKPT ...] [--include-ai]
                        [--iterations 40] [--seed 0] [--out OUT.pt] [--report R.json]
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

from fheroes2_agent import train_ppo  # noqa: E402
from fheroes2_agent.policy import load_policy  # noqa: E402
from fheroes2_agent.scenarios import Matchup, measure  # noqa: E402
from fheroes2_agent.selfplay import OpponentPool, SelfPlayEnv  # noqa: E402

POOL_FILE = pathlib.Path(__file__).resolve().parents[1] / "docs" / "archive" / "experiments" / "files" \
    / "2026-08-05-run-reports" / "pool_value.json"


def duel_rate(worker: str, learner, opponent_path: str | None, matchups: list[dict], episodes: int = 8) -> float:
    pool = OpponentPool([opponent_path], seed=1)
    env = SelfPlayEnv(worker, matchups, pool, reward_margin="two_sided")
    wins = 0
    total = 0
    try:
        for _ in range(episodes):
            obs, mask = env.reset()
            while True:
                with torch.no_grad():
                    logits, _ = learner(torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0))
                step = env.step(int(torch.distributions.Categorical(logits=logits).sample()))
                if step.done:
                    wins += step.info["termination"] == "victory"
                    total += 1
                    break
                obs, mask = step.observation, step.mask
    finally:
        env.close()
    return wins / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("learner")
    parser.add_argument("--pool", nargs="+", required=True)
    parser.add_argument("--include-ai", action="store_true")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matchups", type=int, default=12)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    started = time.time()
    entries = json.loads(POOL_FILE.read_text())["matchups"][: args.matchups]
    train_matchups = [dict(attacker=e["attacker"], defender=e["defender"],
                           attacker_hero=e.get("attacker_hero"), defender_hero=e.get("defender_hero"),
                           allow_wide=bool(e.get("allow_wide"))) for e in entries]
    pool_paths: list[str | None] = list(args.pool) + ([None] if args.include_ai else [])
    pool = OpponentPool(pool_paths, seed=args.seed)
    env = SelfPlayEnv(args.worker, train_matchups, pool, reward_margin="two_sided",
                      rotation_seed=args.seed)

    result = train_ppo.train(args.worker, checkpoint=args.learner, iterations=args.iterations,
                             seed=args.seed, env=env, quiet=False, out=args.out,
                             value_warmup_iters=5, entropy_floor=0.15)
    env.close()

    learner_before = load_policy(torch.load(args.learner, map_location="cpu", weights_only=True)["state_dict"])
    learner_before.eval()
    learner_after = load_policy(torch.load(args.out, map_location="cpu", weights_only=True)["state_dict"])
    learner_after.eval()

    duels = {}
    for opponent in pool_paths:
        name = pathlib.Path(opponent).name if opponent else "builtin_ai"
        duels[name] = {
            "before": duel_rate(args.worker, learner_before, opponent, train_matchups),
            "after": duel_rate(args.worker, learner_after, opponent, train_matchups),
        }
        print(f"duel vs {name:24s} before {duels[name]['before']:.3f} -> after {duels[name]['after']:.3f}", flush=True)

    held_entries = json.loads(POOL_FILE.read_text())["matchups"][40:50]
    held = [Matchup(e["attacker"], e["defender"], attacker_hero=e.get("attacker_hero"),
                    defender_hero=e.get("defender_hero"), allow_wide=bool(e.get("allow_wide")))
            for e in held_entries]
    vs_ai = {}
    for tag, model in (("before", learner_before), ("after", learner_after)):
        vs_ai[tag] = float(np.mean([measure(model, args.worker, m, episodes=8, seeds=4)["win_rate"] for m in held]))
    print(f"held-out vs built-in AI: before {vs_ai['before']:.3f} -> after {vs_ai['after']:.3f}", flush=True)

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"pool": [str(p) for p in pool_paths], "iterations": args.iterations, "seed": args.seed,
             "history_last": result["history"][-1], "duels": duels, "held_out_vs_ai": vs_ai,
             "seconds": round(time.time() - started, 1)}, indent=1))
    print(f"total {round(time.time() - started)}s")


if __name__ == "__main__":
    main()
