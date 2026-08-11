#!/usr/bin/env python3
"""Does root search improve play exactly where the policy and the teacher both fail?

The popular UCB variant, applied at the root: at every agent decision, candidate actions are
scored by PUCT (Q from rollout returns, prior from the policy, exploration bonus from visit
counts) over a budget of simulations, and the most-visited action plays. Simulations run on the
real engine through reset-continuation: a persistent side environment replays the action prefix
(determinism makes any state reachable by replay), applies the candidate, and rolls out with the
sampling policy to terminal. Rollout returns rather than critic leaves score the branches,
because the critic measures worse than the mean on student-visited states (critic_calibration)
and AlphaStar Unplugged reports MCTS exploiting value error until the policy collapses.

The probe compares searched play against policy-only play on matchups the policy mostly loses,
which is where an improvement operator would have to earn its keep as the next teacher.

Usage:
    ./search_probe.py WORKER CHECKPOINT [--simulations 32] [--episodes 12] [--c-puct 1.5]
                      [--report search_probe.json]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.encoding import encode_mask, encode_observation  # noqa: E402
from fheroes2_agent.env import BattleEnv  # noqa: E402
from fheroes2_agent.policy import load_policy, BattlePolicy  # noqa: E402

POOL = pathlib.Path(__file__).resolve().parents[2] / "agent_play" / "docs" / "archive" / "experiments" / "files" \
    / "2026-08-05-run-reports" / "pool_value.json"
THUNK = {"attacker": "11:1,11:1,11:1,10:2,9:2", "defender": "1:334,1:333,1:333",
         "attacker_hero": "13:12", "allow_wide": True, "label": "thunk_1000"}


# The primitives moved to `fheroes2_agent.search` on 2026-08-10, where nine scripts can import them
# without a path insert. They are re-exported here unchanged so every existing caller, and every
# vendored command line in the archive, keeps working against the same names.
from fheroes2_agent.search import (  # noqa: E402,F401
    policy_action, priors, rollout, search_action, search_action_detail, sync_side_environment)


def play(env: BattleEnv, sim: BattleEnv | None, model: BattlePolicy, simulations: int, c_puct: float) -> tuple[bool, int]:
    observation, mask = env.reset()
    prefix: list[int] = []
    searched = 0
    while True:
        if sim is not None:
            action = search_action(sim, model, prefix, observation, mask, simulations, c_puct)
            searched += 1
        else:
            action = policy_action(model, observation, mask)
        prefix.append(action)
        step = env.step(action)
        if step.done:
            return step.info["termination"] == "victory", searched
        observation, mask = step.observation, step.mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--hard-matchups", type=int, default=3)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    model = load_policy(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()

    entries = json.loads(POOL.read_text())["matchups"][:40]
    share2 = json.loads((POOL.parent / "dagger_share2.json").read_text())["evals"]["train"]
    hard = [entries[i] | {"label": f"pool_{i}"} for i in np.argsort(share2)[: args.hard_matchups]]
    targets = [THUNK] + hard

    started = time.time()
    results = []
    for m in targets:
        kwargs = dict(attacker=m["attacker"], defender=m["defender"], attacker_hero=m.get("attacker_hero"),
                      defender_hero=m.get("defender_hero"), allow_wide=bool(m.get("allow_wide")))
        env = BattleEnv(args.worker, **kwargs)
        sim = BattleEnv(args.worker, **kwargs)
        try:
            row = {"label": m["label"]}
            for arm, use_search in (("policy", False), ("search", True)):
                wins = 0
                decisions = 0
                for _ in range(args.episodes):
                    won, searched = play(env, sim if use_search else None, model, args.simulations, args.c_puct)
                    wins += won
                    decisions += searched
                row[arm] = wins / args.episodes
                if use_search:
                    row["searched_decisions"] = decisions
            results.append(row)
            print(f"{m['label']:12s} policy {row['policy']:.3f} -> search {row['search']:.3f} "
                  f"({row['searched_decisions']} searched decisions)", flush=True)
        finally:
            env.close()
            sim.close()

    lift = np.array([r["search"] - r["policy"] for r in results])
    print(f"\nmean lift {lift.mean():+.3f} over {len(results)} matchups; total {round(time.time() - started)}s")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"results": results, "simulations": args.simulations, "episodes": args.episodes,
             "c_puct": args.c_puct, "checkpoint": pathlib.Path(args.checkpoint).name,
             "seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
