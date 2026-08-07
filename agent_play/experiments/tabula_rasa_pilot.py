#!/usr/bin/env python3
"""Could this have been done without imitation at all, AlphaZero style?

The owner's question, tested rather than argued. AlphaZero learns tabula rasa: search over a
randomly initialized prior generates decisions better than the prior, those decisions are
distilled, and the loop repeats. Nothing about that recipe needs a demonstrator. What it does
need is that search over a random prior already beats the random prior by enough to bootstrap,
and this pilot measures exactly that on a small fixed matchup set, where the scenario axis is
held still so only the learning question moves.

Each round: play the matchups with root-PUCT over the current policy (rollouts scored by the
policy itself), keep every episode, distil a fresh network on the accumulated searched labels,
and evaluate the raw distilled policy. Round zero's evaluation is the random policy itself.

Usage:
    ./tabula_rasa_pilot.py WORKER [--rounds 3] [--matchups 4] [--episodes 6] [--simulations 32]
                           [--report tabula_rasa_pilot.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fheroes2_agent import train_bc  # noqa: E402
from fheroes2_agent.policy import load_policy, BattlePolicy  # noqa: E402
from fheroes2_agent.scenarios import Matchup, measure  # noqa: E402
from search_teacher import collect_matchup  # noqa: E402

POOL = pathlib.Path(__file__).resolve().parents[2] / "agent_play" / "docs" / "archive" / "experiments" / "files" \
    / "2026-08-05-run-reports" / "pool_value.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--matchups", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    # Mid-band matchups by the champion's own evaluation: winnable, not free, so a policy that
    # learns anything shows it, and a policy that learns nothing cannot coast.
    entries = json.loads(POOL.read_text())["matchups"][:40]
    rates = json.loads((POOL.parent / "dagger_share2.json").read_text())["evals"]["train"]
    chosen = [entries[i] for i in np.argsort(np.abs(np.array(rates) - 0.5))[: args.matchups]]
    suite = [Matchup(e["attacker"], e["defender"], attacker_hero=e.get("attacker_hero"),
                     defender_hero=e.get("defender_hero"), allow_wide=bool(e.get("allow_wide")))
             for e in chosen]

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="tabula_rasa_"))
    torch.manual_seed(0)
    model = BattlePolicy()
    model.eval()
    checkpoint = workdir / "round0.pt"
    torch.save({"state_dict": model.state_dict(), "encoding_version": "obs_encoding_v3"}, checkpoint)

    history = []
    started = time.time()
    base = float(np.mean([measure(model, args.worker, m, episodes=args.eval_episodes, seeds=4)["win_rate"]
                          for m in suite]))
    print(f"round 0 (random init): win rate {base:.3f}", flush=True)
    history.append({"round": 0, "win_rate": base, "labels": 0})

    data_root = workdir / "data"
    for rnd in range(1, args.rounds + 1):
        labels = 0
        searched_wins = 0
        for index, entry in enumerate(chosen):
            decisions, wins = collect_matchup(args.worker, model, entry, data_root / f"r{rnd}_m{index}",
                                              args.episodes, args.simulations, 1.5)
            labels += decisions
            searched_wins += wins
        out = workdir / f"round{rnd}.pt"
        train_bc.train([str(data_root)], epochs=15, seed=rnd, out=str(out))
        model = load_policy(torch.load(out, map_location="cpu", weights_only=True)["state_dict"])
        model.eval()
        rate = float(np.mean([measure(model, args.worker, m, episodes=args.eval_episodes, seeds=4)["win_rate"]
                              for m in suite]))
        searched_rate = searched_wins / (args.episodes * len(chosen))
        print(f"round {rnd}: search played {searched_rate:.3f}, distilled policy {rate:.3f}, "
              f"{labels} labels this round", flush=True)
        history.append({"round": rnd, "win_rate": rate, "search_win_rate": searched_rate, "labels": labels})

    print(f"\ntotal {round(time.time() - started)}s; checkpoints under {workdir}")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"history": history, "matchups": chosen, "simulations": args.simulations,
             "episodes_per_matchup": args.episodes}, indent=2))


if __name__ == "__main__":
    main()
