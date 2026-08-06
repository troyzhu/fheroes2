#!/usr/bin/env python3
"""How much does the battlefield matter? Win-rate spread across world seeds, per matchup.

Every training and evaluation number so far was measured on one obstacle layout per matchup,
because the loop never spread world seeds. This measures what that hid: for each matchup, the
same policy plays the same armies on several battlefield variants, and the spread of per-seed
win rates is compared with the binomial noise the episode count alone would produce. Spread
well above binomial means the battlefield is a real difficulty factor and single-seed numbers
are about a layout, not a matchup.

Usage:
    ./battlefield_spread.py WORKER CHECKPOINT [--matchups 12] [--seeds 6] [--episodes 24]
                            [--report battlefield_spread.json]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.env import BattleEnv  # noqa: E402
from fheroes2_agent.policy import BattlePolicy  # noqa: E402

POOL = pathlib.Path(__file__).resolve().parents[2] / "agent_play" / "docs" / "archive" / "experiments" / "files" \
    / "2026-08-05-run-reports" / "pool_value.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--matchups", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=24, help="episodes per battlefield variant")
    parser.add_argument("--torch-seed", type=int, default=0)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.torch_seed)
    model = BattlePolicy()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()

    matchups = json.loads(POOL.read_text())["matchups"][: args.matchups]
    rows = []
    print(f"{'matchup':>8s} {'mean':>6s} {'spread':>7s} {'binom':>6s} {'ratio':>6s}  per-seed rates")
    for index, m in enumerate(matchups):
        env = BattleEnv(args.worker, attacker=m["attacker"], defender=m["defender"],
                        attacker_hero=m.get("attacker_hero"), defender_hero=m.get("defender_hero"),
                        allow_wide=bool(m.get("allow_wide")), seeds=args.seeds)
        wins = defaultdict(list)
        try:
            for _ in range(args.seeds * args.episodes):
                observation, mask = env.reset()
                scenario = env.scenario_id
                while True:
                    with torch.no_grad():
                        logits, _ = model(torch.from_numpy(observation).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0))
                        action = int(torch.distributions.Categorical(logits=logits).sample())
                    step = env.step(action)
                    if step.done:
                        wins[scenario].append(1.0 if step.info["termination"] == "victory" else 0.0)
                        break
                    observation, mask = step.observation, step.mask
        finally:
            env.close()

        rates = [float(np.mean(w)) for w in wins.values()]
        mean = float(np.mean(rates))
        spread = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
        binom = math.sqrt(max(mean * (1.0 - mean), 1e-9) / args.episodes)
        rows.append({"matchup": m, "per_seed": {k: float(np.mean(v)) for k, v in wins.items()},
                     "mean": mean, "spread": spread, "binomial": binom})
        print(f"{index:>8d} {mean:6.3f} {spread:7.3f} {binom:6.3f} {spread / binom if binom else float('nan'):6.2f}  "
              + " ".join(f"{r:.2f}" for r in rates), flush=True)

    spreads = np.array([r["spread"] for r in rows])
    binoms = np.array([r["binomial"] for r in rows])
    print(f"\nmean spread {spreads.mean():.3f} against mean binomial noise {binoms.mean():.3f} "
          f"(ratio {spreads.mean() / binoms.mean():.2f}); "
          f"{int((spreads > 2 * binoms).sum())}/{len(rows)} matchups exceed twice binomial")

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"seeds": args.seeds, "episodes": args.episodes, "checkpoint": pathlib.Path(args.checkpoint).name,
             "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
