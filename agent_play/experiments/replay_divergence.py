#!/usr/bin/env python3
"""How often can a searched candidate not be played at the position the side environment replays?

Root search estimates a candidate by replaying the live action prefix in a side environment and
playing the candidate there. ADR 0008 mandates a nonzero `combat_seed_offset` for every number
quoted against the built-in AI, which keeps the battlefield but makes the dice independent. That
makes the replay a RESAMPLED trajectory rather than a copy: different rolls kill different units,
so the replayed position drifts, and a candidate that is legal in the live battle may not exist
there at all.

Until 2026-08-23 nothing measured that drift and nothing surfaced it. `rollout` played the
candidate regardless; the engine's contract for an illegal selection is to skip the acting unit's
turn (`agent_external_controller.cpp`), so the playout measured a DIFFERENT action and credited its
value to the candidate. A prefix that ended the resampled battle early was worse: every candidate
returned that same terminal, so the search compared identical values and fell through its tie-break
to the lowest legal action index. Both failures were silent.

The rollouts now return None in those cases and the search excludes the candidate instead of
scoring it. This measures what that exclusion is worth: per decision depth, how many live-legal
candidates the replayed position can offer, and how often it can offer none at all. The shared-dice
configuration is measured beside it as the zero control, since there the replay is exact by
construction and any nonzero reading would mean the harness itself is broken.

Usage:
    ./replay_divergence.py WORKER CHECKPOINT [--matchups 6] [--depth 20] [--offset 987631]
                           [--report R.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.env import BattleEnv  # noqa: E402
from fheroes2_agent.policy import load_policy  # noqa: E402
from fheroes2_agent.search import policy_action  # noqa: E402

#: Symmetric mirrors, the suite where search has its largest measured margin and therefore where a
#: corrupted candidate value costs the most.
ARMIES = ["62:3,30:6,15:10", "9:2,11:2,6:12,1:30", "13:3,48:12,12:20",
          "10:4,7:8", "28:3,40:8,2:15", "51:4,50:4,12:16"]


def probe(worker: str, model, army: str, offset: int, depth_limit: int, seed: int) -> list[dict]:
    """Walk one battle, comparing the live legal set against the replayed one at every decision."""
    kwargs = dict(attacker=army, defender=army, attacker_hero="10:10", defender_hero="10:10",
                  allow_wide=True, side="defender", reward_margin="hit_points")
    live = BattleEnv(worker, seeds=1, **kwargs)
    sim = BattleEnv(worker, seeds=1, combat_seed_offset=offset, **kwargs)
    rows: list[dict] = []
    try:
        torch.manual_seed(seed)
        observation, mask = live.reset()
        prefix: list[int] = []
        for depth in range(depth_limit):
            # Replay the prefix exactly as `rollout` does, then compare what each position offers.
            # `sim._pending` always holds the decision the side environment is waiting on, so a
            # reset plus the prefix leaves it at the replayed counterpart of the live decision.
            # An empty prefix needs no special case: the reset alone presents the first decision.
            sim.reset()
            replay_ended = False
            for action in prefix:
                if sim.step(action).done:
                    replay_ended = True
                    break
            live_legal = set(np.flatnonzero(mask).tolist())
            if replay_ended:
                rows.append({"depth": depth, "replay_ended": True, "live_legal": len(live_legal),
                             "appliable": 0, "same_unit": False})
            else:
                pending = sim.pending_decision["observation"]
                sim_legal = {int(a) for a in sim.pending_decision["legal_actions"]}
                rows.append({"depth": depth, "replay_ended": False,
                             "live_legal": len(live_legal),
                             "appliable": len(live_legal & sim_legal),
                             "same_unit": pending["active_uid"] ==
                                          live.pending_decision["observation"]["active_uid"]})
            action = policy_action(model, observation, mask, env=live)
            prefix.append(action)
            step = live.step(action)
            if step.done:
                break
            observation, mask = step.observation, step.mask
    finally:
        live.close()
        sim.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--matchups", type=int, default=6)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--offset", type=int, default=987631)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    model = load_policy(torch.load(args.checkpoint, map_location="cpu",
                                   weights_only=True)["state_dict"])
    model.eval()
    out = {}
    for offset, label in ((args.offset, "honest_offset"), (0, "shared_dice")):
        rows = []
        for i, army in enumerate(ARMIES[: args.matchups]):
            for s in range(args.seeds):
                rows += probe(args.worker, model, army, offset, args.depth, 4242 + 31 * i + s)
        live = sum(r["live_legal"] for r in rows)
        ok = sum(r["appliable"] for r in rows)
        ended = sum(1 for r in rows if r["replay_ended"])
        wrong_unit = sum(1 for r in rows if not r["replay_ended"] and not r["same_unit"])
        blind = sum(1 for r in rows if r["appliable"] == 0)
        out[label] = {"decisions": len(rows), "live_legal": live, "appliable": ok,
                      "un_appliable_rate": 1 - ok / max(live, 1),
                      "replay_ended_early": ended, "different_acting_unit": wrong_unit,
                      "blind_decisions": blind,
                      "by_depth": {str(d): {
                          "un_appliable_rate": 1 - sum(r["appliable"] for r in rows if r["depth"] == d)
                          / max(sum(r["live_legal"] for r in rows if r["depth"] == d), 1),
                          "n": sum(1 for r in rows if r["depth"] == d)}
                          for d in sorted({r["depth"] for r in rows})}}
        print(f"\n{label}: {len(rows)} decisions over {args.matchups} matchups x {args.seeds} seeds")
        print(f"  live-legal candidates          {live}")
        print(f"  appliable at the replay        {ok}  ({ok/max(live,1):.1%})")
        print(f"  UN-APPLIABLE                   {live-ok}  ({1-ok/max(live,1):.1%})")
        print(f"  decisions with zero appliable  {blind}  ({blind/max(len(rows),1):.1%})")
        print(f"  replay ended before the decision {ended}")
        print(f"  replay presenting another unit   {wrong_unit}")
        depths = sorted({r["depth"] for r in rows})
        band = [d for d in depths if d % 4 == 0][:6]
        print("  un-appliable by depth:  " + "  ".join(
            f"d{d}:{out[label]['by_depth'][str(d)]['un_appliable_rate']:.0%}" for d in band))
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(out, indent=1))
    print("\nREPLAY DIVERGENCE COMPLETE")


if __name__ == "__main__":
    main()
