#!/usr/bin/env python3
"""The agent's report card: fresh samples, out-of-distribution stress, and the Thunk ladder.

Three validation surfaces the owner fixed. Freshly sampled matchups come straight from the
value-budget generator with a seed no training or calibration ever used, uncalibrated on
purpose, so the number is about the raw distribution rather than a band any clone selected.
Stress suites probe out of distribution: horde counts beyond anything recorded, armies of only
wide creatures, commander stat extremes, and Thunk rungs beyond the rolled fight. The Thunk
ladder itself stays the standing validation no sampler drew and no training touched.

Every matchup is measured over battlefields for every checkpoint given, so the table reads as
paired columns.

Usage:
    ./validation_battery.py WORKER CHECKPOINT [CHECKPOINT ...] [--episodes 24] [--eval-seeds 4]
                            [--fresh 24] [--report validation_battery.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fheroes2_agent.env import REWARD_MARGINS  # noqa: E402
from fheroes2_agent.policy import BattlePolicy  # noqa: E402
from fheroes2_agent.scenarios import Matchup, measure, sample_budget_matchup  # noqa: E402
from sampling_policies import Sampler  # noqa: E402

# The suites moved to `fheroes2_agent.suites` on 2026-08-10 so the battery, the search harnesses and
# the built-in AI baseline share one definition instead of importing this script for two names. The
# lists are byte-identical across the move, verified matchup by matchup. Re-exported here because
# archived command lines and reports refer to these names.
from fheroes2_agent.suites import (  # noqa: E402,F401
    POOL, REAL_MAPS_MANIFEST, SUITE_SIDE, THUNK_ARMY, THUNK_HERO,
    build_suites, real_map_suite, thunk_split)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--eval-seeds", type=int, default=4)
    parser.add_argument("--fresh", type=int, default=24)
    parser.add_argument("--deployment", default="sample", choices=("sample", "greedy", "adaptive"),
                        help="how the policy acts at evaluation; every historical number is `sample`, "
                             "which pays a stochasticity penalty the deterministic engine does not")
    parser.add_argument("--reward-margin", default="two_sided",
                        choices=REWARD_MARGINS,
                        help="which objective the rw column reports; stamped into the report")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    suites = build_suites(args.fresh)
    started = time.time()
    report = {"suites": {name: {"matchups": len(ms)} for name, ms in suites.items()},
              "episodes": args.episodes, "eval_seeds": args.eval_seeds,
              # Self-describing: reports without this key predate 2026-08-08 and their rw column
              # is the hit-point margin whatever the checkpoint trained on.
              "reward_margin": args.reward_margin, "deployment": args.deployment, "results": {}}

    for path in args.checkpoints:
        name = pathlib.Path(path).name
        from fheroes2_agent.policy import load_policy
        model = load_policy(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
        if args.deployment != "sample":
            # `measure` samples the distribution it is handed, so a deployment rule is a logits
            # transform wrapped around the checkpoint rather than a change to the harness: greedy
            # collapses to the argmax, adaptive keeps the entropy-scaled nucleus.
            scheme = "greedy" if args.deployment == "greedy" else "adaptive"
            model = Sampler(model, scheme)
        model.eval()
        report["results"][name] = {}
        report.setdefault("quality", {})[name] = {}
        for suite, matchups in suites.items():
            side = SUITE_SIDE.get(suite, "attacker")
            measured = [measure(model, args.worker, m, episodes=args.episodes, seeds=args.eval_seeds,
                                side=side, reward_margin=args.reward_margin)
                        for m in matchups]
            rates = [m["win_rate"] for m in measured]
            report["results"][name][suite] = rates
            # The owner's reporting requirements, 2026-08-07: win rate alone hides too much, so
            # each suite also carries win quality (engine strength kept when winning) and mean
            # episode length, and graded suites are never collapsed to one number.
            survs = [m["surviving_strength"] for m in measured if m["surviving_strength"] is not None]
            damages = [m["loss_damage"] for m in measured if m["loss_damage"] is not None]
            margins = [m["strength_margin"] for m in measured]
            lens = [m["mean_length"] for m in measured]
            rewards = [m["mean_reward"] for m in measured]
            report["quality"][name][suite] = {
                "surviving_strength": [m["surviving_strength"] for m in measured],
                "loss_damage": [m["loss_damage"] for m in measured],
                "strength_margin": margins,
                "mean_reward": rewards,
                "mean_length": lens,
                "normalized_entropy": [m["normalized_entropy"] for m in measured],
                "effective_actions": [m["effective_actions"] for m in measured],
                "support_at_1pct": [m["support_at_1pct"] for m in measured],
                "legal_actions": [m["legal_actions"] for m in measured],
                "mean_rounds": [m["mean_rounds"] for m in measured],
                "reward_on_wins": [m["reward_on_wins"] for m in measured],
                "reward_on_losses": [m["reward_on_losses"] for m in measured],
                "mean_reward_commanded": [m["mean_reward_commanded"] for m in measured],
            }
            surv_txt = f" wq {np.mean(survs):.2f}" if survs else " wq  --"
            dmg_txt = f" lq {np.mean(damages):.2f}" if damages else " lq  --"
            mg_txt = f" mg {np.mean(margins):+.2f}"
            rw_txt = f" rw {np.mean(rewards):+.2f}"
            tail = f"{surv_txt}{dmg_txt}{mg_txt}{rw_txt} len {np.mean(lens):.0f}"
            if suite == "thunk_ladder":
                rungs = "/".join(f"{r:.2f}" for r in rates)
                print(f"{name:24s} {suite:18s} rungs {rungs}{tail}", flush=True)
            else:
                print(f"{name:24s} {suite:18s} mean {np.mean(rates):.3f}{tail}  " +
                      " ".join(f"{r:.2f}" for r in rates[:6]) + (" ..." if len(rates) > 6 else ""), flush=True)

    print(f"\ntotal {round(time.time() - started)}s")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
