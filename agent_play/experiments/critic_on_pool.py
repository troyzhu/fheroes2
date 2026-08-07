#!/usr/bin/env python3
"""Does a pre-fitted critic help where a single matchup could not show it?

On one army pair, cloning plus PPO solves the matchup every time from either critic, and the
comparison came back at $+0.033 \\pm 0.027$ over 95 runs. A matchup every run solves cannot show
which run solved it better, so that result is about the matchup as much as about the critic.

A rotating pool does not have that ceiling. Training gain on the 140-matchup pool was 0.234 and
held-out gain 0.047, both far from saturated, so a better critic has room to show. The pool also
tests something the single matchup cannot: whether a critic fitted across many army pairs
generalizes, since the value of an opening position is mostly the matchup rather than the tactics.

Both arms train with PPO and GAE, which is the point. The group-relative trainer is critic-free by
construction, so it cannot answer this question at all.

Usage:
    ./critic_on_pool.py POOL.json CLONED.pt PREFITTED.pt WORKER [--holdout 50]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.env import MatchupPool  # noqa: E402
from fheroes2_agent.policy import load_policy, BattlePolicy  # noqa: E402
from fheroes2_agent.scenarios import measure, policy_fingerprint, pool_matchups  # noqa: E402
from fheroes2_agent.train_ppo import train  # noqa: E402


def load(checkpoint: str) -> BattlePolicy:
    model = load_policy(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    return model


def evaluate(model: BattlePolicy, worker: str, matchups: list, episodes: int) -> dict:
    """Errors across matchups rather than across episodes, because the claim is about the pool."""
    rates = [measure(model, worker, m, episodes=episodes)["win_rate"] for m in matchups]
    return {"mean": statistics.mean(rates),
            "stderr": statistics.stdev(rates) / len(rates) ** 0.5 if len(rates) > 1 else 0.0,
            "n": len(rates)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pool")
    parser.add_argument("cloned", help="stage 1 checkpoint, value head at initialization")
    parser.add_argument("prefitted", help="same policy with the value head fitted on teacher play")
    parser.add_argument("worker")
    parser.add_argument("--holdout", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    pool = json.loads(pathlib.Path(args.pool).read_text())
    matchups = pool_matchups(pool)
    held, trained = matchups[: args.holdout], matchups[args.holdout :]

    cloned = load(args.cloned)
    # The two checkpoints must differ only in the value head, or the comparison is not about the
    # critic. The fingerprint covers the policy head alone, so equality is exactly that claim.
    if policy_fingerprint(cloned) != policy_fingerprint(load(args.prefitted)):
        raise SystemExit("the two checkpoints differ in their policy head, so this compares two policies")
    if pool.get("policy_fingerprint") not in (None, policy_fingerprint(cloned)):
        raise SystemExit("pool was calibrated against a different policy")

    print(f"{len(matchups)} matchups: {len(trained)} training, {len(held)} held out", flush=True)
    started = time.time()
    before = {"training": evaluate(cloned, args.worker, trained, args.eval_episodes),
              "held_out": evaluate(cloned, args.worker, held, args.eval_episodes)}
    print(f"before: training {before['training']['mean']:.3f}, held out {before['held_out']['mean']:.3f} "
          f"({time.time() - started:.0f}s)", flush=True)

    work = pathlib.Path(args.report).parent if args.report else pathlib.Path(".")
    rows = []
    for arm, checkpoint in [("cold critic", args.cloned), ("pre-fitted critic", args.prefitted)]:
        for seed in range(args.seeds):
            out = work / f"pool_{arm.split()[0]}_{seed}.pt"
            r = train(args.worker, checkpoint=checkpoint, iterations=args.iterations,
                      episodes_per_iter=args.episodes, seed=seed, quiet=True, out=str(out),
                      env=MatchupPool(args.worker, trained, seed=seed))
            model = load(str(out))
            after = {"training": evaluate(model, args.worker, trained, args.eval_episodes),
                     "held_out": evaluate(model, args.worker, held, args.eval_episodes)}
            rows.append({"arm": arm, "seed": seed, "after": after,
                         "value_loss_first": r["history"][0]["value_loss"],
                         "value_loss_last": r["history"][-1]["value_loss"],
                         "floored_iterations": r["floored_iterations"]})
            print(f"  {arm:18s} seed {seed}  training {after['training']['mean']:.3f}  "
                  f"held out {after['held_out']['mean']:.3f}  "
                  f"(value loss {r['history'][0]['value_loss']:.2f} -> {r['history'][-1]['value_loss']:.2f})",
                  flush=True)

    print(f"\n  {'arm':18s} {'training gain':>16s} {'held-out gain':>16s}")
    summary = []
    for arm in ("cold critic", "pre-fitted critic"):
        a = [r for r in rows if r["arm"] == arm]
        s = {"arm": arm, "seeds": len(a)}
        for key in ("training", "held_out"):
            vals = [r["after"][key]["mean"] - before[key]["mean"] for r in a]
            s[key] = {"gain": statistics.mean(vals),
                      "stderr": statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0}
        summary.append(s)
        print(f"  {arm:18s} {s['training']['gain']:+.3f} +- {s['training']['stderr']:.3f}  "
              f"{s['held_out']['gain']:+.3f} +- {s['held_out']['stderr']:.3f}")

    print(f"\n  {time.time() - started:.0f}s total")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"before": before, "summary": summary, "runs": rows,
             "split": {"training": len(trained), "held_out": len(held)},
             "iterations": args.iterations, "episodes_per_iteration": args.episodes,
             "seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
