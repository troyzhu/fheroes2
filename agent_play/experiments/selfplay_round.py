#!/usr/bin/env python3
"""One self-play training round, parameterized: the driver the ad-hoc round scripts became.

Rounds two through four each ran from a copy of the previous round's scratch driver, which is how
a chair setting stayed at its default through three rounds while the scoreboard said the defending
chair was the untrained one. This is that driver as one indexed script: the pool, the leash, the
budget, the chair and the matchup source are flags, and the round's configuration is written
beside its checkpoints so a later reader can tell what produced them.

Matchups are validated before training, and validated per chair. The generator emits compositions
the worker refuses, and separately a pairing can be unplayable from one chair without being
unplayable from the other: a lone weak defender can die before its first turn, so that chair
yields no decisions at all. Alternating rounds therefore keep only the matchups playable from
both chairs, so a chair comparison is never confounded by a different matchup set, and the
report says how many were dropped for each reason.

Usage:
    ./selfplay_round.py WORKER --out-dir DIR --tag NAME [--beta 0.5] [--iterations 1000]
                        [--seeds 0 1 2] [--chair attacker|alternate] [--matchups 200]
                        [--pool anchor|ai-only] [--report R.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent import train_ppo  # noqa: E402
from fheroes2_agent.env import REWARD_MARGINS, BattleEnv, ScenarioRejected  # noqa: E402
from fheroes2_agent.scenarios import sample_matchups  # noqa: E402
from fheroes2_agent.selfplay import OpponentPool, SelfPlayEnv  # noqa: E402

FILES = pathlib.Path(__file__).resolve().parents[1] / "docs" / "archive" / "experiments" / "files"
CHECKPOINTS = FILES / "2026-08-05-checkpoints"
ANCHOR = CHECKPOINTS / "policy_gen1.pt"


def playable(worker: str, kwargs: dict, side: str) -> bool:
    """Does this pairing yield at least one decision from this chair?"""
    try:
        env = BattleEnv(worker, side=side, **kwargs)
    except ScenarioRejected:
        return False
    try:
        env.reset()
        return True
    except ScenarioRejected:
        return False
    finally:
        env.close()


def build_matchups(worker: str, count: int, seed: int, chairs: tuple) -> tuple[list, dict]:
    kept, stats = [], {"drawn": 0, "rejected_scenario": 0, "unplayable_chair": 0}
    batch_seed = seed
    while len(kept) < count:
        for m in sample_matchups(50, seed=batch_seed):
            stats["drawn"] += 1
            kwargs = dict(attacker=m.attacker, defender=m.defender, attacker_hero=m.attacker_hero,
                          defender_hero=m.defender_hero, allow_wide=m.allow_wide)
            if not playable(worker, kwargs, "attacker"):
                stats["rejected_scenario"] += 1
                continue
            if "defender" in chairs and not playable(worker, kwargs, "defender"):
                stats["unplayable_chair"] += 1
                continue
            kept.append(kwargs)
            if len(kept) >= count:
                break
        batch_seed += 1
    return kept, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", required=True, help="names the checkpoints, e.g. bothchair")
    parser.add_argument("--beta", type=float, default=0.5, help="KL leash to the frozen anchor")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--chair", default="attacker", choices=("attacker", "defender", "alternate"))
    parser.add_argument("--matchups", type=int, default=200)
    parser.add_argument("--matchup-seed", type=int, default=777)
    parser.add_argument("--pool", default="anchor", choices=("anchor", "ai-only"),
                        help="anchor: share2 and clone v4 beside the built-in AI; ai-only: the AI alone")
    parser.add_argument("--anchor", default=str(ANCHOR))
    parser.add_argument("--reward-margin", default="two_sided", choices=REWARD_MARGINS,
                        help="the objective the round trains on; every round through 2026-08-08 was "
                             "two_sided, and it is written into the report so an arm cannot be "
                             "mistaken for one trained on another")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chairs = ("attacker", "defender") if args.chair == "alternate" else (args.chair,)
    cache = out_dir / f"matchups_{args.tag}.json"
    if cache.exists():
        payload = json.loads(cache.read_text())
        matchups, stats = payload["matchups"], payload["stats"]
    else:
        matchups, stats = build_matchups(args.worker, args.matchups, args.matchup_seed, chairs)
        cache.write_text(json.dumps({"matchups": matchups, "stats": stats}, indent=1))
    print(f"matchups {len(matchups)} kept from {stats['drawn']} drawn "
          f"({stats['rejected_scenario']} refused by the worker, "
          f"{stats['unplayable_chair']} unplayable from a required chair)", flush=True)

    pool_paths = [None] if args.pool == "ai-only" else [
        str(CHECKPOINTS / "policy_share2.pt"), str(CHECKPOINTS / "policy_v4.pt"), None]

    runs = []
    for seed in args.seeds:
        out = out_dir / f"{args.tag}_s{seed}.pt"
        if out.exists():
            print(f"skip {out} (exists)", flush=True)
            continue
        started = time.time()
        env = SelfPlayEnv(args.worker, matchups, OpponentPool(pool_paths, seed=seed),
                          learner_side=args.chair, reward_margin=args.reward_margin, rotation_seed=seed)
        try:
            result = train_ppo.train(args.worker, checkpoint=args.anchor, iterations=args.iterations,
                                     seed=seed, env=env, quiet=True, out=str(out),
                                     value_warmup_iters=5, entropy_floor=0.15,
                                     anchor_kl_coef=args.beta)
        finally:
            skipped = env.skipped_resets
            env.close()
        runs.append({"seed": seed, "checkpoint": str(out), "skipped_resets": skipped,
                     "final_win_rate": result["final_win_rate"], "seconds": round(time.time() - started, 1)})
        print(f"{args.tag} s{seed} done in {runs[-1]['seconds']:.0f}s, "
              f"{skipped} chair skips -> {out}", flush=True)

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"tag": args.tag, "beta": args.beta, "iterations": args.iterations, "chair": args.chair,
             "pool": args.pool, "reward_margin": args.reward_margin, "matchup_stats": stats,
             "matchups": len(matchups), "runs": runs}, indent=1))
    print(f"ROUND {args.tag} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
