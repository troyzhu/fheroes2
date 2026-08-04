#!/usr/bin/env python3
"""Does a value head fitted on teacher play improve reinforcement learning, or only look better?

Explained variance says the fitted head predicts returns well under the teacher's policy. That is
not the claim that matters. What matters is whether starting PPO from it learns faster, further,
or more reliably than starting from an untrained head, and the only way to know is to run both.

Paired by seed. The two arms share a seed, so the same action-sampling stream drives both and the
difference is attributable to the critic rather than to which run got luckier. Twenty seeds,
because the effect this is looking for turned out to be in the spread rather than in the mean, and
a spread needs more samples than a mean does.

Collapse is defined rather than eyeballed: a run that reached a win rate of 0.95 or better at some
iteration and finished with a last-five mean below 0.5. That is a run which solved the matchup and
then lost it again, which is a different failure from never solving it.

Usage:
    ./critic_pre_fitting.py DATA_DIR CHECKPOINT WORKER --attacker 2:6,1:10 --defender 1:121
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import tempfile
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.dataset import load_dir, split_by_episode  # noqa: E402
from fheroes2_agent.policy import BattlePolicy  # noqa: E402
from fheroes2_agent.train_critic import train as fit_critic  # noqa: E402
from fheroes2_agent.train_ppo import train as train_ppo  # noqa: E402

SOLVED = 0.95
COLLAPSED = 0.5


def agreement(checkpoint: str, data_dir: str) -> float:
    """Held-out teacher agreement, to price what fitting the critic cost the cloned policy."""
    samples = load_dir(data_dir)
    _, held = split_by_episode(samples, 0.2, 0)
    model = BattlePolicy()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()
    obs, masks = torch.from_numpy(held.observations), torch.from_numpy(held.masks)
    actions = torch.from_numpy(held.actions)
    hits = 0
    with torch.no_grad():
        for i in range(0, len(actions), 4096):
            logits, _ = model(obs[i : i + 4096], masks[i : i + 4096])
            hits += int((logits.argmax(-1) == actions[i : i + 4096]).sum())
    return hits / len(actions)


def summarize(rows: list[dict], arm: str) -> dict:
    a = [r for r in rows if r["arm"] == arm]
    finals = [r["final5"] for r in a]
    collapses = [r for r in a if r["best"] >= SOLVED and r["final5"] < COLLAPSED]
    return {
        "arm": arm,
        "seeds": len(a),
        "mean_final5": statistics.mean(finals),
        "stderr": statistics.stdev(finals) / len(finals) ** 0.5 if len(finals) > 1 else 0.0,
        "stdev": statistics.stdev(finals) if len(finals) > 1 else 0.0,
        "solved": sum(1 for r in a if r["best"] >= SOLVED),
        "collapsed": len(collapses),
        "collapsed_seeds": [r["seed"] for r in collapses],
        "value_loss_first": statistics.mean(r["value_loss_first"] for r in a),
        "value_loss_last": statistics.mean(r["value_loss_last"] for r in a),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", help="recorded teacher episodes, for fitting the critic")
    parser.add_argument("checkpoint", help="the cloned policy from stage 1")
    parser.add_argument("worker")
    parser.add_argument("--attacker", required=True)
    parser.add_argument("--defender", required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--critic-epochs", type=int, default=20)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    work = pathlib.Path(tempfile.mkdtemp())
    started = time.time()

    # Part one: fit the critic two ways, and price what each costs the cloned policy.
    print("fitting the value head on teacher play")
    frozen = fit_critic(args.data_dir, args.checkpoint, out=str(work / "frozen.pt"),
                        epochs=args.critic_epochs, freeze_policy=True)
    thawed = fit_critic(args.data_dir, args.checkpoint, out=str(work / "thawed.pt"),
                        epochs=args.critic_epochs, freeze_policy=False)

    fits = [
        {"variant": "cloned, value head at initialization", "explained_variance": frozen["before"]["explained_variance"],
         "agreement": agreement(args.checkpoint, args.data_dir)},
        {"variant": "value head only, trunk frozen", "explained_variance": frozen["after"]["explained_variance"],
         "agreement": agreement(str(work / "frozen.pt"), args.data_dir)},
        {"variant": "end to end", "explained_variance": thawed["after"]["explained_variance"],
         "agreement": agreement(str(work / "thawed.pt"), args.data_dir)},
    ]
    print(f"\n  {'variant':38s} {'explained var':>14s} {'agreement':>11s}")
    for f in fits:
        print(f"  {f['variant']:38s} {f['explained_variance']:>+14.3f} {f['agreement']:>11.4f}")

    # Part two: does it help? Only the frozen fit is carried forward, since the end-to-end one
    # damages the very warm start stage 3 depends on.
    print(f"\nrunning {args.seeds} paired seeds per arm on {args.attacker} against {args.defender}")
    rows = []
    for arm, ckpt in [("cold critic", args.checkpoint), ("pre-fitted critic", str(work / "frozen.pt"))]:
        for seed in range(args.seeds):
            r = train_ppo(args.worker, checkpoint=ckpt, attacker=args.attacker, defender=args.defender,
                          iterations=args.iterations, episodes_per_iter=args.episodes, seed=seed, quiet=True)
            wins = [h["win_rate"] for h in r["history"]]
            rows.append({"arm": arm, "seed": seed, "initial": r["initial_win_rate"],
                         "final5": statistics.mean(wins[-5:]), "best": max(wins), "history": wins,
                         "value_loss_first": r["history"][0]["value_loss"],
                         "value_loss_last": r["history"][-1]["value_loss"]})
            print(f"  {arm:18s} seed {seed:2d}  {r['initial_win_rate']:.3f} -> {statistics.mean(wins[-5:]):.3f}"
                  f"  (best {max(wins):.3f})", flush=True)

    arms = [summarize(rows, a) for a in ("cold critic", "pre-fitted critic")]
    print(f"\n  {'arm':18s} {'last-five':>16s} {'spread':>9s} {'solved':>8s} {'collapsed':>11s}")
    for s in arms:
        print(f"  {s['arm']:18s} {s['mean_final5']:.3f} +- {s['stderr']:.3f}  {s['stdev']:>8.3f}"
              f"  {s['solved']:>3d}/{s['seeds']:<4d} {s['collapsed']:>6d}/{s['seeds']:<4d}")

    cold = {r["seed"]: r["final5"] for r in rows if r["arm"] == "cold critic"}
    warm = {r["seed"]: r["final5"] for r in rows if r["arm"] == "pre-fitted critic"}
    diff = [warm[s] - cold[s] for s in sorted(cold)]
    se = statistics.stdev(diff) / len(diff) ** 0.5
    print(f"\n  paired difference {statistics.mean(diff):+.3f} +- {se:.3f} "
          f"({abs(statistics.mean(diff)) / se if se else 0:.1f} SE over {len(diff)} seeds)")
    print(f"  {time.time() - started:.0f}s total")

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"fits": fits, "arms": arms, "runs": rows,
             "paired_difference": {"mean": statistics.mean(diff), "stderr": se, "n": len(diff)},
             "matchup": {"attacker": args.attacker, "defender": args.defender},
             "iterations": args.iterations, "episodes_per_iteration": args.episodes,
             "seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
