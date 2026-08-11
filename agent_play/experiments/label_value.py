#!/usr/bin/env python3
"""When the search teacher overrules the policy, is it right?

`teacher_leakage.py` measured that two identically informed searches reproduce only about a quarter
of each other's overrules at the collection budget, so an individual informative label is close to a
coin flip. That does not settle whether the labels are worth learning: averaging thousands of noisy
but unbiased labels still recovers a real signal, and the 2026-08-09 regret weighting did measure
$+0.063$ held-out from upweighting exactly these decisions. The two facts are in tension and this
resolves it directly rather than by another distillation run.

The test is causal and does not ask any search to grade itself, which is what made the first
teacher comparison uninterpretable. At a state where search overruled the policy's argmax, the
battle is played to termination many times from the search's action and many times from the
policy's action, each replay on an independently drawn combat stream. Whichever action leads to
better terminal outcomes is better, measured rather than estimated, and the comparison is paired
because both branches start from the identical position.

Three quantities come out of it. The advantage of the overrule, which is what a distilled student
would be learning. Its spread across states, which says whether a few decisions carry it. And the
same advantage restricted to the overrules the search was most confident about, by visit count,
which is the obvious way to keep the good labels and drop the rest if the mean is near zero.

Usage:
    ./label_value.py WORKER CHECKPOINT [--states 40] [--replays 12] [--simulations 48]
                     [--coverage-forced] [--report R.json]
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

from fheroes2_agent.env import REWARD_MARGINS, BattleEnv, _side_won  # noqa: E402
from fheroes2_agent.policy import load_policy  # noqa: E402
from fheroes2_agent.search import policy_action, search_action_detail  # noqa: E402

ARMIES = ["9:2,11:2,6:12,1:30", "62:3,30:6,15:10", "13:3,48:12,12:20",
          "10:4,7:8", "28:3,40:8,2:15", "51:4,50:4,12:16"]
#: The stream the teacher searches under. Kept away from the replay offsets below so a teacher can
#: never be evaluated on the dice it planned against.
TEACHER_OFFSET = 987631
#: Replays draw from here upward, one distinct stream each.
REPLAY_OFFSET_BASE = 5_000_011


def play_out(worker: str, base: dict, offset: int, combat_offset: int, prefix: list[int],
             first: int, model, side: str) -> float | None:
    """Replay the prefix, take `first`, then let the policy finish. Returns 1.0 for a win.

    The whole battle is replayed rather than resumed because the engine owns the call stack, and
    determinism makes the prefix reproduce the position exactly on a given world seed.
    """
    env = BattleEnv(worker, seeds=1, seed_offset=offset, combat_seed_offset=combat_offset, **base)
    try:
        observation, mask = env.reset()
        for action in prefix:
            step = env.step(action)
            if step.done:
                return None  # the prefix ended the battle; cannot happen mid-episode
            observation, mask = step.observation, step.mask
        if not mask[first]:
            # A different combat stream can make the teacher's action illegal here, which is itself
            # a fact about the label rather than an error, so it is counted and dropped.
            return None
        step = env.step(first)
        while not step.done:
            action = policy_action(model, step.observation, step.mask, env=env)
            step = env.step(action)
        return 1.0 if _side_won(step.info, side) else 0.0
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--armies", nargs="+", default=ARMIES)
    parser.add_argument("--sides", nargs="+", default=["attacker", "defender"])
    parser.add_argument("--states", type=int, default=40, help="overruled decisions to test")
    parser.add_argument("--replays", type=int, default=12, help="independent playouts per action")
    parser.add_argument("--battlefields", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--coverage-forced", action="store_true")
    parser.add_argument("--margin", default="hit_points", choices=REWARD_MARGINS)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    model = load_policy(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()

    started = time.time()
    tested, skipped, illegal = [], 0, 0
    for side in args.sides:
        for army in args.armies:
            for offset in range(args.battlefields):
                if len(tested) >= args.states:
                    break
                base = dict(side=side, attacker=army, defender=army, attacker_hero="10:10",
                            defender_hero="10:10", allow_wide=True, reward_margin=args.margin)
                env = BattleEnv(args.worker, seeds=1, seed_offset=offset, **base)
                sim = BattleEnv(args.worker, seeds=1, seed_offset=offset,
                                combat_seed_offset=TEACHER_OFFSET, **base)
                try:
                    torch.manual_seed(31337 + offset)
                    observation, mask = env.reset()
                    prefix: list[int] = []
                    while len(tested) < args.states:
                        chosen, means, visits, prior = search_action_detail(
                            sim, model, prefix, observation, mask, args.simulations, 1.5,
                            live=env, coverage_forced=args.coverage_forced)
                        greedy = max(prior, key=prior.get)
                        if chosen != greedy:
                            # Both branches are replayed on the same set of fresh streams, so the
                            # comparison is paired on the dice as well as on the position.
                            wins = {"search": [], "policy": []}
                            for r in range(args.replays):
                                co = REPLAY_OFFSET_BASE + r * 7919
                                for label, action in (("search", chosen), ("policy", greedy)):
                                    out = play_out(args.worker, base, offset, co, prefix, action,
                                                   model, side)
                                    if out is None:
                                        continue
                                    wins[label].append(out)
                            if wins["search"] and wins["policy"]:
                                tested.append({
                                    "side": side, "army": army, "battlefield": offset,
                                    "depth": len(prefix),
                                    "search_win": float(np.mean(wins["search"])),
                                    "policy_win": float(np.mean(wins["policy"])),
                                    "replays": min(len(wins["search"]), len(wins["policy"])),
                                    "visits_chosen": visits.get(chosen, 0),
                                    "visits_greedy": visits.get(greedy, 0),
                                    "search_value_gap": float(means.get(chosen, 0.0) - means.get(greedy, 0.0)),
                                    "prior_greedy": float(prior[greedy]),
                                    "candidates": len(prior)})
                                print(f"  state {len(tested):3d}/{args.states}  {side:9s} "
                                      f"depth {len(prefix):2d}  search {tested[-1]['search_win']:.2f} "
                                      f"policy {tested[-1]['policy_win']:.2f}  "
                                      f"visits {visits.get(chosen,0)}v{visits.get(greedy,0)}  "
                                      f"({time.time()-started:.0f}s)", flush=True)
                            else:
                                illegal += 1
                        else:
                            skipped += 1
                        prefix.append(chosen)
                        step = env.step(chosen)
                        if step.done:
                            break
                        observation, mask = step.observation, step.mask
                finally:
                    env.close()
                    sim.close()

    if not tested:
        print("no overruled decisions found")
        return
    adv = np.array([t["search_win"] - t["policy_win"] for t in tested])
    sem = adv.std(ddof=1) / np.sqrt(len(adv))
    print(f"\n{len(tested)} overruled decisions, {args.replays} independent replays per branch, "
          f"{args.simulations} simulations, coverage_forced={args.coverage_forced}")
    print(f"  decisions where search agreed with the policy and were skipped: {skipped}")
    print(f"  overrules dropped because a branch never produced a legal replay: {illegal}\n")
    print(f"  search action wins        {np.mean([t['search_win'] for t in tested]):.3f}")
    print(f"  policy action wins        {np.mean([t['policy_win'] for t in tested]):.3f}")
    print(f"  advantage of the overrule {adv.mean():+.3f}  SE {sem:.3f}  "
          f"({int((adv > 0).sum())} better / {int((adv < 0).sum())} worse / {int((adv == 0).sum())} equal)")
    # If the mean is near zero, confidence is the obvious filter, so it is reported without being
    # proposed: a positive slope here is what would justify keeping only the confident overrules.
    order = np.argsort([-(t["visits_chosen"] - t["visits_greedy"]) for t in tested])
    top = order[: max(len(tested) // 3, 1)]
    print(f"  ... restricted to the third search was most confident about "
          f"{adv[top].mean():+.3f}  SE {adv[top].std(ddof=1)/np.sqrt(len(top)):.3f}")
    verdict = ("the overrules carry real value" if adv.mean() > 2 * sem else
               "the overrules are worse than the prior" if adv.mean() < -2 * sem else
               "the overrules are not distinguishable from the policy's own choice")
    print(f"\nVERDICT: {verdict}.")

    report = {"checkpoint": pathlib.Path(args.checkpoint).name, "simulations": args.simulations,
              "coverage_forced": args.coverage_forced, "replays": args.replays,
              "margin": args.margin, "teacher_offset": TEACHER_OFFSET,
              "states": tested, "skipped_agreements": skipped, "dropped": illegal,
              "mean_advantage": float(adv.mean()), "advantage_sem": float(sem),
              "confident_third_advantage": float(adv[top].mean()),
              "seconds": round(time.time() - started, 1)}
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=1))
    print("LABEL VALUE COMPLETE")


if __name__ == "__main__":
    main()
