#!/usr/bin/env python3
"""Does knowing the dice change what the search teacher actually labels?

Every corpus this project has distilled was labelled by `search_teacher.py`, which builds its side
environment with the live episode's `seed_offset` and no combat offset. The battlefield matches,
which the prefix replay requires, but the combat stream is derived from the same world seed, so the
teacher chose while seeing the rolls the battle was about to make. `search_leakage.py` showed that
this is worth $+0.323$ to the teacher's own win rate, which says the leaky teacher plays a
different and much stronger game.

Whether that matters for the corpus is a separate question, and a cheaper one. A student imitates
actions, not win rates. If the two teachers pick the same move at the same state almost always, the
labels are nearly the same and re-collecting buys little; if they diverge often, every corpus on
record encodes advice conditioned on a realization the student will never see, and re-collection is
the first thing to do.

Both teachers are run at the *same* states, taken from one live episode driven by the leaky
teacher, so this is a paired comparison of choices rather than two separate playthroughs.

A control arm is mandatory here and its absence made the first run of this script uninterpretable.
Two effects masquerade as leakage. A budget spread thinly over many candidates estimates each from
very few playouts, so two searches disagree often from sampling noise alone, whatever they know
about the dice. And the value gap one search assigns to the other's pick is negative by
construction, because each search chose the argmax of its own estimates and everything else it
scored is below that: the winner's curse, not evidence. So a third teacher runs with a *different*
independent dice offset, and every quantity is reported for the honest-versus-leaky pair beside the
honest-versus-honest pair. Only the difference between those two columns is attributable to knowing
the dice; the honest-versus-honest column is what noise alone produces.

Usage:
    ./teacher_leakage.py WORKER CHECKPOINT [--armies A B ...] [--episodes 6] [--simulations 16]
                         [--report R.json]
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
from fheroes2_agent.search import search_action_detail  # noqa: E402

ARMIES = ["9:2,11:2,6:12,1:30", "62:3,30:6,15:10", "13:3,48:12,12:20",
          "10:4,7:8", "28:3,40:8,2:15", "51:4,50:4,12:16"]
COMBAT_OFFSET = 987631
#: A second, unrelated stream. The honest-versus-honest pair built from these two is the noise floor
#: every leakage claim has to clear.
CONTROL_OFFSET = 424243


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--armies", nargs="+", default=ARMIES)
    parser.add_argument("--sides", nargs="+", default=["attacker", "defender"])
    parser.add_argument("--episodes", type=int, default=6, help="per army per side")
    parser.add_argument("--battlefields", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--margin", default="hit_points", choices=REWARD_MARGINS)
    parser.add_argument("--coverage-forced", action="store_true",
                        help="the collector's own rule, which is what a corpus is actually labelled under")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    model = load_policy(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()

    started = time.time()
    rows = []
    for side in args.sides:
        for army in args.armies:
            for offset in range(args.battlefields):
                base = dict(side=side, attacker=army, defender=army, attacker_hero="10:10",
                            defender_hero="10:10", allow_wide=True, reward_margin=args.margin)
                env = BattleEnv(args.worker, seeds=1, seed_offset=offset, **base)
                leaky = BattleEnv(args.worker, seeds=1, seed_offset=offset, **base)
                honest = BattleEnv(args.worker, seeds=1, seed_offset=offset,
                                   combat_seed_offset=COMBAT_OFFSET, **base)
                control = BattleEnv(args.worker, seeds=1, seed_offset=offset,
                                    combat_seed_offset=CONTROL_OFFSET, **base)
                try:
                    for _ in range(max(args.episodes // args.battlefields, 1)):
                        torch.manual_seed(555 + offset)
                        observation, mask = env.reset()
                        prefix: list[int] = []
                        while True:
                            # Both teachers are asked at the identical state, so the comparison is
                            # of choices rather than of two divergent playthroughs.
                            a_leak, m_leak, _, prior = search_action_detail(
                                leaky, model, prefix, observation, mask, args.simulations, 1.5,
                                live=env, coverage_forced=args.coverage_forced)
                            a_hon, m_hon, _, _ = search_action_detail(
                                honest, model, prefix, observation, mask, args.simulations, 1.5,
                                live=env, coverage_forced=args.coverage_forced)
                            a_ctl, _, _, _ = search_action_detail(
                                control, model, prefix, observation, mask, args.simulations, 1.5,
                                live=env, coverage_forced=args.coverage_forced)
                            greedy = max(prior, key=prior.get)
                            rows.append({
                                "agree": a_leak == a_hon,
                                # A label only teaches something new where search left the prior.
                                "informative": a_leak != greedy,
                                "agree_informative": (a_leak == a_hon) if a_leak != greedy else None,
                                # What the honest teacher thinks the leaky teacher's pick is worth,
                                # relative to its own pick. Negative means the leaky choice is worse
                                # under honest dice, which is the shape leakage would produce.
                                "value_gap": float(m_hon.get(a_leak, 0.0) - m_hon.get(a_hon, 0.0)),
                                # The control: a second honest teacher, differing from the first only
                                # in its dice. Whatever this column shows is noise.
                                "agree_control": a_ctl == a_hon,
                                "value_gap_control": float(m_hon.get(a_ctl, 0.0) - m_hon.get(a_hon, 0.0)),
                                "candidates": len(prior)})
                            # The leaky teacher drives, because that is what collection did.
                            prefix.append(a_leak)
                            step = env.step(a_leak)
                            if step.done:
                                rows[-1]["won"] = bool(_side_won(step.info, side))
                                break
                            observation, mask = step.observation, step.mask
                finally:
                    env.close()
                    leaky.close()
                    honest.close()
                    control.close()
            print(f"  {side:9s} {army:22s} {len(rows)} decisions compared "
                  f"({time.time() - started:.0f}s)", flush=True)

    informative = [r for r in rows if r["informative"]]
    agree = float(np.mean([r["agree"] for r in rows]))
    agree_ctl = float(np.mean([r["agree_control"] for r in rows]))
    agree_inf = float(np.mean([r["agree"] for r in informative])) if informative else float("nan")
    agree_inf_ctl = float(np.mean([r["agree_control"] for r in informative])) if informative else float("nan")
    gaps = np.array([r["value_gap"] for r in rows])
    gaps_ctl = np.array([r["value_gap_control"] for r in rows])
    # Paired per decision, so the difference is what knowing the dice adds over pure sampling noise.
    d_gap = gaps - gaps_ctl
    sem = lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))  # noqa: E731
    print(f"\n{len(rows)} decisions at identical states, {args.simulations} simulations over "
          f"{np.mean([r['candidates'] for r in rows]):.1f} legal moves, coverage_forced={args.coverage_forced}\n")
    print(f"{'':44s}{'leaky vs honest':>17s}{'honest vs honest':>18s}{'difference':>12s}")
    print(f"{'picks the same move':44s}{agree:17.3f}{agree_ctl:18.3f}{agree - agree_ctl:+12.3f}")
    print(f"{'... where search left the policy prior':44s}{agree_inf:17.3f}{agree_inf_ctl:18.3f}"
          f"{agree_inf - agree_inf_ctl:+12.3f}")
    print(f"{'value the honest teacher assigns the pick':44s}{gaps.mean():+17.4f}{gaps_ctl.mean():+18.4f}"
          f"{d_gap.mean():+12.4f}")
    print(f"{'  paired standard error':44s}{sem(gaps):17.4f}{sem(gaps_ctl):18.4f}{sem(d_gap):12.4f}")
    print(f"\nThe honest-versus-honest column is the noise floor: two searches differing only in their")
    print(f"dice. Only the difference column can be attributed to knowing the live battle's rolls.")
    if abs(d_gap.mean()) < 2 * sem(d_gap):
        print(f"\nVERDICT: the leaky teacher's labels are NOT distinguishable from an honest teacher's")
        print(f"at this budget. The apparent divergence is sampling noise, not leakage.")
    else:
        print(f"\nVERDICT: knowing the dice changes the label by {d_gap.mean():+.4f} beyond noise, "
              f"{abs(d_gap.mean())/sem(d_gap):.1f} standard errors.")

    report = {"checkpoint": pathlib.Path(args.checkpoint).name, "simulations": args.simulations,
              "coverage_forced": args.coverage_forced, "combat_seed_offset": COMBAT_OFFSET,
              "margin": args.margin, "decisions": len(rows), "agreement": agree,
              "informative_decisions": len(informative), "agreement_on_informative": agree_inf,
              "agreement_control": agree_ctl, "agreement_on_informative_control": agree_inf_ctl,
              "mean_value_gap": float(gaps.mean()), "value_gap_sem": sem(gaps),
              "mean_value_gap_control": float(gaps_ctl.mean()), "value_gap_control_sem": sem(gaps_ctl),
              "paired_gap_difference": float(d_gap.mean()), "paired_gap_difference_sem": sem(d_gap),
              "seconds": round(time.time() - started, 1)}
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=1))
    print("TEACHER LEAKAGE COMPLETE")


if __name__ == "__main__":
    main()
