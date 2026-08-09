#!/usr/bin/env python3
"""Print the canonical multi-metric comparison block for a battery report.

Written 2026-08-09 after the owner had to ask four times for more than win rate. The reports have
always carried every column and the conventions have always required them; what kept failing was
the step where a human summary got written from a report, which collapsed to the rate because the
rate is one number and the rest are five. This makes the full block the cheap thing to produce:
point it at a report and paste what it prints.

Every suite prints rate, win quality, loss quality, unconditional margin, the trained reward and
episode length, with the built-in AI's own columns beside them when its baseline is available, and
graded suites print per rung. A comparison of arms prints the paired per-seed deltas too, because
a mean of three seeds hides whether the sign was consistent.

Usage:
    ./report_summary.py REPORT.json [--baseline builtin_ai_baseline_v2.json]
                        [--group PREFIX ...] [--suites S ...]
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

COLUMNS = (("win_rate", "rate"), ("surviving_strength", "wq"), ("loss_damage", "lq"),
           ("strength_margin", "mg"), ("mean_reward", "rw"), ("reward_on_wins", "rwW"),
           ("reward_on_losses", "rwL"), ("mean_reward_commanded", "rwC"),
           ("mean_length", "dec"), ("mean_rounds", "rnds"), ("normalized_entropy", "Hnorm"),
           ("effective_actions", "effA"), ("legal_actions", "legal"))
DEFAULT_SUITES = ("held_out_pool", "thunk_ladder", "held_out_as_defender", "mirrors_attacker",
                  "mirrors_defender", "stress_commanders", "stress_hordes", "fresh_sampled", "real_maps")


def mean_of(values) -> float:
    kept = [v for v in values if isinstance(v, (int, float))]
    return float(np.mean(kept)) if kept else float("nan")


def suite_rate(results, name, suite) -> float:
    value = results.get(name, {}).get(suite)
    if value is None:
        return float("nan")
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return float(np.mean([r["win_rate"] for r in value]))
    return mean_of(value)


def rungs_of(results, name, suite):
    value = results.get(name, {}).get(suite)
    if not isinstance(value, list) or not value:
        return None
    if isinstance(value[0], dict):
        return [r["win_rate"] for r in value]
    return value if suite == "thunk_ladder" and len(value) <= 6 else None


def column(report, name, suite, key) -> float:
    if key == "win_rate":
        return suite_rate(report["results"], name, suite)
    return mean_of(report.get("quality", {}).get(name, {}).get(suite, {}).get(key, []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report")
    parser.add_argument("--baseline", default=None, help="a builtin_ai_baseline report for the AI columns")
    parser.add_argument("--group", nargs="+", default=None,
                        help="checkpoint-name prefixes to average as arms; default is one arm per checkpoint")
    parser.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES))
    args = parser.parse_args()

    report = json.loads(pathlib.Path(args.report).read_text())
    results = report["results"]
    names = list(results)
    groups = {g: [n for n in names if g in n] for g in args.group} if args.group else \
        {pathlib.Path(n).stem: [n] for n in names}
    groups = {g: ns for g, ns in groups.items() if ns}

    ai = None
    if args.baseline:
        ai = json.loads(pathlib.Path(args.baseline).read_text())

    stamp = [f"deployment={report.get('deployment', 'sample')}",
             f"reward_margin={report.get('reward_margin', 'hit_points (pre-2026-08-08)')}",
             f"episodes={report.get('episodes')}", f"eval_seeds={report.get('eval_seeds')}"]
    print(" ".join(stamp))

    for suite in args.suites:
        if not any(suite in results[n] for n in names):
            continue
        print(f"\n{suite}")
        header = f"  {'arm':22s}" + "".join(f"{tag:>8s}" for _, tag in COLUMNS)
        print(header)
        if ai is not None and suite in ai["results"]["builtin_ai"]:
            row = f"  {'built-in AI':22s}"
            for key, _ in COLUMNS:
                value = mean_of(ai["results"]["builtin_ai"][suite]) if key == "win_rate" else \
                    mean_of(ai.get("quality", {}).get("builtin_ai", {}).get(suite, {}).get(key, []))
                row += f"{value:8.3f}" if value == value else f"{'--':>8s}"
            print(row)
        for group, members in groups.items():
            row = f"  {group:22s}"
            for key, _ in COLUMNS:
                value = float(np.mean([column(report, n, suite, key) for n in members]))
                row += f"{value:8.3f}" if value == value else f"{'--':>8s}"
            print(row)
            # The two numbers that say whether a difference is readable at all: the spread of the
            # arm's own seeds, and how many of the suite's matchups can discriminate anything.
            if len(members) > 1:
                seed_rates = [suite_rate(results, n, suite) for n in members]
                spread = float(np.std(seed_rates, ddof=1) / np.sqrt(len(seed_rates)))
                print(f"  {'':22s}seed SE {spread:.3f} over {len(members)} seeds "
                      f"({', '.join(f'{r:.3f}' for r in sorted(seed_rates))})")
            per_matchup = results.get(members[0], {}).get(suite)
            if isinstance(per_matchup, list) and per_matchup and not isinstance(per_matchup[0], dict):
                decided = sum(1 for r in per_matchup if r <= 0.1 or r >= 0.9)
                print(f"  {'':22s}decided matchups {decided}/{len(per_matchup)} "
                      f"(already at or beyond 0.9 or 0.1, so they cannot separate policies)")
            rungs = rungs_of(results, members[0], suite)
            if rungs and len(members) == 1:
                print(f"  {'':22s}rungs " + "/".join(f"{r:.2f}" for r in rungs))
            elif rungs:
                stacked = np.array([rungs_of(results, n, suite) for n in members], dtype=float)
                print(f"  {'':22s}rungs " + "/".join(f"{r:.2f}" for r in stacked.mean(axis=0)))

    if len(groups) == 2:
        a, b = list(groups)
        print(f"\npaired per-seed deltas, {a} minus {b}, held_out_pool:")
        pairs = list(zip(sorted(groups[a]), sorted(groups[b])))
        deltas = [suite_rate(results, x, "held_out_pool") - suite_rate(results, y, "held_out_pool")
                  for x, y in pairs]
        signs = "all positive" if all(d > 0 for d in deltas) else \
            "all negative" if all(d < 0 for d in deltas) else "mixed signs"
        print("  " + ", ".join(f"{d:+.3f}" for d in deltas) + f"  ({signs})")


if __name__ == "__main__":
    main()
