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
           ("mean_length", "dec"), ("mean_rounds", "rnds"), ("entropy", "H"),
           ("normalized_entropy", "Hnorm"), ("effective_actions", "effA"),
           ("support_at_1pct", "sup"), ("legal_actions", "legal"),
           ("search_visit_entropy", "Hvis"))
DEFAULT_SUITES = ("held_out_pool", "thunk_ladder", "held_out_as_defender", "mirrors_attacker",
                  "mirrors_defender", "stress_commanders", "stress_hordes", "fresh_sampled", "real_maps")


def as_battery(report: dict) -> dict:
    """Normalize a search-battery report onto the policy battery's shape.

    Two harnesses produce report cards and they nest differently: `validation_battery.py` writes
    `results[checkpoint][suite]` beside `quality[checkpoint][suite][column]`, and
    `search_agent_battery.py` writes `arms[arm][suite]` with the columns inline. That difference is
    why the full column block had no printer for searched arms and every searched summary collapsed
    to the rate, which is the failure this script exists to prevent. Rather than teach every reader
    both shapes, the search shape is translated into the battery shape here.
    """
    if "arms" not in report:
        return report
    results: dict = {}
    quality: dict = {}
    for arm, suites in report["arms"].items():
        results[arm] = {}
        quality[arm] = {}
        for suite, cell in suites.items():
            per = cell.get("per_matchup", [])
            results[arm][suite] = [m.get("win_rate") for m in per]
            quality[arm][suite] = {k: [m.get(k) for m in per]
                                   for k in {k for m in per for k in m}}
    stamp = {k: report.get(k) for k in ("episodes", "eval_seeds", "deployment", "seed",
                                        "simulations", "search_combat_offset")}
    return {"results": results, "quality": quality,
            "reward_margin": report.get("search_objective", report.get("reward_margin")),
            **{k: v for k, v in stamp.items() if v is not None}}


def merge_seeds(reports: list[dict]) -> dict:
    """Average several seeded runs of one configuration, matchup by matchup.

    Seeding the battery made repeats necessary, and a mean of three runs is the unit a suite verdict
    is quoted in now. Averaging per matchup rather than per suite keeps the per-matchup vectors
    intact, so the decided-matchup count and the paired comparisons downstream still work on the
    merged report. The per-seed suite means are kept beside them for the spread line.
    """
    base = reports[0]
    merged = {k: v for k, v in base.items() if k not in ("results", "quality")}
    merged["seed"] = "+".join(str(r.get("seed")) for r in reports)
    merged["results"], merged["quality"], merged["per_seed"] = {}, {}, {}
    for arm in base["results"]:
        merged["results"][arm], merged["quality"][arm], merged["per_seed"][arm] = {}, {}, {}
        for suite in base["results"][arm]:
            stacks = [r["results"][arm][suite] for r in reports if suite in r["results"].get(arm, {})]
            width = min(len(s) for s in stacks)
            merged["results"][arm][suite] = [
                float(np.mean([s[i] for s in stacks if s[i] is not None])) for i in range(width)]
            merged["per_seed"][arm][suite] = [float(np.mean([x for x in s if x is not None])) for s in stacks]
            keys = set().union(*(r["quality"][arm][suite] for r in reports if suite in r["quality"].get(arm, {})))
            merged["quality"][arm][suite] = {}
            for k in keys:
                cols = [r["quality"][arm][suite].get(k, []) for r in reports]
                cols = [c for c in cols if c]
                if not cols:
                    continue
                w = min(len(c) for c in cols)
                merged["quality"][arm][suite][k] = [
                    float(np.mean([c[i] for c in cols if isinstance(c[i], (int, float))]))
                    if any(isinstance(c[i], (int, float)) for c in cols) else None for i in range(w)]
    return merged


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
    parser.add_argument("report", nargs="+",
                        help="one report, or several seeded runs of one configuration, which are "
                             "averaged per matchup so the block reads as one arm with a seed spread "
                             "beneath it. Mixing configurations here would silently average them")
    parser.add_argument("--baseline", default=None, help="a builtin_ai_baseline report for the AI columns")
    parser.add_argument("--group", nargs="+", default=None,
                        help="checkpoint-name prefixes to average as arms; default is one arm per checkpoint")
    parser.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES))
    args = parser.parse_args()

    loaded = [as_battery(json.loads(pathlib.Path(r).read_text())) for r in args.report]
    report = merge_seeds(loaded) if len(loaded) > 1 else loaded[0]
    results = report["results"]
    names = list(results)
    groups = {g: [n for n in names if g in n] for g in args.group} if args.group else \
        {pathlib.Path(n).stem: [n] for n in names}
    groups = {g: ns for g, ns in groups.items() if ns}

    ai = None
    if args.baseline:
        ai = as_battery(json.loads(pathlib.Path(args.baseline).read_text()))

    stamp = [f"deployment={report.get('deployment', 'sample')}",
             *( [f"simulations={report['simulations']}",
                 f"search_combat_offset={report.get('search_combat_offset', 0)}"]
                if "simulations" in report else [] ),
             *( [f"seed={report['seed']}"] if report.get("seed") is not None else [] ),
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
            per_seed = report.get("per_seed", {}).get(members[0], {}).get(suite)
            if per_seed and len(per_seed) > 1:
                arr = np.array(per_seed)
                print(f"  {'':22s}seed spread {arr.std(ddof=1):.3f} over {len(arr)} runs "
                      f"({', '.join(f'{x:.3f}' for x in sorted(arr))}), SE {arr.std(ddof=1)/np.sqrt(len(arr)):.3f}")
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
