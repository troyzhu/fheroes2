#!/usr/bin/env python3
"""Does the searched ladder saturate because search saturates, or because PUCT allocates?

ADR 0008 read the ladder at 0.594, 0.700, 0.750 and 0.744 playing 4, 8, 16 and 32 playouts, with 64
below 32 on the mirror grid, and concluded that sixteen to thirty-two is the deployment range. That
conclusion attributes the flattening to search. There is a second explanation it did not separate.

PUCT was designed to minimise cumulative regret, which is the right objective when a node's estimate
feeds a parent. The root has no parent, so only the action finally played matters and the root is a
simple-regret problem (Bubeck et al. 2011; Danihelka et al. 2022 for the AlphaZero-specific
argument). This project's search is a single ply, so the root is the whole search and the mismatch is
total. Under PUCT extra budget mostly refines a leader the prior had already chosen, which is exactly
what a saturating curve looks like. Sequential Halving splits a doubled budget into doubled per-arm
samples in every phase, so it has a mechanism for continuing to pay.

The two therefore make different predictions about budget, and this reports both curves side by side
on the same checkpoints, suites, dice and evaluation seed, with every quality column the battery
carries. If halving keeps climbing where PUCT flattens, the ADR's range is a statement about PUCT
rather than about search, and it needs qualifying rather than repeating.

The candidate cap is not an optional extra here. Uncapped at 26 legal actions and 48 playouts the
schedule can afford one visit per candidate in phase one and the winner ends on four, which is the
uniform coverage that already measured negative. Capping is what makes the algorithm applicable, and
it is also a confound: a capped halving arm changes both the candidate set and the allocation, so
`--allocator puct --candidates m` exists as the control that separates them.

Usage:
    ./allocator_scaling.py --reports DIR [--baseline puct] [--report R.json]

Reads reports named n{budget}_{allocator}_m{cap}_s{seed}.json, which is what the sweep driver writes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

#: Every column that has ever changed a verdict on this project. The rate is deliberately not first:
#: the strength margin has been the sharpest instrument at every dose measured, and the rate alone
#: would have called three separate real effects noise.
COLUMNS = ("win_rate", "mean_reward", "strength_margin", "reward_on_wins", "reward_on_losses",
           "effective_actions", "normalized_entropy", "search_visit_entropy", "mean_rounds")
NAME = re.compile(r"n(\d+)_([a-z_]+)_m(\d+)_s(\d+)\.json$")


def load(reports_dir: str) -> dict:
    """Group every report by (budget, allocator, cap), keeping one entry per seed."""
    arms: dict = {}
    for path in sorted(glob.glob(os.path.join(reports_dir, "n*_*_m*_s*.json"))):
        m = NAME.search(os.path.basename(path))
        if not m:
            continue
        budget, alloc, cap, seed = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        blob = json.load(open(path))
        # A report that forgot its offset measured the leaky ceiling, not the agent, so it is
        # dropped rather than averaged in. ADR 0008 retracted a headline over exactly this.
        if blob.get("simulations", 0) > 0 and blob.get("search_combat_offset", 0) == 0:
            print(f"  DROPPED {os.path.basename(path)}: shared-dice ceiling, not comparable")
            continue
        suites = blob["arms"]["search"]
        row = {c: float(np.mean([suites[s][c] for s in suites if c in suites[s]]))
               for c in COLUMNS if any(c in suites[s] for s in suites)}
        arms.setdefault((budget, alloc, cap), {})[seed] = row
    return arms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", required=True)
    parser.add_argument("--baseline", default="puct", help="allocator to pair every arm against")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    arms = load(args.reports)
    if not arms:
        print("no reports found")
        return

    print(f"\n{'budget':>7s}{'allocator':>20s}{'cap':>5s}{'seeds':>6s}" +
          "".join(f"{c[:11]:>12s}" for c in COLUMNS))
    print("-" * (38 + 12 * len(COLUMNS)))
    for (budget, alloc, cap) in sorted(arms):
        rows = list(arms[(budget, alloc, cap)].values())
        cells = "".join(f"{np.mean([r[c] for r in rows if c in r]):12.3f}"
                        if any(c in r for r in rows) else f"{'-':>12s}" for c in COLUMNS)
        print(f"{budget:7d}{alloc:>20s}{cap if cap else '-':>5}{len(rows):6d}{cells}")

    # Paired within seed against the baseline allocator at the SAME budget, which is the only
    # comparison that holds the checkpoint, the matchups and the dice fixed.
    print(f"\nPaired against {args.baseline} at the same budget, within seed:")
    paired = {}
    for (budget, alloc, cap) in sorted(arms):
        if alloc == args.baseline and not cap:
            continue
        base = arms.get((budget, args.baseline, 0))
        if not base:
            continue
        shared = sorted(set(base) & set(arms[(budget, alloc, cap)]))
        if len(shared) < 2:
            print(f"  n={budget:<4d} {alloc} m={cap}: only {len(shared)} shared seed(s), not paired")
            continue
        line = []
        # Every column, not a chosen three. Restricting the paired block to the rate and two
        # companions is how a summary ends up quoting the rate again, which has happened on this
        # project repeatedly; the strength margin has been the sharpest instrument at every dose
        # measured and the rate the bluntest.
        for c in COLUMNS:
            d = np.array([arms[(budget, alloc, cap)][s][c] - base[s][c] for s in shared])
            se = d.std(ddof=1) / np.sqrt(len(d))
            line.append(f"{c[:11]:>11s} {d.mean():+.4f} ({abs(d.mean())/se if se else 0:.1f}t)")
            paired[f"n{budget}_{alloc}_m{cap}_{c}"] = {"delta": float(d.mean()), "se": float(se),
                                                       "seeds": len(shared)}
        # Three seeds give a paired t with two degrees of freedom, whose 95 percent critical value
        # is 4.30, not the ~2 a normal approximation suggests. The ratio is labelled t rather than
        # SE so it is not read against the wrong distribution, which is an error this project made.
        crit = {2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57}.get(len(shared) - 1, 2.0)
        print(f"  n={budget:<4d} {alloc:>20s} m={cap:<3d} seeds={len(shared)} "
              f"(95% needs t>{crit})")
        for chunk in [line[i:i + 3] for i in range(0, len(line), 3)]:
            print("      " + "   ".join(chunk))

    # The question the sweep exists to answer: does each allocator's own curve still rise?
    print("\nBudget response per allocator, every column, so a flat rate cannot hide a moving margin:")
    for alloc, cap in sorted({(a, c) for (_, a, c) in arms}):
        pts = sorted((b, arms[(b, a, c)]) for (b, a, c) in arms if a == alloc and c == cap)
        if len(pts) < 2:
            continue
        print(f"  {alloc} m={cap if cap else 'all'}")
        for c in COLUMNS:
            series = [f"{b}:{np.mean([r[c] for r in d.values() if c in r]):+.3f}" for b, d in pts
                      if any(c in r for r in d.values())]
            if series:
                print(f"      {c:>20s}  " + "  ->  ".join(series))

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"arms": {f"n{b}_{a}_m{c}": v for (b, a, c), v in arms.items()}, "paired": paired}, indent=1))
    print("\nALLOCATOR SCALING COMPLETE")


if __name__ == "__main__":
    main()
