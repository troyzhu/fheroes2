#!/usr/bin/env python3
"""Convergence analysis over training heartbeats: is the run still moving, and where.

The owner asked whether convergence is ever assessed, and it was not: heartbeats recorded every
iteration and no verdict ever read their trends. This report does, per metric: the trailing-third
mean against the middle-third mean (is the level still shifting), an ordinary least-squares slope
over the trailing third scaled to per-hundred-iterations (is it still trending), and the
trailing-third standard deviation (is it noise or drift). The verdict per metric is `converged`
when the trailing slope is small against the trailing noise, `trending` when it is not, and
`oscillating` when level shift is small but variance is large.

A converged win rate with a trending value loss is a policy at rest under a critic still moving,
and vice versa; the whole point is that "converged" is per-metric, never one word for a run.

Usage:
    ./convergence_report.py HEARTBEAT.jsonl [HEARTBEAT.jsonl ...] [--report R.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

METRICS = ("win_rate", "mean_terminal_reward", "value_loss", "entropy", "normalized_entropy",
           "raw_advantage_std", "loss_policy", "loss_total")


def analyze(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"iterations": n}
    if n < 9:
        out["note"] = "too short for trend analysis"
        return out
    third = n // 3
    for metric in METRICS:
        series = np.array([r[metric] for r in rows if metric in r], dtype=float)
        if len(series) < 9:
            continue
        mid = series[third:2 * third]
        tail = series[2 * third:]
        x = np.arange(len(tail), dtype=float)
        slope = float(np.polyfit(x, tail, 1)[0]) * 100.0
        noise = float(tail.std(ddof=1))
        level_shift = float(tail.mean() - mid.mean())
        # Slope over the tail in units of tail noise per hundred iterations.
        slope_in_noise = abs(slope) / noise if noise > 1e-9 else 0.0
        if slope_in_noise < 0.5 and abs(level_shift) < noise:
            verdict = "converged"
        elif slope_in_noise >= 0.5:
            verdict = "trending"
        else:
            verdict = "oscillating"
        out[metric] = {"tail_mean": round(float(tail.mean()), 4), "mid_mean": round(float(mid.mean()), 4),
                       "level_shift": round(level_shift, 4), "slope_per_100_iters": round(slope, 4),
                       "tail_noise": round(noise, 4), "verdict": verdict}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("heartbeats", nargs="+")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    results = {}
    for path in args.heartbeats:
        rows = [json.loads(line) for line in pathlib.Path(path).read_text().splitlines()]
        name = pathlib.Path(path).name.replace(".heartbeat.jsonl", "")
        results[name] = analyze(rows)
        print(f"== {name} ({results[name].get('iterations')} iterations)")
        for metric, stats in results[name].items():
            if isinstance(stats, dict):
                print(f"  {metric:22s} {stats['verdict']:12s} tail {stats['tail_mean']:+.3f} "
                      f"shift {stats['level_shift']:+.3f} slope/100 {stats['slope_per_100_iters']:+.4f} "
                      f"noise {stats['tail_noise']:.4f}")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
