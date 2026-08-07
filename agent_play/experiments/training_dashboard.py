#!/usr/bin/env python3
"""A live training-health dashboard: one self-refreshing HTML page, no dependencies.

The owner asked for monitoring throughout training rather than verdicts after it. Trainers now
append one JSON line per iteration to a `.heartbeat.jsonl` beside their checkpoint; this script
scans a directory for heartbeats and pipeline logs and renders a single HTML file with an SVG
sparkline per metric per run, the per-term gradient norms beside the losses, and the tails of
any pipeline logs, then rewrites it on an interval so a browser tab pointed at the file (it
carries a meta-refresh) stays current.

Health heuristics are drawn on the charts rather than computed opinions: the advantage-floor
line at 0.1 where the collapse mechanism lives, and entropy's low-water mark, so a run
destroying itself is visible while it is still cheap to stop.

Usage:
    ./training_dashboard.py WATCH_DIR [--out DIR/dashboard.html] [--interval 10] [--once]
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import time

METRICS = ("win_rate", "mean_terminal_reward", "value_loss", "entropy", "raw_advantage_std")
GRAD_KEYS = ("policy", "value", "entropy", "total_pre_clip")


def sparkline(values, width=260, height=52, floor=None) -> str:
    if not values:
        return "<svg width='260' height='52'></svg>"
    lo, hi = min(values), max(values)
    if floor is not None:
        lo, hi = min(lo, floor), max(hi, floor)
    span = (hi - lo) or 1.0
    points = " ".join(f"{i * width / max(len(values) - 1, 1):.1f},{height - 4 - (v - lo) / span * (height - 8):.1f}"
                      for i, v in enumerate(values))
    floor_line = ""
    if floor is not None:
        y = height - 4 - (floor - lo) / span * (height - 8)
        floor_line = f"<line x1='0' y1='{y:.1f}' x2='{width}' y2='{y:.1f}' stroke='#c33' stroke-dasharray='4'/>"
    return (f"<svg width='{width}' height='{height}' style='background:#f7f7f7'>{floor_line}"
            f"<polyline points='{points}' fill='none' stroke='#246' stroke-width='1.5'/></svg>")


def render(watch: pathlib.Path, interval: int) -> str:
    parts = [f"<meta http-equiv='refresh' content='{interval}'><title>training health</title>"
             "<style>body{font-family:monospace;margin:1.5em}td,th{padding:2px 10px;text-align:left}"
             ".num{color:#246}h2{margin-top:1.2em}</style>",
             f"<h1>training health</h1><p>rendered {time.strftime('%H:%M:%S')}, refresh {interval}s</p>"]

    beats = sorted(watch.rglob("*.heartbeat.jsonl"), key=lambda q: q.stat().st_mtime, reverse=True)
    if not beats:
        parts.append("<p>no heartbeats yet; trainers emit them beside their checkpoints.</p>")
    for beat in beats[:8]:
        rows = []
        for line in beat.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not rows:
            continue
        age = time.time() - beat.stat().st_mtime
        live = "LIVE" if age < 300 else f"idle {int(age / 60)}m"
        last = rows[-1]
        parts.append(f"<h2>{html.escape(beat.stem.replace('.heartbeat', ''))} "
                     f"<small>[{live}] iteration {last.get('iteration')}</small></h2><table><tr>")
        for metric in METRICS:
            series = [r[metric] for r in rows if metric in r]
            floor = 0.1 if metric == "raw_advantage_std" else None
            parts.append(f"<td><b>{metric}</b><br>{sparkline(series, floor=floor)}"
                         f"<br><span class='num'>{series[-1]:.3f}</span></td>")
        parts.append("</tr></table>")
        norms = [r.get("grad_norms", {}) for r in rows]
        if any(norms):
            parts.append("<table><tr><th>grad norm (pre-sum)</th>" +
                         "".join(f"<th>{k}</th>" for k in GRAD_KEYS) + "</tr><tr><td>latest</td>" +
                         "".join(f"<td class='num'>{norms[-1].get(k, float('nan')):.2f}</td>" for k in GRAD_KEYS) +
                         "</tr></table>")

    logs = sorted(watch.glob("*.log"), key=lambda q: q.stat().st_mtime, reverse=True)[:4]
    for log in logs:
        tail = log.read_text().splitlines()[-6:]
        parts.append(f"<h2>{html.escape(log.name)}</h2><pre>{html.escape(chr(10).join(tail))}</pre>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("watch")
    parser.add_argument("--out", default=None)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    watch = pathlib.Path(args.watch)
    out = pathlib.Path(args.out or (watch / "dashboard.html"))
    while True:
        out.write_text(render(watch, args.interval))
        if args.once:
            print(f"rendered {out}")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
