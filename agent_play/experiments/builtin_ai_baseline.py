#!/usr/bin/env python3
"""The built-in AI's own win rates on the validation suites: the baseline everyone recognizes.

Every trained checkpoint here descends from this AI's demonstrations, so "better than the clone"
answers a question about our pipeline while "better than the engine's own AI" answers the
question a player would ask. The engine plays both sides natively without any controller
attached, so the measurement is just episodes: for each suite matchup, run the worker AI against
AI over many battlefields and count how often the measured side wins.

The teacher is deterministic per battlefield, so the rate comes from battlefield variety rather
than from resampling; the suites and episode counts otherwise match `validation_battery.py`.

Usage:
    ./builtin_ai_baseline.py WORKER [--episodes 24] [--report builtin_ai_baseline.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from validation_battery import SUITE_SIDE, build_suites  # noqa: E402


def ai_win_rate(worker: str, matchup, side: str, episodes: int) -> float:
    """Fraction of battlefields the given side wins with the built-in AI commanding both armies."""
    with tempfile.TemporaryDirectory(prefix="ai_baseline_") as tmp:
        cmd = [worker, "--attacker", matchup.attacker, "--defender", matchup.defender,
               "--seeds", str(episodes), "--fixture", "m1_tiny_melee",
               "--trajectory-dir", tmp, "--audit-coverage", "--quiet"]
        if matchup.attacker_hero:
            cmd += ["--attacker-hero", matchup.attacker_hero]
        if matchup.defender_hero:
            cmd += ["--defender-hero", matchup.defender_hero]
        if matchup.allow_wide:
            cmd.append("--allow-wide")
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            detail = (run.stderr or "").strip().splitlines()
            raise RuntimeError(detail[-1] if detail else f"worker exited {run.returncode}")

        wins = total = 0
        for path in sorted(pathlib.Path(tmp).rglob("*.jsonl")):
            for line in path.read_text().splitlines():
                record = json.loads(line)
                if record.get("record") != "terminal":
                    continue
                total += 1
                own, foe = ("attacker", "defender") if side == "attacker" else ("defender", "attacker")
                wins += record[own]["live_stacks"] > 0 and record[foe]["live_stacks"] == 0
        return wins / total if total else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("--episodes", type=int, default=24, help="battlefields per matchup")
    parser.add_argument("--fresh", type=int, default=24)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    suites = build_suites(args.fresh)
    started = time.time()
    results = {}
    for suite, matchups in suites.items():
        side = SUITE_SIDE.get(suite, "attacker")
        rates = []
        for m in matchups:
            try:
                rates.append(ai_win_rate(args.worker, m, side, args.episodes))
            except RuntimeError as error:
                print(f"  {suite}: matchup rejected ({error})", flush=True)
        results[suite] = rates
        if rates:
            print(f"built-in AI  {suite:22s} mean {np.mean(rates):.3f}  " +
                  " ".join(f"{r:.2f}" for r in rates[:8]) + (" ..." if len(rates) > 8 else ""), flush=True)

    print(f"\ntotal {round(time.time() - started)}s")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"results": {"builtin_ai": results}, "episodes": args.episodes}, indent=2))


if __name__ == "__main__":
    main()
