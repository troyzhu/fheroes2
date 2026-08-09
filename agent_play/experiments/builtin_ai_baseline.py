#!/usr/bin/env python3
"""The built-in AI's own win rates on the validation suites: the baseline everyone recognizes.

Every trained checkpoint here descends from this AI's demonstrations, so "better than the clone"
answers a question about our pipeline while "better than the engine's own AI" answers the
question a player would ask. The engine plays both sides natively without any controller
attached, so the measurement is just episodes: for each suite matchup, run the worker AI against
AI over many battlefields and count how often the measured side wins.

The teacher is deterministic per battlefield, so the rate comes from battlefield variety rather
than from resampling; the suites and episode counts otherwise match `validation_battery.py`.

The baseline carries the same quality columns the battery reports for policies, added 2026-08-08
because the convention of quoting the AI column beside every claim could otherwise be honored on
win rate alone. Definitions match `scenarios.measure` exactly: win quality is engine strength kept
on wins, loss quality is the fraction of the enemy destroyed on losses, the margin is own kept
minus enemy kept over every episode, and the reward is the trained two-sided objective. Two
columns do not transfer and are named apart rather than silently mismatched: the battery's `len`
counts the learner's own decisions, so the AI reports `mean_rounds` (engine rounds) and
`mean_decisions_both_sides` instead, and the historical rate counts a side's win as clearing the
board, so the termination-based rate the battery uses is reported beside it.

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

from fheroes2_agent.env import terminal_reward_two_sided  # noqa: E402
from validation_battery import SUITE_SIDE, build_suites  # noqa: E402


def ai_win_rate(worker: str, matchup, side: str, episodes: int) -> dict:
    """The measured side's rate and quality columns, with the AI commanding both armies."""
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

        wins = wins_by_termination = total = 0
        survival, damage, margins, rewards, rounds, decisions = [], [], [], [], [], []
        own_key, foe_key = ("attacker", "defender") if side == "attacker" else ("defender", "attacker")
        won_termination = "victory" if side == "attacker" else "defeat"
        for path in sorted(pathlib.Path(tmp).rglob("*.jsonl")):
            for line in path.read_text().splitlines():
                record = json.loads(line)
                if record.get("record") != "terminal":
                    continue
                total += 1
                own, foe = record[own_key], record[foe_key]
                cleared = own["live_stacks"] > 0 and foe["live_stacks"] == 0
                wins += cleared
                won = record["termination"] == won_termination
                wins_by_termination += won
                own_initial = float(own.get("initial_strength", 0.0))
                foe_initial = float(foe.get("initial_strength", 0.0))
                own_kept = float(own.get("strength", 0.0)) / own_initial if own_initial > 0 else 0.0
                foe_kept = float(foe.get("strength", 0.0)) / foe_initial if foe_initial > 0 else 0.0
                margins.append(own_kept - foe_kept)
                (survival if won else damage).append(own_kept if won else 1.0 - foe_kept)
                rewards.append(terminal_reward_two_sided(record, side))
                rounds.append(record.get("rounds", 0))
                decisions.append(record.get("decision_count", 0))
    if not total:
        return {"win_rate": float("nan")}
    return {
        "win_rate": wins / total,
        "win_rate_by_termination": wins_by_termination / total,
        "surviving_strength": float(np.mean(survival)) if survival else None,
        "loss_damage": float(np.mean(damage)) if damage else None,
        "strength_margin": float(np.mean(margins)),
        "mean_reward": float(np.mean(rewards)),
        "mean_rounds": float(np.mean(rounds)),
        "mean_decisions_both_sides": float(np.mean(decisions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("--episodes", type=int, default=24, help="battlefields per matchup")
    parser.add_argument("--fresh", type=int, default=24)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    suites = build_suites(args.fresh)
    started = time.time()
    results, quality = {}, {}
    columns = ("surviving_strength", "loss_damage", "strength_margin", "mean_reward",
               "mean_rounds", "mean_decisions_both_sides", "win_rate_by_termination")
    for suite, matchups in suites.items():
        side = SUITE_SIDE.get(suite, "attacker")
        measured = []
        for m in matchups:
            try:
                measured.append(ai_win_rate(args.worker, m, side, args.episodes))
            except RuntimeError as error:
                print(f"  {suite}: matchup rejected ({error})", flush=True)
        rates = [d["win_rate"] for d in measured]
        results[suite] = rates
        quality[suite] = {c: [d.get(c) for d in measured] for c in columns}
        if rates:
            def column(name):
                vals = [v for v in quality[suite][name] if isinstance(v, (int, float))]
                return float(np.mean(vals)) if vals else float("nan")
            print(f"built-in AI  {suite:22s} mean {np.mean(rates):.3f}  "
                  f"wq {column('surviving_strength'):.2f} lq {column('loss_damage'):.2f} "
                  f"mg {column('strength_margin'):+.2f} rw {column('mean_reward'):+.2f} "
                  f"rounds {column('mean_rounds'):.0f}", flush=True)

    print(f"\ntotal {round(time.time() - started)}s")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"results": {"builtin_ai": results}, "quality": {"builtin_ai": quality},
             "episodes": args.episodes}, indent=2))


if __name__ == "__main__":
    main()
