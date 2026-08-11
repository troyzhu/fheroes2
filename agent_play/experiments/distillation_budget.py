#!/usr/bin/env python3
"""How long should the student train, and how sharp should it be allowed to get?

Two questions that turn out to be one. Every distillation arm on record peaked at its final epoch
and none of them was ever given a longer horizon: the 2026-08-07 sharpness sweep tested a budget
cut to eight epochs, so the shorter direction is measured and the longer one never was. That gap
matters because `CosineAnnealingLR` is armed `T_max=epochs`, which drives the learning rate to
`eta_min` exactly at the boundary. A run annealed that way flattens at whatever budget it was
given, and a flat tail is not evidence of convergence. It is evidence of a schedule ending. That
is a property of this single-cycle setup rather than of cosine schedules at large: torch's own
scheduler is periodic with period 2*T_max, and under `--restart-period` the rate returns to its
maximum every cycle, where a flat tail would mean what this one cannot.

The trap this exists to avoid is selecting on the wrong number. The loop ships best-agreement
checkpointing, and agreement is top-1 match against the teacher on a held-out episode split. Two
independent reasons say that cannot referee a budget. This project's own sharpness sweep found the
arm with the *worst* agreement was the only one that played better afterwards, and the imitation
literature has the general form of it: Codevilla et al. (ICCV 2019) report that offline prediction
error is not necessarily correlated with driving quality, and that two models with identical
prediction error can differ dramatically in what they actually do.

So every arm here is judged on four things that can disagree, and they are reported together:
held-out agreement, held-out cross-entropy, both entropy forms, and play on the suites that can
separate players. Agreement and loss disagree about where to stop. Loss and play disagree again.

The entropy arm followed from the budget arm rather than being an independent idea, and it is kept
here as a measured negative. The reasoning was that if play tracks the student's entropy rather
than its agreement, sharpening is the damage a long budget does and paying the student to stay
uncertain should recover it. It does not: the motivating correlation is confounded by the budget
moving both, it reverses under intervention, and every dose costs play. `--entropy-bonus` adds
$-\\beta H(\\pi_\\theta(\\cdot \\mid s))$ to the loss and remains sweepable so the result can be
reproduced or overturned, not because it is a knob worth turning.

Both questions are asked of the network alone, because a searched arm costs roughly half a second
per decision and the collapse being measured is a property of the weights. The searched regime is
the separate question of whether a sharp prior starves PUCT, whose exploration term scales with
the prior itself, and `--simulations` is here so that arm can be run on the same suites.

Usage:
    ./distillation_budget.py WORKER --roots DIR [DIR ...] --soft-root DIR --out-dir DIR
                             [--budgets 25 100] [--entropy-bonus 0.0 0.15] [--seeds 3]
                             [--episodes 8] [--simulations 0] [--report R.json]
    ./distillation_budget.py --from-reports DIR [--report R.json]     # aggregate an earlier run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))

#: The suites that can actually separate two players. The battery builds ten; five are left out.
#: Four are saturated (real_maps 22 matchups of 24, stress_hordes 4 of 5, stress_commanders 3 of 4,
#: thunk_ladder 2 of 4) and a suite mean is only readable when its matchups are contested. The
#: fifth, stress_wide_only, is outside the standing nine every earlier scoreboard was quoted on,
#: so including it would make results here incomparable to those.
LIVE_SUITES = ("held_out_pool", "held_out_as_defender", "mirrors_attacker",
               "mirrors_defender", "fresh_sampled")
#: Play columns worth carrying. The rate alone has misreported this program's results more than
#: once, so the quality columns travel with it and the entropy pair says how the rate was earned.
PLAY_KEYS = ("win_rate", "mean_reward", "reward_on_wins", "reward_on_losses", "strength_margin",
             "entropy", "normalized_entropy", "effective_actions", "support_at_1pct", "mean_rounds")
#: Training columns, read from the heartbeat the trainer appends per epoch.
TRAIN_KEYS = ("holdout_agreement", "holdout_loss", "holdout_normalized_entropy",
              "holdout_effective_actions")


def train_arm(roots, soft_root, out_dir, epochs, bonus, seed) -> tuple[str, str]:
    """Train one arm and return (checkpoint path, heartbeat path)."""
    tag = f"b{epochs}_e{bonus}_s{seed}"
    out = str(pathlib.Path(out_dir) / f"{tag}_soft.pt")
    cmd = [sys.executable, str(HERE / "soft_distill.py"),
           "--roots", *roots, "--soft-root", *soft_root,
           "--lam", "0.5", "--soft-weight", "2.0",
           "--epochs", str(epochs), "--seed", str(seed), "--entropy-bonus", str(bonus),
           "--checkpoint-every", "25",
           "--out", out, "--hard-out", str(pathlib.Path(out_dir) / f"{tag}_hard.pt"),
           "--report", str(pathlib.Path(out_dir) / f"{tag}.json")]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return out, out + ".heartbeat.jsonl"


def evaluate(worker, checkpoint, out_dir, suites, episodes, simulations, seed) -> str:
    """Play one checkpoint on the live suites and return the report path."""
    report = str(pathlib.Path(out_dir) / ("play_" + pathlib.Path(checkpoint).name + ".json"))
    cmd = [sys.executable, str(HERE / "search_agent_battery.py"), worker, checkpoint,
           "--suites", *suites, "--episodes", str(episodes),
           "--simulations", str(simulations), "--seed", str(seed), "--report", report]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return report


def read_play(report_path: str, suites) -> dict:
    """Average the play columns over the suites, keeping the per-suite rates beside them."""
    arm = json.load(open(report_path))["arms"]["search"]
    present = [s for s in suites if s in arm]
    out = {k: float(np.mean([arm[s][k] for s in present])) for k in PLAY_KEYS if
           all(k in arm[s] for s in present)}
    out["per_suite_win_rate"] = {s: arm[s]["win_rate"] for s in present}
    return out


def read_training(heartbeat_path: str) -> dict:
    """Both selectors and the trajectory they disagree about."""
    rows = [json.loads(line) for line in open(heartbeat_path) if line.strip()]
    if not rows:
        return {}
    best_agree = max(rows, key=lambda r: r["holdout_agreement"])
    out = {k: rows[-1][k] for k in TRAIN_KEYS if k in rows[-1]}
    out["epochs_run"] = len(rows)
    out["best_agreement_epoch"] = best_agree["epoch"]
    out["best_agreement"] = best_agree["holdout_agreement"]
    if "holdout_loss" in rows[-1]:
        best_loss = min(rows, key=lambda r: r["holdout_loss"])
        out["best_loss_epoch"] = best_loss["epoch"]
        out["best_loss"] = best_loss["holdout_loss"]
        # The whole point of carrying both: when these two epochs differ, the shipped selector is
        # choosing a model that held-out loss says is already past its best.
        out["selector_disagreement_epochs"] = best_agree["epoch"] - best_loss["epoch"]
    return out


def print_table(arms: dict) -> None:
    """One row per arm, every metric that separates them, seeds pooled with their spread."""
    if not arms:
        print("no arms to report")
        return
    print(f"\n{'arm':22s}{'seeds':>6s}{'agree':>8s}{'hLoss':>8s}{'effA_h':>8s}"
          f"{'PLAY':>9s}{'+/-':>7s}{'reward':>8s}{'margin':>8s}{'effA_p':>8s}{'rounds':>8s}")
    print("-" * 108)
    for name, rows in arms.items():
        def col(key, source):
            vals = [r[source].get(key) for r in rows if key in r.get(source, {})]
            return float(np.mean(vals)) if vals else float("nan")
        play = [r["play"]["win_rate"] for r in rows if "win_rate" in r.get("play", {})]
        spread = float(np.std(play, ddof=1)) if len(play) > 1 else float("nan")
        print(f"{name:22s}{len(rows):6d}{col('best_agreement','train'):8.4f}"
              f"{col('best_loss','train'):8.4f}{col('holdout_effective_actions','train'):8.2f}"
              f"{np.mean(play) if play else float('nan'):9.3f}{spread:7.3f}"
              f"{col('mean_reward','play'):8.3f}{col('strength_margin','play'):8.3f}"
              f"{col('effective_actions','play'):8.2f}{col('mean_rounds','play'):8.2f}")
    print("\nagree/hLoss/effA_h are held out against the teacher; PLAY and the columns after it are")
    print("play on the live suites. When agree and PLAY disagree, PLAY is the one that decides.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker", nargs="?")
    parser.add_argument("--roots", nargs="+")
    parser.add_argument("--soft-root", nargs="+")
    parser.add_argument("--out-dir")
    parser.add_argument("--budgets", type=int, nargs="+", default=[25, 100])
    parser.add_argument("--entropy-bonus", type=float, nargs="+", default=[0.0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--suites", nargs="+", default=list(LIVE_SUITES))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=0,
                        help="0 plays the network alone, which is what the collapse question is "
                             "about; a nonzero budget answers the separate question of whether a "
                             "sharp prior starves PUCT")
    parser.add_argument("--eval-seed", type=int, default=17)
    parser.add_argument("--from-reports", default=None,
                        help="skip training and evaluation, aggregate an out-dir written earlier")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    started = time.time()
    arms: dict[str, list] = {}

    if args.from_reports:
        # Re-read a completed run. Arm identity is recovered from the checkpoint name, which is
        # why the naming above is structured rather than free text.
        for play_path in sorted(glob.glob(os.path.join(args.from_reports, "play_*.json"))):
            stem = os.path.basename(play_path)[len("play_"):-len(".json")]
            heartbeat = os.path.join(args.from_reports, stem + ".heartbeat.jsonl")
            name = stem.replace("_soft.pt", "").rsplit("_s", 1)[0]
            row = {"checkpoint": stem, "play": read_play(play_path, args.suites),
                   "train": read_training(heartbeat) if os.path.exists(heartbeat) else {}}
            arms.setdefault(name, []).append(row)
    else:
        if not (args.worker and args.roots and args.soft_root and args.out_dir):
            parser.error("worker, --roots, --soft-root and --out-dir are required unless --from-reports")
        pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        for epochs in args.budgets:
            for bonus in args.entropy_bonus:
                name = f"epochs{epochs}_beta{bonus}"
                for seed in range(args.seeds):
                    print(f"  {name} seed {seed}: training ({time.time()-started:.0f}s)", flush=True)
                    checkpoint, heartbeat = train_arm(args.roots, args.soft_root, args.out_dir,
                                                      epochs, bonus, seed)
                    print(f"  {name} seed {seed}: playing ({time.time()-started:.0f}s)", flush=True)
                    play_path = evaluate(args.worker, checkpoint, args.out_dir, args.suites,
                                         args.episodes, args.simulations, args.eval_seed)
                    arms.setdefault(name, []).append(
                        {"checkpoint": os.path.basename(checkpoint),
                         "play": read_play(play_path, args.suites),
                         "train": read_training(heartbeat)})

    print_table(arms)
    report = {"suites": args.suites, "episodes": args.episodes, "simulations": args.simulations,
              "eval_seed": args.eval_seed, "budgets": args.budgets,
              "entropy_bonus": args.entropy_bonus, "arms": arms,
              "seconds": round(time.time() - started, 1)}
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=1))
    print("DISTILLATION BUDGET COMPLETE")


if __name__ == "__main__":
    main()
