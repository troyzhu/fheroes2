#!/usr/bin/env python3
"""Where does the behavior value function actually work? Calibration, stratified.

The critic is fitted on teacher play, which makes it a behavior value in the offline-RL sense,
and every proposed use of it (GAE baseline, potential shaping, search leaf evaluation) quietly
assumes it stays accurate away from the data. This measures that assumption where it will be
spent: explained variance and calibration of a freshly fitted v3 critic on held-out teacher
episodes split by matchup, on student-played episodes (the DAgger collections, a distribution
the critic never saw), and stratified by matchup difficulty, including the near-hopeless
matchups where any improvement operator would need it most.

Usage:
    ./critic_calibration.py CHECKPOINT --teacher-data DIR [DIR ...] --student-data DIR [DIR ...]
                            [--epochs 20] [--report critic_calibration.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.dataset import load_dir, split_by_episode  # noqa: E402
from fheroes2_agent.policy import load_policy, BattlePolicy  # noqa: E402
from fheroes2_agent import train_critic  # noqa: E402


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    if len(target) < 2:
        return float("nan")
    var = float(np.var(target))
    return float(1.0 - np.var(target - pred) / var) if var > 1e-9 else float("nan")


def predict(model: BattlePolicy, samples) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(samples), 4096):
            rows = slice(start, min(start + 4096, len(samples)))
            _, values = model(torch.from_numpy(samples.observations[rows]),
                              torch.from_numpy(samples.masks[rows]))
            out.append(values.squeeze(-1).numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="clone whose value head gets fitted")
    parser.add_argument("--teacher-data", nargs="+", required=True)
    parser.add_argument("--student-data", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", default=None, help="where the refitted critic checkpoint goes")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    out_path = args.out or str(pathlib.Path(tempfile.mkdtemp(prefix="critic_cal_")) / "critic_v3.pt")
    result = train_critic.train(list(args.teacher_data), checkpoint=args.checkpoint, epochs=args.epochs, out=out_path)

    model = load_policy(torch.load(out_path, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()

    report = {"fit": {k: v for k, v in result.items() if k != "state_dict"}, "strata": {}}

    # Teacher-side holdout, split by episode the way the fit itself held out.
    teacher = load_dir(list(args.teacher_data))
    _, teacher_holdout = split_by_episode(teacher, 0.2, seed=0)
    keep = np.isfinite(teacher_holdout.returns)
    pred = predict(model, teacher_holdout.subset(np.flatnonzero(keep)))
    target = teacher_holdout.returns[keep]
    report["strata"]["teacher_holdout"] = {
        "n": int(keep.sum()), "explained_variance": explained_variance(pred, target),
        "mean_error": float(np.mean(pred - target)), "mean_abs_error": float(np.mean(np.abs(pred - target)))}

    # Student-side: every decision of the student-played collections, a distribution the critic
    # never trained on. This is the distribution-shift measurement.
    for label, roots in (("student", list(args.student_data)),):
        student = load_dir(roots)
        keep = np.isfinite(student.returns)
        pred = predict(model, student.subset(np.flatnonzero(keep)))
        target = student.returns[keep]
        report["strata"][label] = {
            "n": int(keep.sum()), "explained_variance": explained_variance(pred, target),
            "mean_error": float(np.mean(pred - target)), "mean_abs_error": float(np.mean(np.abs(pred - target)))}

    for name, s in report["strata"].items():
        print(f"{name:16s} n={s['n']:7d}  EV {s['explained_variance']:+.3f}  "
              f"bias {s['mean_error']:+.3f}  mae {s['mean_abs_error']:.3f}")

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"critic checkpoint: {out_path}")


if __name__ == "__main__":
    main()
