#!/usr/bin/env python3
"""Do explicit ability features earn their place, on the champion corpus?

The plateau finding says the escalations past generation one are architectural, and this is the
first: each slot's input extended by its creature's fixed ability profile (the capability
audit's layer-1 records: movement and attack flags, log-scaled hit points and strength, ability
and weakness counts), computed inside the model from the one-hot the observation already
carries. No observation bytes change, so no encoding version moves; the change is pure
inductive bias, and per the ADR discipline it ships only if this ablation says it helps.

Both arms clone the same corpus with the same budget and seed; the battery judges them paired.

Usage:
    ./ability_ablation.py WORKER --data DIR [DIR ...] [--epochs 25] [--seed 0]
                          [--report ability_ablation.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent import train_bc  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="ability_ablation_"))
    checkpoints = {}
    fits = {}
    for arm, kwargs in (("plain", {}), ("ability", {"ability_features": True})):
        out = workdir / f"{arm}.pt"
        result = train_bc.train(list(args.data), epochs=args.epochs, seed=args.seed,
                                out=str(out), model_kwargs=kwargs)
        checkpoints[arm] = str(out)
        fits[arm] = {k: v for k, v in result.items() if k != "state_dict"}
        print(f"{arm}: holdout agreement {result.get('best_holdout_agreement')}", flush=True)

    battery = pathlib.Path(__file__).resolve().parent / "validation_battery.py"
    report_path = workdir / "battery.json"
    subprocess.run([sys.executable, str(battery), args.worker, checkpoints["plain"], checkpoints["ability"],
                    "--report", str(report_path)], check=True)

    if args.report:
        payload = {"fits": fits, "checkpoints": checkpoints,
                   "battery": json.loads(report_path.read_text())}
        pathlib.Path(args.report).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
