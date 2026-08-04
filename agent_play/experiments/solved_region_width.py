#!/usr/bin/env python3
"""How wide is the solved region around a trained policy, matchup by matchup?

Motivated by a chain of refutations. The advantage-normalization collapse reproduces on one
contested matchup at roughly ten percent of seeds and on no other matchup tested, and every simple
predicate for which matchups are vulnerable has failed: a reward spread of exactly zero is not
sufficient, a sharp policy is not the discriminator, and a contested matchup that heavily amplifies
after solving ran twenty-four seeds without a dip. What remains is a geometric hypothesis, that the
collapse needs a losing policy adjacent to the solved one, so that an amplified-noise step can
actually reach it.

This measures that adjacency directly. Train one floored run per matchup to its plateau, then add
Gaussian noise to every parameter, scaled per tensor by that tensor's own spread so layers of
different magnitude are perturbed proportionally, and measure the win rate as the noise grows. A
wide solved region degrades slowly; a narrow one craters. The prediction, registered before the
numbers: the one matchup that collapses degrades fastest.

Usage:
    ./solved_region_width.py CHECKPOINT WORKER --matchup "2:6,1:10=1:121" --matchup "7:1=7:1" ...
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import tempfile
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.policy import BattlePolicy  # noqa: E402
from fheroes2_agent.scenarios import Matchup, measure  # noqa: E402
from fheroes2_agent.train_ppo import train  # noqa: E402

SCALES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]


def perturbed(model: BattlePolicy, scale: float, draw: int) -> BattlePolicy:
    """A copy of the policy with per-tensor proportional Gaussian noise added."""
    copy = BattlePolicy()
    copy.load_state_dict(model.state_dict())
    generator = torch.Generator().manual_seed(draw)
    with torch.no_grad():
        for name, tensor in copy.named_parameters():
            # A one-element tensor has no spread; skip it rather than compute std of nothing.
            spread = float(tensor.std()) if tensor.numel() > 1 else 0.0
            if spread > 0 and scale > 0:
                tensor += scale * spread * torch.randn(tensor.shape, generator=generator)
    return copy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="cloned policy to train from")
    parser.add_argument("worker")
    parser.add_argument("--matchup", action="append", required=True,
                        help="attacker=defender, repeatable")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=16, help="evaluation episodes per noise draw")
    parser.add_argument("--draws", type=int, default=3, help="independent noise draws per scale")
    parser.add_argument("--scales", default=None, help="comma-separated noise scales, overriding the default")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    scales = [float(x) for x in args.scales.split(",")] if args.scales else SCALES

    work = pathlib.Path(tempfile.mkdtemp())
    started = time.time()
    results = []

    for spec in args.matchup:
        attacker, defender = spec.split("=")
        out = work / f"{spec.replace(':', '_').replace(',', '-').replace('=', '__')}.pt"
        r = train(args.worker, checkpoint=args.checkpoint, attacker=attacker, defender=defender,
                  iterations=args.iterations, episodes_per_iter=32, seed=0, quiet=True, out=str(out))
        plateau = statistics.mean(h["win_rate"] for h in r["history"][-5:])

        model = BattlePolicy()
        model.load_state_dict(torch.load(out, map_location="cpu", weights_only=True)["state_dict"])
        model.eval()

        curve = []
        for scale in scales:
            rates = []
            for draw in range(args.draws if scale > 0 else 1):
                m = perturbed(model, scale, draw)
                rates.append(measure(m, args.worker, Matchup(attacker, defender),
                                     episodes=args.episodes)["win_rate"])
            curve.append({"scale": scale, "win_rate": statistics.mean(rates), "rates": rates})

        results.append({"matchup": spec, "plateau": plateau, "curve": curve})
        line = "  ".join(f"{c['scale']:.2f}:{c['win_rate']:.2f}" for c in curve)
        print(f"  {spec:28s} plateau {plateau:.3f}   {line}", flush=True)

    print(f"\n  {time.time() - started:.0f}s total")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"results": results, "scales": scales, "iterations": args.iterations,
             "episodes": args.episodes, "draws": args.draws,
             "seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
