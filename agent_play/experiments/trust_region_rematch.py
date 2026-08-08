#!/usr/bin/env python3
"""The DPPO rematch in the actor-critic line: divergence gates against the ratio clip.

The 2026-08-03 arc tested DPPO's divergence trust region only inside the group-relative trainer,
where it never separated cleanly from the clip. This reruns the question inside `train_ppo`'s
actor-critic path at current budgets, and the owner's re-read of the paper sets the arms: the
ratio clip's opponents are the exact total-variation gate (affordable at this action count) and
the paper's own deployed form, the binary lower bound that gates on the sampled action's moved
probability mass. Everything else matches the self-play round's control trio, same anchor, same
AI-only opponent path, same reward, warmup and floor, so those runs are the ratio-clip arm and
this script trains only the divergence arms.

No inline evaluation: verdicts come from the full battery, symmetry gauge, convergence report
and the heartbeats' gate_fraction column, per the single-suite rule.

Usage:
    ./trust_region_rematch.py WORKER --out-dir DIR [--iterations 1000] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent import train_ppo  # noqa: E402
from fheroes2_agent.selfplay import OpponentPool, SelfPlayEnv  # noqa: E402

FILES = pathlib.Path(__file__).resolve().parents[1] / "docs" / "archive" / "experiments" / "files"
ANCHOR = str(FILES / "2026-08-05-checkpoints" / "policy_gen1.pt")
POOL_FILE = FILES / "2026-08-05-run-reports" / "pool_value.json"

ARMS = (
    ("binary_005", "binary", 0.05),
    ("exact_005", "exact", 0.05),
    ("exact_020", "exact", 0.20),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = json.loads(POOL_FILE.read_text())["matchups"][:12]
    matchups = [dict(attacker=e["attacker"], defender=e["defender"],
                     attacker_hero=e.get("attacker_hero"), defender_hero=e.get("defender_hero"),
                     allow_wide=bool(e.get("allow_wide"))) for e in entries]

    for name, kind, threshold in ARMS:
        for seed in args.seeds:
            out = out_dir / f"tr_{name}_s{seed}.pt"
            if out.exists():
                print(f"skip {out} (exists)", flush=True)
                continue
            started = time.time()
            env = SelfPlayEnv(args.worker, matchups, OpponentPool([None], seed=seed),
                              reward_margin="two_sided", rotation_seed=seed)
            try:
                result = train_ppo.train(args.worker, checkpoint=ANCHOR, iterations=args.iterations,
                                         seed=seed, env=env, quiet=True, out=str(out),
                                         trust_region="divergence", divergence_kind=kind,
                                         divergence_threshold=threshold,
                                         value_warmup_iters=5, entropy_floor=0.15)
            finally:
                env.close()
            # The run's own report, stamp included, so the arm is provable from the artifact
            # rather than from memory of which script produced it.
            (out_dir / f"tr_{name}_s{seed}.json").write_text(json.dumps(result, indent=1))
            print(f"{name} s{seed} done in {round(time.time() - started)}s -> {out} "
                  f"[{result['trust_region']}/{result['divergence_kind']}@{result['divergence_threshold']}]", flush=True)

    print("TRUST REGION REMATCH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
