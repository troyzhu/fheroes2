#!/usr/bin/env python3
"""Record teacher demonstrations across the whole bestiary, counts and commanders included.

The clone that anchors calibration, the critic and every reinforcement-learning start was trained
on five fixtures holding three creature types. This records the dataset that removes that bound:
sampled armies over every `wide_v1` creature, three count regimes, and hero commanders on a coin
flip per side, because real maps always have one.

The regimes, each an archetype the earlier data lacked:
  skirmish   the proven small-stack sampler, one to three stacks worth 15 to 40 hit points
  battle     up to five stacks worth 60 to 150 hit points a side
  horde      a small elite army against one cheap creature in the hundreds, split like the
             engine splits a neutral stack, which is the Thunk opening fight's shape

Each sampled matchup runs through the worker's own recorder with coverage auditing on, so every
decision carries the observation, the legal set and the teacher's index, and the run fails loudly
if any teacher decision fails to resolve to an enumerated candidate.

Usage:
    ./record_diverse.py WORKER OUT_DIR [--matchups 400] [--seeds 6]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.scenarios import load_wide_roster, sample_diverse_matchup  # noqa: E402


def sample_matchup(rng: random.Random, horde_total_range=(60, 900), horde_only=False) -> dict:
    """The shared diverse sampler, reshaped into the recorder's spec dict."""
    m = sample_diverse_matchup(rng, horde_total_range=horde_total_range, horde_only=horde_only)
    # A horde defender is three stacks of one creature. The first version of this label counted
    # colons instead of stacks and misfiled every horde, which only the manifest noticed.
    parts = m.defender.split(",")
    horde = len(parts) == 3 and len({p.split(":")[0] for p in parts}) == 1
    spec = {"regime": "horde" if horde else ("battle" if len(m.attacker.split(",")) > 3 or len(parts) > 3 else "skirmish"),
            "attacker": m.attacker, "defender": m.defender}
    if m.attacker_hero:
        spec["attacker_hero"] = m.attacker_hero
    if m.defender_hero:
        spec["defender_hero"] = m.defender_hero
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("out_dir")
    parser.add_argument("--matchups", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", default=None, help="where to record what was sampled")
    parser.add_argument("--horde-only", action="store_true")
    parser.add_argument("--horde-max", type=int, default=900, help="upper bound of the horde total hit points")
    args = parser.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    roster = load_wide_roster()
    print(f"{len(roster)} creatures in the wide roster", flush=True)

    rng = random.Random(args.seed)
    sampled, failures = [], 0
    started = time.time()

    for index in range(args.matchups):
        spec = sample_matchup(rng, horde_total_range=(60, args.horde_max), horde_only=args.horde_only)
        sub = out / f"m{index:04d}"
        sub.mkdir(exist_ok=True)
        cmd = [args.worker, "--runs", "1", "--seeds", str(args.seeds), "--allow-wide",
               "--attacker", spec["attacker"], "--defender", spec["defender"],
               "--audit-coverage", "--trajectory-dir", str(sub), "--quiet"]
        if "attacker_hero" in spec:
            cmd += ["--attacker-hero", spec["attacker_hero"]]
        if "defender_hero" in spec:
            cmd += ["--defender-hero", spec["defender_hero"]]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                                env={"HOME": str(out)})
        episodes = len(list(sub.glob("*.jsonl")))
        spec["episodes"] = episodes
        # The verdict line is the worker's own judgement, and it goes to stdout. The first
        # version of this script kept stderr alone, so 68 coverage-incomplete verdicts read as
        # inexplicable failures with benign-looking diagnostics.
        verdict = next((l for l in (result.stdout or "").splitlines() if l.startswith("VERDICT")), "")
        spec["verdict"] = verdict
        sampled.append(spec)
        if result.returncode != 0 or episodes == 0 or "INCOMPLETE" in verdict or "deterministic=no" in verdict:
            failures += 1
            spec["error"] = [verdict] + (result.stderr or "").strip().splitlines()[-1:]
        if (index + 1) % 50 == 0:
            total = sum(s["episodes"] for s in sampled)
            print(f"  {index + 1}/{args.matchups} matchups, {total} episodes, "
                  f"{failures} failures, {time.time() - started:.0f}s", flush=True)

    total = sum(s["episodes"] for s in sampled)
    by_regime = {}
    for s in sampled:
        by_regime.setdefault(s["regime"], []).append(s["episodes"])
    print(f"\n{total} episodes over {args.matchups} matchups, {failures} failures")
    for regime, counts in sorted(by_regime.items()):
        print(f"  {regime:9s} {len(counts)} matchups, {sum(counts)} episodes")

    if args.manifest:
        pathlib.Path(args.manifest).write_text(json.dumps(
            {"matchups": sampled, "seeds": args.seeds, "sampler_seed": args.seed,
             "failures": failures, "seconds": round(time.time() - started, 1)}, indent=1))


if __name__ == "__main__":
    main()
