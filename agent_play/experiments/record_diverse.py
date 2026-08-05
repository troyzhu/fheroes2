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


def load_wide_roster() -> tuple[list[tuple[int, int, bool]], list[tuple[int, int, bool]]]:
    """(wide_v1 roster, the cheap-creature subset for hordes), from the capability audit."""
    path = pathlib.Path(__file__).resolve().parents[2] / "python" / "fheroes2_agent" / "data" / "monster_capabilities_v1.json"
    records = json.loads(path.read_text())
    roster = [(r["monster_id"], int(r["hit_points"]), bool(r["is_archer"]))
              for r in records if r["wide_v1_supported"]]
    roster.sort()
    cheap = [entry for entry in roster if entry[1] <= 5]
    return roster, cheap


def sample_side(rng: random.Random, roster, total_hp: float, max_stacks: int) -> str:
    stacks = rng.randint(1, max_stacks)
    share = total_hp / stacks
    parts = []
    for _ in range(stacks):
        monster, hp, _ = rng.choice(roster)
        parts.append(f"{monster}:{max(1, min(500, int(round(share / hp))))}")
    return ",".join(parts)


def sample_matchup(rng: random.Random, roster, cheap) -> dict:
    regime = rng.choices(("skirmish", "battle", "horde"), weights=(4, 4, 2))[0]
    if regime == "skirmish":
        strength = rng.choice([15, 20, 25, 30, 40])
        ratio = rng.uniform(0.85, 1.15)
        attacker = sample_side(rng, roster, strength, 3)
        defender = sample_side(rng, roster, strength * ratio, 3)
    elif regime == "battle":
        strength = rng.choice([60, 90, 120, 150])
        ratio = rng.uniform(0.85, 1.15)
        attacker = sample_side(rng, roster, strength, 5)
        defender = sample_side(rng, roster, strength * ratio, 5)
    else:
        # Elite against horde, the opening-fight archetype. The horde is one cheap creature split
        # into three near-equal stacks, the way Army::ArrangeForBattle splits a neutral stack.
        attacker = sample_side(rng, roster, rng.choice([80, 120, 160]), 4)
        monster, hp, _ = rng.choice(cheap)
        total = rng.randint(60, 900) // max(hp, 1)
        a = total // 3 + (1 if total % 3 else 0)
        defender = f"{monster}:{a},{monster}:{total // 3},{monster}:{max(1, total - a - total // 3)}"
    spec = {"regime": regime, "attacker": attacker, "defender": defender}
    # Commanders on a coin flip per side, with map-hero-like stats. The dumped Thunk heroes span
    # attack 1 to 30 and defense 0 to 27.
    for side in ("attacker", "defender"):
        if rng.random() < 0.5:
            spec[f"{side}_hero"] = f"{rng.randint(0, 25)}:{rng.randint(0, 20)}"
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("out_dir")
    parser.add_argument("--matchups", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", default=None, help="where to record what was sampled")
    args = parser.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    roster, cheap = load_wide_roster()
    print(f"{len(roster)} creatures in the wide roster, {len(cheap)} cheap enough for hordes", flush=True)

    rng = random.Random(args.seed)
    sampled, failures = [], 0
    started = time.time()

    for index in range(args.matchups):
        spec = sample_matchup(rng, roster, cheap)
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
