#!/usr/bin/env python3
"""Can the built-in planner be queried without advancing the arena? The DAgger precondition.

Code reading says yes: AI::BattlePlanner has no Rand:: call site, planning recomputes its
analysis members from the arena on every call, and the pathfinder cache it warms is the same one
the action-space enumeration already warms under digest-proven gates. This experiment is the
empirical half: every configuration runs twice under a deterministic scripted policy, once plain
and once with --probe-teacher querying the planner at every controlled decision, and the paired
terminal state digests must be identical. Any divergence means the query perturbed the battle.

It also measures what the probe is for: how often the teacher's choice at student-visited states
resolves inside simple_v1 (the DAgger label rate), and what the probe costs in wall time.

Usage:
    ./planner_query.py WORKER [--episodes-per-config 2] [--report planner_query.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
import zlib

REPO = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = ("m1_tiny_melee", "m1_three_stack", "m1_five_stack", "m1_ranged_heavy", "m1_longer_balanced")
POOL = REPO / "agent_play" / "docs" / "archive" / "experiments" / "files" / "2026-08-05-run-reports" / "pool_value.json"


def choose(record: dict) -> int:
    """Deterministic pure function of the decision record, so both arms pick identically as
    long as the battles are identical, which is exactly what the digest then certifies."""
    legal = record["legal_actions"]
    key = json.dumps(record["observation"], sort_keys=True).encode()
    return legal[zlib.crc32(key) % len(legal)]


def run(worker: str, args: list[str], probe: bool) -> tuple[list[dict], float]:
    cmd = [worker, "--protocol"] + args + (["--probe-teacher"] if probe else [])
    started = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    terminals, decisions = [], 0
    for line in proc.stdout:
        record = json.loads(line)
        if record["record"] == "decision":
            decisions += 1
            proc.stdin.write(f"{choose(record)}\n")
            proc.stdin.flush()
        elif record["record"] == "terminal":
            record["_decisions"] = decisions
            terminals.append(record)
            decisions = 0
    proc.stdin.close()
    proc.wait()
    return terminals, time.time() - started


def run_pair(worker: str, args: list[str], label: str, results: dict) -> None:
    plain, t_plain = run(worker, args, probe=False)
    probed, t_probe = run(worker, args, probe=True)
    if len(plain) != len(probed):
        results["mismatches"].append({"config": label, "kind": "episode count", "plain": len(plain), "probed": len(probed)})
        return
    for a, b in zip(plain, probed):
        results["pairs"] += 1
        if a["state_digest"] != b["state_digest"]:
            results["mismatches"].append({"config": label, "scenario": a["scenario_id"],
                                          "plain_digest": a["state_digest"], "probed_digest": b["state_digest"]})
        results["probes_resolved"] += b.get("probes_resolved", 0)
        results["probes_outside"] += b.get("probes_outside", 0)
    results["seconds_plain"] += t_plain
    results["seconds_probed"] += t_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("--seeds", type=int, default=4, help="world seeds per fixture configuration")
    parser.add_argument("--pool-matchups", type=int, default=20)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    results = {"pairs": 0, "mismatches": [], "probes_resolved": 0, "probes_outside": 0,
               "seconds_plain": 0.0, "seconds_probed": 0.0}

    for fixture in FIXTURES:
        for side in ("attacker", "defender", "both"):
            run_pair(args.worker, ["--fixture", fixture, "--seeds", str(args.seeds), "--side", side],
                     f"{fixture}/{side}", results)
        print(f"{fixture}: pairs so far {results['pairs']}, mismatches {len(results['mismatches'])}", flush=True)

    pool = json.loads(POOL.read_text())["matchups"][: args.pool_matchups]
    for i, m in enumerate(pool):
        extra = ["--fixture", "m1_tiny_melee", "--seeds", "2", "--side", "attacker",
                 "--attacker", m["attacker"], "--defender", m["defender"]]
        if m.get("attacker_hero"):
            extra += ["--attacker-hero", m["attacker_hero"]]
        if m.get("defender_hero"):
            extra += ["--defender-hero", m["defender_hero"]]
        if m.get("allow_wide"):
            extra += ["--allow-wide"]
        run_pair(args.worker, extra, f"pool[{i}]", results)
    print(f"pool: pairs total {results['pairs']}, mismatches {len(results['mismatches'])}", flush=True)

    probes = results["probes_resolved"] + results["probes_outside"]
    overhead = results["seconds_probed"] / results["seconds_plain"] if results["seconds_plain"] else float("nan")
    print(f"\npaired episodes: {results['pairs']}, digest mismatches: {len(results['mismatches'])}")
    print(f"teacher probes: {probes}, resolved in simple_v1: {results['probes_resolved']} "
          f"({results['probes_resolved'] / probes:.3f})" if probes else "no probes")
    print(f"wall time probed/plain: {overhead:.3f}")

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(results, indent=2))
    sys.exit(1 if results["mismatches"] else 0)


if __name__ == "__main__":
    main()
