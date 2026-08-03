#!/usr/bin/env python3
"""Check that recorded decisions are complete behaviour-cloning samples.

A decision recorded under --audit-coverage must carry three things: the board the teacher saw,
the legal set it chose from, and the index it chose. An action without an observation is a label
with no input and cannot train anything, which is the state this project was in until the
observation emitter existed.

Usage: check_bc_samples.py <trajectory-dir>
Prints a one-line summary. Exits non-zero if any sample is incomplete or inconsistent.
"""
import json
import pathlib
import sys


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    bad: list[str] = []
    samples = 0
    stacks = 0

    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("record") != "decision":
                continue

            samples += 1
            where = f"{path.name}#{record['engine_decision_index']}"
            observation = record.get("observation")
            legal = record.get("legal_actions")

            if observation is None or legal is None:
                bad.append(f"{where}: missing observation or legal set")
                continue

            if not legal:
                bad.append(f"{where}: empty legal set")

            # The teacher's own action has to be one the enumeration offered, otherwise the
            # label is unreachable through the action interface being trained.
            if record.get("teacher_resolved") and record["teacher_action"] not in legal:
                bad.append(f"{where}: teacher action outside the legal set")

            # Exactly one stack is on turn, and it is the one that acted.
            active = [u for u in observation["units"] if u["active"]]
            if len(active) != 1:
                bad.append(f"{where}: {len(active)} active stacks, expected 1")
            elif active[0]["uid"] != record["unit_uid"]:
                bad.append(f"{where}: active stack is not the deciding stack")

            # Observations carry living stacks only, so every one holds creatures.
            for unit in observation["units"]:
                stacks += 1
                if unit["count"] < 1 or unit["hit_points"] < 1:
                    bad.append(f"{where}: dead stack in a living-only observation")
                if unit["count"] > unit["initial_count"]:
                    bad.append(f"{where}: stack grew beyond its initial count")

    if bad:
        print("; ".join(bad[:5]) + (f" (+{len(bad) - 5} more)" if len(bad) > 5 else ""))
        return 1
    if samples == 0:
        print("no decision records found")
        return 1
    print(f"{samples} samples, {stacks} observed stacks, all consistent")
    return 0


sys.exit(main())
