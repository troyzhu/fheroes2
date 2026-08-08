#!/usr/bin/env python3
"""Scripted stdin and stdout tests for blocking external control.

This is Milestone 4's exit criterion in miniature: a client drives both sides of a battle
through the protocol without a single invalid command reaching the engine. The cases that matter
are the unhappy ones, because a decision hook that blocks is the first thing in this project that
can deadlock or leave a half-applied turn behind.

Usage: test_protocol.py <path-to-worker>
"""
import json
import os
import pathlib
import subprocess
import sys

WORKER = sys.argv[1] if len(sys.argv) > 1 else "src/agent_worker/fheroes2_agent_worker"
ENV = dict(os.environ, HOME="/tmp")

passed = failed = 0


def check(condition, name, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
        passed += 1
    else:
        print(f"  FAIL  {name}" + (f"  [{detail}]" if detail else ""))
        failed += 1


def drive(policy, fixture="m1_tiny_melee", side="attacker", stop_after=None, timeout=30, extra=()):
    """Run one episode. `policy(record) -> str|None`; None closes stdin."""
    proc = subprocess.Popen(
        [WORKER, "--protocol", "--fixture", fixture, "--side", side, *extra],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env=ENV,
    )
    decisions, terminal, closed = [], None, False
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            record = json.loads(line)
            if record["record"] == "decision":
                decisions.append(record)
                if stop_after is not None and len(decisions) >= stop_after:
                    proc.stdin.close()
                    closed = True
                    stop_after = None
                    continue
                if closed:
                    continue
                answer = policy(record)
                if answer is None:
                    proc.stdin.close()
                    closed = True
                    continue
                proc.stdin.write(answer + "\n")
                proc.stdin.flush()
            elif record["record"] == "terminal":
                terminal = record
        if not closed:
            proc.stdin.close()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return decisions, None, "timeout"
    return decisions, terminal, proc.returncode


# 1. A full episode driven to termination, every selection legal.
decisions, terminal, rc = drive(lambda r: str(r["legal_actions"][0]))
check(terminal is not None, "an episode driven externally reaches a terminal record")
check(rc == 0, "the worker exits cleanly", f"rc={rc}")
check(terminal and terminal["rejected"] == 0, "no selection was rejected")
check(terminal and terminal["decisions_answered"] == terminal["decisions_seen"],
      "every decision the engine asked for was answered",
      terminal and f"{terminal['decisions_answered']}/{terminal['decisions_seen']}")
check(all(r["observation"]["units"] for r in decisions), "every decision carried a non-empty board")
check(all(r["legal_actions"] for r in decisions), "every decision carried a non-empty legal set")

# 2. A policy that always skips must lose. If it wins, the actions are not reaching the engine.
_, skip_terminal, _ = drive(lambda r: "0")
check(skip_terminal and skip_terminal["termination"] == "defeat",
      "a policy that always skips loses the battle", skip_terminal and skip_terminal["termination"])

# 3. An illegal selection is recoverable: counted, skipped, episode continues, worker survives.
_, bad_terminal, bad_rc = drive(lambda r: "792" if 792 not in r["legal_actions"] else str(r["legal_actions"][0]))
check(bad_terminal is not None and bad_rc == 0, "an illegal selection does not kill the worker", f"rc={bad_rc}")
check(bad_terminal and bad_terminal["rejected"] > 0, "an illegal selection is counted as rejected",
      bad_terminal and f"rejected={bad_terminal['rejected']}")

# 4. Garbage on the wire is recoverable the same way.
_, junk_terminal, junk_rc = drive(lambda r: "not-a-number")
check(junk_terminal is not None and junk_rc == 0, "unparseable input does not kill the worker", f"rc={junk_rc}")

# 5. Closing stdin mid-battle must unwind rather than hang, and must say so.
_, closed_terminal, closed_rc = drive(lambda r: str(r["legal_actions"][0]), stop_after=2)
check(closed_rc == 0, "closing the client mid-battle unwinds cleanly", f"rc={closed_rc}")
check(closed_terminal is not None, "a terminal record is still produced after the client leaves")
check(closed_terminal and closed_terminal["client_closed"], "the terminal record reports the client left")

# 6. Determinism: the same scripted actions must reproduce the same terminal state.
_, a, _ = drive(lambda r: str(r["legal_actions"][0]))
_, b, _ = drive(lambda r: str(r["legal_actions"][0]))
check(a and b and a["state_digest"] == b["state_digest"],
      "identical action sequences reproduce the state digest", a and a["state_digest"][:12])

# 7. Driving the defender instead must be accepted and must change the outcome.
_, d_terminal, _ = drive(lambda r: "0", side="defender")
check(d_terminal and d_terminal["termination"] == "victory",
      "always-skipping as the defender loses to the attacker", d_terminal and d_terminal["termination"])

# 8. A hero commander's stats must appear in every observed unit of that side, and only that
# side. Peasant base attack and defense are both 1, so the expected values are exact.
hero_decisions, hero_terminal, hero_rc = drive(lambda r: str(r["legal_actions"][0]),
                                               extra=("--attacker", "1:5", "--defender", "1:5",
                                                      "--attacker-hero", "30:20"))
check(hero_rc == 0 and hero_decisions, "a commander is accepted by the protocol path", f"rc={hero_rc}")
first = hero_decisions[0]["observation"]["units"] if hero_decisions else []
atk = [u for u in first if u["side"] == "attacker"]
dfn = [u for u in first if u["side"] == "defender"]
check(atk and all(u["attack"] == 31 and u["defense"] == 21 for u in atk),
      "the commander's stats reach every unit on its side",
      atk and f"attack={atk[0]['attack']} defense={atk[0]['defense']}")
check(dfn and all(u["attack"] == 1 and u["defense"] == 1 for u in dfn),
      "the other side's units are untouched by it")
check(hero_terminal and hero_terminal["termination"] == "victory",
      "the buffed side wins a mirror it would otherwise trade evenly",
      hero_terminal and hero_terminal["termination"])

# 9. Wide units under external control: the wide_v1 profile admits Champions, the observation
# reports them wide with a tail cell, and a scripted episode completes with legal actions only.
wide_decisions, wide_terminal, wide_rc = drive(lambda r: str(r["legal_actions"][0]),
                                               extra=("--attacker", "9:2,1:10", "--defender", "1:40",
                                                      "--allow-wide"))
check(wide_rc == 0 and wide_terminal is not None, "a wide-unit battle completes under external control",
      f"rc={wide_rc}")
wide_units = [u for d in wide_decisions for u in d["observation"]["units"] if u["wide"]]
check(bool(wide_units) and all(u["tail_cell"] >= 0 for u in wide_units),
      "wide units appear in observations with a tail cell",
      wide_units and f"head={wide_units[0]['head_cell']} tail={wide_units[0]['tail_cell']}")
check(wide_terminal and wide_terminal["rejected"] == 0,
      "no candidate offered for a wide unit was rejected by the engine",
      wide_terminal and f"rejected={wide_terminal['rejected']}")

print(f"{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
