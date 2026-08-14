#!/usr/bin/env python3
"""Battle a checkpoint yourself, through the game's own interface.

Launches the replay tool in play mode: you command one side with the mouse in a real battle
window, and the checkpoint answers the other side's decisions over the worker's line protocol.
The observation is encoded per the checkpoint's stamped version, actions sample from the masked
softmax exactly as in evaluation, so what you are fighting is the measured policy, not a
determinized imitation of it.

Usage:
    ./play_vs.py CHECKPOINT [--human-side attacker] [--attacker SPEC] [--defender SPEC]
                 [--attacker-hero A:D] [--defender-hero A:D] [--allow-wide] [--speed 5]

Defaults give you the Thunk armies against the checkpoint's Peasant horde command; swap sides
or armies freely. Close the battle window or finish the fight to end.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

import json  # noqa: E402

import torch  # noqa: E402

from fheroes2_agent import encoding as enc  # noqa: E402
from fheroes2_agent.policy import BattlePolicy  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from capture_replay import load_model  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
TOOL = REPO / "src" / "agent_replay" / "fheroes2_agent_replay"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--human-side", default="attacker", choices=("attacker", "defender"))
    parser.add_argument("--attacker", default="11:1,11:1,11:1,10:2,9:2")
    parser.add_argument("--defender", default="1:334,1:333,1:333")
    parser.add_argument("--attacker-hero", default="13:12")
    parser.add_argument("--defender-hero", default=None)
    parser.add_argument("--allow-wide", action="store_true", default=True)
    parser.add_argument("--speed", type=int, default=5)
    parser.add_argument("--simulations", type=int, default=0,
                        help="wrap the checkpoint in root search at this playout budget; 16 is the "
                             "deployment rung. The side environment models YOUR chair with the "
                             "engine's built-in AI, so search plans against an AI-shaped opponent "
                             "rather than a copy of you; the further your play is from the AI's, "
                             "the staler its plan, which is the documented resampled semantics")
    parser.add_argument("--allow-flying", action="store_true")
    parser.add_argument("--rollout-opponent", default="ai", choices=("ai", "policy"),
                        help="who answers YOUR chair inside search playouts. 'ai' models you with "
                             "the engine's planner; 'policy' is self-play search, the checkpoint "
                             "modelling you with itself (scores playouts by the record-only "
                             "'strength' margin)")
    args = parser.parse_args()

    if not TOOL.exists():
        raise SystemExit(f"{TOOL} not built; run src/agent_replay/build_replay.sh")

    model, encode, version = load_model(args.checkpoint)
    searched = f" + search at {args.simulations} playouts" if args.simulations else ""
    print(f"you play {args.human_side}; {pathlib.Path(args.checkpoint).name} ({version}){searched} plays the other side")

    sim = None
    prefix: list[int] = []
    if args.simulations:
        from fheroes2_agent.env import BattleEnv
        from fheroes2_agent.search import search_action_detail
        agent_side = "defender" if args.human_side == "attacker" else "attacker"
        worker = REPO / "src" / "agent_worker" / "fheroes2_agent_worker"
        if not worker.exists():
            raise SystemExit(f"{worker} not built; run src/agent_worker/build_worker.sh")
        # Same scenario flags as the play window, one battlefield, independent dice (ADR 0008's
        # honest configuration). The prefix holds only the checkpoint's own actions; your moves are
        # answered inside playouts by the built-in AI, which is what the --simulations help states.
        sim_side = "both" if args.rollout_opponent == "policy" else agent_side
        sim_margin = {"reward_margin": "strength"} if args.rollout_opponent == "policy" else {}
        sim = BattleEnv(str(worker), seeds=1, side=sim_side,
                        attacker=args.attacker, defender=args.defender,
                        attacker_hero=args.attacker_hero, defender_hero=args.defender_hero,
                        allow_wide=args.allow_wide, allow_flying=args.allow_flying,
                        combat_seed_offset=987631, **sim_margin)

    cmd = [str(TOOL), "--play", args.human_side, "--speed", str(args.speed),
           "--attacker", args.attacker, "--defender", args.defender]
    if args.attacker_hero:
        cmd += ["--attacker-hero", args.attacker_hero]
    if args.defender_hero:
        cmd += ["--defender-hero", args.defender_hero]
    if args.allow_wide:
        cmd.append("--allow-wide")
    if args.allow_flying:
        cmd.append("--allow-flying")

    env = dict(FHEROES2_DATA=str(REPO), PATH="/usr/bin:/bin", HOME=str(pathlib.Path.home()))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
    try:
        for line in proc.stdout:
            record = json.loads(line)
            if record.get("record") == "decision":
                observation = record["observation"]
                mask = enc.encode_mask(record["legal_actions"])
                if sim is not None:
                    action, _, _, _ = search_action_detail(
                        sim, model, prefix, encode(observation), mask, args.simulations, 1.5,
                        rollout_opponent=args.rollout_opponent, agent_side=agent_side)
                    prefix.append(action)
                else:
                    with torch.no_grad():
                        logits, _ = model(torch.from_numpy(encode(observation)).unsqueeze(0),
                                          torch.from_numpy(mask).unsqueeze(0))
                        action = int(torch.distributions.Categorical(logits=logits).sample())
                proc.stdin.write(f"{action}\n")
                proc.stdin.flush()
            elif record.get("record") == "replay_terminal":
                human_won = (record["termination"] == "victory") == (args.human_side == "attacker")
                print(f"battle over: {record['termination']} — {'you win' if human_won else 'the checkpoint wins'}")
    finally:
        proc.wait()
        if sim is not None:
            sim.close()


if __name__ == "__main__":
    main()
