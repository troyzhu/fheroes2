#!/usr/bin/env python3
"""Capture a battle as a frame-by-frame JSON replay, for rendering.

The text viewer in `watch.py` prints boards; this records them. Each frame carries the raw unit
list before the policy's decision, the action it chose in words, and the board geometry a renderer
needs. The policy is fed observations encoded per the checkpoint's own stamped version, so a
checkpoint trained under the older linear count scaling replays faithfully even though the
deployed encoder has moved on.

Usage:
    ./capture_replay.py WORKER CHECKPOINT --attacker SPEC --defender SPEC [--attacker-hero A:D]
                        [--allow-wide] [--want victory|defeat|any] [--tries 6] --out replay.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent import encoding as enc  # noqa: E402
from fheroes2_agent.env import BattleEnv  # noqa: E402
from fheroes2_agent.policy import BattlePolicy  # noqa: E402
from fheroes2_agent.render import describe_action, monster_name  # noqa: E402


def encode_v2(observation: dict) -> np.ndarray:
    """The retired linear scaling, for replaying checkpoints stamped obs_encoding_v2."""
    out = enc.encode_observation(observation).copy()
    for slot, unit in enumerate(observation["units"]):
        base = slot * enc.SLOT_FEATURES
        initial = max(unit["initial_count"], 1)
        out[base + 4] = unit["count"] / 100.0
        out[base + 5] = unit["initial_count"] / 100.0
        out[base + 6] = unit["count"] / initial
        out[base + 7] = unit["hit_points"] / 100.0
        out[base + 8] = unit["top_hit_points"] / 100.0
    return out


def frame_of(raw: dict, action: int | None, caption: str | None) -> dict:
    units = [{"uid": u["uid"], "name": monster_name(u["monster_id"]), "side": u["side"],
              "count": u["count"], "initial": u["initial_count"], "hp": u["hit_points"],
              "head": u["head_cell"], "tail": u["tail_cell"], "wide": u["wide"],
              "archer": u["archer"], "active": u["uid"] == raw["active_uid"]}
             for u in raw["units"]]
    return {"round": raw["round"], "units": units, "action": action, "caption": caption}


def load_model(checkpoint: str):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = BattlePolicy()
    model.load_state_dict(state["state_dict"])
    model.eval()
    version = state.get("encoding_version", "obs_encoding_v2")
    return model, (encode_v2 if version == "obs_encoding_v2" else enc.encode_observation), version


def capture(worker: str, checkpoint: str, defender_checkpoint: str | None = None, **env_kwargs) -> dict:
    """One policy per side when a defender checkpoint is given: the worker runs side=both and
    each decision routes to the model owning the active side, which is what lets two checkpoints
    fight each other on the record."""
    attacker_model, attacker_encode, version = load_model(checkpoint)
    defender_model, defender_encode, defender_version = (attacker_model, attacker_encode, version)
    if defender_checkpoint is not None:
        defender_model, defender_encode, defender_version = load_model(defender_checkpoint)
        env_kwargs["side"] = "both"

    env = BattleEnv(worker, **env_kwargs)
    frames = []
    try:
        env.reset()
        while True:
            raw = env._pending
            observation = raw["observation"]
            mask = enc.encode_mask(raw["legal_actions"])
            attacker_turn = bool(observation.get("active_is_attacker"))
            model, encode = (attacker_model, attacker_encode) if attacker_turn or defender_checkpoint is None \
                else (defender_model, defender_encode)
            with torch.no_grad():
                logits, _ = model(torch.from_numpy(encode(observation)).unsqueeze(0),
                                  torch.from_numpy(mask).unsqueeze(0))
                action = int(torch.distributions.Categorical(logits=logits).sample())
            frames.append(frame_of(observation, action, describe_action(action)))
            step = env.step(action)
            if step.done:
                return {"frames": frames, "termination": step.info["termination"],
                        "encoding": version, "defender_encoding": defender_version, "reward": step.reward}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--attacker", required=True)
    parser.add_argument("--defender", required=True)
    parser.add_argument("--attacker-hero", default=None)
    parser.add_argument("--defender-hero", default=None)
    parser.add_argument("--allow-wide", action="store_true")
    parser.add_argument("--side", default="attacker")
    parser.add_argument("--defender-checkpoint", default=None,
                        help="a second policy controlling the defender, so two checkpoints battle each other")
    parser.add_argument("--want", default="any", choices=("victory", "defeat", "any"),
                        help="retry until an episode ends this way, so a replay shows what the win rate says")
    parser.add_argument("--tries", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    kwargs = dict(side=args.side, attacker=args.attacker, defender=args.defender,
                  attacker_hero=args.attacker_hero, defender_hero=args.defender_hero,
                  allow_wide=args.allow_wide)
    replay = None
    for attempt in range(args.tries):
        candidate = capture(args.worker, args.checkpoint, defender_checkpoint=args.defender_checkpoint, **kwargs)
        replay = candidate
        if args.want == "any" or candidate["termination"] == args.want:
            break
    replay["checkpoint"] = pathlib.Path(args.checkpoint).name
    if args.defender_checkpoint:
        replay["defender_checkpoint"] = pathlib.Path(args.defender_checkpoint).name
    replay["fixture"] = "m1_tiny_melee"  # BattleEnv default; the worker derives the world seed from it
    replay["attacker"] = args.attacker
    replay["defender"] = args.defender
    replay["attacker_hero"] = args.attacker_hero
    replay["defender_hero"] = args.defender_hero
    # A duel runs the worker side=both, and the stamp must say so: an earlier version stamped
    # args.side here unconditionally, which clobbered a duel's "both" into "attacker" and made
    # its replay desynchronize at the first defender decision (#43).
    replay["side"] = "both" if args.defender_checkpoint else args.side
    replay["allow_wide"] = args.allow_wide
    pathlib.Path(args.out).write_text(json.dumps(replay))
    print(f"{replay['checkpoint']}: {replay['termination']} in {len(replay['frames'])} decisions "
          f"(reward {replay['reward']:+.2f}) -> {args.out}")


if __name__ == "__main__":
    main()
