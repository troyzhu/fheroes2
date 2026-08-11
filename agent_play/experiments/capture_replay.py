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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fheroes2_agent import encoding as enc  # noqa: E402
from fheroes2_agent.env import REWARD_MARGINS, BattleEnv, _side_won  # noqa: E402
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
    from fheroes2_agent.policy import load_policy
    model = load_policy(state["state_dict"])
    model.eval()
    version = state.get("encoding_version", "obs_encoding_v2")
    return model, (encode_v2 if version == "obs_encoding_v2" else enc.encode_observation), version


def capture(worker: str, checkpoint: str, defender_checkpoint: str | None = None,
            search_simulations: int = 0, coverage_forced: bool = False,
            search_objective: str | None = None, search_combat_offset: int = 0, **env_kwargs) -> dict:
    """One policy per side when a defender checkpoint is given: the worker runs side=both and
    each decision routes to the model owning the active side, which is what lets two checkpoints
    fight each other on the record."""
    attacker_model, attacker_encode, version = load_model(checkpoint)
    defender_model, defender_encode, defender_version = (attacker_model, attacker_encode, version)
    if defender_checkpoint is not None:
        defender_model, defender_encode, defender_version = load_model(defender_checkpoint)
        env_kwargs["side"] = "both"

    wants_planes = bool(getattr(attacker_model, "planes", False)) or bool(getattr(defender_model, "planes", False))
    env = BattleEnv(worker, planes=wants_planes, **env_kwargs)
    # A searched capture plays the same position the policy would, with root search choosing:
    # recorded against a policy capture on the same matchup it shows what the 2026-08-08 deviation
    # probe measured, since the positions where the two disagree are the ones that decide battles.
    # The side environment gets its own objective and its own dice, for the same reasons the
    # battery does. Sharing the live environment's settings, which this did until 2026-08-10, meant
    # the filmed agent both maximized whatever the reward column happened to report and planned
    # against the rolls the battle was about to make: a stronger agent than any measurement
    # describes, so the footage showed something the numbers did not.
    sim_kwargs = dict(env_kwargs)
    if search_objective is not None:
        sim_kwargs["reward_margin"] = search_objective
    sim = (BattleEnv(worker, planes=wants_planes, combat_seed_offset=search_combat_offset, **sim_kwargs)
           if search_simulations else None)
    prefix: list[int] = []
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
            plane_arg = ()
            if getattr(model, "planes", False):
                plane_arg = (torch.from_numpy(env.last_planes).unsqueeze(0),)
            if sim is not None:
                from search_probe import search_action_detail
                # Plain UCB by default, because that is the rule the playing measurements use:
                # `search_agent_battery.py` and the simulation ladder both take the default, and a
                # replay filmed under forced coverage would not be the agent those numbers describe.
                # Forcing belongs to the soft-target collector, which needs support on every
                # candidate rather than a good move, so it stays available behind the flag.
                action, means, _, prior = search_action_detail(
                    sim, model, prefix, encode(observation), mask, search_simulations, 1.5,
                    live=env, coverage_forced=coverage_forced)
                greedy = max(prior, key=prior.get)
                caption = describe_action(action)
                if action != greedy:
                    # The caption says when search overruled the policy and what it was worth, so
                    # a viewer can see the disagreements rather than infer them from the outcome.
                    caption += f"  (search overrules the policy, +{means[action] - means.get(greedy, 0.0):.2f})"
            else:
                with torch.no_grad():
                    logits, _ = model(torch.from_numpy(encode(observation)).unsqueeze(0),
                                      torch.from_numpy(mask).unsqueeze(0), *plane_arg)
                    action = int(torch.distributions.Categorical(logits=logits).sample())
                caption = describe_action(action)
            frames.append(frame_of(observation, action, caption))
            prefix.append(action)
            step = env.step(action)
            if step.done:
                return {"frames": frames, "termination": step.info["termination"],
                        "encoding": version, "defender_encoding": defender_version, "reward": step.reward}
    finally:
        env.close()
        if sim is not None:
            sim.close()


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
                        help="retry until the recorded side ends the battle this way, so a replay shows "
                             "what the win rate says; read from --side, not from the engine's "
                             "attacker-perspective termination string")
    parser.add_argument("--tries", type=int, default=6)
    parser.add_argument("--search-simulations", type=int, default=0,
                        help="record the searching agent instead of the raw policy, captioning every "
                             "decision where search overrules the policy and what it was worth")
    parser.add_argument("--reward-margin", default="two_sided", choices=REWARD_MARGINS,
                        help="the objective the recorded battle is scored by, and the one root search "
                             "maximizes, since `rollout` returns the side environment's reward. This "
                             "defaulted to hit_points until 2026-08-09 while every battery measurement "
                             "used two_sided, so a filmed agent searched by a different rule than the "
                             "one its win rates were measured under")
    parser.add_argument("--search-objective", default=None, choices=REWARD_MARGINS,
                        help="what root search maximizes, separate from what --reward-margin reports. "
                             "Defaults to following --reward-margin, which is what it silently did before")
    parser.add_argument("--search-combat-offset", type=int, default=0,
                        help="perturbs the side environment's dice while keeping its battlefield. Zero "
                             "lets the filmed agent plan against the rolls the battle will make, which "
                             "is a ceiling rather than the agent the win rates describe")
    parser.add_argument("--coverage-forced", action="store_true",
                        help="visit every candidate once before UCB takes over; this is the soft-target "
                             "collector's rule, not the playing rule the win rates were measured under")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--battlefield", type=int, default=None,
                        help="which obstacle variant to fight on; by default every try replays variant 0, "
                             "so a near-deterministic agent returns the same outcome however many times "
                             "it retries, and --want cannot be satisfied. Given a number, each try steps "
                             "to the next variant from there.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    kwargs = dict(side=args.side, attacker=args.attacker, defender=args.defender,
                  attacker_hero=args.attacker_hero, defender_hero=args.defender_hero,
                  allow_wide=args.allow_wide, reward_margin=args.reward_margin)
    replay = None
    # `--want` is read from the chair being recorded, not from the engine's `termination` string,
    # which is written from the attacker's seat whoever is playing. Compared raw, `--want victory
    # --side defender` retried until the recorded agent *lost*, and the line below then announced
    # that loss as a victory; the 2026-08-09 mirror captures came back labelled "victory ... reward
    # -1.00" and "defeat ... reward +1.54", which is what exposed it. Attacker-side captures, which
    # is every replay recorded before that date, are unaffected either way.
    wanted_win = {"victory": True, "defeat": False}.get(args.want)
    satisfied = wanted_win is None
    for attempt in range(args.tries):
        if args.battlefield is not None:
            kwargs["seed_offset"] = args.battlefield + attempt
        candidate = capture(args.worker, args.checkpoint, defender_checkpoint=args.defender_checkpoint,
                            search_simulations=args.search_simulations,
                            coverage_forced=args.coverage_forced,
                            search_objective=args.search_objective,
                            search_combat_offset=args.search_combat_offset, **kwargs)
        replay = candidate
        if wanted_win is None or _side_won(candidate, args.side) == wanted_win:
            satisfied = True
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
    # The world seed the battle was actually fought under, so `render_replay.py` can reproduce it.
    # Without the stamp a capture made on any variant but the default replays on the default and
    # fails verification part-way through the action stream.
    replay["battlefield"] = kwargs.get("seed_offset", 0)
    replay["reward_margin"] = args.reward_margin
    replay["search_objective"] = args.search_objective or args.reward_margin
    replay["search_combat_offset"] = args.search_combat_offset
    replay["search_simulations"] = args.search_simulations
    pathlib.Path(args.out).write_text(json.dumps(replay))
    outcome = "won" if _side_won(replay, args.side) else "lost"
    if not satisfied:
        # Silence here is how a replay ends up captioned as the outcome that was asked for rather
        # than the one that happened, which is the same mislabelling `--want` itself had.
        print(f"WARNING: wanted the {args.side} to have {args.want} but it {outcome} every one of "
              f"{args.tries} tries; keeping the last. Vary --battlefield to search more positions.")
    print(f"{replay['checkpoint']}: {args.side} {outcome} ({replay['termination']}) in "
          f"{len(replay['frames'])} decisions (reward {replay['reward']:+.2f}) -> {args.out}")


if __name__ == "__main__":
    main()
