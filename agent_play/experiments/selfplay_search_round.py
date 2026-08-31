#!/usr/bin/env python3
"""One expert-iteration round in TRUE self-play: search labels both chairs of policy-vs-policy games.

The owner's curriculum, stated 2026-08-13: extensive initial training and self-distillation run
against the built-in AI, and once the model is in good range the round should be played against
itself. Every search-taught corpus before this script was collected against the engine, opponent
in the live game and opponent model inside the playouts alike. Here both chairs of the live game
are driven by search over the policy, every playout models the other chair with the policy
(`rollout_self_play`), and every decision of both chairs becomes a label, so one episode yields
roughly twice the decisions of a one-chair collection.

Labels are scored by a record-only margin, because a both-sides environment's step reward is not
perspectived to either chair and `hit_points` has no record-only form. The default is `contested`,
the plain strength difference: self-play is a zero-sum game and `contested` is the only margin here
that is zero sum, so what one chair gains is exactly what the other loses. It also has no stall
exploit, where `strength` pays an evading defender its maximum. The combat offset keeps the label
honest exactly as in the AI-opponent collector: nonzero, so search cannot see the live game's rolls.

Usage:
    ./selfplay_search_round.py WORKER CHECKPOINT --out-dir DIR [--matchups N] [--episodes 2]
                               [--simulations 32] [--sample-seed 97] [--report R.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fheroes2_agent.env import REWARD_MARGINS, BattleEnv, reward_from_record  # noqa: E402
from fheroes2_agent.policy import load_policy  # noqa: E402
from fheroes2_agent.search import search_action_detail  # noqa: E402
from fheroes2_agent.suites import POOL  # noqa: E402

SEARCH_OFFSET = 987631
#: OpenAI Five trained 80 percent of games against the current policy and 20 percent against past
#: selves, explicitly to avoid strategy collapse, a narrow set of tactics that beats the current
#: self and nothing else (Dota 2 with Large Scale Deep RL, arXiv 1912.06680). Mirror self-play
#: labels only the states one policy reaches against itself, and this project's own prose already
#: argued for a pool on the PPO path while the first search collector shipped without one.
PAST_FRACTION = 0.2


def draw_opponent(models, rng, index, episode):
    """Which checkpoint answers the opposing chair inside the live game and its playouts.

    Returns (model, name). With no pool the current policy answers both chairs, which is mirror
    self-play; with a pool, PAST_FRACTION of episodes draw a frozen past self instead.
    """
    current, past = models[0], models[1:]
    if past and rng.random() < PAST_FRACTION:
        i = rng.randrange(len(past))
        return past[i][0], past[i][1]
    return current[0], current[1]


def collect_matchup(worker, models, entry, out_dir, episodes, simulations, c_puct, margin,
                    index, rng) -> tuple[int, dict]:
    kwargs = dict(attacker=entry["attacker"], defender=entry["defender"],
                  attacker_hero=entry.get("attacker_hero"), defender_hero=entry.get("defender_hero"),
                  allow_wide=bool(entry.get("allow_wide")), side="both", reward_margin=margin)
    live = BattleEnv(worker, **kwargs)
    sim = BattleEnv(worker, combat_seed_offset=SEARCH_OFFSET, **kwargs)
    decisions, outcomes = 0, {"victory": 0, "defeat": 0, "stalemate": 0}
    try:
        for episode in range(episodes):
            torch.manual_seed(1009 * index + episode)
            opponent, opponent_name = draw_opponent(models, rng, index, episode)
            model, _ = models[0]
            # Alternate which chair the learner occupies, so labels cover both sides evenly even
            # when a past self holds the other; mirror self-play got both chairs for free.
            learner_side = "attacker" if (index + episode) % 2 == 0 else "defender"
            observation, mask = live.reset()
            prefix: list[int] = []
            rows = []
            while True:
                acting = live.acting_side
                # The learner's chair is searched and labelled; the opposing chair is played by the
                # drawn opponent (the current policy, or a past self) and is NOT labelled, because a
                # past self's choices are not the target the student should imitate.
                # Who plays this chair, and is the decision worth keeping as a label? The learner
                # always is. The opposing chair is too whenever the drawn opponent IS the current
                # policy, which is every episode of mirror self-play, because then both chairs are
                # the same searched policy and labelling one of them would halve the yield for
                # nothing. A past self's choices are not a target to imitate, so those are played
                # but not recorded.
                learner_turn = acting == learner_side
                actor = model if learner_turn else opponent
                keep_label = learner_turn or opponent is model
                action, means, visits, prior = search_action_detail(
                    sim, actor, prefix, observation, mask, simulations, c_puct,
                    rollout_opponent="policy", agent_side=acting, full_prefix=True)
                if keep_label:
                    rows.append({
                        "record": "decision", "side": acting,
                        "observation": live.pending_decision["observation"],
                        "legal_actions": [int(a) for a in np.flatnonzero(mask)],
                        "teacher_action": int(action),
                        "search_values": {str(a): float(v) for a, v in means.items()},
                        "search_visits": {str(a): int(v) for a, v in visits.items()},
                        "prior": {str(a): float(p) for a, p in prior.items()},
                        "opponent": opponent_name,
                    })
                prefix.append(action)
                step = live.step(action)
                if step.done:
                    outcomes[step.info["termination"]] = outcomes.get(step.info["termination"], 0) + 1
                    for row in rows:
                        row["episode_reward"] = reward_from_record(step.info, row["side"], margin)
                    path = out_dir / f"matchup_{index:03d}" / f"episode_{episode:04d}.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w") as f:
                        for row in rows:
                            f.write(json.dumps(row) + "\n")
                        f.write(json.dumps({"record": "terminal", **step.info}) + "\n")
                    decisions += len(rows)
                    break
                observation, mask = step.observation, step.mask
    finally:
        live.close()
        sim.close()
    return decisions, outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--matchups", type=int, default=24)
    parser.add_argument("--episodes", type=int, default=2, help="per matchup; both chairs label every episode")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--margin", default="contested",
                        choices=[m for m in REWARD_MARGINS if m != "hit_points"],
                        help="how a playout's terminal is scored. `contested` is the default "
                             "because self-play is zero sum and it is the only record-computable "
                             "margin that is: r(attacker) = -r(defender) at every terminal. The "
                             "earlier default `strength` is neither, and it carries the stall "
                             "exploit, paying an evading defender +2.0, its maximum, so a stalling "
                             "playout returned the best value available. Corpora collected before "
                             "2026-08-23 used `strength`; their manifests record it")
    parser.add_argument("--past", nargs="*", default=[],
                        help="frozen past-self checkpoints; PAST_FRACTION of episodes draw one as "
                             "the opposing chair, the OpenAI Five 80/20 remedy for strategy collapse")
    parser.add_argument("--sample-seed", type=int, default=97)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    def _load(path):
        m = load_policy(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
        m.eval()
        return (m, pathlib.Path(path).name)
    models = [_load(args.checkpoint)] + [_load(p) for p in args.past]
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.sample_seed)
    pool = json.loads(POOL.read_text())["matchups"]
    entries = [pool[i] for i in rng.permutation(len(pool))[:args.matchups]]

    started = time.time()
    total, all_outcomes = 0, {}
    for index, entry in enumerate(entries):
        n, outcomes = collect_matchup(args.worker, models, entry, out, args.episodes,
                                      args.simulations, args.c_puct, args.margin, index,
                                      random.Random(7717 + index))
        total += n
        for k, v in outcomes.items():
            all_outcomes[k] = all_outcomes.get(k, 0) + v
        print(f"  matchup {index + 1}/{len(entries)}: {n} decisions, outcomes so far {all_outcomes} "
              f"({time.time() - started:.0f}s)", flush=True)

    manifest = {"checkpoint": pathlib.Path(args.checkpoint).name,
                "past_selves": [pathlib.Path(p).name for p in args.past],
                "past_fraction": PAST_FRACTION if args.past else 0.0, "matchups": len(entries),
                "episodes_per_matchup": args.episodes, "simulations": args.simulations,
                "margin": args.margin, "search_combat_offset": SEARCH_OFFSET,
                "rollout_opponent": "policy", "sides": "both",
                "decisions": total, "outcomes": all_outcomes,
                "seconds": round(time.time() - started, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(manifest, indent=1))
    print(f"SELF-PLAY COLLECTION COMPLETE: {total} decisions over {len(entries)} matchups")


if __name__ == "__main__":
    main()
