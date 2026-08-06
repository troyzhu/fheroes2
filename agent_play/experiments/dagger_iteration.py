#!/usr/bin/env python3
"""DAgger's first iteration: the student walks, the teacher labels, the clone retrains.

Behavior cloning learns the teacher's action on the teacher's states, and its known failure is
compounding error on the states the student reaches instead. DAgger's repair is to collect
exactly those states and ask the teacher what it would have done, which the planner probe made
possible today: the student policy plays its training matchups over rotated battlefields with
`--probe-teacher`, every decision is written in the recorder's dataset schema with the teacher's
canonical index as the label, and the clone retrains on teacher demonstrations plus the
relabeled student states as one corpus.

Measurement is on what DAgger claims to fix, student-reached play: pool win rate over
battlefields against the previous clone, and the Thunk ladder. The cloning holdout agreement of
the retrain is reported for continuity but is not the point.

Usage:
    ./dagger_iteration.py WORKER CHECKPOINT --dagger-dir DIR --teacher-data DIR [DIR ...]
                          [--matchups 40] [--episodes-per-matchup 25] [--battlefields 4]
                          [--epochs 25] [--out policy_dagger.pt] [--report dagger.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.env import BattleEnv  # noqa: E402
from fheroes2_agent.policy import BattlePolicy  # noqa: E402
from fheroes2_agent.scenarios import Matchup, measure  # noqa: E402
from fheroes2_agent import train_bc  # noqa: E402

POOL = pathlib.Path(__file__).resolve().parents[2] / "agent_play" / "docs" / "archive" / "experiments" / "files" \
    / "2026-08-05-run-reports" / "pool_value.json"
EVAL_SEEDS = 4


def as_matchup(entry: dict) -> Matchup:
    return Matchup(entry["attacker"], entry["defender"], attacker_hero=entry.get("attacker_hero"),
                   defender_hero=entry.get("defender_hero"), allow_wide=bool(entry.get("allow_wide")))


def collect(worker: str, model: BattlePolicy, matchups: list[Matchup], out_dir: pathlib.Path,
            episodes_per: int, battlefields: int) -> dict:
    """Student-played episodes with teacher labels, one JSONL per episode in the dataset schema."""
    decisions = 0
    labeled = 0
    episodes = 0
    for index, matchup in enumerate(matchups):
        matchup_dir = out_dir / f"matchup_{index:03d}"
        matchup_dir.mkdir(parents=True, exist_ok=True)
        env = BattleEnv(worker, attacker=matchup.attacker, defender=matchup.defender,
                        attacker_hero=matchup.attacker_hero, defender_hero=matchup.defender_hero,
                        allow_wide=matchup.allow_wide, seeds=battlefields, probe_teacher=True)
        try:
            for episode in range(episodes_per):
                observation, mask = env.reset()
                records = []
                while True:
                    raw = env._pending
                    decisions += 1
                    if "teacher_action" in raw:
                        labeled += 1
                        records.append({"record": "decision", "observation": raw["observation"],
                                        "legal_actions": raw["legal_actions"], "teacher_resolved": True,
                                        "teacher_action": int(raw["teacher_action"])})
                    with torch.no_grad():
                        logits, _ = model(torch.from_numpy(observation).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0))
                        action = int(torch.distributions.Categorical(logits=logits).sample())
                    step = env.step(action)
                    if step.done:
                        records.append(step.info)
                        break
                    observation, mask = step.observation, step.mask
                episodes += 1
                (matchup_dir / f"episode_{episode:04d}.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in records) + "\n")
        finally:
            env.close()
        if (index + 1) % 10 == 0:
            print(f"collected {index + 1}/{len(matchups)} matchups, {episodes} episodes, "
                  f"{labeled}/{decisions} decisions labeled", flush=True)
    return {"episodes": episodes, "decisions": decisions, "labeled": labeled}


def evaluate(model: BattlePolicy, worker: str, matchups: list[Matchup], episodes: int) -> list[float]:
    return [measure(model, worker, m, episodes=episodes, seeds=EVAL_SEEDS)["win_rate"] for m in matchups]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint", help="the student whose states get relabeled, and the comparison clone")
    parser.add_argument("--dagger-dir", required=True, help="where collected episodes go")
    parser.add_argument("--teacher-data", nargs="+", required=True, help="existing teacher demonstration roots")
    parser.add_argument("--matchups", type=int, default=40)
    parser.add_argument("--episodes-per-matchup", type=int, default=25)
    parser.add_argument("--battlefields", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--skip-collect", action="store_true", help="reuse an existing --dagger-dir")
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    started = time.time()
    entries = json.loads(POOL.read_text())["matchups"]
    train_set = [as_matchup(e) for e in entries[: args.matchups]]
    held_set = [as_matchup(e) for e in entries[args.matchups: args.matchups + 20]]

    student = BattlePolicy()
    student.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    student.eval()

    dagger_dir = pathlib.Path(args.dagger_dir)
    if args.skip_collect:
        stats = {"episodes": "reused", "decisions": "reused", "labeled": "reused"}
    else:
        stats = collect(args.worker, student, train_set, dagger_dir, args.episodes_per_matchup, args.battlefields)
        print(f"collection: {stats}", flush=True)

    result = train_bc.train(list(args.teacher_data) + [str(dagger_dir)], epochs=args.epochs, out=args.out)

    dagger_model = BattlePolicy()
    dagger_model.load_state_dict(torch.load(args.out, map_location="cpu", weights_only=True)["state_dict"])
    dagger_model.eval()

    evals = {"train": evaluate(dagger_model, args.worker, train_set, args.eval_episodes),
             "held": evaluate(dagger_model, args.worker, held_set, args.eval_episodes)}
    print(f"dagger clone over {EVAL_SEEDS} battlefields: train {np.mean(evals['train']):.3f}, "
          f"held-out {np.mean(evals['held']):.3f}")

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"collection": stats, "bc": {k: v for k, v in result.items() if k != "state_dict"},
             "evals": evals, "eval_seeds": EVAL_SEEDS, "battlefields": args.battlefields,
             "seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
