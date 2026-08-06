#!/usr/bin/env python3
"""A value network fitted on search play, and search that uses it instead of rollouts.

The measured objection to the current critic is distributional, not statistical: fitted on
teacher play it explains 0.30 of return variance there and less than zero on student-visited
states. The repair AlphaZero uses is to fit the value on the very distribution search generates,
and this project now has that distribution, tens of thousands of searched episodes with terminal
outcomes recorded.

Two questions, measured in order. Does a value fitted on searched play explain returns on
searched play, and how does it behave on policy play. Then, with that value as the leaf
evaluator instead of a full engine rollout, does search hold its quality, and at what cost:
a rollout costs a whole battle of engine steps, a value costs one forward pass, and that ratio
is what decides whether multi-ply search is affordable at all.

Usage:
    ./search_value.py WORKER POLICY --search-data DIR [DIR ...] [--policy-data DIR ...]
                      [--epochs 20] [--episodes 8] [--simulations 32]
                      [--report search_value.json]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fheroes2_agent import train_critic  # noqa: E402
from fheroes2_agent.dataset import load_dir, split_by_episode  # noqa: E402
from fheroes2_agent.env import BattleEnv  # noqa: E402
from fheroes2_agent.policy import load_policy  # noqa: E402
from search_probe import policy_action, priors  # noqa: E402

THUNK = dict(attacker="11:1,11:1,11:1,10:2,9:2", defender="1:334,1:333,1:333",
             attacker_hero="13:12", allow_wide=True)


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    var = float(np.var(target))
    return float(1.0 - np.var(target - pred) / var) if var > 1e-9 else float("nan")


def evaluate_value(model, samples) -> dict:
    keep = np.flatnonzero(np.isfinite(samples.returns))
    subset = samples.subset(keep)
    out = []
    with torch.no_grad():
        for start in range(0, len(subset), 4096):
            rows = slice(start, min(start + 4096, len(subset)))
            _, values = model(torch.from_numpy(subset.observations[rows]), torch.from_numpy(subset.masks[rows]))
            out.append(values.squeeze(-1).numpy())
    pred = np.concatenate(out)
    return {"n": len(subset), "explained_variance": explained_variance(pred, subset.returns),
            "bias": float(np.mean(pred - subset.returns))}


def value_leaf_search(sim: BattleEnv, policy, value_model, prefix: list[int], observation, mask,
                      simulations: int, c_puct: float = 1.5) -> tuple[int, int]:
    """One-ply search scored by the value network: apply each candidate, evaluate the resulting
    state, no rollout. Returns the action and the number of engine steps spent, which is the
    quantity a rollout search pays a whole battle for."""
    prior = priors(policy, observation, mask)
    actions = list(prior)
    if len(actions) == 1:
        return actions[0], 0
    if not actions:
        return 0, 0
    budget = min(simulations, len(actions))
    ranked = sorted(actions, key=lambda a: -prior[a])[:budget]
    scores = {}
    steps = 0
    for action in ranked:
        obs, msk = sim.reset()
        for earlier in prefix:
            step = sim.step(earlier)
            steps += 1
            if step.done:
                break
            obs, msk = step.observation, step.mask
        if sim._pending is None:
            # The prefix ended the battle in the side-environment, which the seed pinning should
            # prevent; score nothing rather than crash, and the caller's outcome comes from the
            # live environment either way.
            continue
        step = sim.step(action)
        steps += 1
        if step.done:
            scores[action] = step.reward
            continue
        with torch.no_grad():
            _, value = value_model(torch.from_numpy(step.observation).unsqueeze(0),
                                   torch.from_numpy(step.mask).unsqueeze(0))
        # The critic is fitted on the return of whoever acts at that state, because the teacher
        # plays both sides and the dataset labels each decision with its own actor's outcome. So
        # a successor where the opponent acts must be negated, exactly the negamax convention;
        # scoring it unsigned makes search prefer moves that are good for the opponent, which is
        # what the first run of this probe did.
        actor_is_ours = bool(sim._pending["observation"]["active_is_attacker"]) == (sim.side == "attacker")
        scores[action] = float(value) if actor_is_ours else -float(value)
    return max(scores, key=scores.get), steps


def play(worker: str, policy, value_model, episodes: int, simulations: int, matchup: dict | None = None,
         seeds: int = 1) -> dict:
    spec = dict(matchup) if matchup else dict(THUNK)
    env = BattleEnv(worker, **spec, seeds=seeds)
    sim = None
    wins, surv, steps, seconds = [], [], 0, time.time()
    try:
        for episode in range(episodes):
            obs, mask = env.reset()
            # The side-environment must replay the exact battlefield variant the live episode is
            # on, or prefix replay silently diverges; seed_offset pins it, which is the purpose
            # it was built for. One worker per episode is ~100 ms.
            if sim is not None:
                sim.close()
            sim = BattleEnv(worker, **spec, seeds=1, seed_offset=episode % max(seeds, 1))
            prefix: list[int] = []
            while True:
                action, spent = value_leaf_search(sim, policy, value_model, prefix, obs, mask, simulations)
                steps += spent
                prefix.append(action)
                step = env.step(action)
                if step.done:
                    won = step.info["termination"] == "victory"
                    wins.append(won)
                    if won:
                        surv.append(step.info["attacker"]["strength"] / max(step.info["attacker"]["initial_strength"], 1e-9))
                    break
                obs, mask = step.observation, step.mask
    finally:
        env.close()
        if sim is not None:
            sim.close()
    return {"win_rate": float(np.mean(wins)), "surviving_strength": float(np.mean(surv)) if surv else float("nan"),
            "engine_steps": steps, "seconds": round(time.time() - seconds, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("policy")
    parser.add_argument("--search-data", nargs="+", required=True)
    parser.add_argument("--policy-data", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--held-out", type=int, default=0,
                        help="instead of the Thunk fight, run value-leaf search over this many held-out pool matchups")
    args = parser.parse_args()

    out_path = args.out or str(pathlib.Path(tempfile.mkdtemp(prefix="search_value_")) / "value.pt")
    fit = train_critic.train(list(args.search_data), checkpoint=args.policy, epochs=args.epochs, out=out_path)

    value_model = load_policy(torch.load(out_path, map_location="cpu", weights_only=True)["state_dict"])
    value_model.eval()
    policy = load_policy(torch.load(args.policy, map_location="cpu", weights_only=True)["state_dict"])
    policy.eval()

    report = {"fit": {k: v for k, v in fit.items() if k != "state_dict"}, "calibration": {}}
    searched = load_dir(list(args.search_data))
    _, holdout = split_by_episode(searched, 0.2, seed=0)
    report["calibration"]["searched_holdout"] = evaluate_value(value_model, holdout)
    if args.policy_data:
        report["calibration"]["policy_states"] = evaluate_value(value_model, load_dir(list(args.policy_data)))
    for name, stats in report["calibration"].items():
        print(f"{name:20s} n={stats['n']:7d} EV {stats['explained_variance']:+.3f} bias {stats['bias']:+.3f}", flush=True)

    if args.held_out:
        entries = json.loads((pathlib.Path(__file__).resolve().parents[2] / "agent_play" / "docs" / "archive"
                              / "experiments" / "files" / "2026-08-05-run-reports" / "pool_value.json").read_text())
        rates = []
        for entry in entries["matchups"][40:40 + args.held_out]:
            spec = dict(attacker=entry["attacker"], defender=entry["defender"],
                        attacker_hero=entry.get("attacker_hero"), defender_hero=entry.get("defender_hero"),
                        allow_wide=bool(entry.get("allow_wide")))
            result = play(args.worker, policy, value_model, args.episodes, args.simulations, matchup=spec,
                          seeds=args.episodes)
            rates.append(result["win_rate"])
            print(f"  held-out {entry['attacker']} vs {entry['defender']}: {result['win_rate']:.2f}", flush=True)
        arr = np.array(rates)
        report["held_out_value_search"] = {"rates": rates, "mean": float(arr.mean()),
                                           "se": float(arr.std(ddof=1) / np.sqrt(len(arr)))}
        print(f"value-leaf search on held-out: mean {arr.mean():.3f} +/- {arr.std(ddof=1)/np.sqrt(len(arr)):.3f} "
              f"(built-in AI reads 0.660)")
        if args.report:
            pathlib.Path(args.report).write_text(json.dumps(report, indent=2))
        return

    report["value_leaf_search"] = play(args.worker, policy, value_model, args.episodes, args.simulations)
    print(f"value-leaf search on Thunk-1000: win {report['value_leaf_search']['win_rate']:.2f}, "
          f"surviving strength {report['value_leaf_search']['surviving_strength']:.3f}, "
          f"{report['value_leaf_search']['engine_steps']} engine steps, "
          f"{report['value_leaf_search']['seconds']}s")

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"value checkpoint: {out_path}")


if __name__ == "__main__":
    main()
