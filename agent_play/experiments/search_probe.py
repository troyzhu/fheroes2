#!/usr/bin/env python3
"""Does root search improve play exactly where the policy and the teacher both fail?

The popular UCB variant, applied at the root: at every agent decision, candidate actions are
scored by PUCT (Q from rollout returns, prior from the policy, exploration bonus from visit
counts) over a budget of simulations, and the most-visited action plays. Simulations run on the
real engine through reset-continuation: a persistent side environment replays the action prefix
(determinism makes any state reachable by replay), applies the candidate, and rolls out with the
sampling policy to terminal. Rollout returns rather than critic leaves score the branches,
because the critic measures worse than the mean on student-visited states (critic_calibration)
and AlphaStar Unplugged reports MCTS exploiting value error until the policy collapses.

The probe compares searched play against policy-only play on matchups the policy mostly loses,
which is where an improvement operator would have to earn its keep as the next teacher.

Usage:
    ./search_probe.py WORKER CHECKPOINT [--simulations 32] [--episodes 12] [--c-puct 1.5]
                      [--report search_probe.json]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.encoding import encode_mask, encode_observation  # noqa: E402
from fheroes2_agent.env import BattleEnv  # noqa: E402
from fheroes2_agent.policy import load_policy, BattlePolicy  # noqa: E402

POOL = pathlib.Path(__file__).resolve().parents[2] / "agent_play" / "docs" / "archive" / "experiments" / "files" \
    / "2026-08-05-run-reports" / "pool_value.json"
THUNK = {"attacker": "11:1,11:1,11:1,10:2,9:2", "defender": "1:334,1:333,1:333",
         "attacker_hero": "13:12", "allow_wide": True, "label": "thunk_1000"}


def _plane_arg(model: BattlePolicy, source) -> tuple:
    """The planes tensor for a planes-built policy, from whichever env presented the state.

    A planes policy hard-fails without its tensor rather than silently reading zeros, so every
    search path threads the presenting environment through; entity policies get an empty tuple
    and are untouched."""
    if not getattr(model, "planes", False):
        return ()
    planes = getattr(source, "last_planes", None)
    if planes is None:
        raise ValueError("planes policy searched through an env constructed without planes=True")
    return (torch.from_numpy(planes).unsqueeze(0),)


def policy_action(model: BattlePolicy, observation: np.ndarray, mask: np.ndarray, sample: bool = True,
                  env=None) -> int:
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(observation).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0),
                          *_plane_arg(model, env))
        if sample:
            return int(torch.distributions.Categorical(logits=logits).sample())
        return int(logits.argmax())


def priors(model: BattlePolicy, observation: np.ndarray, mask: np.ndarray, env=None) -> dict[int, float]:
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(observation).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0),
                          *_plane_arg(model, env))
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    legal = np.flatnonzero(mask)
    return {int(a): float(probs[a]) for a in legal}


def rollout(sim: BattleEnv, model: BattlePolicy, prefix: list[int], first: int) -> float:
    """Replay the prefix, apply the candidate, sample the policy to terminal; the return is the
    episode's terminal reward. Deterministic engine plus identical action sequence reproduces
    the prefix state exactly (the replay-rendering machinery's own guarantee)."""
    observation, mask = sim.reset()
    for action in prefix:
        step = sim.step(action)
        if step.done:
            return step.reward  # prefix ended the battle; cannot happen when called mid-episode
        observation, mask = step.observation, step.mask
    step = sim.step(first)
    while not step.done:
        action = policy_action(model, step.observation, step.mask, env=sim)
        step = sim.step(action)
    return step.reward


def search_action_detail(sim: BattleEnv, model: BattlePolicy, prefix: list[int],
                         observation: np.ndarray, mask: np.ndarray, simulations: int,
                         c_puct: float, live: BattleEnv | None = None,
                         coverage_forced: bool = False) -> tuple[int, dict, dict, dict]:
    """The search decision plus its whole measurement: per-candidate mean rollout values, visit
    counts, and the prior. The values are the counterfactuals only search produces (a real
    playout per candidate it tried), which is what makes them valid soft-distillation targets
    where fitted state values and behavior Q measured 0.00; the prior anchors the target on
    support per Grill et al.

    `coverage_forced` is the demonstrated prerequisite from the soft-target program: UCB left
    alone visits about two candidates per state, which starves every downstream consumer of
    support, so the forced variant spends the first rollouts visiting every candidate once, in
    descending prior order, before UCB takes over. With more candidates than simulations the
    sweep is truncated at the simulation budget, still widest-support-first."""
    # The observation is the live environment's state, so a planes policy needs the live
    # env's tensor here; the sim's belongs to whatever state its own replay last presented.
    prior = priors(model, observation, mask, env=live if live is not None else sim)
    actions = list(prior)
    if len(actions) == 1:
        return actions[0], {actions[0]: 0.0}, {actions[0]: 1}, prior
    visits = {a: 0 for a in actions}
    total_return = {a: 0.0 for a in actions}
    sweep = sorted(actions, key=lambda a: -prior[a]) if coverage_forced else []
    for n in range(simulations):
        unvisited = [a for a in sweep if visits[a] == 0]
        if unvisited:
            chosen = unvisited[0]
        else:
            scores = {}
            for a in actions:
                q = total_return[a] / visits[a] if visits[a] else 0.0
                u = c_puct * prior[a] * math.sqrt(n + 1) / (1 + visits[a])
                scores[a] = q + u
            chosen = max(scores, key=scores.get)
        value = rollout(sim, model, prefix, chosen)
        visits[chosen] += 1
        total_return[chosen] += value
    means = {a: (total_return[a] / visits[a] if visits[a] else 0.0) for a in actions}
    # Ties on visit count are broken by the mean rollout value, not by action index. `visits` is
    # keyed in ascending legal-action order, so a plain argmax over it returns the lowest index,
    # and under coverage forcing ties are the common case rather than the rare one: 11.12 percent
    # of the 15,007 decisions in the first scaled corpus tied at the top and every one of them
    # was labeled by array position. The value is the signal the search actually measured.
    best = max(actions, key=lambda a: (visits[a], means[a]))
    return best, means, visits, prior


def search_action(sim: BattleEnv, model: BattlePolicy, prefix: list[int],
                  observation: np.ndarray, mask: np.ndarray, simulations: int, c_puct: float,
                  live: BattleEnv | None = None, coverage_forced: bool = False) -> int:
    action, _, _, _ = search_action_detail(sim, model, prefix, observation, mask, simulations, c_puct,
                                           live=live, coverage_forced=coverage_forced)
    return action


def play(env: BattleEnv, sim: BattleEnv | None, model: BattlePolicy, simulations: int, c_puct: float) -> tuple[bool, int]:
    observation, mask = env.reset()
    prefix: list[int] = []
    searched = 0
    while True:
        if sim is not None:
            action = search_action(sim, model, prefix, observation, mask, simulations, c_puct)
            searched += 1
        else:
            action = policy_action(model, observation, mask)
        prefix.append(action)
        step = env.step(action)
        if step.done:
            return step.info["termination"] == "victory", searched
        observation, mask = step.observation, step.mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--hard-matchups", type=int, default=3)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    model = load_policy(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()

    entries = json.loads(POOL.read_text())["matchups"][:40]
    share2 = json.loads((POOL.parent / "dagger_share2.json").read_text())["evals"]["train"]
    hard = [entries[i] | {"label": f"pool_{i}"} for i in np.argsort(share2)[: args.hard_matchups]]
    targets = [THUNK] + hard

    started = time.time()
    results = []
    for m in targets:
        kwargs = dict(attacker=m["attacker"], defender=m["defender"], attacker_hero=m.get("attacker_hero"),
                      defender_hero=m.get("defender_hero"), allow_wide=bool(m.get("allow_wide")))
        env = BattleEnv(args.worker, **kwargs)
        sim = BattleEnv(args.worker, **kwargs)
        try:
            row = {"label": m["label"]}
            for arm, use_search in (("policy", False), ("search", True)):
                wins = 0
                decisions = 0
                for _ in range(args.episodes):
                    won, searched = play(env, sim if use_search else None, model, args.simulations, args.c_puct)
                    wins += won
                    decisions += searched
                row[arm] = wins / args.episodes
                if use_search:
                    row["searched_decisions"] = decisions
            results.append(row)
            print(f"{m['label']:12s} policy {row['policy']:.3f} -> search {row['search']:.3f} "
                  f"({row['searched_decisions']} searched decisions)", flush=True)
        finally:
            env.close()
            sim.close()

    lift = np.array([r["search"] - r["policy"] for r in results])
    print(f"\nmean lift {lift.mean():+.3f} over {len(results)} matchups; total {round(time.time() - started)}s")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"results": results, "simulations": args.simulations, "episodes": args.episodes,
             "c_puct": args.c_puct, "checkpoint": pathlib.Path(args.checkpoint).name,
             "seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
