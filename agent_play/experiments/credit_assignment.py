#!/usr/bin/env python3
"""How often does trajectory-level credit mis-sign a decision, judged by search?

The owner's critique: a terminal-only, trajectory-level advantage upweights every action in a
won episode and downweights every action in a lost one, so a bad action redeemed by later play
is reinforced and a good action inside a lost game is punished. This measures how often that
actually happens here, using root-PUCT Q-values over engine rollouts as the per-decision ground
truth: at every decision of sampled episodes, the taken action's rollout value is compared with
the state's best, and the episode's group-relative advantage sign says what trajectory credit
would have done with it.

Reported: among decisions trajectory credit would reinforce, the fraction search calls bad
(regret at least half a win); among decisions it would punish, the fraction search calls good
(taken action within a tenth of best). Those are the critique's two error rates.

Usage:
    ./credit_assignment.py WORKER CHECKPOINT [--matchups 4] [--episodes 4] [--group 8]
                           [--simulations 32] [--report credit_assignment.json]
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

from fheroes2_agent.env import BattleEnv  # noqa: E402
from fheroes2_agent.policy import BattlePolicy  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from search_probe import policy_action, priors, rollout  # noqa: E402

POOL = pathlib.Path(__file__).resolve().parents[2] / "agent_play" / "docs" / "archive" / "experiments" / "files" \
    / "2026-08-05-run-reports" / "pool_value.json"


def q_values(sim: BattleEnv, model: BattlePolicy, prefix: list[int],
             observation: np.ndarray, mask: np.ndarray, simulations: int, c_puct: float = 1.5) -> dict[int, float]:
    """PUCT-allocated rollout values per legal action at this state."""
    prior = priors(model, observation, mask)
    actions = list(prior)
    visits = {a: 0 for a in actions}
    total = {a: 0.0 for a in actions}
    for n in range(simulations):
        scores = {}
        for a in actions:
            q = total[a] / visits[a] if visits[a] else 0.0
            u = 1.5 * prior[a] * math.sqrt(n + 1) / (1 + visits[a])
            scores[a] = q + u
        chosen = max(scores, key=scores.get)
        total[chosen] += rollout(sim, model, prefix, chosen)
        visits[chosen] += 1
    return {a: (total[a] / visits[a]) for a in actions if visits[a]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--matchups", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=4, help="analyzed episodes per matchup")
    parser.add_argument("--group", type=int, default=8, help="plain episodes forming the group baseline")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--bad-regret", type=float, default=0.5)
    parser.add_argument("--good-regret", type=float, default=0.1)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    model = BattlePolicy()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()

    # Mid-band matchups, where outcomes vary and the group baseline is informative.
    entries = json.loads(POOL.read_text())["matchups"][:40]
    rates = json.loads((POOL.parent / "dagger_share2.json").read_text())["evals"]["train"]
    order = np.argsort(np.abs(np.array(rates) - 0.5))
    chosen = [entries[i] for i in order[: args.matchups]]

    started = time.time()
    reinforced_bad = 0
    reinforced_total = 0
    punished_good = 0
    punished_total = 0
    decisions = []
    for m in chosen:
        kwargs = dict(attacker=m["attacker"], defender=m["defender"], attacker_hero=m.get("attacker_hero"),
                      defender_hero=m.get("defender_hero"), allow_wide=bool(m.get("allow_wide")))
        env = BattleEnv(args.worker, **kwargs)
        sim = BattleEnv(args.worker, **kwargs)
        try:
            group_rewards = []
            for _ in range(args.group):
                obs, mask = env.reset()
                while True:
                    step = env.step(policy_action(model, obs, mask))
                    if step.done:
                        group_rewards.append(step.reward)
                        break
                    obs, mask = step.observation, step.mask
            baseline = float(np.mean(group_rewards))

            for _ in range(args.episodes):
                obs, mask = env.reset()
                prefix: list[int] = []
                per_decision = []
                while True:
                    action = policy_action(model, obs, mask)
                    qs = q_values(sim, model, prefix, obs, mask, args.simulations)
                    if action in qs and qs:
                        regret = max(qs.values()) - qs[action]
                        per_decision.append(regret)
                    prefix.append(action)
                    step = env.step(action)
                    if step.done:
                        advantage = step.reward - baseline
                        for regret in per_decision:
                            if advantage > 0:
                                reinforced_total += 1
                                reinforced_bad += regret >= args.bad_regret
                            elif advantage < 0:
                                punished_total += 1
                                punished_good += regret <= args.good_regret
                            decisions.append({"advantage": advantage, "regret": regret})
                        break
                    obs, mask = step.observation, step.mask
        finally:
            env.close()
            sim.close()
        print(f"matchup done: reinforced {reinforced_total} (bad {reinforced_bad}), "
              f"punished {punished_total} (good {punished_good})", flush=True)

    rb = reinforced_bad / reinforced_total if reinforced_total else float("nan")
    pg = punished_good / punished_total if punished_total else float("nan")
    print(f"\nreinforced-but-search-bad rate: {rb:.3f} ({reinforced_bad}/{reinforced_total})")
    print(f"punished-but-search-good rate:  {pg:.3f} ({punished_good}/{punished_total})")
    print(f"total {round(time.time() - started)}s")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"reinforced_bad": reinforced_bad, "reinforced_total": reinforced_total,
             "punished_good": punished_good, "punished_total": punished_total,
             "bad_regret": args.bad_regret, "good_regret": args.good_regret,
             "simulations": args.simulations, "decisions": decisions}, indent=2))


if __name__ == "__main__":
    main()
