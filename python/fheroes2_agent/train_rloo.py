"""Policy improvement without a critic, using a leave-one-out baseline.

A learned critic is one way to get a baseline and not the only one. Drawing several episodes from
the same starting state and using the mean return of the others gives a baseline directly, with
no value network to fit, no bootstrapping, and therefore none of the bias bootstrapping carries.
`agent_play/docs/rl/rlhf-transfer.md` argues this deserves a first attempt here rather than a
fallback, and the reason is that the usual objection does not apply: in a language model, drawing
K completions per prompt is the dominant cost, while this environment runs thousands of episodes
per second from a reproducible seed.

The measured badly-calibrated critic is the immediate motivation. Value estimates sat near -0.7
in positions the policy went on to win, because the critic had been fitted on one matchup and was
being applied to another. A leave-one-out baseline has nothing to miscalibrate.

    b_k = (1 / (K - 1)) * sum_{i != k} G_i        A_k = G_k - b_k

Excluding the sample itself is what keeps this exactly unbiased: b_k is then independent of the
actions in episode k, so the baseline term vanishes by the control-variate argument. Including it,
as group-relative optimization does, leaves an O(1/K) bias.

The cost is coarse credit. One advantage covers every decision in an episode, so a battle that
turned on one decision out of thirty spreads the blame evenly. That is the trade against the
critic, not a defect to be fixed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from .encoding import ENCODING_VERSION
from .env import BattleEnv
from .policy import BattlePolicy


def collect_group(env: BattleEnv, model: BattlePolicy, k: int) -> dict:
    """K episodes from the same starting state, which is what makes the baseline conditional."""
    episodes = []
    for _ in range(k):
        observation, mask = env.reset()
        obs, masks, actions, logps = [], [], [], []
        while True:
            with torch.no_grad():
                logits, _ = model(torch.from_numpy(observation).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0))
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
                logp = distribution.log_prob(action)

            obs.append(observation)
            masks.append(mask)
            actions.append(int(action))
            logps.append(float(logp))

            step = env.step(int(action))
            if step.done:
                episodes.append({"observations": obs, "masks": masks, "actions": actions,
                                 "logps": logps, "return": step.reward, "info": step.info})
                break
            observation, mask = step.observation, step.mask
    return {"episodes": episodes}


def leave_one_out_advantages(returns: np.ndarray) -> np.ndarray:
    """A_k = G_k - mean of the other returns. Undefined for K < 2."""
    k = len(returns)
    if k < 2:
        raise ValueError("a leave-one-out baseline needs at least two episodes per group")
    total = returns.sum()
    baselines = (total - returns) / (k - 1)
    return returns - baselines


def train(
    worker: str,
    checkpoint: str | None = None,
    attacker: str | None = None,
    defender: str | None = None,
    fixture: str = "m1_tiny_melee",
    side: str = "attacker",
    iterations: int = 20,
    group_size: int = 8,
    groups_per_iter: int = 4,
    epochs: int = 4,
    minibatch: int = 256,
    lr: float = 1e-4,
    clip: float = 0.2,
    entropy_coef: float = 0.01,
    seed: int = 0,
    out: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    model = BattlePolicy()
    started_from = "random initialization"
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("encoding_version") != ENCODING_VERSION:
            raise ValueError(f"checkpoint encoding {state.get('encoding_version')} does not match {ENCODING_VERSION}")
        model.load_state_dict(state["state_dict"])
        started_from = f"cloned policy ({checkpoint})"
    print(f"starting from {started_from}, leave-one-out baseline, K={group_size}")

    env = BattleEnv(worker, fixture=fixture, side=side, attacker=attacker, defender=defender)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    started = time.time()
    initial = None

    for iteration in range(iterations):
        obs_rows, mask_rows, act_rows, logp_rows, adv_rows = [], [], [], [], []
        outcomes = []
        degenerate = 0

        for _ in range(groups_per_iter):
            group = collect_group(env, model, group_size)["episodes"]
            returns = np.array([e["return"] for e in group], dtype=np.float32)
            outcomes.extend(e["info"] for e in group)

            # Every episode scoring alike leaves every advantage at zero. That is the degenerate
            # case scenario-distribution.md describes, and counting it is more useful than
            # silently contributing a batch of zeros.
            if float(returns.std()) < 1e-6:
                degenerate += 1
                continue

            advantages = leave_one_out_advantages(returns)
            for episode, advantage in zip(group, advantages):
                # One advantage per episode, broadcast to every decision in it.
                obs_rows.extend(episode["observations"])
                mask_rows.extend(episode["masks"])
                act_rows.extend(episode["actions"])
                logp_rows.extend(episode["logps"])
                adv_rows.extend([advantage] * len(episode["actions"]))

        wins = float(np.mean([o["termination"] == ("victory" if side == "attacker" else "defeat") for o in outcomes]))
        if initial is None:
            initial = wins

        if not act_rows:
            history.append({"iteration": iteration, "win_rate": wins, "steps": 0, "degenerate_groups": degenerate})
            print(f"iter {iteration:3d}  win_rate {wins:.3f}  no gradient, all {degenerate} groups degenerate")
            continue

        obs = torch.from_numpy(np.stack(obs_rows).astype(np.float32))
        masks = torch.from_numpy(np.stack(mask_rows))
        actions = torch.from_numpy(np.asarray(act_rows, dtype=np.int64))
        old_logps = torch.from_numpy(np.asarray(logp_rows, dtype=np.float32))
        adv = torch.from_numpy(np.asarray(adv_rows, dtype=np.float32))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(actions)
        for _ in range(epochs):
            order = torch.randperm(n)
            for start in range(0, n, minibatch):
                rows = order[start : start + minibatch]
                logits, _ = model(obs[rows], masks[rows])
                distribution = torch.distributions.Categorical(logits=logits)
                logps = distribution.log_prob(actions[rows])
                ratio = torch.exp(logps - old_logps[rows])
                surrogate = torch.min(ratio * adv[rows], torch.clamp(ratio, 1 - clip, 1 + clip) * adv[rows])
                # No value term: there is no critic to fit.
                loss = -surrogate.mean() - entropy_coef * distribution.entropy().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        history.append({"iteration": iteration, "win_rate": wins, "steps": int(n), "degenerate_groups": degenerate})
        print(f"iter {iteration:3d}  win_rate {wins:.3f}  steps {n}  degenerate groups {degenerate}/{groups_per_iter}")
        if out:
            torch.save({"state_dict": model.state_dict(), "encoding_version": ENCODING_VERSION, "win_rate": wins}, out)

    env.close()
    result = {
        "method": "leave-one-out baseline, no critic",
        "encoding_version": ENCODING_VERSION,
        "started_from": started_from,
        "group_size": group_size,
        "initial_win_rate": initial,
        "final_win_rate": history[-1]["win_rate"] if history else initial,
        "seconds": round(time.time() - started, 1),
        "history": history,
    }
    print(f"\nwin rate {result['initial_win_rate']:.3f} -> {result['final_win_rate']:.3f} ({result['seconds']}s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Improve a battle policy with a leave-one-out baseline.")
    parser.add_argument("worker")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--attacker", default=None)
    parser.add_argument("--defender", default=None)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    result = train(args.worker, checkpoint=args.checkpoint, attacker=args.attacker, defender=args.defender,
                   iterations=args.iterations, group_size=args.group_size, groups_per_iter=args.groups, out=args.out)
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
