"""Group-relative policy improvement, with the advantage and the trust region as separate choices.

One trainer covering leave-one-out, GRPO and Dr. GRPO advantages, under either PPO's ratio clip or
DPPO's divergence mask. Holding everything else fixed is the point: a comparison that changes two
things at once cannot attribute its result to either.

Everything is grounded in the notes under `agent_play/docs/`. What is new here is only that the
divergence trust region is computed exactly rather than approximated, which
`research/works/dppo-trust-region.md` records as affordable at this action-space size and
prohibitive at a language model's.
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
from .objectives import ADVANTAGE_MODES, TRUST_REGIONS, clip_fraction, group_advantages, surrogate, total_variation
from .policy import BattlePolicy


def collect_group(env: BattleEnv, model: BattlePolicy, k: int) -> list[dict]:
    episodes = []
    for _ in range(k):
        observation, mask = env.reset()
        obs, masks, actions, logps, logits_list = [], [], [], [], []
        while True:
            with torch.no_grad():
                logits, _ = model(torch.from_numpy(observation).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0))
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
            obs.append(observation)
            masks.append(mask)
            actions.append(int(action))
            logps.append(float(distribution.log_prob(action)))
            # The behaviour policy's logits, needed by the divergence trust region and free to keep.
            logits_list.append(logits.squeeze(0).numpy())

            step = env.step(int(action))
            if step.done:
                episodes.append({"observations": obs, "masks": masks, "actions": actions, "logps": logps,
                                 "logits": logits_list, "return": step.reward, "info": step.info})
                break
            observation, mask = step.observation, step.mask
    return episodes


def train(
    worker: str,
    checkpoint: str | None = None,
    attacker: str | None = None,
    defender: str | None = None,
    side: str = "attacker",
    advantage: str = "loo",
    trust_region: str = "ratio",
    iterations: int = 20,
    group_size: int = 8,
    groups_per_iter: int = 4,
    epochs: int = 4,
    minibatch: int = 256,
    lr: float = 1e-4,
    clip: float = 0.2,
    divergence_threshold: float = 0.05,
    entropy_coef: float = 0.01,
    seed: int = 0,
    out: str | None = None,
    quiet: bool = False,
) -> dict:
    if advantage not in ADVANTAGE_MODES:
        raise ValueError(f"advantage must be one of {ADVANTAGE_MODES}")
    if trust_region not in TRUST_REGIONS:
        raise ValueError(f"trust_region must be one of {TRUST_REGIONS}")

    torch.manual_seed(seed)
    model = BattlePolicy()
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("encoding_version") != ENCODING_VERSION:
            raise ValueError(f"checkpoint encoding {state.get('encoding_version')} does not match {ENCODING_VERSION}")
        model.load_state_dict(state["state_dict"])

    env = BattleEnv(worker, side=side, attacker=attacker, defender=defender)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    target = "victory" if side == "attacker" else "defeat"

    history, initial = [], None
    started = time.time()

    for iteration in range(iterations):
        rows_obs, rows_mask, rows_act, rows_logp, rows_adv, rows_logits = [], [], [], [], [], []
        outcomes, degenerate = [], 0

        for _ in range(groups_per_iter):
            group = collect_group(env, model, group_size)
            returns = np.array([e["return"] for e in group], dtype=np.float32)
            outcomes.extend(e["info"] for e in group)

            if float(returns.std()) < 1e-6:
                # Equal returns leave every advantage at zero whichever mode is chosen, so the
                # group is dropped and counted rather than contributing a batch of zeros.
                degenerate += 1
                continue

            for episode, adv in zip(group, group_advantages(returns, advantage)):
                rows_obs.extend(episode["observations"])
                rows_mask.extend(episode["masks"])
                rows_act.extend(episode["actions"])
                rows_logp.extend(episode["logps"])
                rows_logits.extend(episode["logits"])
                rows_adv.extend([adv] * len(episode["actions"]))

        wins = float(np.mean([o["termination"] == target for o in outcomes]))
        if initial is None:
            initial = wins

        if not rows_act:
            history.append({"iteration": iteration, "win_rate": wins, "steps": 0, "degenerate_groups": degenerate})
            if not quiet:
                print(f"iter {iteration:3d}  win {wins:.3f}  all {degenerate} groups degenerate, no gradient")
            continue

        obs = torch.from_numpy(np.stack(rows_obs).astype(np.float32))
        masks = torch.from_numpy(np.stack(rows_mask))
        actions = torch.from_numpy(np.asarray(rows_act, dtype=np.int64))
        old_logps = torch.from_numpy(np.asarray(rows_logp, dtype=np.float32))
        old_logits = torch.from_numpy(np.stack(rows_logits).astype(np.float32))
        adv_all = torch.from_numpy(np.asarray(rows_adv, dtype=np.float32))
        adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

        n = len(actions)
        clipped, blocked_total, batches = 0.0, 0.0, 0
        for _ in range(epochs):
            order = torch.randperm(n)
            for start in range(0, n, minibatch):
                rows = order[start : start + minibatch]
                logits, _ = model(obs[rows], masks[rows])
                distribution = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(distribution.log_prob(actions[rows]) - old_logps[rows])

                divergence = None
                if trust_region == "divergence":
                    divergence = total_variation(logits, old_logits[rows], masks[rows])
                    blocked_total += float((divergence > divergence_threshold).float().mean())

                obj = surrogate(ratio, adv_all[rows], trust_region=trust_region, clip=clip,
                                divergence=divergence, threshold=divergence_threshold)
                loss = -obj.mean() - entropy_coef * distribution.entropy().mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                clipped += clip_fraction(ratio.detach(), clip)
                batches += 1

        entry = {"iteration": iteration, "win_rate": wins, "steps": int(n), "degenerate_groups": degenerate,
                 "clip_fraction": clipped / max(batches, 1), "shifted_fraction": blocked_total / max(batches, 1)}
        history.append(entry)
        if not quiet:
            print(f"iter {iteration:3d}  win {wins:.3f}  steps {n:4d}  clipped {entry['clip_fraction']:.3f}"
                  + (f"  shifted {entry['shifted_fraction']:.3f}" if trust_region == "divergence" else ""))
        if out:
            torch.save({"state_dict": model.state_dict(), "encoding_version": ENCODING_VERSION, "win_rate": wins}, out)

    env.close()
    finals = [h["win_rate"] for h in history[-5:]] or [initial]
    result = {
        "advantage": advantage,
        "trust_region": trust_region,
        "encoding_version": ENCODING_VERSION,
        "group_size": group_size,
        "initial_win_rate": initial,
        "final_win_rate": float(np.mean(finals)),
        "best_win_rate": max(h["win_rate"] for h in history) if history else initial,
        "seconds": round(time.time() - started, 1),
        "history": history,
    }
    if not quiet:
        print(f"\n{advantage} + {trust_region}: {initial:.3f} -> {result['final_win_rate']:.3f} "
              f"(last five), best {result['best_win_rate']:.3f}, {result['seconds']}s")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Group-relative policy improvement.")
    parser.add_argument("worker")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--attacker", default=None)
    parser.add_argument("--defender", default=None)
    parser.add_argument("--advantage", default="loo", choices=ADVANTAGE_MODES)
    parser.add_argument("--trust-region", default="ratio", choices=TRUST_REGIONS)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    result = train(args.worker, checkpoint=args.checkpoint, attacker=args.attacker, defender=args.defender,
                   advantage=args.advantage, trust_region=args.trust_region, iterations=args.iterations,
                   group_size=args.group_size, groups_per_iter=args.groups,
                   divergence_threshold=args.threshold, out=args.out)
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
