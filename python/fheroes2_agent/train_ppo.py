"""Stage 3: refine the cloned policy with masked PPO.

The clipped surrogate and GAE are derived in `agent_play/docs/rl/rl-methods.md`. Two integration
details decide whether this works at all, and both are asserted rather than assumed.

The mask is applied when sampling and again when recomputing log-probabilities, or the ratio is
not one at the current iterate and the clipping window is centred on the wrong point.

Truncation is distinguished from termination. A battle that hit the round limit has a future that
was cut off, so its final value must be bootstrapped rather than treated as zero; treating the
two alike biases every value estimate downward, and is the most common environment-side bug in
reinforcement learning.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from .encoding import ACTION_SPACE_SIZE, ENCODING_VERSION, OBSERVATION_SIZE
from .env import BattleEnv
from .objectives import normalize_advantages
from .policy import BattlePolicy


def collect(env: BattleEnv, model: BattlePolicy, episodes: int) -> dict:
    """Roll out whole episodes. Battles are 5 to 40 decisions, so an episode is the natural unit."""
    obs_buf, mask_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], [], []
    outcomes = []

    for _ in range(episodes):
        observation, mask = env.reset()
        while True:
            obs_t = torch.from_numpy(observation).unsqueeze(0)
            mask_t = torch.from_numpy(mask).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(obs_t, mask_t)
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
                logp = distribution.log_prob(action)

            step = env.step(int(action))
            obs_buf.append(observation)
            mask_buf.append(mask)
            act_buf.append(int(action))
            logp_buf.append(float(logp))
            val_buf.append(float(value))
            rew_buf.append(step.reward)
            done_buf.append(step.done)

            if step.done:
                outcomes.append(step.info)
                break
            observation, mask = step.observation, step.mask

    return {
        "observations": np.stack(obs_buf).astype(np.float32),
        "masks": np.stack(mask_buf),
        "actions": np.asarray(act_buf, dtype=np.int64),
        "logps": np.asarray(logp_buf, dtype=np.float32),
        "values": np.asarray(val_buf, dtype=np.float32),
        "rewards": np.asarray(rew_buf, dtype=np.float32),
        "dones": np.asarray(done_buf, dtype=bool),
        "outcomes": outcomes,
    }


def compute_gae(rewards, values, dones, truncated, gamma=0.99, lam=0.95) -> tuple[np.ndarray, np.ndarray]:
    """Backward recursion, which is the O(T) form rather than the O(T^2) sum.

    `truncated` marks a step that ended because the round limit was reached. Its successor value
    is bootstrapped; a genuine termination's is zero.
    """
    advantages = np.zeros_like(rewards)
    running = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            # A truncated episode still has a future, so bootstrap it; a finished one does not.
            next_value = values[t] if truncated[t] else 0.0
            running = 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
        next_value = values[t]
    return advantages, advantages + values


def win_rate(outcomes: list[dict], side: str) -> float:
    target = "victory" if side == "attacker" else "defeat"
    return float(np.mean([o["termination"] == target for o in outcomes])) if outcomes else 0.0


def train(
    worker: str,
    checkpoint: str | None = None,
    fixture: str = "m1_tiny_melee",
    side: str = "attacker",
    attacker: str | None = None,
    defender: str | None = None,
    attacker_hero: str | None = None,
    defender_hero: str | None = None,
    allow_wide: bool = False,
    iterations: int = 20,
    episodes_per_iter: int = 32,
    epochs: int = 4,
    minibatch: int = 256,
    lr: float = 1e-4,
    clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    seed: int = 0,
    out: str | None = None,
    heartbeat: str | None = None,
    quiet: bool = False,
    advantage_std_floor: float = 0.1,
    value_warmup_iters: int = 0,
    env: object | None = None,
    model_kwargs: dict | None = None,
) -> dict:
    torch.manual_seed(seed)
    # Width overrides exist for capacity experiments. A checkpoint records no widths, so loading
    # one trained at another size fails loudly on shape mismatch rather than silently.
    model = BattlePolicy(**(model_kwargs or {}))

    started_from = "random initialization"
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("encoding_version") != ENCODING_VERSION:
            raise ValueError(f"checkpoint encoding {state.get('encoding_version')} does not match {ENCODING_VERSION}")
        model.load_state_dict(state["state_dict"])
        started_from = f"cloned policy ({checkpoint})"
    if not quiet:
        print(f"starting from {started_from}")

    # A caller may pass its own environment. `collect` needs only reset and step, so a MatchupPool
    # rotating over many army pairs substitutes for a single fixed one, which is what turns a
    # result about one matchup into a result about a distribution. Matches `train_group`.
    if env is None:
        env = BattleEnv(worker, fixture=fixture, side=side, attacker=attacker, defender=defender,
                        attacker_hero=attacker_hero, defender_hero=defender_hero, allow_wide=allow_wide)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    baseline = collect(env, model, episodes_per_iter)
    initial_win = win_rate(baseline["outcomes"], side)
    initial_reward = float(np.mean([r for r, d in zip(baseline["rewards"], baseline["dones"]) if d]))
    if not quiet:
        print(f"before training: win rate {initial_win:.3f}, mean terminal reward {initial_reward:.3f}")

    history = []
    degenerate = 0
    started = time.time()

    # Value warmup (diagnosed 2026-08-07 by the per-term gradient norms): when the anchor's
    # value head has never seen the objective being optimized, its early errors dominate the
    # shared trunk, 11.9 against the policy term's 2.2 in the first live reading. Warmup
    # iterations update the value head alone, everything else frozen, so the head lands on the
    # new reward scale before any gradient touches what the policy relies on.
    head_only = [parameter for name, parameter in model.named_parameters() if name.startswith("value_head")]
    warmup_optimizer = torch.optim.Adam(head_only, lr=1e-3) if value_warmup_iters else None
    for warmup in range(value_warmup_iters):
        batch = collect(env, model, episodes_per_iter)
        truncated = np.array([o["termination"] == "round_limit" for o in batch["outcomes"]])
        step_truncated = np.zeros_like(batch["dones"])
        step_truncated[np.flatnonzero(batch["dones"])] = truncated
        _, warm_returns = compute_gae(batch["rewards"], batch["values"], batch["dones"], step_truncated)
        obs_w = torch.from_numpy(batch["observations"])
        masks_w = torch.from_numpy(batch["masks"])
        ret_w = torch.from_numpy(warm_returns.astype(np.float32))
        for _ in range(epochs):
            _, values_w = model(obs_w, masks_w)
            loss_w = ((values_w - ret_w) ** 2).mean()
            warmup_optimizer.zero_grad(set_to_none=True)
            loss_w.backward()
            warmup_optimizer.step()
        if not quiet:
            print(f"value warmup {warmup}: mse {float(loss_w):.3f}")

    grad_norms_first: dict = {}
    for iteration in range(iterations):
        batch = collect(env, model, episodes_per_iter)
        # Only the round-limit cap is a truncation, a battle cut off with a future still worth
        # bootstrapping. A stalemate is decisive as of the 2026-08-06 semantics: the reward
        # already carries the engine's own resolution, defender wins and the attacker's army is
        # forfeit, so bootstrapping a value on top would count a future the outcome says does
        # not exist.
        truncated = np.array([o["termination"] == "round_limit" for o in batch["outcomes"]])
        # Expand per-episode truncation onto the step that ended each episode.
        step_truncated = np.zeros_like(batch["dones"])
        step_truncated[np.flatnonzero(batch["dones"])] = truncated

        advantages, returns = compute_gae(batch["rewards"], batch["values"], batch["dones"], step_truncated)
        # Recorded before normalization. As a matchup is solved every episode returns nearly the
        # same reward, so this shrinks toward zero, and dividing by it turns whatever noise is
        # left into full-sized advantages. That is a mechanism for a converged run to destroy
        # itself, and it is invisible after normalization, which always reports a spread of one.
        raw_advantage_std = float(advantages.std())
        episode_rewards = [r for r, d in zip(batch["rewards"], batch["dones"]) if d]
        reward_std = float(np.std(episode_rewards))
        degenerate += raw_advantage_std < advantage_std_floor
        advantages = normalize_advantages(advantages, advantage_std_floor)

        obs = torch.from_numpy(batch["observations"])
        masks = torch.from_numpy(batch["masks"])
        actions = torch.from_numpy(batch["actions"])
        old_logps = torch.from_numpy(batch["logps"])
        adv_t = torch.from_numpy(advantages.astype(np.float32))
        ret_t = torch.from_numpy(returns.astype(np.float32))

        n = len(actions)
        # Value loss on the rollout, before any update touches it. This is what a critic
        # pre-fitted on teacher play is supposed to reduce, and it is measured at the first
        # minibatch of the iteration so it reflects the critic the rollout was collected with
        # rather than the one left behind after the update.
        with torch.no_grad():
            logits_before, values_before = model(obs, masks)
            value_loss_before = float(((values_before - ret_t) ** 2).mean())
            # Entropy over the legal set alone, so it measures the policy's indecision rather than
            # the mask. A policy that has learned is sharp, and a sharp policy is what a noisy
            # update can push off a cliff; a diffuse one absorbs the same noise harmlessly. This is
            # the diagnostic that distinguishes those two states, and it was missing when the
            # collapse was first traced.
            entropy_before = float(torch.distributions.Categorical(logits=logits_before).entropy().mean())

        for _ in range(epochs):
            order = torch.randperm(n)
            for start in range(0, n, minibatch):
                rows = order[start : start + minibatch]
                # The mask is applied here too. Without it the ratio is not one at theta_old and
                # the clipping window is centred on the wrong point.
                logits, values = model(obs[rows], masks[rows])
                distribution = torch.distributions.Categorical(logits=logits)
                logps = distribution.log_prob(actions[rows])

                ratio = torch.exp(logps - old_logps[rows])
                surrogate = torch.min(ratio * adv_t[rows], torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[rows])
                # Entropy over the legal set alone, or it measures the mask rather than the
                # policy's indecision.
                entropy = distribution.entropy().mean()
                policy_term = -surrogate.mean()
                value_term = value_coef * ((values - ret_t[rows]) ** 2).mean()
                entropy_term = -entropy_coef * entropy
                loss = policy_term + value_term + entropy_term

                # Per-term gradient norms, measured before the sum, on the epoch's first
                # minibatch (owner-requested diagnostic, 2026-08-07). A shared trunk means the
                # heads compete for the same weights, and the post-sum clipped norm cannot say
                # which term dominated; three extra backward passes once per epoch can.
                if start == 0:
                    term_norms = {}
                    for term_name, term in (("policy", policy_term), ("value", value_term),
                                            ("entropy", entropy_term)):
                        optimizer.zero_grad(set_to_none=True)
                        term.backward(retain_graph=True)
                        # Decomposed per top-level module (owner-requested 2026-08-07), because
                        # the interference question is not how big a term's gradient is but
                        # where it lands: the value term's share inside the shared trunk is the
                        # number the warmup exists to shrink.
                        per_module: dict[str, float] = {}
                        for name, parameter in model.named_parameters():
                            if parameter.grad is not None:
                                module = name.split(".")[0]
                                per_module[module] = per_module.get(module, 0.0) + float(parameter.grad.norm()) ** 2
                        term_norms[term_name] = {module: value ** 0.5 for module, value in per_module.items()}
                        term_norms[term_name]["total"] = sum(per_module.values()) ** 0.5
                    grad_norms_first = term_norms

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                pre_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5))
                if start == 0:
                    grad_norms_first["total_pre_clip"] = pre_clip
                optimizer.step()

        wr = win_rate(batch["outcomes"], side)
        if out:
            torch.save({"state_dict": model.state_dict(), "encoding_version": ENCODING_VERSION,
                        "win_rate": wr, "iteration": iteration}, out)
        mean_reward = float(np.mean(episode_rewards))
        history.append({"iteration": iteration, "win_rate": wr, "mean_terminal_reward": mean_reward,
                        "steps": int(n), "value_loss": value_loss_before,
                        "raw_advantage_std": raw_advantage_std, "reward_std": reward_std,
                        "entropy": entropy_before,
                        "grad_norms": grad_norms_first})
        # Live monitoring heartbeat (owner-requested 2026-08-07): one JSON line per iteration,
        # appended as it happens, so a dashboard can watch training health without waiting for
        # the end-of-run report. Defaults on whenever a checkpoint path exists.
        beat_path = heartbeat or (out + ".heartbeat.jsonl" if out else None)
        if beat_path:
            with open(beat_path, "a") as beat:
                beat.write(json.dumps(history[-1]) + "\n")
        if not quiet:
            print(f"iter {iteration:3d}  win_rate {wr:.3f}  mean_terminal_reward {mean_reward:+.3f}  "
                  f"value_loss {value_loss_before:.3f}  entropy {entropy_before:.3f}  steps {n}")

    env.close()
    final_win = history[-1]["win_rate"] if history else initial_win
    result = {
        "encoding_version": ENCODING_VERSION,
        "started_from": started_from,
        "fixture": fixture,
        "side": side,
        "attacker": attacker,
        "defender": defender,
        "initial_win_rate": initial_win,
        "final_win_rate": final_win,
        "initial_mean_terminal_reward": initial_reward,
        "iterations": iterations,
        "episodes_per_iteration": episodes_per_iter,
        # Reported rather than buried. A run that spent most of its budget below the floor was
        # training on batches with almost no outcome spread, which is a statement about the
        # matchup having been solved rather than about the method.
        "advantage_std_floor": advantage_std_floor,
        "floored_iterations": int(degenerate),
        "seconds": round(time.time() - started, 1),
        "history": history,
    }
    if not quiet:
        print(f"\nwin rate {initial_win:.3f} -> {final_win:.3f} over {iterations} iterations ({result['seconds']}s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine a battle policy with masked PPO.")
    parser.add_argument("worker")
    parser.add_argument("--checkpoint", default=None, help="cloned policy to start from")
    parser.add_argument("--fixture", default="m1_tiny_melee")
    parser.add_argument("--side", default="attacker")
    parser.add_argument("--attacker", default=None, help="army override, monsterId:count,...")
    parser.add_argument("--defender", default=None)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--report", default=None)
    parser.add_argument("--out", default=None, help="checkpoint path for the refined policy")
    # Rollouts sample from the policy, so the seed is what makes one run differ from another on
    # an identical configuration. Without it on the command line a comparison across methods
    # cannot be separated from a comparison across noise.
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = train(args.worker, checkpoint=args.checkpoint, fixture=args.fixture, side=args.side,
                   attacker=args.attacker, defender=args.defender,
                   iterations=args.iterations, episodes_per_iter=args.episodes,
                   seed=args.seed, out=args.out)
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
