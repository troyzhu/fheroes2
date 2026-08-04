"""Stage 2b: fit the value head on teacher play before any reinforcement learning.

Proposed in `agent_play/docs/rl/training-design.md` and unbuilt until now. Stage 1 trains a policy
head and leaves the value head at initialization, which discards something the setup gives away:
the teacher plays both sides of every battle, so realized returns are available in unlimited
quantity without a learner.

Regressing the value head on those returns is Monte Carlo policy evaluation, a supervised problem,
and it fits the value of the teacher's policy against a teacher opponent.

Why the mismatch does not break it. That is not the value of the policy being trained, and the two
diverge as soon as reinforcement learning starts. A baseline only has to be independent of the
action to leave the gradient unbiased, and a value that depends on the observation alone satisfies
that however far the learner has drifted. It buys less variance reduction than the correct critic,
nothing worse. Bias enters only through bootstrapping.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from .dataset import load_dir, split_by_episode
from .encoding import ENCODING_VERSION
from .policy import BattlePolicy


def evaluate(model: BattlePolicy, observations: torch.Tensor, masks: torch.Tensor, targets: torch.Tensor,
             batch: int = 4096) -> dict[str, float]:
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(targets), batch):
            stop = start + batch
            _, values = model(observations[start:stop], masks[start:stop])
            errors.append(values - targets[start:stop])
    residual = torch.cat(errors)
    mse = float((residual ** 2).mean())
    variance = float(targets.var())
    # Fraction of the return's variance the critic explains. Zero means it is no better than
    # predicting the mean, which is the bar a critic has to clear to be worth its cost.
    return {"mse": mse, "mae": float(residual.abs().mean()), "explained_variance": 1.0 - mse / variance if variance > 0 else 0.0}


def train(
    data_dir: str,
    checkpoint: str,
    out: str | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    freeze_policy: bool = True,
) -> dict:
    torch.manual_seed(seed)

    samples = load_dir(data_dir)
    if samples.returns is None or not np.isfinite(samples.returns).all():
        raise ValueError("episodes carry no returns; were they recorded with --audit-coverage?")
    train_s, hold_s = split_by_episode(samples, holdout_fraction, seed)
    print(f"{len(samples)} decisions, {len(train_s)} training and {len(hold_s)} held out, split by episode")

    model = BattlePolicy()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("encoding_version") != ENCODING_VERSION:
        raise ValueError(f"checkpoint encoding {state.get('encoding_version')} does not match {ENCODING_VERSION}")
    model.load_state_dict(state["state_dict"])

    # The trunk is shared, so training the value head end to end would move the cloned policy as a
    # side effect. Freezing everything but the value head keeps stage 1's result intact, at the
    # cost of a critic that can only read features chosen for predicting actions.
    if freeze_policy:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("value_head")
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"training {sum(p.numel() for p in trainable):,} parameters"
          + (" (value head only, trunk frozen)" if freeze_policy else " (whole network)"))

    to_t = lambda s: (torch.from_numpy(s.observations), torch.from_numpy(s.masks), torch.from_numpy(s.returns))
    train_obs, train_masks, train_targets = to_t(train_s)
    hold_obs, hold_masks, hold_targets = to_t(hold_s)

    baseline = evaluate(model, hold_obs, hold_masks, hold_targets)
    print(f"before: held-out mse {baseline['mse']:.4f}, explained variance {baseline['explained_variance']:+.3f}")

    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    history = []
    started = time.time()

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_targets))
        running = 0.0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            _, values = model(train_obs[rows], train_masks[rows])
            loss = ((values - train_targets[rows]) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)
            optimizer.step()
            running += float(loss.detach()) * len(rows)

        metrics = evaluate(model, hold_obs, hold_masks, hold_targets)
        history.append({"epoch": epoch, "train_mse": running / len(train_targets), **metrics})
        if epoch % 5 == 4 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  train mse {running / len(train_targets):.4f}  "
                  f"held-out mse {metrics['mse']:.4f}  explained variance {metrics['explained_variance']:+.3f}")

    final = evaluate(model, hold_obs, hold_masks, hold_targets)
    if out:
        torch.save({"state_dict": model.state_dict(), "encoding_version": ENCODING_VERSION,
                    "explained_variance": final["explained_variance"]}, out)

    result = {
        "encoding_version": ENCODING_VERSION,
        "samples": len(samples),
        "frozen_policy": freeze_policy,
        "before": baseline,
        "after": final,
        "seconds": round(time.time() - started, 1),
        "history": history,
    }
    print(f"\nexplained variance {baseline['explained_variance']:+.3f} -> {final['explained_variance']:+.3f} "
          f"({result['seconds']}s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the value head on recorded teacher play.")
    parser.add_argument("data_dir")
    parser.add_argument("checkpoint", help="cloned policy whose value head is being fitted")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--unfreeze", action="store_true", help="train the whole network, not just the value head")
    args = parser.parse_args()

    result = train(args.data_dir, args.checkpoint, out=args.out, epochs=args.epochs, freeze_policy=not args.unfreeze)
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
