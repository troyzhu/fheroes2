"""Stage 1: clone the built-in AI.

Supervised classification over the masked action space, per
`agent_play/docs/rl/training-design.md`. The reported metric is teacher agreement on held-out
episodes, which is the fraction of decisions where the policy's highest-scoring legal action is
the one the teacher took.

Agreement is an upper bound on what cloning can achieve here rather than a target to maximize.
The teacher plays both sides, so a perfect clone equals the teacher and does not beat it, and the
minimum achievable loss is the teacher's own conditional entropy given what the observation
shows, not zero.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from .dataset import Samples, load_dir, split_by_episode, summarize
from .encoding import ENCODING_VERSION
from .policy import BattlePolicy, masked_cross_entropy, parameter_count


def to_tensors(samples: Samples, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(samples.observations).to(device),
        torch.from_numpy(samples.masks).to(device),
        torch.from_numpy(samples.actions).to(device),
    )


@torch.no_grad()
def evaluate(model: BattlePolicy, observations: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor, batch: int = 4096) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    for start in range(0, len(actions), batch):
        stop = start + batch
        logits, _ = model(observations[start:stop], masks[start:stop])
        total_loss += float(masked_cross_entropy(logits, actions[start:stop])) * (stop - start if stop <= len(actions) else len(actions) - start)
        correct += int((logits.argmax(dim=-1) == actions[start:stop]).sum())
    n = len(actions)
    return {"loss": total_loss / n, "agreement": correct / n}


def train(
    data_dir: str,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    out: str | None = None,
    model_kwargs: dict | None = None,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cpu")  # small model, small data; MPS is slower here than it is worth

    samples = load_dir(data_dir)
    print(summarize(samples))
    train_s, holdout_s = split_by_episode(samples, holdout_fraction, seed)
    print(f"train {len(train_s)} decisions, holdout {len(holdout_s)} decisions, split by episode")

    train_obs, train_masks, train_actions = to_tensors(train_s, device)
    hold_obs, hold_masks, hold_actions = to_tensors(holdout_s, device)

    # Width overrides exist for capacity experiments; the default is the deployed size, and a
    # checkpoint records no widths, so whoever loads one must pass the same kwargs.
    model = BattlePolicy(**(model_kwargs or {})).to(device)
    print(f"policy has {parameter_count(model):,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # A policy that always picks the teacher's most common action, and one that picks uniformly
    # among legal actions. Any agreement below these is worse than not training.
    majority = int(np.bincount(train_s.actions, minlength=793).argmax())
    baseline_majority = float((holdout_s.actions == majority).mean())
    baseline_uniform = float((1.0 / holdout_s.masks.sum(axis=1)).mean())
    print(f"baselines: majority-action {baseline_majority:.3f}, uniform-over-legal {baseline_uniform:.3f}")

    history = []
    best = {"agreement": -1.0}
    started = time.time()

    for epoch in range(epochs):
        model.train()
        # The rate this epoch actually trains at, recorded before the anneal advances it, so
        # the history is self-contained about the schedule rather than implying it.
        epoch_lr = schedule.get_last_lr()[0]
        order = torch.randperm(len(train_actions))
        running = 0.0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            logits, _ = model(train_obs[rows], train_masks[rows])
            loss = masked_cross_entropy(logits, train_actions[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            running += float(loss) * len(rows)
        schedule.step()

        train_loss = running / len(train_actions)
        metrics = evaluate(model, hold_obs, hold_masks, hold_actions)
        history.append({"epoch": epoch, "train_loss": train_loss, "lr": epoch_lr, **metrics})
        print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  holdout_loss {metrics['loss']:.4f}  agreement {metrics['agreement']:.4f}")

        if metrics["agreement"] > best["agreement"]:
            best = {"epoch": epoch, **metrics}
            if out:
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "encoding_version": ENCODING_VERSION,
                        "agreement": metrics["agreement"],
                        "epoch": epoch,
                    },
                    out,
                )

    result = {
        "encoding_version": ENCODING_VERSION,
        "samples": len(samples),
        "train": len(train_s),
        "holdout": len(holdout_s),
        "parameters": parameter_count(model),
        "epochs": epochs,
        "baseline_majority_action": baseline_majority,
        "baseline_uniform_over_legal": baseline_uniform,
        "best": best,
        "seconds": round(time.time() - started, 1),
        "history": history,
    }
    print(f"\nbest holdout agreement {best['agreement']:.4f} at epoch {best['epoch']} ({result['seconds']}s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Behaviour-clone the built-in battle AI.")
    parser.add_argument("data_dir", help="directory of recorded .jsonl episodes")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", default=None, help="checkpoint path for the best epoch")
    parser.add_argument("--report", default=None, help="write the run's metrics as JSON")
    args = parser.parse_args()

    result = train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, out=args.out)
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
