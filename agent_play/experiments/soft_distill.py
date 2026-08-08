#!/usr/bin/env python3
"""Distill search's whole measurement, not just its argmax: the owner's soft-target proposal.

A searched decision carries a value for every candidate search tried. The prior-anchored target
of Grill et al., pi_bar(a) proportional to prior(a) * exp(Q(a)/lambda), turns them into a
distribution that stays on support by construction, and the loss becomes cross-entropy against
that distribution instead of a one-hot. The paired twin trains on the identical corpus with the
identical pilot decisions as hard argmax labels, so the only difference between the arms is
whether the label keeps one number per state or all of them.

Usage:
    ./soft_distill.py --roots DIR [DIR ...] --soft-root DIR --out soft.pt --hard-out hard.pt
                      [--lam 0.5] [--soft-weight 2.0] [--epochs 25] [--seed 0] [--report R.json]
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

from fheroes2_agent.dataset import load_dir, split_by_episode  # noqa: E402
from fheroes2_agent.encoding import ACTION_SPACE_SIZE, ENCODING_VERSION, encode_mask, encode_observation  # noqa: E402
from fheroes2_agent.policy import BattlePolicy  # noqa: E402


def load_soft(roots, lam: float, target_kind: str = "values") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Soft rows: (observations, masks, hard argmax actions, dense pi_bar targets)."""
    if isinstance(roots, str):
        roots = [roots]
    observations, masks, actions, targets = [], [], [], []
    for path in sorted(q for root in roots for q in pathlib.Path(root).rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("record") != "decision" or "search_values" not in record:
                continue
            observations.append(encode_observation(record["observation"]))
            masks.append(encode_mask(record["legal_actions"]))
            actions.append(int(record["teacher_action"]))
            dense = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            if target_kind == "visits":
                # AlphaZero's own anti-collapse target: pi proportional to root visit counts at
                # temperature tau, softness encoding search's deliberation rather than value
                # arithmetic; tau = 1 is the canonical opening-move setting.
                for a, n in record["search_visits"].items():
                    if n > 0:
                        dense[int(a)] = float(n) ** (1.0 / lam)
            else:
                logits = {int(a): np.log(max(record["prior"][a], 1e-9)) + record["search_values"][a] / lam
                          for a in record["search_values"]}
                peak = max(logits.values())
                for a, l in logits.items():
                    dense[a] = np.exp(l - peak)
            dense /= dense.sum()
            targets.append(dense)
    return (np.stack(observations), np.stack(masks), np.asarray(actions), np.stack(targets))


def train_arm(hard, soft_rows, soft_as: str, soft_weight: float, epochs: int, seed: int, out: str) -> dict:
    """soft_as='distribution' trains on pi_bar; soft_as='argmax' trains the same rows one-hot."""
    torch.manual_seed(seed)
    train_s, holdout_s = split_by_episode(hard, 0.2, seed)
    obs = torch.from_numpy(np.concatenate([train_s.observations, soft_rows[0]]))
    masks = torch.from_numpy(np.concatenate([train_s.masks, soft_rows[1]]))
    actions = torch.from_numpy(np.concatenate([train_s.actions, soft_rows[2]]))
    dense = torch.from_numpy(soft_rows[3])
    n_hard = len(train_s.actions)
    weights = torch.ones(len(actions))
    weights[n_hard:] = soft_weight

    model = BattlePolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    hobs, hmasks, hactions = (torch.from_numpy(holdout_s.observations), torch.from_numpy(holdout_s.masks),
                              torch.from_numpy(holdout_s.actions))
    best = {"agreement": -1.0}
    # Per-epoch training diagnostics (owner requirement, 2026-08-08): loss decomposed into its
    # hard and soft terms before the sum, the live learning rate, and holdout agreement, kept in
    # the report and appended per epoch to a heartbeat the dashboard and the convergence report
    # can read. The first coverage-corpus verdict was drawn without these, which was the gap.
    history = []
    beat_path = out + ".heartbeat.jsonl"
    for epoch in range(epochs):
        model.train()
        epoch_lr = schedule.get_last_lr()[0]
        perm = torch.randperm(len(actions))
        running_hard = running_soft = 0.0
        for start in range(0, len(actions), 256):
            batch = perm[start:start + 256]
            logits, _ = model(obs[batch], masks[batch])
            log_probs = torch.log_softmax(logits, dim=-1)
            hard_mask = batch < n_hard
            loss = torch.zeros((), dtype=torch.float32)
            if hard_mask.any():
                rows = batch[hard_mask]
                ce = torch.nn.functional.nll_loss(log_probs[hard_mask], actions[rows], reduction="none")
                hard_term = (ce * weights[rows]).sum()
                loss = loss + hard_term
                running_hard += float(hard_term)
            soft_mask = ~hard_mask
            if soft_mask.any():
                rows = batch[soft_mask]
                if soft_as == "distribution":
                    ce = -(dense[rows - n_hard] * log_probs[soft_mask]).sum(-1)
                else:
                    ce = torch.nn.functional.nll_loss(log_probs[soft_mask], actions[rows], reduction="none")
                soft_term = (ce * weights[rows]).sum()
                loss = loss + soft_term
                running_soft += float(soft_term)
            loss = loss / len(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedule.step()
        model.eval()
        with torch.no_grad():
            agree = hits = 0
            for start in range(0, len(hactions), 4096):
                sl = slice(start, start + 4096)
                logits, _ = model(hobs[sl], hmasks[sl])
                agree += int((logits.argmax(-1) == hactions[sl]).sum())
                hits += len(hactions[sl])
        agreement = agree / hits
        row = {"epoch": epoch, "train_loss_hard": round(running_hard / len(actions), 5),
               "train_loss_soft": round(running_soft / len(actions), 5),
               "train_loss": round((running_hard + running_soft) / len(actions), 5),
               "holdout_agreement": round(agreement, 5), "lr": epoch_lr}
        history.append(row)
        with open(beat_path, "a") as beat:
            beat.write(json.dumps(row) + "\n")
        if agreement > best["agreement"]:
            best = {"epoch": epoch, "agreement": agreement}
            torch.save({"state_dict": model.state_dict(), "encoding_version": ENCODING_VERSION}, out)
    return best | {"history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--soft-root", nargs="+", required=True)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--target", default="values", choices=("values", "visits"),
                        help="visits builds AlphaZero-style pi proportional to N^(1/lam)")
    parser.add_argument("--soft-weight", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hard-out", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    started = time.time()
    hard = load_dir(list(args.roots))
    soft_rows = load_soft(args.soft_root, args.lam, args.target)
    entropy = float(np.mean([-(t[t > 0] * np.log(t[t > 0])).sum() for t in soft_rows[3]]))
    print(f"{len(hard.actions)} hard decisions + {len(soft_rows[2])} soft decisions; "
          f"target entropy {entropy:.3f} nats at lambda {args.lam}", flush=True)

    soft = train_arm(hard, soft_rows, "distribution", args.soft_weight, args.epochs, args.seed, args.out)
    print(f"soft-target arm: best agreement {soft['agreement']:.4f} at epoch {soft['epoch']}", flush=True)
    hard_arm = train_arm(hard, soft_rows, "argmax", args.soft_weight, args.epochs, args.seed, args.hard_out)
    print(f"hard-label twin: best agreement {hard_arm['agreement']:.4f} at epoch {hard_arm['epoch']}", flush=True)

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"roots": args.roots, "soft_root": args.soft_root, "lam": args.lam,
             "soft_weight": args.soft_weight, "seed": args.seed, "target_entropy": entropy,
             "soft": soft, "hard_twin": hard_arm, "seconds": round(time.time() - started, 1)}, indent=2))
    print(f"total {round(time.time() - started)}s")


if __name__ == "__main__":
    main()
