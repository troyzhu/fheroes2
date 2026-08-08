#!/usr/bin/env python3
"""Do spatial planes earn their place, with capacity controlled? The ADR 0004 ablation.

Three arms from one seed on one planes-recorded corpus. The entity baseline is the shipped
network. The planes arm adds the conv fusion (about 441k extra parameters). The control arm is
the entity network with its trunk widened to land near the planes arm's parameter count,
because the capacity law measured on 2026-08-04 says cloning gains from width alone, so a
planes win over the baseline proves nothing until the control shows width alone does less.

The corpus must be recorded with the worker's --planes flag; the obstacle channel of episodes
recorded without it is all zeros, which would silently turn the planes arm into a capacity arm.

Usage:
    ./planes_ablation.py WORKER CORPUS_DIR --out-dir DIR [--epochs 25] [--seed 0]
                         [--report planes_ablation.json]
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

from fheroes2_agent.encoding import ENCODING_VERSION, encode_mask, encode_observation, encode_planes  # noqa: E402
from fheroes2_agent.policy import BattlePolicy  # noqa: E402


def load_planes_corpus(roots, require_planes: bool = True):
    """(observations, masks, actions, planes fp16, episode ids), planes verified nonzero.

    A list of roots concatenates, and repeating a root repeats its rows, the same double-weight
    convention the champion recipe uses on the flat corpus."""
    if isinstance(roots, str):
        roots = [roots]
    observations, masks, actions, planes, episodes = [], [], [], [], []
    episode_id = 0
    for path in sorted(q for root in roots for q in pathlib.Path(root).rglob("*.jsonl")):
        rows = 0
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("record") != "decision" or "observation" not in record:
                continue
            if not record.get("teacher_resolved") or record.get("teacher_action") is None:
                continue
            observations.append(encode_observation(record["observation"]))
            masks.append(encode_mask(record["legal_actions"]))
            actions.append(int(record["teacher_action"]))
            planes.append(encode_planes(record["observation"]).astype(np.float16))
            episodes.append(episode_id)
            rows += 1
        if rows:
            episode_id += 1
    planes_arr = np.stack(planes)
    if require_planes and float(np.abs(planes_arr[:, 6]).sum()) == 0.0:
        raise SystemExit("obstacle channel is all zeros: this corpus was not recorded with --planes")
    return (np.stack(observations), np.stack(masks), np.asarray(actions), planes_arr, np.asarray(episodes))


def split(episodes: np.ndarray, fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    ids = np.unique(episodes)
    rng.shuffle(ids)
    holdout = set(ids[: int(len(ids) * fraction)].tolist())
    mask = np.isin(episodes, list(holdout))
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def train_arm(data, arm: str, epochs: int, seed: int, out: str) -> dict:
    observations, masks, actions, planes, episodes = data
    torch.manual_seed(seed)
    train_idx, hold_idx = split(episodes, 0.2, seed)
    if arm == "planes":
        model = BattlePolicy(planes=True)
    elif arm == "wide":
        # Approximate capacity match for the planes arm, trunk-only widening; the report carries
        # the exact parameter counts so the approximation is visible.
        model = BattlePolicy(trunk_hidden=360)
    elif arm == "mean":
        model = BattlePolicy(pooling="mean")
    elif arm == "mean_wide":
        # Mean pooling drops parameters (the trunk narrows to one embedding), so this arm widens
        # the trunk back toward the concat baseline's count, the capacity control in reverse.
        model = BattlePolicy(pooling="mean", trunk_hidden=300)
    elif arm == "softplus":
        model = BattlePolicy(activation="softplus")
    elif arm in ("ent001", "ent005", "smooth005", "early8"):
        # The owner's sharpness program, 2026-08-07: a deterministic teacher drives imitation
        # toward one-hot, which starves later exploration, so these arms keep entropy alive.
        # ent*: the confidence penalty, loss = CE - beta * H, beta encouraging entropy.
        # smooth005: label smoothing spread over the legal set only.
        # early8: plain loss, training stopped at epoch 8 before the softmax saturates.
        model = BattlePolicy()
    else:
        model = BattlePolicy()
    parameters = sum(p.numel() for p in model.parameters())
    uses_planes = arm == "planes"
    entropy_beta = {"ent001": 0.01, "ent005": 0.05}.get(arm, 0.0)
    smooth_eps = 0.05 if arm == "smooth005" else 0.0
    if arm == "early8":
        epochs = min(epochs, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    obs = torch.from_numpy(observations)
    msk = torch.from_numpy(masks)
    act = torch.from_numpy(actions)
    pl = torch.from_numpy(planes)
    best = {"agreement": -1.0}
    beat_path = out + ".heartbeat.jsonl"
    for epoch in range(epochs):
        model.train()
        perm = torch.from_numpy(np.random.default_rng(seed * 1000 + epoch).permutation(train_idx))
        for start in range(0, len(perm), 256):
            batch = perm[start:start + 256]
            plane_arg = (pl[batch].float(),) if uses_planes else ()
            logits, _ = model(obs[batch], msk[batch], *plane_arg)
            if smooth_eps > 0.0:
                log_probs = torch.log_softmax(logits, dim=-1)
                legal = msk[batch].float()
                uniform_legal = legal / legal.sum(-1, keepdim=True)
                target = torch.zeros_like(log_probs).scatter_(1, act[batch].unsqueeze(1), 1.0)
                target = (1.0 - smooth_eps) * target + smooth_eps * uniform_legal
                loss = -(target * log_probs).sum(-1).mean()
            else:
                loss = torch.nn.functional.cross_entropy(logits, act[batch])
            if entropy_beta > 0.0:
                probs = torch.softmax(logits, dim=-1).clamp_min(1e-12)
                loss = loss - entropy_beta * (-(probs * probs.log()).sum(-1).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedule.step()
        model.eval()
        agree = 0
        with torch.no_grad():
            for start in range(0, len(hold_idx), 4096):
                batch = torch.from_numpy(hold_idx[start:start + 4096])
                plane_arg = (pl[batch].float(),) if uses_planes else ()
                logits, _ = model(obs[batch], msk[batch], *plane_arg)
                agree += int((logits.argmax(-1) == act[batch]).sum())
        agreement = agree / len(hold_idx)
        with open(beat_path, "a") as beat:
            beat.write(json.dumps({"iteration": epoch, "train_loss": float(loss),
                                   "holdout_agreement": agreement}) + "\n")
        if agreement > best["agreement"]:
            best = {"epoch": epoch, "agreement": agreement, "parameters": parameters}
            torch.save({"state_dict": model.state_dict(), "encoding_version": ENCODING_VERSION}, out)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker", help="recorded in the report for provenance; the battery runs separately")
    parser.add_argument("corpus", nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", default=None)
    parser.add_argument("--arms", nargs="+", default=["entity", "planes", "wide"],
                        choices=["entity", "planes", "wide", "mean", "mean_wide", "softplus", "ent001", "ent005", "smooth005", "early8"])
    args = parser.parse_args()

    started = time.time()
    data = load_planes_corpus(args.corpus)
    print(f"{len(data[2])} decisions, obstacle cells per state mean "
          f"{float(data[3][:, 6].sum(axis=(1, 2)).mean()):.1f}", flush=True)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in args.arms:
        results[arm] = train_arm(data, arm, args.epochs, args.seed, str(out_dir / f"policy_{arm}.pt"))
        print(f"{arm:7s} arm: agreement {results[arm]['agreement']:.4f} at epoch {results[arm]['epoch']}, "
              f"{results[arm]['parameters']:,} parameters", flush=True)

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"corpus": args.corpus, "worker": args.worker, "seed": args.seed,
             "results": results, "seconds": round(time.time() - started, 1)}, indent=2))
    print(f"total {round(time.time() - started)}s; battery the three checkpoints next")


if __name__ == "__main__":
    main()
