#!/usr/bin/env python3
"""Which encoding earns its features, on data diverse in creatures and in counts?

Same demonstrations, same architecture, same training budget; the encoding is the only variable.
The variants target the three audit findings: counts scaled linearly across three orders of
magnitude, identity one-hots covering `simple_v1` only so every wide creature encodes with
all-zero identity, and no tail cell for wide units.

  v2          the deployed encoding, unchanged, the baseline
  v2log       counts and hit points log-scaled, everything else as v2
  v2log_wid   v2log with the one-hot extended to the wide_v1 roster and a tail-cell feature
  v2log_noid  v2log with the one-hot removed entirely, identity carried by stats alone

Two splits, because they answer different questions. The episode split is the usual one, does the
encoding clone better overall. The count-extrapolation split trains on episodes whose largest
stack is at most 200 creatures and tests on the rest, which is the direct form of the owner's
question about numbers: does the policy generalize to counts it never saw.

Usage:
    ./encoding_ablation.py DATA_DIR [--epochs 20]
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

from fheroes2_agent import encoding as enc  # noqa: E402
from fheroes2_agent.policy import BattlePolicy, masked_cross_entropy  # noqa: E402

AUDIT = json.loads((pathlib.Path(__file__).resolve().parents[2] / "python" / "fheroes2_agent"
                    / "data" / "monster_capabilities_v1.json").read_text())
WIDE_V1 = tuple(sorted(r["monster_id"] for r in AUDIT if r["wide_v1_supported"]))
WIDE_SLOT = {m: i for i, m in enumerate(WIDE_V1)}

LOG_COUNT = float(np.log1p(1000.0))
LOG_HP = float(np.log1p(50000.0))


def encode_variant(observation: dict, variant: str) -> np.ndarray:
    """Encode one observation under a named variant. v2 delegates to the deployed encoder."""
    if variant == "v2":
        return enc.encode_observation(observation)

    identity_ids = () if variant == "v2log_noid" else (WIDE_V1 if variant == "v2log_wid" else enc.SIMPLE_V1_MONSTERS)
    slot_map = WIDE_SLOT if variant == "v2log_wid" else enc.MONSTER_SLOT
    tail = variant == "v2log_wid"
    base_features = 22 + (2 if tail else 0)
    slot_features = base_features + len(identity_ids)
    size = enc.SLOT_COUNT * slot_features + enc.GLOBAL_FEATURES

    out = np.zeros(size, dtype=np.float32)
    units = observation["units"]
    active_is_attacker = bool(observation["active_is_attacker"])
    own = enemy = 0

    for slot, unit in enumerate(units):
        b = slot * slot_features
        is_attacker = unit["side"] == "attacker"
        is_own = is_attacker == active_is_attacker
        own += is_own
        enemy += not is_own
        cell = unit["head_cell"]
        row, column = (cell // enc.BOARD_WIDTH, cell % enc.BOARD_WIDTH) if cell >= 0 else (0, 0)
        initial = max(unit["initial_count"], 1)

        out[b + 0] = 1.0
        out[b + 1] = float(is_own)
        out[b + 2] = float(is_attacker)
        out[b + 3] = float(unit["active"])
        out[b + 4] = np.log1p(unit["count"]) / LOG_COUNT
        out[b + 5] = np.log1p(unit["initial_count"]) / LOG_COUNT
        out[b + 6] = unit["count"] / initial
        out[b + 7] = np.log1p(unit["hit_points"]) / LOG_HP
        out[b + 8] = np.log1p(unit["top_hit_points"]) / LOG_HP
        out[b + 9] = unit["attack"] / 10.0
        out[b + 10] = unit["defense"] / 10.0
        out[b + 11] = unit["speed"] / 10.0
        out[b + 12] = unit["shots"] / 20.0
        out[b + 13] = unit["morale"] / 3.0
        out[b + 14] = unit["luck"] / 3.0
        out[b + 15] = row / (enc.BOARD_HEIGHT - 1)
        out[b + 16] = column / (enc.BOARD_WIDTH - 1)
        out[b + 17] = max(cell, 0) / (enc.BOARD_CELLS - 1)
        out[b + 18] = float(unit["wide"])
        out[b + 19] = float(unit["flying"])
        out[b + 20] = float(unit["archer"])
        out[b + 21] = float(unit["hand_fighting"])
        if tail:
            tcell = unit.get("tail_cell", -1)
            trow, tcol = (tcell // enc.BOARD_WIDTH, tcell % enc.BOARD_WIDTH) if tcell >= 0 else (0, 0)
            out[b + 22] = trow / (enc.BOARD_HEIGHT - 1)
            out[b + 23] = tcol / (enc.BOARD_WIDTH - 1)
        if identity_ids:
            index = slot_map.get(unit["monster_id"])
            if index is not None:
                out[b + base_features + index] = 1.0

    g = enc.SLOT_COUNT * slot_features
    out[g + 0] = observation["round"] / 20.0
    out[g + 1] = float(active_is_attacker)
    out[g + 2] = own / enc.SLOT_COUNT
    out[g + 3] = enemy / enc.SLOT_COUNT
    return out


def slot_features_of(variant: str) -> int:
    if variant == "v2":
        return enc.SLOT_FEATURES
    if variant == "v2log":
        return 22 + len(enc.SIMPLE_V1_MONSTERS)
    if variant == "v2log_wid":
        return 24 + len(WIDE_V1)
    if variant == "v2log_noid":
        return 22
    raise ValueError(variant)


def load_raw(data_dir: str) -> list[dict]:
    """Raw decision records with episode id, max stack count and creature ids attached."""
    rows = []
    files = sorted(pathlib.Path(data_dir).rglob("*.jsonl"))
    for episode_id, path in enumerate(files):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        decisions = [r for r in records if r.get("record") == "decision"
                     and "observation" in r and r.get("teacher_resolved")]
        if not decisions:
            continue
        biggest = max(u["initial_count"] for r in decisions for u in r["observation"]["units"])
        for r in decisions:
            rows.append({"episode": episode_id, "observation": r["observation"],
                         "legal": r["legal_actions"], "action": int(r["teacher_action"]),
                         "max_count": biggest,
                         "creatures": frozenset(u["monster_id"] for u in r["observation"]["units"])})
    return rows


def train_eval(rows_train, rows_test, variant: str, epochs: int, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    sf = slot_features_of(variant)

    def tensors(rows):
        obs = np.stack([encode_variant(r["observation"], variant) for r in rows])
        masks = np.stack([enc.encode_mask(r["legal"]) for r in rows])
        actions = np.asarray([r["action"] for r in rows], dtype=np.int64)
        return (torch.from_numpy(obs.astype(np.float32)), torch.from_numpy(masks),
                torch.from_numpy(actions))

    to, tm, ta = tensors(rows_train)
    ho, hm, ha = tensors(rows_test)

    # The architecture reads its slot width from a module global at construction AND at every
    # forward pass, where it slices the observation. The patch therefore has to hold for the
    # whole of training and evaluation, not just the constructor; restoring it early sliced a
    # wide variant's tensors back to the narrow width, which is the shape error the first run of
    # this script died on.
    import fheroes2_agent.policy as pol
    saved = pol.SLOT_FEATURES
    pol.SLOT_FEATURES = sf
    try:
        return _train_eval_inner(to, tm, ta, ho, hm, ha, variant, epochs)
    finally:
        pol.SLOT_FEATURES = saved


def _train_eval_inner(to, tm, ta, ho, hm, ha, variant, epochs) -> dict:
    model = BattlePolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    best = 0.0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(ta))
        for start in range(0, len(order), 256):
            idx = order[start : start + 256]
            logits, _ = model(to[idx], tm[idx])
            loss = masked_cross_entropy(logits, ta[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        model.eval()
        hits = 0
        with torch.no_grad():
            for start in range(0, len(ha), 4096):
                logits, _ = model(ho[start : start + 4096], hm[start : start + 4096])
                hits += int((logits.argmax(-1) == ha[start : start + 4096]).sum())
        best = max(best, hits / len(ha))
    return {"variant": variant, "agreement": best, "parameters": sum(p.numel() for p in model.parameters())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--count-split", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    rows = load_raw(args.data_dir)
    episodes = sorted({r["episode"] for r in rows})
    rng = np.random.default_rng(args.seed)
    held = set(rng.permutation(episodes)[: max(1, int(len(episodes) * args.holdout))].tolist())
    train_rows = [r for r in rows if r["episode"] not in held]
    test_rows = [r for r in rows if r["episode"] in held]
    print(f"{len(rows)} decisions from {len(episodes)} episodes; episode split "
          f"{len(train_rows)} train, {len(test_rows)} held out", flush=True)

    small = [r for r in rows if r["max_count"] <= args.count_split]
    large = [r for r in rows if r["max_count"] > args.count_split]
    print(f"count split at {args.count_split}: {len(small)} small-count decisions train, "
          f"{len(large)} large-count test", flush=True)

    variants = ("v2", "v2log", "v2log_wid", "v2log_noid")
    results = {"episode_split": [], "count_extrapolation": []}
    started = time.time()
    for variant in variants:
        r1 = train_eval(train_rows, test_rows, variant, args.epochs, args.seed)
        results["episode_split"].append(r1)
        print(f"  episode split       {variant:11s} agreement {r1['agreement']:.4f} "
              f"({r1['parameters']:,} params)", flush=True)
    for variant in variants:
        r2 = train_eval(small, large, variant, args.epochs, args.seed)
        results["count_extrapolation"].append(r2)
        print(f"  count extrapolation {variant:11s} agreement {r2['agreement']:.4f}", flush=True)

    print(f"\n{time.time() - started:.0f}s total")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"results": results, "epochs": args.epochs, "count_split": args.count_split,
             "decisions": len(rows), "seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
