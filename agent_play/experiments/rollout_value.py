#!/usr/bin/env python3
"""A value trained on search's own rollouts, the owner's proposal, tested.

Every prior value here was fitted on played trajectories: one return per visited state, and the
lab measured where that fails. The owner's observation is that search already produces a far
richer dataset and discards it: each decision fans thirty-two rollouts across different
candidate branches, so one searched episode yields many outcome-labeled (state, action) pairs
covering counterfactual subtrees, and every labeled action was genuinely played out, which is
the support-safety the off-support survey demands by construction.

The collector has recorded exactly this since 2026-08-06: `search_values` carries each visited
candidate's mean rollout return and `search_visits` its rollout count. This script trains a
move-level Q on those targets, weighted by visit count since a mean of one rollout is noisier
than a mean of ten, and answers three questions in order: does it fit its holdout, does its
argmax reproduce search's choice, and does acting greedily on it play well, on the Thunk ladder
and a held-out slice, against the raw policy that generated the rollouts.

Usage:
    ./rollout_value.py WORKER POLICY --roots DIR [DIR ...] [--epochs 12] [--seed 0]
                       [--report rollout_value.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fheroes2_agent.encoding import ACTION_SPACE_SIZE, GLOBAL_FEATURES, SLOT_COUNT, SLOT_FEATURES, encode_mask, encode_observation  # noqa: E402
from fheroes2_agent.policy import load_policy  # noqa: E402
from fheroes2_agent.scenarios import Matchup, measure  # noqa: E402


class RolloutQ(nn.Module):
    """The dedicated-trunk shape the lab settled on, with a 793-wide head read at candidates."""

    def __init__(self, slot_hidden: int = 96, trunk: int = 192):
        super().__init__()
        self.slot = nn.Sequential(nn.Linear(SLOT_FEATURES, slot_hidden), nn.ReLU(),
                                  nn.Linear(slot_hidden, slot_hidden), nn.ReLU())
        self.glob = nn.Sequential(nn.Linear(GLOBAL_FEATURES, 32), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(SLOT_COUNT * slot_hidden + 32, trunk), nn.ReLU(),
                                   nn.Linear(trunk, trunk), nn.ReLU())
        self.head = nn.Linear(trunk, ACTION_SPACE_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        slots = x[:, : SLOT_COUNT * SLOT_FEATURES].view(b, SLOT_COUNT, SLOT_FEATURES)
        embedded = self.slot(slots) * slots[:, :, :1]
        joined = torch.cat([embedded.flatten(1), self.glob(x[:, SLOT_COUNT * SLOT_FEATURES:])], dim=1)
        return self.head(self.trunk(joined))


def load_rollout_dataset(roots):
    observations, masks, episode_ids = [], [], []
    cand_rows, cand_actions, cand_values, cand_weights = [], [], [], []
    search_choice = []
    episode = 0
    for root in roots:
        for path in sorted(pathlib.Path(root).rglob("*.jsonl")):
            rows_here = 0
            for line in path.read_text().splitlines():
                record = json.loads(line)
                if record.get("record") != "decision" or "search_values" not in record:
                    continue
                index = len(observations)
                observations.append(encode_observation(record["observation"]))
                masks.append(encode_mask(record["legal_actions"]))
                episode_ids.append(episode)
                search_choice.append(int(record["teacher_action"]))
                for action_str, value in record["search_values"].items():
                    visits = record["search_visits"].get(action_str, 1)
                    if visits < 1:
                        continue
                    cand_rows.append(index)
                    cand_actions.append(int(action_str))
                    cand_values.append(float(value))
                    cand_weights.append(float(visits))
                rows_here += 1
            if rows_here:
                episode += 1
    return (np.stack(observations), np.stack(masks), np.asarray(episode_ids),
            np.asarray(cand_rows), np.asarray(cand_actions),
            np.asarray(cand_values, dtype=np.float32), np.asarray(cand_weights, dtype=np.float32),
            np.asarray(search_choice))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("policy")
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    started = time.time()
    (obs, masks, episodes, rows, actions, values, weights, choices) = load_rollout_dataset(args.roots)
    per_state = len(rows) / max(len(obs), 1)
    print(f"{len(obs)} decisions, {len(rows)} rollout-valued (state, action) pairs, "
          f"{per_state:.1f} per state; one played episode would have given 1.0", flush=True)

    rng = np.random.default_rng(args.seed)
    unique_eps = np.unique(episodes)
    rng.shuffle(unique_eps)
    holdout_eps = set(unique_eps[: max(len(unique_eps) // 5, 1)].tolist())
    hold_state = np.isin(episodes, list(holdout_eps))
    hold_pair = hold_state[rows]

    torch.manual_seed(args.seed)
    model = RolloutQ()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    obs_t = torch.from_numpy(obs)
    rows_t = torch.from_numpy(rows)
    act_t = torch.from_numpy(actions)
    val_t = torch.from_numpy(values)
    w_t = torch.from_numpy(weights / weights.mean())
    train_idx = np.flatnonzero(~hold_pair)
    hold_idx = np.flatnonzero(hold_pair)

    for epoch in range(args.epochs):
        perm = torch.from_numpy(rng.permutation(train_idx))
        model.train()
        for start in range(0, len(perm), 512):
            batch = perm[start:start + 512]
            q = model(obs_t[rows_t[batch]])
            predicted = q.gather(1, act_t[batch].unsqueeze(1)).squeeze(1)
            loss = (w_t[batch] * (predicted - val_t[batch]) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            hb = torch.from_numpy(hold_idx)
            predicted = model(obs_t[rows_t[hb]]).gather(1, act_t[hb].unsqueeze(1)).squeeze(1)
        ev = 1.0 - float(np.var(values[hold_idx] - predicted.numpy()) / np.var(values[hold_idx]))
        print(f"epoch {epoch}: holdout EV on rollout values {ev:+.3f}", flush=True)

    # Does the fitted Q reproduce search's own choice at held-out states?
    hold_states = np.flatnonzero(hold_state)
    agree = 0
    with torch.no_grad():
        for start in range(0, len(hold_states), 2048):
            sl = hold_states[start:start + 2048]
            q = model(obs_t[sl])
            q[~torch.from_numpy(masks[sl])] = -1e9
            agree += int((q.argmax(-1).numpy() == choices[sl]).sum())
    search_agreement = agree / len(hold_states)
    print(f"argmax-Q matches search's chosen action on {search_agreement:.3f} of held-out states", flush=True)

    # And does acting greedily on it play? Same harness as every policy, Thunk rungs + held-out.
    class GreedyQ(nn.Module):
        def __init__(self, q):
            super().__init__()
            self.q = q
        def forward(self, observation, mask, planes=None):
            q = self.q(observation)
            logits = q * 50.0
            logits = logits.masked_fill(~mask, -1e9)
            return logits, q.max(-1).values

    greedy = GreedyQ(model)
    greedy.eval()
    raw = load_policy(torch.load(args.policy, map_location="cpu", weights_only=True)["state_dict"])
    raw.eval()
    thunk = Matchup("11:1,11:1,11:1,10:2,9:2", "1:334,1:333,1:333", attacker_hero="13:12", allow_wide=True)
    pool = json.loads((pathlib.Path(__file__).resolve().parents[1] / "docs" / "archive" / "experiments"
                       / "files" / "2026-08-05-run-reports" / "pool_value.json").read_text())["matchups"][40:50]
    results = {}
    for name, agent in (("greedy_rollout_q", greedy), ("raw_policy", raw)):
        held = float(np.mean([measure(agent, args.worker, Matchup(e["attacker"], e["defender"],
                                                                 attacker_hero=e.get("attacker_hero"),
                                                                 defender_hero=e.get("defender_hero"),
                                                                 allow_wide=bool(e.get("allow_wide"))),
                                      episodes=8, seeds=4)["win_rate"] for e in pool]))
        rung = measure(agent, args.worker, thunk, episodes=12, seeds=4)["win_rate"]
        results[name] = {"held_out10": held, "thunk_1000": rung}
        print(f"{name:18s} held-out(10) {held:.3f}  Thunk-1000 {rung:.2f}", flush=True)

    if args.out:
        torch.save({"state_dict": model.state_dict()}, args.out)
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"roots": args.roots, "decisions": int(len(obs)), "pairs": int(len(rows)),
             "pairs_per_state": round(per_state, 2), "holdout_ev": round(ev, 4),
             "search_agreement": round(search_agreement, 4), "play": results,
             "seconds": round(time.time() - started, 1)}, indent=1))


if __name__ == "__main__":
    main()
