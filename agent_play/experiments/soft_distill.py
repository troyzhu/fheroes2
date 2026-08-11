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


def load_soft(roots, lam: float, target_kind: str = "values") -> tuple:
    """Soft rows: (observations, masks, hard argmax actions, dense targets, regret, visit margin).

    The visit margin is how many more playouts the labeled action received than the one the prior
    would have taken. It is a confidence reading the regret does not carry: a margin of one means
    search explored the two candidates equally and the value tie-break decided, so the label rests
    on a single-playout difference. Measured causally on 2026-08-10, overrules at a margin of one
    are worth $+0.057$ against roughly $+0.20$ for wider margins (`label_value.py`), which is why
    it is loaded rather than inferred from the regret.
    """
    if isinstance(roots, str):
        roots = [roots]
    observations, masks, actions, targets, regrets, margins = [], [], [], [], [], []
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
            # Measured regret: what the labeled action is worth over the one the prior would have
            # taken unaided, in the reward units the rollouts measured. On the first scaled corpus
            # 93.2 percent of decisions carry none, so an unweighted loss spends almost all of its
            # gradient confirming choices the policy already makes.
            values = {int(a): v for a, v in record["search_values"].items()}
            priors = {int(a): v for a, v in record["prior"].items()}
            prior_pick = max(priors, key=priors.get) if priors else None
            regrets.append(max(values.get(int(record["teacher_action"]), 0.0)
                               - values.get(prior_pick, 0.0), 0.0) if prior_pick is not None else 0.0)
            visits = {int(a): v for a, v in record.get("search_visits", {}).items()}
            margins.append(float(visits.get(int(record["teacher_action"]), 0)
                                 - visits.get(prior_pick, 0)) if prior_pick is not None else 0.0)
    return (np.stack(observations), np.stack(masks), np.asarray(actions), np.stack(targets),
            np.asarray(regrets, dtype=np.float32), np.asarray(margins, dtype=np.float32))


def train_arm(hard, soft_rows, soft_as: str, soft_weight: float, epochs: int, seed: int, out: str,
              regret_weighted: bool = False, min_visit_margin: float = 0.0,
              checkpoint_every: int = 0, entropy_bonus: float = 0.0,
              restart_period: int = 0) -> dict:
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
    if min_visit_margin > 0:
        # Drop the overrules search was not confident about rather than down-weighting them. A
        # rank transform keeps every row at some weight, and the causal measurement says the
        # single-playout tie-breaks carry almost nothing, so this asks whether removing them
        # outright beats keeping them cheap. Mass is renormalised over the survivors so the arm
        # carries the same total soft mass as its twin and only the distribution differs.
        keep = soft_rows[5] >= min_visit_margin
        kept = float(keep.sum())
        if kept:
            scale = torch.from_numpy((keep.astype("float32") * (len(keep) / kept)))
            weights[n_hard:] = weights[n_hard:] * scale
    if regret_weighted:
        # Rank-transformed, because a maximum over about thirty candidates each priced by one
        # rollout is upward biased, and renormalized to mean one so the arm carries exactly the
        # same total soft mass as its unweighted twin: the comparison isolates where the mass
        # sits, not how much of it there is.
        regret = soft_rows[4]
        order = np.argsort(np.argsort(regret))
        percentile = order / max(len(order) - 1, 1)
        multiplier = 0.25 + 1.5 * percentile
        multiplier = multiplier / multiplier.mean()
        # Composes with the filter above: the rank weighting redistributes among whatever survived.
        surviving = weights[n_hard:].numpy() / soft_weight
        combined = surviving * multiplier
        combined = combined / max(combined.mean(), 1e-9)
        weights[n_hard:] = torch.from_numpy((soft_weight * combined).astype(np.float32))

    model = BattlePolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    if restart_period > 0:
        # SGDR (Loshchilov and Hutter, ICLR 2017): the rate jumps back to its maximum every T_0
        # epochs instead of decaying once. This exists to separate two things the single-cosine
        # budget comparison confounds, since a longer run differs from a shorter one in total
        # steps AND in how long it spends at a high rate. Restarts give the extra steps while
        # still ending each cycle annealed, so the arms differ in optimisation rather than shape.
        schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=restart_period, T_mult=1, eta_min=1e-5)
    else:
        # Armed to the run length on purpose. torch's CosineAnnealingLR is periodic with period
        # 2*T_max, so it climbs back to the full rate by epoch 2*T_max rather than resting at
        # eta_min; tying T_max to `epochs` is what keeps a run on the descending half. Passing a
        # larger epoch count against a fixed T_max would silently produce warm restarts instead.
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    hobs, hmasks, hactions = (torch.from_numpy(holdout_s.observations), torch.from_numpy(holdout_s.masks),
                              torch.from_numpy(holdout_s.actions))
    best = {"agreement": -1.0}
    # Per-epoch training diagnostics (owner requirement, 2026-08-08): loss decomposed into its
    # hard and soft terms before the sum, the live learning rate, and holdout agreement, kept in
    # the report and appended per epoch to a heartbeat the dashboard and the convergence report
    # can read. The first coverage-corpus verdict was drawn without these, which was the gap.
    history, snapshots = [], []
    beat_path = out + ".heartbeat.jsonl"
    pathlib.Path(beat_path).write_text("")  # truncate: see train_ppo, a rerun must not append
    for epoch in range(epochs):
        model.train()
        epoch_lr = schedule.get_last_lr()[0]
        perm = torch.randperm(len(actions))
        running_hard = running_soft = running_entropy = 0.0
        entropy_batches = 0
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
                running_hard += float(hard_term.detach())
            soft_mask = ~hard_mask
            if soft_mask.any():
                rows = batch[soft_mask]
                if soft_as == "distribution":
                    ce = -(dense[rows - n_hard] * log_probs[soft_mask]).sum(-1)
                else:
                    ce = torch.nn.functional.nll_loss(log_probs[soft_mask], actions[rows], reduction="none")
                soft_term = (ce * weights[rows]).sum()
                loss = loss + soft_term
                running_soft += float(soft_term.detach())
            if entropy_bonus > 0:
                # Maximum-entropy imitation: pay the student to stay uncertain where the corpus does
                # not force a choice. Guarded rather than multiplied by zero so an unset bonus leaves
                # the graph and the arithmetic exactly as they were, and arms stay comparable.
                #
                # Measured and rejected, 2026-08-11, on both regimes. Playing the network alone it
                # costs at every dose: -0.050 win rate and -0.080 strength margin at beta 0.15,
                # -0.066 and -0.105 at 0.40. Under root search it is worth +0.0000 win rate at a
                # standard error of 0.0446 over three seeds. The mechanism is nonetheless real and
                # replicates: search visit entropy roughly doubles, so a broad prior does widen
                # PUCT exactly as its exploration term predicts. Breadth just does not pay at this
                # budget, which is what coverage forcing also found (ADR 0008).
                probs = log_probs.exp()
                row_entropy = -(probs * log_probs.clamp_min(-30.0)).sum(-1)
                # Weighted like the cross-entropy above, so the bonus lands in the same proportion
                # on soft rows as the loss it offsets rather than being diluted by the corpus mix.
                loss = loss - entropy_bonus * (row_entropy * weights[batch]).sum()
                running_entropy += float(row_entropy.detach().mean())
                entropy_batches += 1
            loss = loss / len(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedule.step()
        model.eval()
        with torch.no_grad():
            agree = hits = 0
            nll_sum = entropy_sum = normalized_sum = 0.0
            for start in range(0, len(hactions), 4096):
                sl = slice(start, start + 4096)
                logits, _ = model(hobs[sl], hmasks[sl])
                log_p = torch.log_softmax(logits, dim=-1)
                agree += int((logits.argmax(-1) == hactions[sl]).sum())
                nll_sum += float(torch.nn.functional.nll_loss(log_p, hactions[sl], reduction="sum"))
                # Masked actions carry probability exactly zero, so clamping the log keeps the
                # 0 * -inf product at zero rather than NaN, whichever way the policy masks.
                probs = log_p.exp()
                entropy = -(probs * log_p.clamp_min(-30.0)).sum(-1)
                # Normalised against this state's own legal count, because a state offering six
                # moves and one offering ninety are not comparable in raw nats. Same convention as
                # search_agent_battery.py, so the two harnesses' entropy columns can be read together.
                legal = hmasks[sl].sum(-1).clamp_min(2).float()
                entropy_sum += float(entropy.sum())
                normalized_sum += float((entropy / legal.log()).sum())
                hits += len(hactions[sl])
        agreement = agree / hits
        row = {"epoch": epoch, "train_loss_hard": round(running_hard / len(actions), 5),
               "train_loss_soft": round(running_soft / len(actions), 5),
               "train_loss": round((running_hard + running_soft) / len(actions), 5),
               "holdout_agreement": round(agreement, 5), "lr": epoch_lr,
               # Agreement alone cannot referee a longer budget: the 2026-08-07 sharpness sweep
               # found the arm with the worst agreement was the only one that played better. Held
               # out loss separates fitting from overfitting where argmax agreement cannot, and the
               # entropy pair says whether the extra epochs bought skill or only confidence.
               "holdout_loss": round(nll_sum / hits, 5),
               "holdout_entropy": round(entropy_sum / hits, 5),
               "holdout_normalized_entropy": round(normalized_sum / hits, 5),
               "holdout_effective_actions": round(float(np.exp(entropy_sum / hits)), 4)}
        if entropy_batches:
            row["train_entropy"] = round(running_entropy / entropy_batches, 5)
        history.append(row)
        with open(beat_path, "a") as beat:
            beat.write(json.dumps(row) + "\n")
        if agreement > best["agreement"]:
            best = {"epoch": epoch, "agreement": agreement}
            torch.save({"state_dict": model.state_dict(), "encoding_version": ENCODING_VERSION}, out)
        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            # A snapshot at a fixed epoch rather than at an agreement peak, so play strength can be
            # read at several budgets from one run. Without these, comparing budgets means retraining
            # per budget and paying for the seed spread separately at each.
            snapshot = f"{out}.epoch{epoch + 1:03d}.pt"
            torch.save({"state_dict": model.state_dict(), "encoding_version": ENCODING_VERSION}, snapshot)
            snapshots.append({"epoch": epoch + 1, "path": snapshot, "agreement": round(agreement, 5)})
    return best | {"history": history, "snapshots": snapshots}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--soft-root", nargs="+", required=True)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--target", default="values", choices=("values", "visits"),
                        help="visits builds AlphaZero-style pi proportional to N^(1/lam)")
    parser.add_argument("--soft-weight", type=float, default=2.0)
    parser.add_argument("--min-visit-margin", type=float, default=0.0,
                        help="drop soft rows whose labeled action got fewer than this many extra playouts "
                             "than the prior's pick. A margin of one means the value tie-break decided on a "
                             "single playout, and those overrules measure +0.057 against +0.20 for wider "
                             "margins (label_value.py). Mass is renormalised over the survivors")
    parser.add_argument("--regret-weighted", action="store_true",
                        help="weight soft rows by rank-transformed measured regret at equal total mass")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="also save a snapshot every N epochs, beside the best-agreement checkpoint. "
                             "The default of 0 keeps the standing behaviour. Snapshots are what let a "
                             "budget be judged by play rather than by the agreement peak that selected it")
    parser.add_argument("--restart-period", type=int, default=0,
                        help="SGDR warm restarts with this T_0 instead of one cosine armed to the "
                             "budget. Separates 'more optimisation steps' from 'more time at a high "
                             "rate', which a single-cosine budget comparison confounds")
    parser.add_argument("--entropy-bonus", type=float, default=0.0,
                        help="add -beta*H(pi) to the loss, paying the student to stay uncertain. "
                             "Measured negative 2026-08-11 for the network alone and neutral under "
                             "search (+0.0000, SE 0.0446, three seeds). It does widen the search as "
                             "intended; breadth does not pay at this budget. Kept as a rejected arm")
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

    soft = train_arm(hard, soft_rows, "distribution", args.soft_weight, args.epochs, args.seed,
                     args.out, regret_weighted=args.regret_weighted, min_visit_margin=args.min_visit_margin,
                     checkpoint_every=args.checkpoint_every, entropy_bonus=args.entropy_bonus,
                     restart_period=args.restart_period)
    print(f"soft-target arm: best agreement {soft['agreement']:.4f} at epoch {soft['epoch']}", flush=True)
    hard_arm = train_arm(hard, soft_rows, "argmax", args.soft_weight, args.epochs, args.seed, args.hard_out,
                         checkpoint_every=args.checkpoint_every, entropy_bonus=args.entropy_bonus,
                     restart_period=args.restart_period)
    print(f"hard-label twin: best agreement {hard_arm['agreement']:.4f} at epoch {hard_arm['epoch']}", flush=True)

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(
            {"roots": args.roots, "soft_root": args.soft_root, "lam": args.lam,
             "soft_weight": args.soft_weight, "seed": args.seed, "target_entropy": entropy,
             "soft": soft, "hard_twin": hard_arm, "seconds": round(time.time() - started, 1)}, indent=2))
    print(f"total {round(time.time() - started)}s")


if __name__ == "__main__":
    main()
