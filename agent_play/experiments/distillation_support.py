#!/usr/bin/env python3
"""Where do search's labels sit under the policy that has to learn them?

The owner's question, and it needs the loss form stated before it can be answered. `soft_distill.py`
trains a mixture, not one objective. The hard rows carry `nll_loss` against a single recorded action,
which is plain log loss on an argmax. The soft rows carry `-(pi_bar * log pi_theta).sum()`, which is
cross-entropy against a distribution and therefore KL up to the target's own entropy, a constant in
the gradient. On the standing recipe the hard rows outnumber the soft ones 242,570 to 5,143 and the
soft rows carry weight 2.0, so the soft term is about four percent of the loss mass. The student is
overwhelmingly being told to copy one action per state.

That makes the support question the right one. If the action a label names already sits at the top
of the policy's distribution, the gradient teaches nothing, and if it sits far out in the tail the
label is asking for a move the policy has almost no probability on. Both are ways for a corpus to
be large and uninformative, and they call for opposite fixes.

So this reports, per decision, where the labeled action sits under the prior that collected it:
its probability, its rank, and whether it clears a support threshold at all. It splits confirming
decisions, where search agreed with the prior's own pick, from informative ones where search
overruled it, because only the second kind can move the policy anywhere.

For the soft rows it also reports the divergence the target actually asks for, both directions, and
the mass the target places on the labeled action. A soft target that is nearly the prior is asking
for nothing however many rows it has.

With a checkpoint given, the same quantities are recomputed under that trained student, which
answers the second half: of what the labels asked for, how much did the student absorb?

Usage:
    ./distillation_support.py --soft-root DIR [DIR ...] [--checkpoint C.pt] [--lam 0.5]
                              [--hard-rows 242570] [--soft-weight 2.0] [--report R.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from fheroes2_agent.encoding import ACTION_SPACE_SIZE, encode_mask, encode_observation  # noqa: E402

#: Probability below which an action is treated as effectively off-support for a policy that has to
#: reach it by gradient. One percent is the same threshold the battery's `support_at_1pct` uses, so
#: the two readings are comparable.
SUPPORT = 0.01


def percentiles(values, qs=(5, 25, 50, 75, 95)) -> dict:
    if len(values) == 0:
        return {}
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--soft-root", nargs="+", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="also recompute every quantity under this trained student")
    parser.add_argument("--lam", type=float, default=0.5, help="the pi_bar temperature the corpus is distilled at")
    parser.add_argument("--hard-rows", type=int, default=0, help="hard-row count, for the loss-mass split")
    parser.add_argument("--soft-weight", type=float, default=2.0)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    rows = []
    for path in sorted(q for root in args.soft_root for q in pathlib.Path(root).rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("record") != "decision" or "search_values" not in record:
                continue
            prior = {int(a): float(p) for a, p in record["prior"].items()}
            if not prior:
                continue
            labeled = int(record["teacher_action"])
            top = max(prior, key=prior.get)
            # pi_bar, the prior-anchored target the soft arm actually trains against.
            logits = {int(a): np.log(max(record["prior"][a], 1e-9)) + record["search_values"][a] / args.lam
                      for a in record["search_values"]}
            peak = max(logits.values())
            bar = {a: np.exp(l - peak) for a, l in logits.items()}
            total = sum(bar.values())
            bar = {a: v / total for a, v in bar.items()}
            shared = [a for a in bar if a in prior]
            kl_bar_prior = float(sum(bar[a] * np.log(bar[a] / max(prior[a], 1e-12))
                                     for a in shared if bar[a] > 0))
            kl_prior_bar = float(sum(prior[a] * np.log(prior[a] / max(bar[a], 1e-12))
                                     for a in shared if prior[a] > 0))
            order = sorted(prior, key=prior.get, reverse=True)
            rows.append({
                "labeled": labeled,
                "informative": labeled != top,
                "prior_on_label": prior.get(labeled, 0.0),
                "prior_rank": order.index(labeled) + 1 if labeled in prior else -1,
                "legal": len(prior),
                "bar_on_label": float(bar.get(labeled, 0.0)),
                "kl_bar_prior": kl_bar_prior,
                "kl_prior_bar": kl_prior_bar,
                "observation": record["observation"] if args.checkpoint else None,
                "legal_actions": record["legal_actions"] if args.checkpoint else None,
            })

    if not rows:
        print("no searched decisions found")
        return
    inf = [r for r in rows if r["informative"]]
    con = [r for r in rows if not r["informative"]]
    total_soft_mass = len(rows) * args.soft_weight
    print(f"\n{len(rows)} searched decisions, {len(inf)} informative "
          f"({len(inf)/len(rows):.1%}), {len(con)} confirming the prior's own pick")
    if args.hard_rows:
        share = total_soft_mass / (args.hard_rows + total_soft_mass)
        print(f"loss mass: {args.hard_rows} hard rows at weight 1.0 against {len(rows)} soft rows at "
              f"weight {args.soft_weight}, so the soft term is {share:.1%} of the total.")
        print("The hard term is nll_loss against one recorded action, i.e. log loss on an argmax.")
        print("The soft term is cross-entropy against pi_bar, i.e. KL up to the target's own entropy.\n")

    def block(name, subset):
        if not subset:
            print(f"{name}: none")
            return {}
        p = np.array([r["prior_on_label"] for r in subset])
        rank = np.array([r["prior_rank"] for r in subset])
        legal = np.array([r["legal"] for r in subset])
        off = float((p < SUPPORT).mean())
        print(f"{name} ({len(subset)} decisions, {legal.mean():.1f} legal moves on average)")
        print(f"   prior probability on the labeled action   mean {p.mean():.4f}   "
              + "  ".join(f"{k} {v:.4f}" for k, v in percentiles(p).items()))
        print(f"   its rank under the prior                  mean {rank.mean():.2f}   "
              + "  ".join(f"{k} {v:.0f}" for k, v in percentiles(rank).items()))
        print(f"   share below the {SUPPORT:.0%} support threshold      {off:.1%}")
        return {"n": len(subset), "prior_on_label_mean": float(p.mean()),
                "prior_on_label_percentiles": percentiles(p), "rank_mean": float(rank.mean()),
                "rank_percentiles": percentiles(rank), "below_support": off}

    stats = {"informative": block("INFORMATIVE, search overruled the prior", inf),
             "confirming": block("CONFIRMING, search agreed with the prior", con)}

    kb = np.array([r["kl_bar_prior"] for r in rows])
    kp = np.array([r["kl_prior_bar"] for r in rows])
    bl = np.array([r["bar_on_label"] for r in rows])
    print(f"\nWhat the soft target asks for, over all {len(rows)} rows")
    print(f"   KL(pi_bar || prior)   mean {kb.mean():.4f}   " + "  ".join(f"{k} {v:.4f}" for k, v in percentiles(kb).items()))
    print(f"   KL(prior || pi_bar)   mean {kp.mean():.4f}   " + "  ".join(f"{k} {v:.4f}" for k, v in percentiles(kp).items()))
    print(f"   pi_bar mass on label  mean {bl.mean():.4f}   " + "  ".join(f"{k} {v:.4f}" for k, v in percentiles(bl).items()))
    stats["target"] = {"kl_bar_prior_mean": float(kb.mean()), "kl_bar_prior_percentiles": percentiles(kb),
                       "kl_prior_bar_mean": float(kp.mean()), "bar_on_label_mean": float(bl.mean())}

    if args.checkpoint:
        import torch
        from fheroes2_agent.policy import load_policy
        model = load_policy(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
        model.eval()
        obs = torch.from_numpy(np.stack([encode_observation(r["observation"]) for r in rows]))
        masks = torch.from_numpy(np.stack([encode_mask(r["legal_actions"]) for r in rows]))
        with torch.no_grad():
            probs = torch.softmax(model(obs, masks)[0], dim=-1).numpy()
        student = np.array([probs[i, r["labeled"]] for i, r in enumerate(rows)])
        srank = np.array([int((probs[i] > probs[i, r["labeled"]]).sum()) + 1 for i, r in enumerate(rows)])
        imask = np.array([r["informative"] for r in rows])
        print(f"\nUnder the trained student, {pathlib.Path(args.checkpoint).name}")
        for name, sel in (("informative rows", imask), ("confirming rows", ~imask)):
            if sel.sum() == 0:
                continue
            print(f"   {name}: probability on the labeled action mean {student[sel].mean():.4f}, "
                  f"rank mean {srank[sel].mean():.2f}, below support {float((student[sel] < SUPPORT).mean()):.1%}")
        prior_all = np.array([r["prior_on_label"] for r in rows])
        print(f"   absorbed on the informative rows: prior {prior_all[imask].mean():.4f} "
              f"-> student {student[imask].mean():.4f}  ({student[imask].mean() - prior_all[imask].mean():+.4f})")
        stats["student"] = {
            "checkpoint": pathlib.Path(args.checkpoint).name,
            "informative_prob_mean": float(student[imask].mean()) if imask.sum() else None,
            "informative_rank_mean": float(srank[imask].mean()) if imask.sum() else None,
            "informative_below_support": float((student[imask] < SUPPORT).mean()) if imask.sum() else None,
            "confirming_prob_mean": float(student[~imask].mean()) if (~imask).sum() else None}

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(stats, indent=1))
    print("\nDISTILLATION SUPPORT COMPLETE")


if __name__ == "__main__":
    main()
