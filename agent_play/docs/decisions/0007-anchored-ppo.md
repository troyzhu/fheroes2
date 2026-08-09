---
title: "ADR 0007 — Anchored PPO, a KL leash to the frozen supervised checkpoint"
type: adr
status: accepted
updated: 2026-08-08
related_concepts: ["[[0005-training-and-reward]]", "[[../rl/rl-methods]]", "[[../rl/training-design]]", "[[../archive/experiments/2026-08-08-selfplay-round2-and-trust-region]]"]
tags: [adr, agent-env, training, reinforcement]
---

# ADR 0007 — Anchored PPO, a KL leash to the frozen supervised checkpoint

- Status: accepted for the reinforcement stage, on one budget and three seeds per arm, with the beta value open above 0.5
- Implementation: built, `anchor_kl_coef` in `python/fheroes2_agent/train_ppo.py`
- Evidence: [[../archive/experiments/2026-08-08-selfplay-round2-and-trust-region]], rounds three and four
- Supersedes: the recommendation in [[../rl/rlhf-transfer]] to add a reference leash only if anchor degradation appeared, which it did

## The sub-problem

Every reinforcement run from a strong supervised anchor had eroded that anchor's out-of-distribution play while improving on the distribution it trained on. The 2026-08-08 wide round removed the last available explanation from the data side: at 200 generator-sampled matchups instead of twelve the specialization pathology disappeared and the erosion did not, which relocated the problem from what the policy trains on to how it is optimized.

The clip and the divergence gates cannot address it, and the reason is stated in [[../rl/rl-methods]] rather than discovered here. Both constrain the step against the previous iterate, and the previous iterate moves every iteration, so per-step divergences do not telescope: a policy can walk arbitrarily far from where it started while every individual step stays inside the trust region. The 2026-08-08 rematch confirmed the practical half of that, three provably different gates producing indistinguishable outcomes.

## The decision

Add a penalty on the divergence from a reference policy frozen before the first update, the anchored form the reinforcement-from-human-feedback lineage uses against a supervised checkpoint, at coefficient $\beta = 0.5$ as the standing self-play recipe. The reference defaults to the loaded checkpoint itself. The ratio clip stays as the step constraint; the two are complements, one governing the step and the other the destination.

$$L = L^{\text{clip}} + c_v L^{\text{value}} - c_H H + \beta \, D_{\mathrm{KL}}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)$$

The divergence is the forward KL over the legal set, computed from the same masked forward the surrogate uses, so illegal actions carry matching fill values and contribute nothing.

## The evidence

Three arms at 1000 iterations on the wide distribution, three seeds each, judged by the full battery, the symmetry gauge, the convergence report and forty-episode duels. The leash costs nothing where the policy trains: the training reward and rate tails are indistinguishable across $\beta$ of 0, 0.1 and 0.5. What it buys is retention. The Thunk ladder returns to anchor level at $\beta = 0.5$, rungs 1.00/1.00/0.94/0.64 against the anchor's 0.92/0.58 and the unleashed arm's collapse to 0.14, and under the trained objective the reward columns match the anchor on held-out and on the ladder. The leash tension itself converges, settling at KL 0.11 from the anchor at $\beta = 0.5$ and 0.40 at $\beta = 0.1$.

## What this does not claim

It is not a crossing. Held-out play remains below the supervised anchor and well below the built-in AI, and no reinforcement configuration in this project's record has yet produced a policy better than the anchor it started from; the leash makes reinforcement non-destructive, not productive. The value 0.5 is chosen over 0.1 on ladder and commander retention rather than on held-out, where the two are inside the measured noise band. Higher coefficients are untested, and the reference could be a moving average or a best-so-far checkpoint rather than the start.

## Costs

One extra forward pass through a frozen copy of the network per minibatch, and one more term in the per-term gradient decomposition. The coefficient and the measured divergence are stamped into every heartbeat row as `anchor_kl_coef` and `kl_to_anchor`, and the coefficient rides in the run report, so a leashed run is identifiable from its artifacts alone.

<!-- verify
grep    python/fheroes2_agent/train_ppo.py :: anchor_kl_coef
grep    python/fheroes2_agent/train_ppo.py :: kl_to_anchor
exists  agent_play/docs/archive/experiments/files/2026-08-08-run-reports/battery_round4.json
exists  agent_play/docs/archive/experiments/files/2026-08-08-run-reports/convergence_round4.json
-->
