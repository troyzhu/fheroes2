---
title: "The night block: search-teaching generations over the fresh distribution, 2026-08-06"
type: experiment-log
updated: 2026-08-06
tags: [agent-env, archive, experiment, search-teacher, generations]
---

# The night block: search-teaching generations over the fresh distribution, 2026-08-06

The owner left an eight-hour autonomous block with one goal, a further-improved policy, and all standing requirements in force. The plan follows the evening's evidence: search-as-teacher is the one operator with measured headroom, its pool-concentrated first application regressed off-pool, so the generations here collect over the fresh sampled distribution instead, win-filtered per the credit measurement, from both sides per the swap gap, distilled one step per generation with the battery as the gate between generations. Everything headless and silent; reports vendored under `files/2026-08-06-run-reports/`.

Two capabilities were added before generation one finished collecting, both committed with the harnesses: the worker's `--seed-offset` lets a search side-environment replay the exact battlefield variant the live environment plays, which makes battlefield-varied search labels collectible (`--vary-battlefields`), and the battery gained side coverage, the held-out pool from the defender's chair and mirror armies from both chairs.

## Generation one, collected

Six shards, three per side, thirty fresh value-budget matchups each under a virgin sample seed, eight searched episodes per matchup at 32 simulations, keep only matchups search wins at least half of. Fifty minutes of wall time: 83 matchups kept and 97 dropped, 664 episodes, 9,219 labels, with search winning essentially every episode it kept. The drop rate matches the battery's fresh-sampled reading that roughly half the raw distribution is winnable by nobody, and those matchups now contribute nothing rather than poison.

## Generation one, distilled and judged

Corpus: the teacher demonstrations, the unique targeted set, the round-one relabelings at double weight, and the generation-one labels at double weight, 331,000 samples in effect. The nine-suite battery, paired within one run so the columns share evaluation noise:

| Suite | share2 | Generation 1 |
|---|---|---|
| Fresh sampled | 0.391 | 0.389 |
| Held-out pool | 0.548 | 0.512 |
| Stress hordes | 0.192 | 0.158 |
| Stress wide-only | 0.417 | 0.472 |
| Stress commanders | 0.979 | 0.948 |
| Thunk ladder | 0.844 | 0.948 |
| Held-out as defender | 0.273 | 0.254 |
| Mirrors as attacker | 0.146 | 0.236 |
| Mirrors as defender | 0.382 | 0.306 |

The headline is the ladder: 1.000 / 1.000 / 1.000 / 0.792, the project's best on the fight nothing ever trains on, with the 850 rung perfect and the full fight now won four times in five, from supervised distillation alone. Wide-only and attacker-side mirrors also rose, which is what fresh-distribution labels were for. The costs are a marginal held-out dip, one to two standard errors at this suite size, and small declines on hordes, commander extremes, and defender mirrors. The pre-registered rule, proceed on held-out or out-of-distribution improvement, is met on the out-of-distribution side decisively, so generation two collects from this policy, on a new virgin seed, with each matchup on its own battlefield variant now that the offset capability exists.
