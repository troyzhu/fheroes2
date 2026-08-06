---
title: "The afternoon block: offline improvement arms and the planes trio, 2026-08-06"
type: experiment-log
updated: 2026-08-06
tags: [agent-env, archive, experiment, offline-rl, planes]
---

# The afternoon block: offline improvement arms and the planes trio, 2026-08-06

The owner set the frame in the morning conversation: keep pushing toward a policy that exceeds the built-in AI (the standing task), treat the value thread as education as much as engineering, expand the off-support survey with digested references, and build terrain observability. This log carries the measurements; [[../../rl/off-support-and-offline-improvement]] carries the survey and per-arm verdicts, [[../../rl/value-estimation-lab]] the value thread's full record, and [[../../rl/the-policy-network]] the architecture the planes work extends.

## The arms, in the order they ran

The advantage-weighted arm (AWR at $\beta = 1$, the 0.856 bestiary value supplying advantages) lost to its paired unweighted twin nearly everywhere, and the reason is structural: our demonstrator is deterministic, one action per state, so there is no within-state diversity for advantage weighting to select over and the weights degenerate to episode reweighting. Reports `awr_distill.json`, `battery_awr.json`.

The soft-target pilot (1,183 soft rows, $\lambda = 0.5$) read as a wash and exposed its own under-powering: near-one-hot targets and 0.4 percent of the corpus. Reports `soft_distill.json`, `battery_soft.json`.

The generation-scale soft experiment fixed both: 9,018 soft-labeled decisions over 86 kept fresh matchups from both sides (six shards, win-filtered, battlefield-varied), $\lambda = 0.33$ set to the median per-decision value spread. Against the argmax twin on identical data the soft arm leads seven of nine suites, held-out 0.515 to 0.477, commanders 0.990 to 0.969, with the twin keeping the ladder (0.812 to 0.760) and defender mirrors. Promising, not adopted; the multi-seed gate runs tonight. Reports `soft_distill_gen.json`, `battery_soft2.json`.

## The planes trio

`planes_v1` went from unbuilt to measured inside the block: the engine's obstacle layer (worker `--planes`, byte-identical off), the `encode_planes` rasterizer, the convolutional fusion arm on `BattlePolicy`, a planes-recorded 195,644-decision corpus (12,000 episodes, 29 seconds, mean 5.9 obstacle cells per state), and the three-arm ablation the capacity law demands: entity at 396,570 parameters, planes at 838,074, and a trunk-widened control at 789,690, one seed, one corpus.

| Measure | Entity | Planes | Width control |
|---|---|---|---|
| Cloning agreement | 0.847 | 0.898 | 0.880 |
| Held-out pool | 0.502 | 0.535 | 0.569 |
| Thunk ladder | 0.833 | 0.833 | 0.865 |
| Mirrors as attacker | 0.090 | 0.278 | 0.194 |
| Mirrors as defender | 0.194 | 0.167 | 0.229 |
| Stress commanders | 0.896 | 0.938 | 0.958 |

Two signals separate cleanly on the first seed. Most of the play-level gain is capacity: the width control leads held-out, the ladder, and commanders, which is exactly the confound the third arm was built to expose and why no two-arm version of this experiment would have been honest. And the board information itself shows in the two places it mechanically should: cloning agreement, where planes beat the width control by 1.9 points (the teacher's moves are more predictable when you can see what it walks around), and attacker-side mirrors, where symmetric armies cancel composition and position is the only edge, 0.278 against 0.194 against 0.090. Reports `planes_ablation.json`, `battery_planes.json`; seeds 1 and 2 of the trio and of the soft twins were still running as this log closed, and their verdicts belong to whichever log records them.
