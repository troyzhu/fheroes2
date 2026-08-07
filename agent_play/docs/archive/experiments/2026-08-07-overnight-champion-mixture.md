---
title: "The overnight champion mixture: planes meet the proven recipe, 2026-08-07"
type: experiment-log
updated: 2026-08-07
tags: [agent-env, archive, experiment, planes, champion-mixture]
---

# The overnight champion mixture: planes meet the proven recipe, 2026-08-07

The owner's standing directive is a policy that exceeds the rule-based baseline, and the evening's two confirmed facts pointed at one arm: the planes fidelity signal replicates, and the champion data recipe is the strongest supervised mixture the project has. Overnight the whole recipe was re-collected with the board visible, at trivial cost since the engine records twelve thousand episodes in half a minute: the diverse teacher corpus (195,644 decisions), a fresh DAgger relabel corpus on the champion's states (23,463 planner labels), and six shards of fresh both-sides search labels at new sample seeds, all carrying obstacle layers. The three-arm trio then trained on the full mixture at three seeds each, entity, planes, and the width control, batteried behind.

## The verdict, against the baseline that matters

| Suite | Built-in AI | Entity | Planes | Width control |
|---|---|---|---|---|
| Held-out pool | 0.660 | $0.480 \pm 0.009$ | $0.526 \pm 0.013$ | $0.503 \pm 0.010$ |
| Thunk ladder | 0.969 | $0.788 \pm 0.012$ | $0.844 \pm 0.052$ | $0.743 \pm 0.059$ |
| Held-out as defender | 0.338 | $0.307 \pm 0.001$ | $0.256 \pm 0.034$ | $0.281 \pm 0.007$ |
| Mirrors as attacker | 0.361 | $0.248 \pm 0.031$ | $0.271 \pm 0.024$ | $0.220 \pm 0.049$ |
| Mirrors as defender | 0.639 | $0.278 \pm 0.025$ | $0.199 \pm 0.022$ | $0.312 \pm 0.018$ |
| Stress commanders | 0.958 | $0.955 \pm 0.012$ | $0.976 \pm 0.016$ | $0.965 \pm 0.006$ |
| Fresh sampled | 0.446 | $0.365 \pm 0.011$ | $0.388 \pm 0.021$ | $0.362 \pm 0.015$ |

Cloning agreement: planes $0.9085 \pm 0.005$, the highest fidelity this project has recorded, against the width control's $0.884 \pm 0.003$ and entity's $0.8525 \pm 0.002$; the board's contribution replicates on the mixture exactly as it did on the flat corpus.

## What it means

The planes arm on the champion mixture is the project's most stable strong policy: held-out $0.526 \pm 0.013$ at a third of the seed spread earlier recipes showed, the best ladder mean of the trio, commander extremes above the baseline, and the best fidelity ever. And it does not cross the rule-based baseline's main columns. Held-out stands 0.13 below the AI, the ladder 0.13 below, the defender chairs further. One suite is genuinely past the baseline, commander extremes, where planes lead 0.976 to 0.958.

The reading is the one every measurement this week has pointed at: architecture closes fidelity gaps, and fidelity converges to the teacher, not past it. The remaining 0.13 is not an architecture gap but an improvement-operator gap, and the only operator measured above the planner is search. The next rung is therefore expert iteration on top of the planes policy: search with a planes-equipped prior whose rollouts also see the board, collected fresh, distilled back, the round discipline the literature and the night block both fixed. That round launched as this log closed, from the seed-zero planes checkpoint (ladder 0.896, the trio's best single roll).


## The expert-iteration round, and the plateau's second architecture

The round from the planes champion, six fresh both-sides shards whose search rollouts also saw the board, distilled onto the mixture at three seeds, moved nothing: held-out $0.515 \pm 0.054$ against the mixture's $0.526 \pm 0.013$, the ladder $0.795 \pm 0.102$ against $0.844 \pm 0.052$, defender chair up $0.037$, everything inside noise, agreement steady at $0.907$. One round of search teaching built the champion on the flat architecture; the second round stalls on the planes architecture exactly as it stalled on the flat one, so the plateau is now a property of the supervised program rather than of any network.

The launch itself cost a night: the collector's CLI loaded checkpoints with the pre-`load_policy` pattern and crashed on the planes state dict seconds in, which the library-path smoke had not exercised. Sixteen scripts were swept to `load_policy`, the CLI path is smoke-tested explicitly now, and both lessons are in the experiments conventions.

Where this leaves the scoreboard against the rule-based baseline: commander extremes are past it ($0.976$ against $0.958$, replicated), and held-out, the ladder, and the defender chairs are not, with the held-out gap steady at about $0.13$ through every supervised lever tried on two architectures: more labels, deeper labels, soft labels, reweighted labels, ability features, planes, width, and a second generation. The levers that remain are different in kind: the search agent itself measured on the baseline's columns (running as this closes), pooling, true reinforcement learning from the strongest anchor under the strength-margin reward, and a value trained on search returns.
