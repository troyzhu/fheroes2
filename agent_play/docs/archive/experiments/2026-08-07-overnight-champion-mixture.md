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


## The search agent crosses the baseline

The first remaining lever paid immediately. Rollout-scored root search over the planes champion's prior, 32 simulations per decision, measured on the identical held-out pool and win definition as the baseline column: **0.963 plus or minus 0.027 against the built-in AI's 0.660**, nineteen of twenty matchups at or near sweep, four battlefield variants each (`search_agent_heldout.json`, harness vendored beside it). The agent that searches is not eleven points behind the rule-based baseline; it is thirty points past it.

The framing this fixes: task 45's goal is met in the agent regime and open in the weights-only regime. A policy plus one-ply-deep rollouts at about fifteen seconds per episode beats the planner decisively, which quantifies exactly what the supervised program has been unable to distill, and puts a number on the distillation gap itself: the same prior that searches to 0.963 plays 0.526 raw. Closing any fraction of that 0.44 into the weights is the remaining problem, and the levers that have not been tried on it are true reinforcement learning from this anchor under the strength-margin reward, and the value-on-search-returns line whose dataset the collector already records.

## The owner-objective block, and the day the instruments arrived

The afternoon rebuilt the reward as the owner intended it and ran it everywhere reward enters. `reward_margin="two_sided"` grades wins by own strength kept and losses by enemy strength destroyed, composing with the difficulty weighting; the plumbing carries it through the pool, the PPO harness, and the search collector, smoke-read at $-1.668$ for a difficulty-amplified favored-fight loss.

The distillation half is the block's result: labels chosen by search scoring the owner objective lift six of seven suites together, held-out $0.555 \pm 0.043$ against the champion mixture's $0.526 \pm 0.013$, both mirror chairs at once (attacker $0.329$ against $0.271$, defender $0.259$ against $0.199$), commander extremes $0.990$, with the ladder inside noise, the first coherent multi-suite move since generation one and no flagship paid. The objective's distinctive prediction, better losses, is unresolved: loss quality on held-out reads $0.652 \pm 0.020$ against the champion's $0.647 \pm 0.013$, indistinguishable at this power.

The reinforcement half became the instrument story. The un-warmed arm eroded (train $-0.033$, held $-0.071$), and the owner-requested per-term gradient norms, decomposed per module the same afternoon, showed why in the first reading: the anchor's never-trained value head put norm $11.9$ against the policy's $2.2$, three of it landing in the shared trunk. The prescribed value warmup, head-only updates before joint training, dropped iteration zero to value-trunk $1.3$ and flipped the arm to train $+0.028 \pm 0.011$ with held-out erosion halved to $-0.040 \pm 0.025$, the first positive training delta any PPO run here has posted from a strong anchor; held-out rescue is partial, so the anchoring question stays open rather than solved.

Softplus, the owner's activation request, was staged rather than defaulted after the agent gate flagged it, and the three-seed ablation vindicated the staging: agreement rises to $0.8723 \pm 0.003$ against ReLU's $0.8525 \pm 0.002$ and the Thunk ladder collapses to $0.615 \pm 0.038$ against $0.774 \pm 0.037$, the ability-arm pattern again, fidelity up and the flagship down, so ReLU stays the default and softplus stays a marked option.

The day also left the project permanently better instrumented: per-term per-module gradient norms in every PPO heartbeat, loss decompositions, supervised trainers emitting train loss and holdout agreement, the live dashboard rendering all of it beside the advantage floor, and the battery carrying win quality, loss quality, unconditional margin, the trained reward, and per-rung ladders, with real-map opening fights as a standing suite. Reports for every claim above are vendored beside this log.

## Deployment sampling, the owner's nucleus proposal

The evening's last measurement tests trimming the sampling tail at deployment, with the owner's domain refinement: confidence judged by entropy normalized against the uniform maximum over however many actions were legal, so a five-action state and a thirty-action state read on one scale. Six schemes around one checkpoint, ten held-out matchups and the two hard rungs: full sampling 0.675, greedy 0.750, nucleus at 0.5 mass 0.713, and the entropy-adaptive nucleus 0.738 with the best hard-rung line (1.00 at 850, 0.58 at 1000). Every trimming variant reads at or above full sampling everywhere, nothing pays anywhere, and all deltas sit within one standard error at this size, so the verdict is a consistent free-win signal short of adoption power. The adaptive form is the candidate because it keeps genuine forks stochastic while converging toward greedy exactly where the policy claims confidence; a larger-n run decides the deployment default (`sampling_policies.json`).

The owner then asked the grounding question, and the honest split is: nucleus itself is grounded (Holtzman et al. introduced it for language decoding on exactly the transferable mechanism, the softmax tail is where training signal was thinnest, and low-temperature evaluation of stochastic policies is standard reinforcement-learning practice), entropy-conditioned trimming is a real family in decoding (typical sampling, Mirostat), but the specific width schedule here is this project's heuristic with no theory behind it, and its validity rests on whether the policy's probabilities are calibrated at all. So the owner's actual proposal, normalized entropy as a diagnostic, came first: `fidelity_report.py` now reports entropy over the uniform maximum on the legal set, plus a ten-bin reliability table and expected calibration error of the top action's confidence against whether it was the teacher's move.

The first reading grounds the sampler after the fact: ECE 0.02 with the policy systematically underconfident in the middle bins (predicted 0.75, actual 0.85) and exact at the extremes, so trimming the tail discards mass the calibration says was overpriced, which is why every trimming variant gained or tied (`fidelity_calibration.json`).

## What "the teacher" means in the calibration table, and the chain made explicit

The owner asked whose moves the reliability table calibrates against, and the census answer is a mixture dominated by the rule-based planner: of the 224,811 decisions in the measured corpus, 97.5 percent carry the built-in planner's action as the label (195,644 passive recordings plus 23,463 DAgger relabelings) and 2.5 percent carry root-search's choice under the owner objective (5,704 across both chairs). So "the policy is calibrated" means calibrated against the same blended label distribution it distilled from, which is the right target for the imitation question and predominantly means the planner.

The reasoning chain the calibration serves, recorded because the owner walked it: the adaptive sampler needs two premises, that high confidence is trustworthy and that tail mass is overpriced; the reliability table certifies both in teacher-currency, exact at the extremes and underconfident in the middle; entropy and top-probability are two views of one confidence axis in this near-one-hot regime, with re-binning by normalized entropy the recorded tightening; and teacher-grounded calibration is structurally silent on the residual where the policy's tail beats the teacher, the walkthrough's winning rank-four action being the recorded example, so realized play, the sampler sweep, arbitrates that trade empirically. An outcome-grounded calibration, confidence against realized return, is the measurement that would close the last gap if the thread warrants the expense.

The owner's final refinement closes the thread's epistemics: the planner is deterministic, so its conditional is a delta and calibration against it is meaningful only at the argmax. The reliability table survives as classification calibration, epistemic uncertainty about which single action the deterministic function picks, and the ECE stands in that reading. Everything distribution-shaped beyond the argmax deflates: the ideal imitator of a delta has zero entropy, all tail mass is error mass by the target's own semantics, so "trimming never hurts" is near-tautological in teacher-currency and only the play sweep carried non-trivial content; the middle-bin underconfidence is sharpening headroom, not action stochasticity, of which the target has none. Graded calibration needs a graded target, and the corpus offers three: search's per-candidate rollout values (do the policy's probability ratios track search's value gaps), outcome-grounded confidence against realized wins, which is aleatoric through combat randomness, and the prior-anchored soft targets. Against the planner, only argmax-frequency questions have answers.