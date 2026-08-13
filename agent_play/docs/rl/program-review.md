---
title: "The program review: every approach, its verdict, and what remains"
type: review
updated: 2026-08-12
related_concepts: ["[[training-design]]", "[[rl-methods]]", "[[value-estimation-lab]]", "[[off-support-and-offline-improvement]]", "[[transfer-and-llm-policies]]", "[[../decisions/0007-anchored-ppo]]"]
tags: [agent-env, rl, review, roadmap]
---

# The program review: every approach, its verdict, and what remains

This page states where the program stands today, one verdict per approach, each claim citing the page or archive log that carries its evidence. It is written for a reader arriving fresh: current state first, history behind links. The chronology of how these conclusions were reached, including every retraction, lives in the dated logs under `../archive/experiments/`. The standing goal is task 45, a policy that exceeds the built-in rule-based AI, judged by the full apparatus rather than any single suite.

## Table of contents

- [[#Decisions awaiting the owner]]
- [[#The state of play]] and [[#The scoreboard]]
- [[#The distillation gap, and its mechanism]], the program's central number explained
- Verdict sections: [[#Imitation]], [[#Search]], [[#Distilling search]], [[#Value estimation]], [[#Reinforcement from strong anchors]], [[#Self-play]], [[#Reward, sampling, and the instruments]]
- [[#What still shows promise, in order]] and [[#Closed lines]]
- [[#Scope limits on every number above]]
- [[#The remaining experiments, concretely]]

## Decisions awaiting the owner

Two calls are parked here rather than taken, because ADR 0005 reserves them.

- The `contested` reward, the plain strength difference that removes the stall exploit, is built, tested and opt-in under [[reward-design]]; adopting it for training is the open call, and it gates training use of `flying_v1`.
- Goal 45's next lever: the budget lever is open again at a wall-clock price, and the covariate-shift finding points at DAgger rounds instead; which to fund first is a direction choice, not a measurement.

## The state of play

The agent regime, the network wrapped in root search, clears the built-in AI on four of the five suites that can separate two players, and loses the fifth. On the three-seed `final_sweep`, means over seeds: held-out 0.700 against the engine's 0.660, held-out defending 0.398 against 0.338, the mirror chairs 0.542 and 0.688 against 0.361 and 0.639, fresh samples 0.391 against 0.446. The unsearched network reads 0.512 on held-out (single-seed `honest_sweep`; the fullmetrics paired arm reads 0.569), so search is the entire margin and the weights alone remain well short either way.

Search's budget pays monotonically as far as it has been measured: PUCT reads 0.597, 0.620, 0.671 and 0.704 at 32, 64, 128 and 256 playouts on the mirror suites with honest dice, strength margin rising from $+0.095$ to $+0.168$, decelerating like a log-compute law rather than a plateau. A simple-regret allocator was built beside PUCT, showed a one-experiment advantage, and did not survive its own budget sweep; PUCT stays the default on evidence ([[../archive/experiments/2026-08-11-distillation-budget-and-checkpoint-selection|the budget log]]). The 512-playout rung is in flight.

Two qualifications belong wherever these numbers are quoted. Against spread across seeds, which asks whether the agent is better on this fixed benchmark, four suites separate; against spread across matchups, which asks whether it would beat the engine on fresh matchups drawn the same way, none do, the per-matchup standard error running 0.066 to 0.135. The benchmark is passed, generalisation is not shown, and the suite that most resembles a fresh draw is the one the agent loses.

The other four suites cannot discriminate and are excluded from that count: real maps sit 22 of 24 saturated, hordes 4 of 5, commanders 3 of 4, the Thunk ladder 2 of 4. The ladder still reads by rung, search carrying 0.96 and 0.92 on the two hard rungs against the unsearched 0.67 and 0.46. [[../decisions/0008-search-configuration]] records what search is configured to do and why figures predating 2026-08-10 are on a different footing.

## The scoreboard

<!-- verify
# Invalidators for the scoreboard and the in-flight sentence below.
exists  agent_play/docs/archive/experiments/files/2026-08-07-run-reports/battery_rl_trio.json
exists  agent_play/docs/archive/experiments/files/2026-08-07-run-reports/convergence_long_runs.json
exists  agent_play/experiments/trust_region_rematch.py
grep    python/fheroes2_agent/train_ppo.py :: trust_region
grep    python/fheroes2_agent/selfplay.py :: OpponentPool
grep    python/fheroes2_agent/train_ppo.py :: anchor_kl_coef
exists  agent_play/docs/archive/experiments/files/2026-08-08-run-reports/battery_round4.json
exists  agent_play/docs/archive/experiments/files/2026-08-08-run-reports/deviation_probe.json
exists  agent_play/docs/decisions/0007-anchored-ppo.md
exists  agent_play/experiments/deviation_probe.py
-->

| Suite | Built-in AI | `policy_gen1.pt` | Leashed self-play | Regret-band distillation | Standing |
|---|---|---|---|---|---|
| Held-out pool | 0.660 / $+0.87$ | 0.525 / $+0.59$ | 0.499 / $+0.57$ | **0.571 / $+0.74$** | Short by 0.089 |
| Thunk ladder | 0.969 / $+1.63$ | 0.875 / $+1.47$ | 0.892 / $+1.48$ | 0.708 / $+1.03$ | Short by 0.08 |
| Held-out as defender | 0.338 / $+0.13$ | 0.271 / $-0.06$ | 0.258 / $-0.10$ | 0.283 / $-0.04$ | Short by 0.08 |
| Mirrors as attacker | 0.361 / $+0.25$ | 0.194 / $-0.11$ | 0.162 / $-0.15$ | 0.167 / $-0.14$ | Short by 0.17 |
| Mirrors as defender | 0.639 / $+0.75$ | 0.250 / $+0.04$ | 0.324 / $+0.17$ | 0.375 / $+0.25$ | Short by 0.32, the largest gap left |
| Commanders | 0.958 / $+1.72$ | 0.958 / $+1.78$ | 0.976 / $+1.79$ | 0.917 / $+1.68$ | At par or ahead |
| Hordes | 0.192 / $-0.33$ | 0.175 / $-0.36$ | 0.167 / $-0.37$ | -- | At par |
| Fresh sampled | 0.446 / $+0.31$ | 0.372 / $+0.13$ | 0.372 / $+0.14$ | 0.417 / $+0.23$ | Short by 0.07 |
| Real maps | 0.568 / $+0.66$ | 0.564 / $+0.64$ | 0.564 / $+0.64$ | 0.562 / $+0.64$ | At par |

Cells are win rate over the trained two-sided reward, weights-only policies against the engine on one scale, quality columns included so a rate cannot hide how it was earned. Three suites sit at par or ahead, so the standing goal reduces to the held-out pool, the two mirror chairs and fresh samples.

The searching agent sits outside the table because it is a different kind of measurement, priced separately: the honest ladder reads 0.594, 0.700, 0.750 and 0.744 at four through thirty-two playouts, 1.6 to 12.7 seconds an episode, about 49 Elo per budget doubling, inside the published 150-to-215-per-decade range for frozen-snapshot search scaling. The 0.750 at sixteen is a single unseeded run; the seeded mean is 0.700 and five identical-configuration repeats span 0.656 to 0.725, so 0.700 is the number to quote. An earlier 0.963 was withdrawn entirely ([[../decisions/0008-search-configuration]]).

Two cautions ride with the table. The mirror suites are the same six symmetric matchups from either chair and the engine's own numbers sum to 1.000, so 0.361 and 0.639 are the game's equilibrium split rather than independent bars. And the frozen anchor re-evaluates between 0.498 and 0.533 across batteries, so a difference under about 0.03 on one suite is not a result.

## The distillation gap, and its mechanism

The same prior that searches to 0.700 on held-out plays 0.512 raw, so about 0.19 of win rate sits in the search process. That gap is the program's central number, it has resisted every transfer attempt, and as of 2026-08-11 it has a mechanism rather than only a size.

The training loss barely asks for what search knows. About 95 percent of the training mass imitates the engine, 242,570 hard rows at weight one against 5,143 search-taught rows at weight two before the 80/20 split. Of the search-taught rows, 88 percent confirm the move the policy already ranks first and carry no gradient. The informative remainder names actions at a median prior probability of 0.0002, ranked eighth of some thirty-three legal moves, 63 percent below a one percent support threshold; the student moves 0.0166 of probability onto them while its rank on them drifts slightly worse. Closing the gap needs the mass ratio, the redundancy and the off-support reach addressed together, which is why regret weighting, which attacks only the redundancy, was still the largest paired effect on record at $+0.063$.

Training longer cannot close it. Extended budgets raise teacher agreement 0.851 to 0.914 while play falls 0.314 to 0.284; held-out loss improves while play falls; SGDR restarts buy the best offline numbers of twelve arms with the worst play. Both offline selectors point away from play, agreement correlating $-0.389$ with win rate across the design, and since the holdout is teacher-visited states the signature is covariate shift, the failure DAgger exists for. Twenty-five epochs is the measured optimum in both directions. By the three-seed standard, $t$ over two degrees of freedom needing 4.30, these read as consistently signed favoured evidence rather than established effects; the full account with every table is [[../archive/experiments/2026-08-11-distillation-budget-and-checkpoint-selection|the budget log]].

Where the deficit lives is also measured. Within an outcome the policy fights as well as the engine, wins keeping as much strength (0.47 against 0.45) and losses destroying as much (0.63 against 0.63); the deficit is in how often the outcome falls the right way. The deviation probe places the fixable part where no corpus has collected: search overrules the prior 0.145 of the time on matchups the policy loses against 0.052 where it wins, each overrule there worth $+0.787$ against $+0.152$ ([[../archive/experiments/2026-08-08-audit-and-the-deviation-finding]]).

## Imitation

Verdict: closed as a line of improvement; imitation converges to the teacher, not past it. Cloning reaches agreement $0.9085 \pm 0.005$ on the champion mixture, every architectural lever raised fidelity without raising play past the teacher, and the held-out gap held near 0.13 through all of them ([[the-policy-network]], [[../archive/experiments/2026-08-07-overnight-champion-mixture]]). The recurring trade is fidelity up, flagship down, which is why graded suites are never collapsed to one number. DAgger's first round behaved as the literature predicts, $+0.094 \pm 0.036$ on the pool with the planner-probe seam proving out. The one honest softener is the eight-epoch cut, and a fuller metric block later showed even that does not survive ([[../archive/experiments/2026-08-11-distillation-budget-and-checkpoint-selection|budget log]]).

## Search

Verdict: alive and load-bearing, the only operator measured above the teacher. Root PUCT over the cloned prior with rollout-scored playouts reads 0.700 held-out over three seeded runs against the engine's 0.660, unsearched prior at 0.512, about six seconds an episode at sixteen playouts. Rollout scoring is the AlphaGo-era scheme, and the substitution AlphaZero made, a learned leaf value, is exactly what our measurements block so far ([[value-estimation-lab]]). The tabula-rasa pilot failed honestly at this budget, search over a random prior winning at most 0.208 of its own episodes, so the imitation start is a compute convenience that remains right here ([[../archive/experiments/2026-08-06-night-block-search-generations]], [[../research/prior-art]]).

## Distilling search

Verdict: built, measured, currently buying a small twice-replicated gain. Support-complete corpora give value-derived soft targets $+0.03$ to $+0.04$ held-out with the reward columns agreeing; visit-derived targets stay null at both scales; the broader pilot pattern did not survive its scale test ([[../archive/experiments/2026-08-08-selfplay-round2-and-trust-region]]). The soft-target functional form is not the blocker: the shipped prior-anchored target is frozen near one-hot at every temperature because the prior bounds it, and the corpus-side diagnosis above says why gentle targets cannot reach the labels that matter.

## Value estimation

Verdict: closed as built, reopened only by coverage-forced data. Nine configurations produced one law, support is destiny at every granularity: a state value at 0.8565 explained variance cannot rank moves at a leaf; the behavior Q collapses from 0.853 to 0.263 when reranking actions the data never took; the rollout-trained value fits branch returns at 0.888 while agreeing with search's argmax 0.105 of the time ([[value-estimation-lab]], [[off-support-and-offline-improvement]]). The next value experiment worth running is the same fits on candidate-complete rollout data, and not before.

## Reinforcement from strong anchors

Verdict: open as a retention mechanism, closed as a climbing one. PPO from cloned anchors erodes more than it earns at every budget tried; the gradient-norm instrumentation found the critic slamming the shared trunk, fixed by head-only value warmup; the trust-region rematch measured DPPO's divergence gate provably different and outcome-indistinguishable from the clip. The KL leash to the frozen anchor at $\beta = 0.5$ is the first configuration that trains without eroding, returning the ladder to anchor level at zero training-distribution cost; it retains rather than climbs, and no reinforcement run here has beaten its own supervised anchor ([[../decisions/0007-anchored-ppo]], [[../archive/experiments/2026-08-07-overnight-champion-mixture]]).

## Self-play

Verdict: alive on a non-destructive base, current phase. An `OpponentPool` over frozen checkpoints with the engine as anchor, the league-lite shape the population literature argues for. Four rounds measured: mastery with erosion, then breadth removing the specialization pathology without the erosion, then the leash removing the erosion. All trained the attacker's chair, which the scoreboard says is the wrong one to train alone; `learner_side="alternate"` exists and its first round is the current measurement. Judged only by full battery, symmetry gauge, convergence verdict and duel counts that clear the measured noise ([[../archive/experiments/2026-08-08-selfplay-round2-and-trust-region]]).

## Reward, sampling, and the instruments

The training reward is ADR 0005's two-sided form; its stall exploit and the built `contested` correction are under [[reward-design]] with the adoption call above. Deployment sampling: greedy helps distilled arms and hurts the supervised anchor, so the rule is per-checkpoint; the entropy-adaptive nucleus sits within one standard error. The instrument stack is the quiet result of the program: the thirteen-column battery with per-rung ladders, symmetry gauge, fidelity and calibration reports, heartbeats with loss decomposition and gradient norms, the dashboard, and the convergence report. The process rule they enforce, no training verdict from single-suite reads, exists because two long-budget headlines died on first contact with them.

## What still shows promise, in order

Two facts frame the ranking. No reinforcement configuration has produced a policy better than its own supervised anchor on held-out. And almost every candidate's forecast sits inside the three-seed suite band of about $\pm 0.03$, at which power two suite signs flipped outright in the scale test, so suite-delta-sized levers cannot be verdicted at this power.

| Rank | Approach | Why, and what grounds it |
|---|---|---|
| 1 | Collect where the policy loses, distil weighted by regret | Measured 2026-08-09 ([[../archive/experiments/2026-08-08-audit-and-the-deviation-finding|the audit log]]): regret weighting at equal soft mass beats its unweighted twin $+0.063$ held-out, $+0.135$ on reward, paired positive every seed; the deviation probe explains why |
| 2 | The deployment rule, per checkpoint | Greedy helps every distilled arm and hurts the supervised anchor; greedy lifted the regret-weighted arm to 0.546 from 0.535, and the best weights-only reading on record is the regret-band arm's 0.571, itself greedy |
| 3 | Search-teacher collection in the regret band | Measured twice at matched soft mass: 3.7 times the regret per label, buys the defender mirror clearly, leads every arm on the trained reward |
| 4 | Anchored reinforcement, closed as retention | Four times the budget climbs back to the anchor and stops there, both runs converged; a crossing must come from elsewhere |
| 5 | Group-relative advantages with shared starts | Shared-start groups reach 0.682 to 0.708 held-out against 0.514 for spanning groups, three seeds ([[../archive/experiments/2026-08-03-training-runs]]) |
| 6 | The deployment compute ladder | The crossing sits between four and eight playouts, 0.594 then 0.700 against 0.660 at 1.6 and 3.2 seconds an episode; budget pays monotonically through 256 on the mirrors, so the open question is price, not ceiling |
| 7 | Search-teacher DAgger at learner-reached states | Every corpus so far is teacher-distribution; the covariate-shift finding now supports this directly |
| 8 | Outcome-grounded calibration | Owner-requested, still unbuilt; pairs naturally with support-complete corpora |

## Closed lines

Not to revisit without new evidence, each with its measurement on record: chair-balanced training at matched budget; label smoothing; softplus as default; visit-temperature targets at current coverage; value networks as leaves at current data; tabula rasa at this budget; AWR against a deterministic teacher; the entropy floor as a benefit; the trust-region choice at narrow-range budgets; cross-game transfer and the language model as policy ([[transfer-and-llm-policies]]); the engine's difficulty setting as an opponent axis; longer or shorter distillation budgets and the entropy bonus in both regimes; confidence-filtered and unforced label collection; and the Sequential Halving allocator as a playing rule ([[../archive/experiments/2026-08-11-distillation-budget-and-checkpoint-selection|budget log]], [[../archive/experiments/2026-08-10-search-configuration-and-two-retractions|retractions log]]).

## Scope limits on every number above

The searching agent's record is thinner than its headline: one checkpoint, one exploration constant, and the honest sweep separates from the engine on no suite once matchup spread is counted. Its largest-sample record is its own collection manifests, 0.464 to 0.564 across five collections read unconditionally, so on generator-sampled fights it is close to a coin flip and the distribution, not the operator, carries the difference to any headline.

The suites themselves bound resolution. The sampler prices creatures at base stats and not the commander, and a 5-attack 5-defense commander wins every battle on otherwise identical armies, so matchups the budget calls balanced can be decided before play; 9 of the 20 held-out matchups are already decided in the engine's own hands. Half the headline suite cannot distinguish two policies, the reported differences are carried by the other half, and a recalibrated suite would measure the same policies more sharply ([[scenario-distribution#The commander is not priced, and it decides identical armies]]).

## The remaining experiments, concretely

In flight or immediately next: the both-chair round and the 4000-iteration leashed pair, judged by the full apparatus; the regret-weighted arm against its twins on the relabeled corpus; the deployment-rule rerun with the engine baseline under the identical protocol; and the 512-playout rung of the budget ladder. Gated on those: search-teacher collection at learner-reached states in the losing band, the group-relative arms inside `train_ppo`, and the evaluation-suite repair, since four suites cannot separate players and the freshly sampled one, the agent's worst, is the least screened. Layer 3 of the ability program, task 33, continues underneath. Each lands in a dated log with its reports vendored, per `agent_play/experiments/README.md`.
