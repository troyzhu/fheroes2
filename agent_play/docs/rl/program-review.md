---
title: "The program review: every approach, its verdict, and what remains"
type: review
updated: 2026-08-07
related_concepts: ["[[training-design]]", "[[rl-methods]]", "[[value-estimation-lab]]", "[[off-support-and-offline-improvement]]"]
tags: [agent-env, rl, review, roadmap]
---

# The program review: every approach, its verdict, and what remains

The owner commissioned this as the master review on 2026-08-07, after the memory compaction: one page that walks every approach the project has experimented with, says which still show promise, and grounds both the verdicts and the remaining program in the measurements and the literature. Each claim cites the page or archive log that carries its evidence; nothing here is new, only assembled. The standing goal it serves is task 45, a policy that exceeds the built-in rule-based AI, judged by the full apparatus rather than any single suite.

## The scoreboard

<!-- verify
# Invalidators for the scoreboard and the in-flight sentence below.
exists  agent_play/docs/archive/experiments/files/2026-08-07-run-reports/battery_rl_trio.json
exists  agent_play/docs/archive/experiments/files/2026-08-07-run-reports/convergence_long_runs.json
exists  agent_play/experiments/trust_region_rematch.py
grep    python/fheroes2_agent/train_ppo.py :: trust_region
grep    python/fheroes2_agent/selfplay.py :: OpponentPool
-->

| Regime | Best measured | Built-in AI | Standing |
|---|---|---|---|
| Weights only, held-out pool | $0.555 \pm 0.043$ (owner-objective labels), $0.526 \pm 0.013$ replicated champion | 0.660 | About 0.13 short, unmoved by every supervised lever |
| Agent (policy plus search), same pool | $0.963 \pm 0.027$ | 0.660 | Past the baseline decisively |
| Commander extremes | 0.976 to 1.000 across strong arms | 0.958 | Past the baseline |

The distillation gap is the program's central number: the same prior that searches to 0.963 plays 0.526 raw, so about 0.44 of win rate exists in the search process and has resisted transfer into the weights. The archive log [[../archive/experiments/2026-08-07-overnight-champion-mixture]] carries the full-apparatus verdicts behind every row, and [[../archive/experiments/2026-08-06-night-block-search-generations]] the search-agent measurement. Two runs are in flight as this page is written: the self-play continuation round at 1000 iterations across seeds, and the DPPO trust-region rematch of task 48.

## Imitation: measured to its ceiling

The supervised program is the most thoroughly mapped component and the verdict is a plateau. Cloning the built-in planner reaches agreement $0.9085 \pm 0.005$ with the planes architecture on the champion mixture, and every architectural lever raised fidelity without raising play past the teacher: capacity is data-limited not saturated, ability features and spatial planes and width all closed agreement gaps, and the held-out gap to the AI held at about 0.13 through all of them ([[the-policy-network]], [[../archive/experiments/2026-08-07-overnight-champion-mixture]]). The recurring trade is fidelity up, flagship down: softplus and the ability arm both lifted agreement while collapsing the Thunk ladder, which is why graded suites are never collapsed to one number.

DAgger ran one round and behaved as [[../research/works/expert-iteration|the literature]] and its own bound predict, $+0.094 \pm 0.036$ on pool win rate with the planner-probe labeling seam proving out ([[training-design]]). The sharpness program closed the mode-collapse question for imitation: a deterministic teacher makes the optimal clone nearly one-hot, entropy bonuses and label smoothing fight the target rather than the problem (smoothing was outright harmful), and the one honest softener is the eight-epoch budget cut, genuinely higher normalized entropy at 0.29, play at par, downstream delta $+0.017$ inside one standard error, kept as an arm rather than a default.

Verdict: closed as a line of improvement. Imitation converges to the teacher, not past it, and the remaining 0.13 is an improvement-operator gap, not an architecture gap.

## Search: the one operator measured above the teacher

Root-PUCT over the cloned prior with rollout-scored leaves, 32 simulations per decision, is the only mechanism that has beaten the planner: $0.963 \pm 0.027$ on the identical held-out pool and win definition. This is the AlphaGo-era evaluation scheme, rollouts where [[../research/works/alphazero|AlphaZero]] substitutes a learned value, and the substitution is exactly what our measurements block so far. The tabula-rasa pilot ran AlphaZero's no-demonstrations claim honestly and it failed at this budget: search over a random prior won at most 0.208 of its own episodes and every distilled policy evaluated at 0.000, because rollout-scored search inherits the rollout policy's competence ([[../archive/experiments/2026-08-06-night-block-search-generations]]). The imitation start is a compute convenience, not a principle, and remains right here; the genre's verified record, assembled 2026-08-08 in [[../research/prior-art]], agrees from both directions, no from-scratch success anywhere without dense shaped rewards, and imitation bootstraps winning exactly where shaping was refused.

Verdict: alive and load-bearing. The search agent is the project's strongest player and the standing source of labels; its cost, about fifteen seconds per episode, is the price the distillation program exists to remove.

## Distilling search: blocked on coverage, not on targets

One round of search teaching built the champion; the second round stalled on both architectures, so the plateau is a property of the supervised program rather than any network. Labels chosen by search scoring the owner objective produced the only coherent multi-suite gain since generation one, six of seven suites together at $0.555 \pm 0.043$. Every soft-target family then nulled for one shared reason: with UCB visiting about two candidates per state, visit-count targets at temperature ([[../research/works/mcts-regularized-policy-optimization|the regularized-policy-optimization view]] of AlphaZero's own target) and value-derived soft labels alike carry almost no spread to distill, 0.231 nats against an effectively one-hot baseline. The demonstrated prerequisite is coverage-forced collection, one rollout per candidate before UCB concentrates, which would give the soft-target program the support it measurably lacks.

Verdict: the highest-leverage open build. Coverage-forced collection unlocks three threads at once, soft targets, per-candidate value fitting, and graded calibration, and nothing else in the search-distillation family is worth rerunning before it exists.

## Value estimation: an educational lab with one hard law

The value thread ran nine measured configurations and produced one law: support is destiny at every granularity. A state-value fit reaches explained value 0.8565 on its own distribution and still cannot rank moves at a leaf; the behavior Q collapses from 0.853 to 0.263 the moment it reranks actions the data never took, [[../research/works/bcq-extrapolation|the extrapolation-error result]] reproduced in miniature; and the owner-proposed rollout-trained value fits branch returns at 0.888 while agreeing with search's argmax 0.105 of the time, because per-candidate support at two visited candidates per state is no support at all. The offline-improvement family ([[../research/works/cql|CQL]], [[../research/works/iql|IQL]], [[../research/works/td3-bc|TD3+BC]], [[../research/works/edac-ensembles|ensembles]], [[../research/works/awr|AWR]]) was mapped against these facts in [[off-support-and-offline-improvement]]; AWR additionally requires within-state action diversity a deterministic teacher cannot supply.

Verdict: closed as built, reopened by coverage-forced data. The lab's conclusions are recorded in [[value-estimation-lab]]; the next value experiment worth running is the same fits on candidate-complete rollout data, and not before.

## Reinforcement from strong anchors: instrumented, and honest about erosion

PPO from cloned anchors erodes more than it earns at every budget tried, and the instrumentation built at the owner's prompting turned that from an impression into a mechanism. The per-term per-module gradient norms exposed the critic slamming the shared trunk, fixed by the head-only value warmup; the normalized-entropy floor works as a controller and bought nothing against a fixed opponent, so it stays as insurance; and the long-budget experiment resolved the training-cost illusion, 400 iterations of pure training cost about eight minutes once calibration evaluations were dropped. The full apparatus then overturned the long-budget headline itself: the converged 400-iteration control had relocated erosion into the Thunk ladder, 0.58 to 0.12, while its held-out number stood still, and the convergence report proves it had settled, so the collapse is a property of the optimum, not of under-training ([[../archive/experiments/2026-08-07-overnight-champion-mixture]]).

The trust-region question reopened tonight on the owner's push: [[../research/works/dppo-trust-region|DPPO]] replaces the clip's sampled-ratio gate with a real divergence gate, our group-relative era measured it as a slower starter with a small late edge that never moved the default, and the actor-critic rematch, exact total variation and the paper's binary form against the clip, is running as task 48 with run-identity stamps in every artifact.

Verdict: open, with rules. No reinforcement conclusion without the full battery, symmetry gauge, and convergence verdict; the single-suite reads that produced two dead headlines are recorded in the conventions as the reason.

## Self-play: the current phase, and the first structural finding

The foundation is deliberately small, an `OpponentPool` over frozen checkpoints with the built-in AI as anchor and a `SelfPlayEnv` that answers the opponent's turns internally, the league-lite shape [[../research/works/alphastar-unplugged|AlphaStar's population]] and [[../research/works/openai-five|OpenAI Five's self-play]] both argue for against latest-self overfitting. The 400-iteration round traded rather than won, more ladder kept than the control at matched budget, held-out 0.448, and produced the first attacker-favored policy measured here, symmetry excess $+0.087$. The owner then supplied the reading the gauge lacked: the engine grants the attacker initiative on speed ties, its own gap is $+0.071$ attacker-favored, so the neutral point is the engine's gap and the anchor's defender lean was the anomaly all along. The convergence report says the run stopped mid-climb, which is what justifies the continuation round now in flight.

Verdict: the owner's declared phase, alive and measurably unfinished. Judged from round two onward by full battery, symmetry against the engine's structural lean, convergence, and duel counts that clear the measured $\pm 0.06$ evaluation noise.

## Reward, sampling, calibration, and the instruments

The reward is settled and owned: terminal, two-sided, strength-priced, wins graded by strength kept and losses by enemy strength destroyed, difficulty weighting opt-in, stalls resolved by the engine's rule so evasion is priced rather than exploitable ([[reward-design]], [[../decisions/0005-training-and-reward]]). Deployment sampling measured the owner's entropy-adaptive nucleus as a free-win signal that sits within one standard error against a near-deterministic policy, worth a larger-n read and nothing stronger yet. Calibration is measured against the imitation target with the deterministic-teacher caveat the owner themselves flagged, and the one requested metric still unbuilt is outcome-grounded calibration, predicted win probability against realized outcomes, which becomes cheap exactly when coverage-forced collection exists.

The instrument stack is the quiet result of the week: the thirteen-column battery with per-rung ladders, the symmetry gauge with the engine as reference, fidelity and reliability reporting, per-iteration heartbeats with loss decomposition, gradient norms, live win and loss reward quality, trust-region stamps, the dashboard, and the convergence report. The process rule they enforce, no training verdict from single-suite reads, exists because both long-budget headlines died on first contact with them.

## What still shows promise, in order

| Rank | Approach | Why, and what grounds it |
|---|---|---|
| 1 | Coverage-forced collection | Demonstrated prerequisite for soft targets, per-candidate values, and graded calibration at once; every null in the distillation family traces to its absence |
| 2 | Self-play continuation at settled budgets | The one line whose convergence read says it was still climbing when stopped; league structure from the literature already in place |
| 3 | DPPO divergence gates | Paper's diagnostic held here (clip and divergence flag different updates); rematch running with exact and binary forms |
| 4 | Expert iteration resumed on coverage-forced targets | The operator that built the champion, rerun only once its labels carry spread |
| 5 | Exploring starts by prefix replay | The endorsed exploration family; the deterministic engine makes mid-battle starts replayable at no engine cost |
| 6 | Outcome-grounded calibration | The owner-requested metric still unbuilt; pairs naturally with rank 1's data |
| 7 | Larger-n deployment sampling | The adaptive nucleus read positive inside noise; cheap to settle properly |

Closed lines, not to revisit without new evidence: label smoothing (harmful), softplus as default (flagship collapse), visit-temperature targets at current coverage, value networks as leaves at current data, tabula rasa at this budget, AWR against a deterministic teacher, and the entropy floor as a benefit rather than insurance against fixed opponents.

## The remaining experiments, concretely

The queue that follows from the verdicts: evaluate the in-flight self-play and trust-region rounds by the full apparatus with forty-episode duels; build coverage-forced collection into the search collector and re-vendor a candidate-complete corpus; rerun soft-target distillation, rollout-value fitting, and outcome-grounded calibration on that corpus; take the strongest surviving arm into the next self-play generation; and keep layer 3 of the ability program, per-candidate effect summaries of task 33, moving underneath, since richer candidate features feed every consumer above. Each lands in the day log with its reports vendored, per the conventions in `agent_play/experiments/README.md`.
