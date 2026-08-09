---
title: "The program review: every approach, its verdict, and what remains"
type: review
updated: 2026-08-09
related_concepts: ["[[training-design]]", "[[rl-methods]]", "[[value-estimation-lab]]", "[[off-support-and-offline-improvement]]", "[[transfer-and-llm-policies]]", "[[../decisions/0007-anchored-ppo]]"]
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
grep    python/fheroes2_agent/train_ppo.py :: anchor_kl_coef
exists  agent_play/docs/archive/experiments/files/2026-08-08-run-reports/battery_round4.json
exists  agent_play/docs/archive/experiments/files/2026-08-08-run-reports/deviation_probe.json
exists  agent_play/docs/decisions/0007-anchored-ppo.md
exists  agent_play/experiments/deviation_probe.py
-->

| Suite | Built-in AI | `policy_gen1.pt` | Leashed self-play | Standing |
|---|---|---|---|---|
| Held-out pool | 0.660 / $+0.87$ | 0.525 / $+0.59$ | 0.499 / $+0.57$ | Short by 0.13; the best distilled arm reads 0.571 to 0.579 greedy, short by about 0.085 |
| Thunk ladder | 0.969 / $+1.63$ | 0.875 / $+1.47$ | 0.892 / $+1.48$ | Short by 0.08 |
| Held-out as defender | 0.338 / $+0.13$ | 0.271 / $-0.06$ | 0.258 / $-0.10$ | Short by 0.08 |
| Mirrors as attacker | 0.361 / $+0.25$ | 0.194 / $-0.11$ | 0.162 / $-0.15$ | Short by 0.17 |
| Mirrors as defender | 0.639 / $+0.75$ | 0.250 / $+0.04$ | 0.324 / $+0.17$ | Short by 0.32, the largest gap left |
| Commanders | 0.958 / $+1.72$ | 0.958 / $+1.78$ | 0.976 / $+1.79$ | At par or ahead |
| Hordes | 0.192 / $-0.33$ | 0.175 / $-0.36$ | 0.167 / $-0.37$ | At par |
| Fresh sampled | 0.446 / $+0.31$ | 0.372 / $+0.13$ | 0.372 / $+0.14$ | Short by 0.07 |
| Real maps | 0.568 / $+0.66$ | 0.564 / $+0.64$ | 0.564 / $+0.64$ | At par |
| Held-out pool, agent regime | 0.660 | search over the prior: $0.963 \pm 0.027$ | | Past the baseline decisively |

Cells are win rate over the trained two-sided reward, measured on one scale on 2026-08-08 with the built-in AI carrying quality columns for the first time. Three suites are already at par or ahead, so the standing goal reduces to the held-out pool, the two mirror chairs and fresh samples. 

Two cautions ride with the table. The mirror suites are the same six symmetric matchups from either chair and the engine's own two numbers sum to 1.000, so 0.361 and 0.639 are the game's equilibrium split rather than two independent bars. And the frozen anchor re-evaluates to held-out 0.498 through 0.533 across the day's four batteries, so a difference under about 0.03 on one suite is not a result.

The distillation gap is the program's central number: the same prior that searches to 0.963 plays about 0.53 raw, so roughly 0.44 of win rate exists in the search process and has resisted transfer into the weights. 

Two facts measured on 2026-08-08 locate it. Within an outcome the policy plays as well as the engine, its wins keeping as much strength (wq 0.47 against 0.45) and its losses destroying as much (lq 0.63 against 0.63), so the deficit is entirely in how often the outcome falls the right way rather than in fight quality. And the deviation probe found the action-level signal concentrated where no corpus has ever collected: search disagrees with the prior 0.145 of the time on matchups the policy loses against 0.052 where it wins, and each disagreement there is worth $+0.787$ against $+0.152$. The archive log [[../archive/experiments/2026-08-07-overnight-champion-mixture]] carries the full-apparatus verdicts behind every row, and [[../archive/experiments/2026-08-06-night-block-search-generations]] the search-agent measurement. The 2026-08-08 rounds are judged in [[../archive/experiments/2026-08-08-selfplay-round2-and-trust-region]] and the audit that followed them in [[../archive/experiments/2026-08-08-audit-and-the-deviation-finding]].

## Imitation: measured to its ceiling

The supervised program is the most thoroughly mapped component and the verdict is a plateau. Cloning the built-in planner reaches agreement $0.9085 \pm 0.005$ with the planes architecture on the champion mixture, and every architectural lever raised fidelity without raising play past the teacher: capacity is data-limited not saturated, ability features and spatial planes and width all closed agreement gaps, and the held-out gap to the AI held at about 0.13 through all of them ([[the-policy-network]], [[../archive/experiments/2026-08-07-overnight-champion-mixture]]). The recurring trade is fidelity up, flagship down: softplus and the ability arm both lifted agreement while collapsing the Thunk ladder, which is why graded suites are never collapsed to one number.

DAgger ran one round and behaved as [[../research/works/expert-iteration|the literature]] and its own bound predict, $+0.094 \pm 0.036$ on pool win rate with the planner-probe labeling seam proving out ([[training-design]]). The sharpness program closed the mode-collapse question for imitation: a deterministic teacher makes the optimal clone nearly one-hot, entropy bonuses and label smoothing fight the target rather than the problem (smoothing was outright harmful), and the one honest softener is the eight-epoch budget cut, genuinely higher normalized entropy at 0.29, play at par, downstream delta $+0.017$ inside one standard error, kept as an arm rather than a default.

Verdict: closed as a line of improvement. Imitation converges to the teacher, not past it, and the remaining 0.13 is an improvement-operator gap, not an architecture gap.

## Search: the one operator measured above the teacher

Root-PUCT over the cloned prior with rollout-scored leaves, 32 simulations per decision, is the only mechanism that has beaten the planner: $0.963 \pm 0.027$ on the identical held-out pool and win definition. This is the AlphaGo-era evaluation scheme, rollouts where [[../research/works/alphazero|AlphaZero]] substitutes a learned value, and the substitution is exactly what our measurements block so far. The tabula-rasa pilot ran AlphaZero's no-demonstrations claim honestly and it failed at this budget: search over a random prior won at most 0.208 of its own episodes and every distilled policy evaluated at 0.000, because rollout-scored search inherits the rollout policy's competence ([[../archive/experiments/2026-08-06-night-block-search-generations]]). The imitation start is a compute convenience, not a principle, and remains right here; the genre's verified record, assembled 2026-08-08 in [[../research/prior-art]], agrees from both directions, no from-scratch success anywhere without dense shaped rewards, and imitation bootstraps winning exactly where shaping was refused.

Verdict: alive and load-bearing. The search agent is the project's strongest player and the standing source of labels; its cost, about fifteen seconds per episode, is the price the distillation program exists to remove.

## Distilling search: blocked on coverage, not on targets

One round of search teaching built the champion; the second round stalled on both architectures, so the plateau is a property of the supervised program rather than any network. Labels chosen by search scoring the owner objective produced the only coherent multi-suite gain since generation one, six of seven suites together at $0.555 \pm 0.043$. Every soft-target family then nulled for one shared reason: with UCB visiting about two candidates per state, visit-count targets at temperature ([[../research/works/mcts-regularized-policy-optimization|the regularized-policy-optimization view]] of AlphaZero's own target) and value-derived soft labels alike carry almost no spread to distill, 0.231 nats against an effectively one-hot baseline. The demonstrated prerequisite is coverage-forced collection, one rollout per candidate before UCB concentrates, which would give the soft-target program the support it measurably lacks.

Verdict: built, and measured twice on 2026-08-08. The support-complete corpora (80 percent of decisions with every candidate visited) give value-derived soft targets a small, twice-replicated held-out gain of about $+0.03$ to $+0.04$ with the reward columns agreeing, while the pilot's broader seven-of-eight suite pattern did not survive the 2.1x scale test, two suite signs flipping outright, and visit-derived targets stay null at both scales. The mechanism is real and cheap, the effect it currently buys is modest, and the open questions are a soft-mass-balanced scale test and the rollout-value and calibration consumers on this data ([[../archive/experiments/2026-08-08-selfplay-round2-and-trust-region]]).

## Value estimation: an educational lab with one hard law

The value thread ran nine measured configurations and produced one law: support is destiny at every granularity. A state-value fit reaches explained value 0.8565 on its own distribution and still cannot rank moves at a leaf; the behavior Q collapses from 0.853 to 0.263 the moment it reranks actions the data never took, [[../research/works/bcq-extrapolation|the extrapolation-error result]] reproduced in miniature; and the owner-proposed rollout-trained value fits branch returns at 0.888 while agreeing with search's argmax 0.105 of the time, because per-candidate support at two visited candidates per state is no support at all. The offline-improvement family ([[../research/works/cql|CQL]], [[../research/works/iql|IQL]], [[../research/works/td3-bc|TD3+BC]], [[../research/works/edac-ensembles|ensembles]], [[../research/works/awr|AWR]]) was mapped against these facts in [[off-support-and-offline-improvement]]; AWR additionally requires within-state action diversity a deterministic teacher cannot supply.

Verdict: closed as built, reopened by coverage-forced data. The lab's conclusions are recorded in [[value-estimation-lab]]; the next value experiment worth running is the same fits on candidate-complete rollout data, and not before.

## Reinforcement from strong anchors: instrumented, and honest about erosion

PPO from cloned anchors erodes more than it earns at every budget tried, and the instrumentation built at the owner's prompting turned that from an impression into a mechanism. The per-term per-module gradient norms exposed the critic slamming the shared trunk, fixed by the head-only value warmup; the normalized-entropy floor works as a controller and bought nothing against a fixed opponent, so it stays as insurance; and the long-budget experiment resolved the training-cost illusion, 400 iterations of pure training cost about eight minutes once calibration evaluations were dropped. The full apparatus then overturned the long-budget headline itself: the converged 400-iteration control had relocated erosion into the Thunk ladder, 0.58 to 0.12, while its held-out number stood still, and the convergence report proves it had settled, so the collapse is a property of the optimum, not of under-training ([[../archive/experiments/2026-08-07-overnight-champion-mixture]]).

The trust-region question closed measured on 2026-08-08. [[../research/works/dppo-trust-region|DPPO]] replaces the clip's sampled-ratio gate with a real divergence gate, and nine runs at matched budget showed the gates provably differing, blocking 7 to 13 percent of samples against 5 to 7 and under 1, while their outcomes stayed indistinguishable. The step constraint was not the binding one, which is what pointed at the destination constraint instead.

That is the leash, and it is the first configuration here that trains without eroding its anchor. A forward KL to the checkpoint frozen before the first update, at $\beta = 0.5$, returns the Thunk ladder to anchor level and matches the anchor's reward columns at zero cost on the training distribution, with the leash tension itself converging at KL 0.11. [[../decisions/0007-anchored-ppo]] records it as the standing recipe and states plainly what it is not: retention, not climbing, since no reinforcement run here has yet beaten the supervised anchor it started from.

Verdict: open, with the platform now stable. No reinforcement conclusion without the full battery, symmetry gauge, and convergence verdict; the single-suite reads that produced two dead headlines are recorded in the conventions as the reason.

## Self-play: the current phase, and the first structural finding

The foundation is deliberately small, an `OpponentPool` over frozen checkpoints with the built-in AI as anchor and a `SelfPlayEnv` that answers the opponent's turns internally, the league-lite shape [[../research/works/alphastar-unplugged|AlphaStar's population]] and [[../research/works/openai-five|OpenAI Five's self-play]] both argue for against latest-self overfitting.

Four rounds are measured. Round two at 1000 iterations converged and bought training-distribution mastery with held-out and ladder erosion in every seed. Round three changed one variable, 200 generator-sampled matchups instead of twelve, and removed the specialization pathology without removing the erosion, which relocated the problem from the data distribution to the optimization and set up the leash. Round four is the leash itself. Through all of them the chair was the attacker's, which the scoreboard says is the wrong one to train alone, since the two mirror chairs carry the largest remaining gaps and the engine's own split favours the defender on symmetric armies; `learner_side="alternate"` exists for that and its first round is the current measurement.

Verdict: alive, and now on a non-destructive base. Judged by full battery, symmetry against the engine's structural lean, convergence, and duel counts that clear the measured noise, which forty episodes does not: the frozen anchor's own duel rate against the engine moves 0.20 between report files.

## Reward, sampling, calibration, and the instruments

The reward is settled and owned: terminal, two-sided, strength-priced, wins graded by strength kept and losses by enemy strength destroyed, difficulty weighting opt-in, stalls resolved by the engine's rule so evasion is priced rather than exploitable ([[reward-design]], [[../decisions/0005-training-and-reward]]). Deployment sampling measured the owner's entropy-adaptive nucleus as a free-win signal that sits within one standard error against a near-deterministic policy, worth a larger-n read and nothing stronger yet. Calibration is measured against the imitation target with the deterministic-teacher caveat the owner themselves flagged, and the one requested metric still unbuilt is outcome-grounded calibration, predicted win probability against realized outcomes, which becomes cheap exactly when coverage-forced collection exists.

The instrument stack is the quiet result of the week: the thirteen-column battery with per-rung ladders, the symmetry gauge with the engine as reference, fidelity and reliability reporting, per-iteration heartbeats with loss decomposition, gradient norms, live win and loss reward quality, trust-region stamps, the dashboard, and the convergence report. The process rule they enforce, no training verdict from single-suite reads, exists because both long-budget headlines died on first contact with them.

## What still shows promise, in order

Two facts frame the ranking. No reinforcement configuration here has yet produced a policy better than its own supervised anchor on held-out, so reinforcement is currently a retention mechanism. And almost every candidate's forecast sits inside the three-seed suite band of about $\pm 0.03$ that the scale test priced, at which power two suite signs flipped outright, so a lever whose expected effect is a suite delta cannot be verdicted at this power.

| Rank | Approach | Why, and what grounds it |
|---|---|---|
| 1 | Collect where the policy loses, distil weighted by regret | Measured 2026-08-08: weighting soft rows by rank-transformed regret at equal soft mass beats its unweighted twin by $+0.063$ held-out and $+0.135$ on reward, paired positive on every seed, and beats the hard twin by $+0.102$; the deviation probe explains why, and collecting more of the 6.8 percent that carries regret is the untested half |
| 2 | The deployment rule, per checkpoint | Measured 2026-08-08: greedy helps every distilled arm and hurts the supervised anchor, so the rule is a property of how well a checkpoint ranks rather than a universal correction; the best weights-only reading is 0.546 held-out under a stated rule |
| 3 | Search-teacher collection in the regret band | Measured 2026-08-08 and re-measured at matched soft mass: the screened corpus carries 3.7 times the regret per label and buys the defender mirror clearly, 0.375 against 0.278, with held-out $+0.021$ inside the band and the ladder regressing; the union of the corpora is at least as good as either and needs more seeds to rank |
| 4 | Anchored reinforcement, closed as retention | Measured 2026-08-08: four times the budget on the leashed base climbs back to the anchor and stops there, held-out 0.507 and the ladder 0.906 against the anchor's 0.498 and 0.906, both runs converged; reinforcement here retains rather than climbs, so a crossing must come from elsewhere |
| 5 | Group-relative advantages with shared starts | The repository's own strongest precedent for the wide distribution's problem: groups sharing one matchup transferred fivefold better than groups spanning eight, and the wide arm's advantage spread looks difficulty-inflated rather than starved |
| 6 | The deployment compute ladder | Only 32 and 48 simulations have ever been run; if the crossing survives at eight the agent regime becomes shippable rather than an oracle |
| 7 | Search-teacher DAgger at learner-reached states | Every corpus was collected from the teacher's own state distribution; gated on the deviation finding, which now supports it |
| 8 | Outcome-grounded calibration | The owner-requested metric still unbuilt; pairs naturally with the support-complete corpora |

Closed lines, not to revisit without new evidence: chair-balanced training at matched budget (worse on its own target suite, mirrors as defender 0.241 against the attacker-only control's 0.326 with no per-seed overlap), label smoothing (harmful), softplus as default (flagship collapse), visit-temperature targets at current coverage, value networks as leaves at current data, tabula rasa at this budget, AWR against a deterministic teacher, the entropy floor as a benefit rather than insurance against fixed opponents, the trust-region choice at narrow-range budgets where the rematch measured all three gates indistinguishable, cross-game weight or representation transfer and the small language model as policy ([[transfer-and-llm-policies]]), and the engine's own difficulty setting as an opponent axis, which returns a non-default only on the easiest setting and gates valuations unreachable under `simple_v1`.

## The honest ceiling

The weights-only regime has not crossed the built-in AI and nothing measured says it is about to. The gap has held near 0.13 on held-out through every architecture and data lever tried, the leash buys retention rather than height, and every remaining forecast is at or inside the noise band. There is no single lever on the list sized to 0.13.

Two things could still move it, and neither is a modelling result. The deployment rule is one: every headline was measured under sampling against a deterministic planner, and if greedy holds at full power against a re-measured engine baseline then part of the gap was self-inflicted reporting. The other is that the gap is not diffuse. It concentrates in a handful of positions the policy loses badly and search wins outright, and the 2026-08-08 probe showed those positions carry an action-level signal that no corpus has ever contained. A lever aimed there is the only kind with the right magnitude, which is why it is ranked first.

The agent regime is a different matter and should be stated as such rather than as a consolation. Root search over the cloned prior reads 0.963 against the engine's 0.660 on the same slice, which is a decisive crossing of the standing goal at about fifteen seconds an episode. Whether that is a deliverable or an oracle is a compute question, not a strength question, and the simulation ladder is what turns it into a measurement.

## The remaining experiments, concretely

In flight or immediately next: the both-chair round and the 4000-iteration leashed pair, both judged by the full apparatus; the regret-weighted distillation arm against its unweighted and hard twins on the relabeled corpus; and the deployment-rule rerun with the engine baseline measured under the identical protocol. Then, gated on those: search-teacher collection at learner-reached states in the losing band, the group-relative advantage arms inside `train_ppo` so their runs are readable by the convergence report, and the simulation ladder from 0 to 32. Layer 3 of the ability program, task 33, continues underneath since richer candidate features feed several of these. Each lands in a dated log with its reports vendored, per the conventions in `agent_play/experiments/README.md`.
