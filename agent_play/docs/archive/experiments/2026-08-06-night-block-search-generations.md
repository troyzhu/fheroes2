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

## Generation two, and the gate that failed

The stronger collector kept more of the distribution, 134 of 180 sampled matchups, 808 episodes, 12,704 labels, battlefield-varied through the new offset, in about an hour. Distilled with both generations at double weight, the battery reads a failed gate: the ladder fell to 1.000 / 1.000 / 0.833 / 0.417 and held-out to 0.506, while the defender suites rose modestly (held-out as defender 0.298 against 0.263, defender mirrors 0.285 against 0.229). Generation one's own columns wobble a few hundredths between battery runs, which calibrates the noise; the ladder drop exceeds it.

The reading is the night's recurring one, mixture proportions rather than label quality: twenty-two thousand fresh labels now compete with the horde-recovery relabelings the ladder depends on, and the axis that gained is exactly where the new labels concentrated, defender play. Before the pre-registered deeper-labels pivot, one surgical arm tests that reading directly: keep generation two's defender-side labels, drop its attacker-side ones, and re-distill, aiming to bank the defender gains without paying the ladder.

## The two arms around generation one

Both quick arms lost to generation one. The defender split, 7,081 defender labels added to the generation-one recipe, restored the perfect 850 rung but left the top rung at 0.417 and dropped the defender mirrors it was meant to bank, so no tested proportion of generation-two labels preserves the peak. Upweighting generation one's own labels to triple weight, no new data at all, also degraded the ladder, 0.833 and 0.542 on the top rungs against double weight's 1.000 and 0.792, with a consolation of perfect commander extremes; the mixture dial is monotone-then-costly even on the winning set. Generation one at double weight stands, and the deep-label arm, forty-eight simulations per decision from a fresh seed, is the night's last pre-registered attempt to climb past it.

## The definitive table, five checkpoints in one paired run

| Suite | Clone v4 | share2 | combined v2 | Generation 1 | Generation 3 |
|---|---|---|---|---|---|
| Fresh sampled | 0.392 | 0.387 | 0.366 | 0.378 | 0.384 |
| Held-out pool | 0.385 | 0.550 | 0.504 | 0.527 | 0.512 |
| Stress hordes | 0.167 | 0.200 | 0.192 | 0.158 | 0.192 |
| Stress wide-only | 0.375 | 0.431 | 0.431 | 0.431 | 0.403 |
| Stress commanders | 0.927 | 0.969 | 0.938 | 0.969 | 0.927 |
| Thunk ladder | 0.573 | 0.906 | 0.802 | 0.917 | 0.917 |
| Held-out as defender | 0.258 | 0.235 | 0.273 | 0.256 | 0.298 |
| Mirrors as attacker | 0.312 | 0.174 | 0.167 | 0.201 | 0.104 |
| Mirrors as defender | 0.299 | 0.368 | 0.354 | 0.271 | 0.264 |

No strict dominator exists, and the sharpest stable discriminator is the ladder's top rung: generation one reads 0.71 to 0.79 across every battery of the night where share2 reads 0.42 to 0.54, while everywhere else the two sit within evaluation noise of each other. The night's verdict is therefore: `policy_gen1` is the improved policy, materially better on the flagship out-of-distribution validation and at parity on the rest; share2 remains the held-out co-champion; and the fresh-sampled suite reads 0.37 to 0.39 for every checkpoint from clone v4 onward, because the raw distribution's hopeless half admits no policy improvement at all.

## The first architectural arm, measured at the plateau's edge

With label scaling saturated, the block's last experiment opened the recorded architectural path: each slot's input extended by its creature's fixed ability profile (`ability_features` on the policy, the capability audit's layer-1 records applied to the one-hot inside the model, 1,056 extra parameters, no observation change). The paired single-seed ablation on the champion corpus is a clean axis trade, not a win: wide-only 0.486 against 0.431, mirrors and defender suites up several points, hordes 0.200 against 0.158, and the Thunk ladder collapsed to 0.667 with the top rung at 0.083. The inductive bias helps exactly where ability profiles bind and spends exactly the capacity the extreme-horde behavior lived in, so it does not ship, and the honest note is one seed.

The arm also replicated the champion for free: its plain arm re-rolls generation one's recipe under a different data order and lands at 0.906 on the ladder and 0.487 held-out, which at the time read as recipe robustness. The deliberate replication that closed the block corrects that reading.

## The seed replication, and what it takes back

Three fresh seeds of the champion recipe, batteried beside the original in one paired run, end the block on its most honest note. Top rungs: the original at 0.542 in this read (its own four reads span 0.542 to 0.792), the re-rolls at 0.500, 0.333 and 0.333; held-out 0.498 to 0.542 throughout. The recipe's expected top rung is therefore near 0.43 with seed noise around 0.10, overlapping share2's 0.42-to-0.54 band, so the recipe-level claim that generation one beats share2 on the ladder is not established; the artifact `policy_gen1` remains the best single checkpoint measured, repeatedly, but part of its crown was a favorable training roll.

What survives at full strength: every checkpoint since the DAgger era towers over clone v4 (ladder 0.77-plus against 0.573, held-out 0.50-plus against 0.385), and the plateau law rests on many arms moving coherently across suites rather than on any single rung. The block's methodological deliverable is that single-roll comparisons at the flagship rung carry a tenth of seed noise, so future arms gate on multi-seed batteries, a convention now recorded in the experiments README.

## The deep-label arm, and the plateau confirmed

Forty-eight simulations per decision bought better labels and the same law. The collection kept 792 episodes and 10,220 labels across the six shards, search again winning essentially everything it kept, and the distill traded axes once more in the same-run pairing: held-out 0.519 against generation one's 0.498, defender suites up (0.304 against 0.250 on the held-out-as-defender), commander extremes perfect, and the ladder down, 0.896 against 0.948 with the top rung at 0.583 against 0.792, wide-only and attacker mirrors down with it. Three attempts from three directions, more data, re-weighted data, and better data, all traded against the peak rather than climbing past it. At this corpus and this network, the supervised mixture is saturated: every label set buys its own axes at the price of others, and generation one's blend happens to sit on the owner's flagship validation.

## The baseline that was missing, and it changes the standing

The owner asked the morning after for the built-in AI's own numbers, on the grounds that beating the clone answers a question about the pipeline while beating the engine's AI answers the question a player asks. The measurement is cheap, since the engine plays both sides natively (`builtin_ai_baseline.py`, one battlefield per episode, the same suites), and it reframes everything above.

| Suite | Built-in AI | Clone v4 | share2 | Generation 1 |
|---|---|---|---|---|
| Held-out pool | 0.660 | 0.385 | 0.550 | 0.527 |
| Thunk ladder | 0.969 | 0.573 | 0.906 | 0.917 |
| Held-out as defender | 0.338 | 0.258 | 0.235 | 0.256 |
| Mirrors as attacker | 0.361 | 0.312 | 0.174 | 0.201 |
| Mirrors as defender | 0.639 | 0.299 | 0.368 | 0.271 |
| Stress hordes | 0.192 | 0.167 | 0.200 | 0.158 |
| Stress wide-only | 0.458 | 0.375 | 0.431 | 0.431 |
| Stress commanders | 0.958 | 0.927 | 0.969 | 0.969 |
| Fresh sampled | 0.446 | 0.392 | 0.387 | 0.378 |

The teacher still leads almost everywhere: held-out by 0.13, the Thunk ladder by 0.05 with 0.88 against 0.67 at the top rung, defender mirrors by 0.37, and commander extremes is the single suite where the policies edge ahead. The honest standing is that the pipeline closed most of the distance from a naive clone to its teacher, on the flagship fight from 0.08 to roughly 0.7 against the teacher's 0.88, and surpassed it nowhere that matters. That is what the empty relabeling band predicted the evening before: imitation converges toward its demonstrator, and every gain since has been recovery of the teacher's competence rather than progress past it.

Two consequences bind the program. Every future improvement claim carries this column beside it, because progress against clone v4 flatters and progress against the AI does not. And the escalation ranking hardens: search is the only operator measured above the planner, by the probe and by 90 percent collection win rates on matchups the policies lose, so passing the teacher means deploying search at decision time or distilling far more of it, not another supervised generation.

## The morning questions: win quality, tabula rasa, and what the agent cannot see

Three owner questions the day after, each answered by measurement.

Win quality, because a win rate hides how a battle was won. Terminal records now carry per-side army strength, so the Thunk fight reads in value rather than only in outcomes: the built-in AI wins it with 0.393 of its starting strength intact, generation one with 0.444, share2 with 0.471. The policies preserve more than their teacher, which the survival term in the reward earns, and every one of them still spends more than half the army on a fight the owner believes is winnable at near-zero loss.

Root search wins that fight every time at 32 simulations and preserves 0.443, and neither pricing survival by creature strength nor weighting it five times moved the number (0.416 and 0.449, inside noise). The explanation is structural: root search evaluates a candidate by rolling the rest of the battle out with the policy, so every branch inherits the policy's aggression after the first move, and no scoring rule applied at the root can express a plan the continuation never plays. Multi-ply search, or a value trained on careful play, is what the owner's line would need.

Tabula rasa, because AlphaZero needs no demonstrations and the question is whether this project did. `tabula_rasa_pilot.py` ran the loop honestly: random initialization, root search, distil, evaluate, three rounds on four mid-band matchups. Search over a random prior won 0.167, 0.208 and 0.208 of its own episodes across the rounds, and every distilled policy evaluated at 0.000. So search does not bootstrap from nothing at this budget, and the reason is visible in the mechanism: our search scores branches by rollouts played with the policy itself, so a random policy makes every branch equally noisy, where AlphaZero substitutes a learned value network and thousands of times the compute. The imitation start was a convenience for compute rather than a necessity in principle, and remains the right choice here.

What the agent cannot see, because the owner's tactic depends on it. Each stack contributes counts, current and top hit points, the six stat fields, position, ability flags and identity, for both sides, so remaining hit points are fully observable. The board is not, and the precise statement matters because the owner corrected a sloppier one.

Obstacles are not unseen so much as unrepresented: the observation carries no passability, geometry, reachability or threat features, and the only trace of the board's shape is which of the acting stack's own moves the mask permits this turn. So the policy has a one-step, own-side view of where it may go and no representation at all of where the enemy may go, which is exactly what a plan that forces a stack to walk around an obstacle would need. A plan that forces an enemy stack to walk around terrain is therefore not expressible in what the policy sees, which makes [[../../decisions/0004-spatial-observation-modality]]'s `planes_v1` the first prerequisite for the owner's line rather than a nice-to-have.

## The symmetry test the owner asked for

The mirror suite gives both sides the same army and asks which chair wins, which measures the game's own bias and the policy's together. The owner's sharper test holds the agent's army fixed and swaps only the chair: command that army as attacker, then command the same army as defender while the built-in AI attacks with the opposing one. A policy without side bias, in a game without side bias, posts the same number twice, and running the identical pair with the engine on both sides supplies the game's contribution rather than assuming it.

| Commander | As attacker | As defender | Gap | Excess over the engine's own |
|---|---|---|---|---|
| Built-in AI | 0.708 | 0.637 | $+0.071 \pm 0.066$ | reference |
| share2 | 0.650 | 0.600 | $+0.050 \pm 0.052$ | $-0.021 \pm 0.062$ |
| Generation 1 | 0.537 | 0.633 | $-0.096 \pm 0.085$ | $-0.167 \pm 0.092$ |

Three readings, and the third is the one worth keeping. The game is close to side-neutral on these ten matchups, the engine's own gap being $+0.071 \pm 0.066$, so a policy gap is mostly the policy's own. share2 is essentially symmetric, its excess over the engine sitting at $-0.021 \pm 0.062$. And generation one, the night's champion on the flagship ladder, is the least symmetric of the three: it plays these matchups better from the defender's chair than the attacker's, an excess of $-0.167 \pm 0.092$ against the engine, which is about 1.8 standard errors and therefore suggestive rather than settled.

That is a defect the battery could not see, because every suite except the mirrors fixes the chair, and it fits the night's collection history: generation one's labels came half from each side, but its predecessors' did not, and the ladder it wins is an attacker-side fight. The recorded consequence is that side symmetry joins the battery as a first-class column rather than living in a side experiment, and that any future champion is checked for it before adoption, since a policy that is strong on one flagship fight and asymmetric elsewhere is not obviously the better agent.
