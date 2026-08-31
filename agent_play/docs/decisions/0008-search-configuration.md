---
title: "ADR 0008 — What root search is configured to do, and what its numbers mean"
type: adr
status: accepted
updated: 2026-08-10
related_concepts: ["[[0005-training-and-reward]]", "[[../rl/reward-design]]", "[[../rl/rl-methods]]", "[[../archive/experiments/2026-08-10-search-configuration-and-two-retractions]]"]
tags: [adr, agent-env, search, measurement]
---

# ADR 0008 — What root search is configured to do, and what its numbers mean

- Status: accepted, on the mirror suite from both chairs with the held-out pool as the confirming surface
- Implementation: built, `python/fheroes2_agent/search.py` and the `combatSeedOffset` field on the scenario
- Evidence: [[../archive/experiments/2026-08-10-search-configuration-and-two-retractions]]
- Supersedes: nothing, because nothing decided any of this before

## The sub-problem

Root search has been this project's strongest operator since 2026-08-05 and has never had a decision record. Four settings that determine what its numbers mean were therefore inherited rather than chosen, and each of them moves results by more than most of what the program deliberately varies. Two of the four had different values in different harnesses, which is how the same checkpoint on the same armies and the same battlefield came to read 0.00 from one loop and 0.50 from another.

The settings are not tuning knobs in the usual sense. Three of them decide what a reported number is a statement about, and getting one wrong does not make the agent worse, it makes the measurement mean something other than what the page around it claims.

## The decision

What search maximizes is passed explicitly and stamped. `rollout` returns the side environment's terminal reward, so that environment's `reward_margin` is the objective. The survival-graded family is adopted, `hit_points` as the default, on $+0.191$ over the destruction-graded `two_sided` across 48 paired cells at a standard error of 0.044. This does not bind the training reward, which remains ADR 0005's; what a policy is trained on and what its search maximizes are separate choices and are now kept separate on purpose.

The side environment is pinned to the live episode's battlefield, through `BattleEnv.current_battlefield` and `sync_side_environment`. This is a correctness requirement rather than a preference: `rollout` replays the action prefix, and only on the same world seed does that replay track the live position. Exactness needs the shared dice as well; under the independent offset this record mandates, the prefix produces a resampled position, stale actions skipped silently, which is the accepted price of honest value estimates.

The side environment's combat stream is made independent of the live battle for every number quoted against the built-in AI, through a nonzero `combat_seed_offset`. Pinning the battlefield alone also hands over the live dice, because the combat seed derives from the map seed, and search then checks what a move will do rather than estimating what it is worth. The shared-dice configuration remains available and is labelled a ceiling wherever it appears.

Coverage forcing is not a playing rule. It loses at every budget measured and loses more as the budget grows, $-0.083$, $-0.104$ and $-0.250$ at eight, sixteen and thirty-two playouts, because it spends the budget on breadth and never refines an estimate. It stays available for the soft-target collector, which needs support on every candidate for a different reason.

The exploration weight stays at 1.5, which wins at every budget measured. It is not an exploration dial: PUCT scales its bonus by the prior, this prior sits on about 2.3 of some 27 legal moves, and raising the weight therefore narrows the search rather than widening it.

Every search harness seeds its sampling per matchup and stamps the seed, and no suite verdict is quoted from a single run.

## The evidence

`search_objective.py` for the objective, 48 paired cells per margin. `search_leakage.py` for the dice, three arms of 96 episodes paired per cell: 0.927 with the live dice, 0.604 with the same battlefield and independent dice, 0.573 with neither, which prices the dice at $+0.323$ and the battlefield at $+0.031$. `search_strength.py` for the budget, the exploration weight and coverage forcing, 96 battles a cell, with the candidates each cell actually visited reported beside the rate. Five seeded repeats of one configuration on the held-out pool for the run-to-run spread, 0.656 through 0.725 at a standard deviation of 0.031.

## What this does not claim

None of it is a training result. Nothing was retrained for any of these numbers, and the network they wrap reads 0.512 on the held-out pool unaided.

The adopted objective is the better of two families on one suite, not an optimum, and the exploration weight is read off a grid on the mirror suite alone. The budget saturates rather than keeps paying: the mirror grid reads 0.281, 0.458, 0.594, 0.740 and 0.698 at zero, eight, sixteen, thirty-two and sixty-four playouts under one seeding, and the held-out ladder reads 0.594, 0.700, 0.750 and 0.744 at four, eight, sixteen and thirty-two playouts. Sixty-four buys nothing measurable over thirty-two and may cost a little. Sixteen to thirty-two was read as the deployment range on that evidence.

A direct budget sweep on 2026-08-12 contradicts the saturation that reading rests on. On the mirror suites with honest dice over three seeds, PUCT improves monotonically at 0.597, 0.620 and 0.671 for thirty-two, sixty-four and a hundred and twenty-eight playouts, with the strength margin running $+0.095$ to $+0.147$. The range therefore stands as a cost recommendation rather than as a ceiling, and [[../archive/experiments/2026-08-11-distillation-budget-and-checkpoint-selection|the budget log]] carries the sweep.

The claim that the dice matter is a claim about what a number means, not about the agent. A search given a perfect model including the randomness is a legitimate and useful diagnostic, and the ceiling it measures says this network's weakness lies more in evaluating positions than in the moves it can reach.

## What the offset costs beyond unknown dice, found 2026-08-23

The offset buys honesty at a price this record priced as one thing and which is really two. Not knowing the rolls is the intended cost. The second was invisible until it was looked for: the replay is a resampled trajectory, so different rolls kill different units and the replayed position drifts from the live one, and a candidate legal in the live battle may not exist there at all.

Measured over 256 decisions on six mirror matchups at three seeds (`agent_play/experiments/replay_divergence.py`), 21.6 percent of live-legal candidates were un-appliable at the replayed position, 0 percent at the first decision rising to 36 percent by depth twelve. Seven decisions found the resampled battle already over, and fifty-four found a different unit acting. The shared-dice control reads exactly 0.0 percent, which is both the design's own claim and the check that the harness measures what it says.

`rollout` plays such a candidate regardless. The engine answers an illegal selection by skipping the acting unit's turn, so the playout measures a different action and credits its value to the candidate. Roughly a fifth of the Q values this record's measurements rest on are therefore values of something else, and the share grows with depth, so late decisions carry the most of it.

The obvious repair is to decline to score a candidate the replay cannot offer, and it is measured worse. Paired over three seeds on the mirror suites, excluding those candidates costs $-0.139$ win rate, $-0.267$ mean reward and $-0.128$ strength margin, every seed negative on every column, at $t$ of 3.8, 3.3 and 2.8 against the 4.30 two degrees of freedom demand. The mechanism is mechanical rather than subtle: un-appliable candidates are 19.2 percent of the legal set but the search chooses one on 30.4 percent of decisions, and the live battle can play them perfectly well, so excluding them forbids the move search wants about a third of the time. A noisy value for every option beats an exact value for two thirds of them.

So the substitution stays the default and `exclude_unappliable` keeps the arm reproducible. What is genuinely repaired is the narrower case where the replayed battle has already ended, 7 of 256 decisions: there is no position to evaluate at all, every candidate used to receive the same terminal, and the tie-break fell through to the lowest legal action index. That now falls back to the prior's own pick.

None of this restates the $+0.323$ dice price. Both configurations are what they were, and the honest arm's deficit is still the missing information plus whatever this approximation costs, which the exclusion arm shows cannot be recovered by refusing to approximate.

## Costs

Independent dice cost real strength, and that is the point rather than a side effect: the honest configuration reads about 0.32 lower on the mirror suite than the one that had been in use. Every search figure recorded before 2026-08-10 was produced under some combination of the wrong objective, the wrong battlefield and the live dice, so none of them is comparable to a number produced after it, and the archive keeps them with that stated rather than restating them.

Seeding per matchup makes a run reproducible and makes repeats necessary. A suite verdict now costs three or five runs rather than one.

<!-- verify
exists  python/fheroes2_agent/search.py
grep    python/fheroes2_agent/search.py :: def sync_side_environment
grep    src/fheroes2/agent/agent_scenario.h :: combatSeedOffset
grep    agent_play/experiments/search_agent_battery.py :: search-combat-offset
grep    agent_play/experiments/search_agent_battery.py :: manual_seed
exists  agent_play/experiments/search_objective.py
exists  agent_play/experiments/search_leakage.py
exists  agent_play/experiments/search_strength.py
exists  agent_play/experiments/replay_divergence.py
-->
