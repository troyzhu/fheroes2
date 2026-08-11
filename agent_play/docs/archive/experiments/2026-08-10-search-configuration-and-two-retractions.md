---
title: "What search was actually configured to do, and two retractions, 2026-08-10"
type: experiment-log
updated: 2026-08-10
tags: [agent-env, archive, experiment, search, reward, retraction]
---

# What search was actually configured to do, and two retractions, 2026-08-10

Superseded within this log:

| Claim, where it first appears | What corrected it |
|---|---|
| The desynced side environment costs between 0.12 and 0.62 win rate, and terrain is the reason | The three-arm ablation: terrain alone is worth $+0.031$ and the shared dice are worth $+0.323$, so that measurement was varying both and mostly reading the dice |
| The searching agent is at or above the built-in AI on all nine suites | The same ablation: that sweep gave search the live battle's dice. Re-measured honestly it ties the engine everywhere, six of nine nominally above and none separated by two standard errors |
| The leaky teacher's labels are worth $-0.48$ less than an honest teacher's | The winner's-curse control: a second honest teacher scores $-0.5721$ against the first's $-0.5724$, a paired difference of $-0.0003$, so the entire signal was the estimator scoring its own argmax |
| The agent ties the engine on held-out at 0.619 | The harness was unseeded: five seeded repeats read 0.695 with a standard deviation of 0.031, and the 0.619 sits below the whole range |

Three configuration choices decided what every search number on this project meant, and none of the three had been made. The day started from the owner's question about the reward's win bonus and ended with a retracted headline, because the same reading that answered the reward question exposed what the search harness had been doing.

## The reward's win bonus, which was the owner's question

The owner asked whether $1 + \text{fraction remaining}$ overweights winning, on the reasoning that a side with force left has already won. The premise holds: forty sampled battles gave forty cases of surviving and winning and none of surviving and losing, the exceptions being the two unfinished terminations at 0.473 percent of training episodes. The algebra says more. The two-sided form's loss branch is already $-f^{\text{foe}}$, so only the win branch ever carried a bonus, and removing it gives the strength margin with $r^{\text{balanced}} = r^{\text{two-sided}} - \mathbb{1}[\text{won}]$ holding on every termination.

Two properties follow and one measurement does not. The balanced form is exactly zero sum on a decided battle where the current one pays the chairs a combined $+1$, and the outcome bit falls from 95 to 82 percent of the reward's variance, a 4.8-fold cut in its squared weight. Trained, it is a null: three paired seeds per objective from one anchor on one matchup set average $-0.0008$ across nine suites, with held-out moving $+0.025$ on all three seeds and four other suites moving the other way. Adopt it for its properties, not as a gain.

## What search maximizes, which nobody had chosen

`rollout` returns the side environment's terminal reward, so that environment's `reward_margin` is the quantity root search maximizes. The battery passed `two_sided`, `capture_replay.py` passed nothing and inherited `hit_points`, and the same checkpoint on the same armies and battlefield read 0.00 from one loop and 0.50 from the other. Over the mirror suite from both chairs, 48 paired cells an arm, the survival-graded margins beat the destruction-graded ones: `hit_points` $+0.191$ and `strength` $+0.174$ against `two_sided`, both at a paired standard error of 0.044. The families split by their loss branch, which is most of what a search sees because most playouts end in defeat.

## The side environment, which was on the wrong battlefield and then knew too much

`rollout` replays the action prefix into the side environment, and that replay reproduces the live position only when both sit on the same world seed. Every harness built its side environment with the live `seeds`, and because `reset` respawns the worker whenever a decision is pending, it returned to variant zero on every rollout while the live episode rotated over four. That is a genuine defect and `BattleEnv.current_battlefield` plus `sync_side_environment` fix it.

Fixing it bought something nobody asked for. The combat seed is computed from the tile index, the map seed and the armies, all of which the world seed fixes, so a pinned side environment inherits the live battle's exact dice and search stops estimating what a move is worth and starts checking what it will do. `combatSeedOffset` was added to the scenario, defaulting to zero so every golden digest stays bit-identical, purely to separate the two. Three arms on the mirror suite, 96 episodes each, paired per cell: with the live dice 0.927, with the same battlefield and independent dice 0.604, with neither 0.573. Knowing the dice is worth $+0.323$ and the battlefield is worth $+0.031$.

That retracts the day's headline. The nine-suite sweep that put the agent at or above the engine everywhere had the live dice. Re-measured with independent dice and a paired unsearched control, six of nine suites land nominally above the engine and not one is separated by two standard errors, so the honest verdict is a tie: held-out 0.619 against 0.660, the mirrors 0.500 and 0.583 against 0.361 and 0.639, the ladder 1.000 against 0.969. What survives is the search's own contribution, $+0.132$ averaged over the suites against the same weights without it, separated at four of them and reaching $+0.354$ on the mirrors from the attacking chair. A network well below the engine at 0.512 held-out, lifted by search to roughly engine level.

The shared-dice arm keeps a use. It is a ceiling, and at 0.927 against a 0.5 baseline it says this policy's weakness is largely in evaluating positions rather than in the moves it can reach. It is not a number to set beside an opponent that gets no model at all.

## The teacher's labels, where a control changed the answer

Every corpus this project has distilled was labelled by a search whose side environment shared the live dice, so the natural next question was whether the corpus is compromised. Asked at identical states, the leaky and honest teachers agree on 26.8 percent of moves and the honest teacher scores the leaky pick $-0.5724$ against its own. Both numbers look damning and neither means anything without a control, because a search picks the argmax of its own estimates and everything else it scored is below that by construction.

The control is a second honest teacher differing only in its dice. It agrees with the first on 35.1 percent of moves and scores its pick $-0.5721$. The paired difference in value is $-0.0003$ at a standard error of 0.0324, so leakage is not detectable in the labels at all and the $-0.48$ was pure winner's curse. Agreement does fall $-0.084$ beyond the noise floor, so knowing the dice does shift the choice, but the effect is small beside what the control exposes.

What the control exposes is the real finding, and it sharpens at the budget the corpus was actually collected under. At 16 simulations two identically informed searches agree on 35.1 percent of moves. At 48, the collection budget, agreement rises to 64.2 percent, which looks like the problem solving itself until the rise is decomposed: search leaves the policy prior far less often at the larger budget, 23.2 percent of decisions against 65.9, and on those decisions agreement is flat at 25.6 percent against 28.3.

So the labels that carry information are exactly the ones more budget does not stabilise. Roughly three in four of the decisions where search overruled the prior are not reproduced by a search that knows the same things. That is a better explanation for students plateauing below their teacher than anything else measured here.

It licenses less than it looks, and the tension with the 2026-08-09 regret weighting, which beat its unweighted twin by $+0.063$ held-out on all three seeds, has to be resolved rather than argued. An individual label being unreliable does not make the aggregate worthless if the noise is unbiased.

`agent_play/experiments/label_value.py` settles it causally and without asking any search to grade itself, which is what made the first comparison uninterpretable. At a state where search overruled the policy's argmax, the battle is replayed to termination ten times from each branch on independently drawn combat streams, so the comparison is paired on the position and on the dice. Over 36 overrules at the collection budget the search action wins 0.509 against the policy action's 0.393, an advantage of $+0.116$ at a standard error of 0.046, nineteen better against five worse and twelve equal. The labels are near coin flips individually and genuinely valuable in aggregate, which is exactly the shape unbiased noise produces and exactly what the regret weighting was picking up.

One reading looked directional and did not survive its own test, which is why it is written here rather than as a recommendation. Twenty-two of the thirty-six overrules had a visit margin of one, meaning search explored the two candidates equally and the value tie-break decided, and those carry only $+0.057$ against roughly $+0.20$ for the fourteen with a wider margin. In the corpus itself those tie-breaks are 47 percent of the informative rows, so dropping them is a large and cheap intervention if the per-label difference transfers.

It does not. `soft_distill.py` gained a `--min-visit-margin` filter and three paired seeds were trained with and without it, on identical corpora at identical soft mass with the regret weighting on in both arms. Across nine suites the filtered arm reads $-0.012$, not separated from zero against an across-suite standard error of 0.016, improving four suites and losing five, with the Thunk ladder paying $-0.125$ on all three seeds. So the label-level advantage does not survive into the student: dropping the noisy half loses more in data than it gains in label quality, and the rank-transformed regret weighting was already handling what there is to handle. The filter is measured and not adopted.

## The harness was never seeded, which is why two of its numbers disagreed

Redoing the simulation ladder honestly produced a number that would not reconcile. The nine-suite sweep read 0.619 on held-out at sixteen playouts and the standalone ladder read 0.750 at the same budget in the same configuration, a gap of 0.131 that no error bar on either report covered. `search_agent_battery.py` never called `torch.manual_seed`, so the policy sampling inside every playout and every played move started from whatever state the process had inherited, and the standard errors it printed described spread across matchups within one run rather than the run itself.

Seeded per matchup, so a suite no longer depends on which suites preceded it, five repeats of one configuration on held-out read 0.656, 0.669, 0.706, 0.719 and 0.725: a mean of 0.695, a standard deviation of 0.031 and a standard error of 0.014. Both unseeded readings sit outside that range, which is the sense in which they were never comparable.

The first seeding was itself defective and the defect is instructive. The per-matchup salt was `hash(suite)`, and Python randomizes string hashing per process, so each run drew a different stream and nothing was reproducible; it surfaced when one configuration re-read 0.750 where the same seed had read 0.675. `zlib.crc32` replaces it. Everything measured under the unstable salt stands as independent sampling, which is exactly what the spread estimate needed, and none of it stands as a repeatable configuration, which is what it was labelled as.

The correction changes the day's verdict again, in the other direction. Against the engine's 0.660 on that suite the seeded mean is $+0.035$, about two and a half between-run standard errors, so the agent is nominally above the engine there rather than at the parity a single sweep reported. The suite is a fixed twenty matchups and the other eight have not been re-run seeded, so this is one suite's answer and not a scoreboard. The seeded sweep is running.

## What actually buys search strength

With the dice question settled, `agent_play/experiments/search_strength.py` measures the three knobs against each other on the mirror suite from both chairs, 96 battles a cell, where an engine-equivalent player scores 0.500 by construction. Budget pays and then stops: 0.281 unsearched, then 0.458, 0.594, 0.740 and 0.698 at eight, sixteen, thirty-two and sixty-four playouts with the exploration weight at 1.5 and forcing off. The held-out ladder agrees, 0.594, 0.700, 0.750 and 0.744 across its first four rungs, so sixteen to thirty-two is the range and beyond it the curve is flat.

Coverage forcing loses at every budget and loses worse as the budget grows, $-0.083$, $-0.104$ and $-0.250$. The candidates-touched column says why: at thirty-two playouts forcing visits 25.3 of about 31 legal moves and plain PUCT visits 5.9, so forcing spends everything on breadth and never refines an estimate. It was built for the soft-target collector, which needs support everywhere, and it has never been a good playing rule.

The exploration weight is not an exploration dial here, which is worth stating because it reads like one. PUCT scales its bonus by the prior, and this prior is concentrated on about 2.3 effective actions, so raising the weight amplifies the already-favoured candidates: at eight playouts the search touches 3.7 candidates at 0.5, 3.1 at 1.5 and 2.2 at 4.0. The middle value wins at every budget anyway, and it is the value every harness already passes.

Two independent lines therefore point at dropping coverage forcing from collection as well as from play, since the label-noise measurement above says forcing at the collection budget gives every candidate a single playout and makes the argmax a coin flip. Collecting the same band again without it confirms the mechanism exactly: the mean visit margin rises from 0.58 to 2.80, and the share of informative labels resting on a real margin rather than a single-playout tie-break goes from 53 to 94 percent. The labels are much more confident, which is what the change was for.

The student does not net anything from it. Three paired seeds on the same hard base at matched soft mass read $-0.012$ across nine suites, not separated from zero, but the shape underneath is not a wash: the attacking mirror suite gains $+0.153$ on all three seeds and the Thunk ladder pays $-0.208$ on all three. That is the axis trade this project has seen from every label change since the first generation, now with a mechanism attached rather than a shrug. The comparison is a recipe comparison and not an isolation, because the two corpora also differ in which matchups the band screen kept, in label count, 3,104 against 5,143, and in their dice, so what is established is that confident labels alone do not lift the student, not which of those differences did what.

## The scoreboard, and which question it answers

Three seeded runs of all nine suites with every correction applied close the day. Agent against the engine, means over the seeds: held-out 0.700 against 0.660, held-out defending 0.398 against 0.338, the mirrors 0.542 and 0.688 against 0.361 and 0.639, the ladder 0.979 against 0.969, real maps 0.569 against 0.568, commanders 0.979 against 0.958, hordes 0.200 against 0.192, fresh samples 0.391 against 0.446. Eight of nine at or above, and 0.605 against 0.570 averaged over suites. The unsearched network reads 0.512 on held-out, so search is the entire margin.

Re-run with the full column block, three seeds and a paired unsearched arm, the rates hold and the rest of the block says what the rate cannot. Win quality beats the engine's on held-out, 0.48 against 0.45, and loss quality does too, 0.66 against 0.63, so the agent both wins more cleanly and loses less badly on the flagship suite. The Thunk ladder is the sharpest case: the rate ties the engine at 0.969, and the rungs read 1.00, 1.00, 0.96 and 0.92 against the unsearched network's 1.00, 1.00, 0.67 and 0.46, so search is carrying the two hard rungs entirely and a single suite number hides it.

The entropy columns describe a very concentrated network. Raw entropy sits between 0.50 and 1.06 nats, 0.16 to 0.29 normalized, and 2.0 to 3.8 effective actions against 26 to 45 legal ones. The search's own visit entropy is the more useful of the two: it reads 0.09 on the commander stress suite and 0.18 on the ladder, where the position resolves, and 1.44 on fresh samples and 2.03 on hordes, where it does not. Those are exactly the two suites where the agent fails to clear the engine, which is a lead rather than a finding.

The real-map suite deserves a sharper verdict than "saturated", because the owner's reading of why is right. Priced by the engine's own creature strength, 17 of its 24 fights are lopsided before anyone moves: eight sit below a 0.5 attacker-to-defender ratio, three of them under 0.35, and nine sit above 2.0, reaching 11.5, 27.9 and 40.7 on Betrayal. These are opening fights harvested from shipped maps by distance from the hero, so a guard stack forty times the hero's strength is exactly what the map intends, and no policy is meant to win it. Only seven fights price as contested and six of those seven are still saturated in play, which leaves one fight in twenty-four able to separate anything.

Screening the existing candidates will not fix it either: the harvest found 161 candidates and only 30 survived deduplication, so the suite already holds 24 of the 30 that exist. Rebuilding it means harvesting a wider distance band and screening for contest rather than proximity.

Four of the nine suites cannot separate two players at all, which the eight-of-nine headline was quietly drawing on. Counting matchups where every arm and the engine sit at the identical extreme, `real_maps` is 22 of 24 saturated, `stress_hordes` 4 of 5, `stress_commanders` 3 of 4 and the ladder 2 of 4. `real_maps` reading 0.569 against 0.568 and 0.568 is a near-constant rather than a finding, on battles averaging 4.6 decisions. Restricted to the five suites that can move, the agent is above the engine on four, held-out by $+0.025$, held-out defending by $+0.033$, the mirrors by $+0.188$ and $+0.062$, and below on fresh samples by $-0.062$. That is the number worth quoting, and the ladder still belongs in the record by rung rather than by mean.

The verdict depends on which error bar is the right one, and both belong in any quotation of this table. Against the spread across seeds, which is the denominator for asking whether the agent is better on this fixed benchmark, it is above on five suites, below on fresh samples, and tied on three. Against the spread across matchups, the denominator for asking whether it would beat the engine on fresh matchups drawn the same way, nothing separates at all: the per-matchup standard error runs 0.066 to 0.135 on the suites that carry the result and swamps deltas of 0.04 to 0.18.

The benchmark is beaten and generalisation is not established, which is a narrower claim than the day's first version and a wider one than its second. The detail that decides how much to like it is that `fresh_sampled`, the suite that most resembles a fresh draw from the generator, is the one suite where the agent is clearly worse.

## The action rule and the search, which had never been crossed

Two things were wrong with how searched arms were reported, and the owner named both. The battery collected six columns and aggregated a hardcoded list of the same six, so the entropy, support, rounds and reward-split columns `scenarios.measure` had carried since 2026-08-09 never reached a searched report and a searched arm could only ever be compared to a policy arm on the rate. The aggregation now walks whatever numeric keys the episodes produce, entropy is reported raw in nats and normalized against the uniform maximum over the state's legal set, and a searched arm additionally reports the entropy of its own visit distribution, which says whether the budget resolved anything rather than how decided the network was. `report_summary.py` reads both report shapes, so one printer serves both harnesses.

The deployment rule had been a flag on the policy battery alone, so every searched number ever recorded was implicitly sampled, even though the rule governs the search prior as well as the played move. Crossed on one seed over three suites, it produces an interaction worth having: the rule matters far less under search than without it, 0.662, 0.675 and 0.669 on held-out against the bare policy's 0.550, 0.600 and 0.575, because search absorbs the difference. And greedy is the worst rule to pair with search, not the best. It sharpens the prior, the search's own visit entropy falls from 0.55 to 0.36 on held-out and from 0.76 to 0.66 on the attacking mirrors, and search's contribution falls with it, from $+0.112$ to $+0.075$ and from $+0.292$ to $+0.167$. A sharper prior starves the thing that was supposed to correct it.

The network is concentrated whatever the rule: entropy sits between 0.55 and 0.93 nats, 0.18 to 0.29 normalized, and 2.2 to 3.2 effective actions against 28 to 32 legal ones. That is the same concentration the exploration-weight measurement ran into from the other side, and it is one seed, so the ordering is suggestive rather than settled.

## Flying, and the incentive the owner found under it

`flying_v1` opened the way `wide_v1` did, because the action space was already generic over flight: moves come from the arena's own pathfinder and every candidate is re-checked by the engine's move validator, both of which handle fliers natively, so the only thing keeping them out was a roster gate. `flyingV1Supported` is simple_v1's criteria with the flight exclusion lifted and nothing else, `allowFlyingUnits` defaults to false, and `--allow-flying` reaches both the worker and the replay binary. The roster goes from 41 creatures to 47, adding Sprite, Gargoyle, Vampire, Vampire Lord, Ghost and Genie. Verified four ways: the gate rejects and accepts correctly, a flying scenario is deterministic across runs, `verify_m1` passes so every golden digest is unchanged with the flag off, and fliers receive real flight reachability rather than a seat at the table, 87 legal actions against a walker's 20 to 27 in the same position.

The evasion demo then reran with Sprites and terminated exactly as the walking version does, at round forty with none looping, which is what the deferral had been waiting on and which I reported as the pricing holding. The owner's response is the finding: a terminated exploit is not a disincentivized one. The measurement is in [[../../rl/reward-design]] and it is unambiguous. Evasion pays the defender the maximum reward the objective can produce and beats fighting by the entire range, so any defender that can kite has a strictly dominant strategy that never engages, and fast walkers reach it too. Flying raises its reachability rather than creating it, `flying_v1` is therefore fit for evaluation and not for training, and the correction is the owner's to choose.

## Housekeeping the day also owed

The root-search primitives and the evaluation suites moved into `python/fheroes2_agent/search.py` and `python/fheroes2_agent/suites.py`, where nine scripts had been reaching for them across a path insert. Both original scripts re-export every name so archived command lines keep working, and the suites were compared matchup by matchup across the move.

Five smaller defects came out of re-deriving the carried-over figures after a context compaction, each recorded where it belongs: the balanced reward bypassing `_side_won` and so paying a stalling attacker at the round cap, `capture_replay.py` reading `--want` against the engine's attacker-perspective termination and recording losses as wins, the same script searching under a different exploration rule and a different objective than any measurement used, `win_rate` and the reward disagreeing about who wins a stalemate, and my own misreport of the search suites as 32 battles per matchup when `episodes` is the total and `seeds` only rotates the battlefield.

<!-- verify
exists  agent_play/experiments/search_objective.py
exists  agent_play/experiments/search_leakage.py
exists  agent_play/experiments/teacher_leakage.py
exists  agent_play/experiments/search_strength.py
exists  python/fheroes2_agent/search.py
exists  python/fheroes2_agent/suites.py
grep    src/fheroes2/agent/agent_scenario.h :: combatSeedOffset
grep    python/fheroes2_agent/env.py :: def current_battlefield
grep    python/fheroes2_agent/env.py :: def terminal_reward_balanced
-->
