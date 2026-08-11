---
title: "Value estimation in practice: the lab record"
type: study
updated: 2026-08-06
related_concepts: ["[[../overview#Notation]]", "[[rl-methods]]", "[[training-design]]", "[[../research/works/bcq-extrapolation]]", "[[../research/works/alphastar-unplugged]]"]
tags: [agent-env, rl, value-estimation, study]
---

# Value estimation in practice: the lab record

The owner asked that the value-based thread serve the project's educational goal as much as its win rates, so this page collects every value estimator this project has fitted, what each one measured, and which concept from the literature each measurement is a first-hand instance of. Symbols follow [[../overview#Notation]]; every number here comes from a run vendored under the archive, chiefly `files/2026-08-06-run-reports/`.

The through-line is one question asked five ways: can a function fitted from recorded play judge positions or moves it did not record. Each row sharpens the answer, and the last column is the general lesson the measurement grounds.

| Estimator | Fitted on | Judged where | Result | The concept it demonstrates |
|---|---|---|---|---|
| $V_\phi$ head on the frozen imitation trunk | teacher returns, discounted margin target | teacher holdout | explained variance 0.30 | a behavior value fits its own distribution |
| same | same | student-played states | $-0.13$, optimistic bias $+0.32$ | extrapolation error, [[../research/works/bcq-extrapolation]] |
| same head | undiscounted $\pm 1$ outcomes | teacher holdout | 0.09 to about 0.19 | target design: discounting and margin-mixing pollute a position judgment, [[../research/works/alphazero]] uses the plain outcome |
| dedicated value network, own trunk | same outcomes, narrow-army corpora | narrow holdout, then wide-army successor ranking | 0.61 on its holdout, 0.00 as a search guide | the frozen trunk was the capacity bind, [[../research/works/muzero]] co-trains; and coverage binds next |
| same architecture | full 270,335-decision bestiary corpus | its own holdout | 0.856, the best value this project has fitted | high state-level accuracy is achievable here |
| that 0.856 value | as a one-ply search leaf | successor states one action apart | wins 0.00 where rollout search wins 1.00 | state-level explained variance does not rank moves |
| behavior $Q(s,a)$ | Monte-Carlo on taken actions | taken-action holdout | 0.853 | on-support, $Q$ fits as well as $V$ |
| same $Q$ | re-ranking the prior's top five | actions the teacher mostly did not take | held-out 0.512 falls to 0.263, Thunk 0.75 to 0.17 | extrapolation error at move level: the argmax hunts the estimator's errors, exactly [[../research/works/bcq-extrapolation]]'s mechanism and the reason [[../research/works/one-step-offline-rl]] constrains improvement to support |

Two rows deserve their stories told rather than tabulated. The dedicated network's first search test read 1.000 on the Thunk fight and was retracted within the hour: the probe module retrained at import time with an unseeded initialization and overwrote its checkpoint on every run, so the winning weights were one draw among many and are gone, and every reproducible draw reads 0.000. The lesson is procedural rather than statistical, and it is the same one the checkpoint-level seed replication taught the night before: an unseeded fit that overwrites its artifact is a lottery ticket, and a result that cannot be re-drawn is not a result. [[training-design#Search leaves and the value question, measured 2026-08-06]] carries the full account.

The $Q$ row is the arc's conclusion. A state value cannot rank moves because it holds no counterfactuals at all, and a behavior $Q$ cannot either, because its counterfactuals are exactly the actions the data never took: 0.853 on the actions it saw, ruin one action to the side. What both are missing is the same thing, evaluations of candidates that were actually tried, and that is precisely what rollout search manufactures, a real playout per candidate. This is why rollout-scored search is the one improvement operator this project has measured above the teacher, and why the literature's route to a usable learned evaluator, [[../research/works/muzero]]'s targets and [[../research/works/mcts-regularized-policy-optimization]]'s solution distribution, trains on search returns, move-level values of candidates search genuinely explored. The project already records those in its collection reports (`search_teacher.py --record-candidates` stores per-candidate rollout values, visits and prior on every decision), so the next estimator's dataset exists before the estimator does.

The ninth row arrived on 2026-08-07 from the owner's proposal: train on search's own rollouts, many branch outcomes per state, instead of one played trajectory. The fit is the best this project has produced anywhere, 0.888 explained variance on held-out rollout values, move-level targets, support-safe by construction. And it still cannot choose: argmax over legal actions matches search's own decision on 0.105 of states, and playing greedily on it collapses.

The dataset explains both halves at once. UCB concentrates its thirty-two simulations on the few candidates the prior already likes, so the recording averages 2.0 visited candidates per state: the fit is superb on the visited support while the argmax ranks hundreds of never-visited actions, the same extrapolation mechanism as every row above, now at candidate granularity, and within-state ordering is exactly the residual the global variance number barely weighs. The prescription writes itself and is recorded rather than run: force one rollout per candidate before UCB concentrates, roughly the same simulation budget, an order more branch coverage per state (`rollout_value.json`).

What this page does not claim: none of this says value learning cannot work here, and the 0.856 bestiary value is kept as the project's strongest behavior value, re-verified from the vendored checkpoint at 0.8565 holdout explained variance on 2026-08-07 (`files/2026-08-07-run-reports/reverified_numbers.json`) after the honesty audit flagged the original session printout as unvendored. It says the order of operations the literature prescribes is now measured rather than taken on faith: coverage before capacity, targets before architecture, support before improvement, and search-generated counterfactuals before any leaf evaluator.
