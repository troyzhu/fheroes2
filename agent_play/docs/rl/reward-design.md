---
title: Reward design, the space and what is known about it
type: design
updated: 2026-08-07
related_concepts: ["[[../decisions/0005-training-and-reward]]", "[[rl-methods]]", "[[scenario-distribution]]", "[[rlhf-transfer]]"]
tags: [agent-env, reward, design]
---

# Reward design, the space and what is known about it

[[../decisions/0005-training-and-reward]] fixes the criteria and names four candidates. Four of those are implemented, the hit-point margin, its strength-priced variant, the owner's two-sided piecewise form, and the difficulty weighting, and the rest are a sentence each, which is thin for what is the single choice that decides what the agent is actually being asked to do. This page carries the design space, what the evidence says about each part of it, and what would have to be measured to choose.

Nothing here is decided. The record decides; this exists so the record has something to decide from.

## Table of contents
- [[#What is implemented, and how it behaves]]
- [[#The terminal family]]
- [[#The dense family]]
- [[#Potential-based shaping, which is the safe way to be dense]]
- [[#Two things that are not reward shape but are often mistaken for it]]
- [[#How a candidate would be chosen]]
- [[#The difficulty-weighted candidate, owner-directed 2026-08-05]]
- [[#The lexicographic alternative, from the owner-supplied guide]]
- [[#The win bonus, and what removing it costs, 2026-08-09]]
- [[#What search maximizes is a separate choice, 2026-08-09]]
- [[#Stalls, and the evasion exploit the owner predicted]]

## What is implemented, and how it behaves

One candidate, the margin-weighted terminal reward.

$$r_T = \underbrace{\pm 1}_{\text{outcome}} + \underbrace{\frac{h_T}{h_0}}_{\text{own force surviving}}, \qquad r_t = 0 \text{ for } t < T$$

where $h_0$ is own hit points at the first decision and $h_T$ at the end. Clean win 2.0, pyrrhic win 1.1, cheap loss $-0.4$, rout $-1.0$. Every intermediate step is exactly zero.

Two things were learned by running it. The obvious form of the margin, $(h^{\text{own}} - h^{\text{foe}})/(h^{\text{own}} + h^{\text{foe}})$, is useless, because a decided battle almost always ends with the loser wiped out and the expression is then 1.0 whether the winner finished with fifty hit points or five. Measuring survival against one's own start instead is what makes the term carry information.

And the claim in [[scenario-distribution]] that a margin-weighted reward keeps hopeless matchups informative holds only for partial losses. In a matchup where the agent is always wiped out, the survival term is zero every time, the reward is $-1$ with no variance, and no reward shape rescues it. That is a statement about the scenario rather than about the reward.

The mirror case briefly looked like a second benefit, and better measurement took it away, which is kept here because the retraction is instructive. On 50 Peasants against 30, which the cloned policy wins every time, pure win-loss would produce a reward with no variance at all, and the margin-weighted form measured 0.166 because surviving force differed between episodes. That number belonged to the cloned policy. A policy trained on the matchup wins with nearly identical survival, the training-time spread averages 0.029 against an advantage floor of 0.1, and unfloored runs there spend 0.87 of their iterations amplifying without ever dipping. So the survival term keeps the spread nonzero rather than healthy, no protection against the collapse in [[training-design#An amplification in advantage normalization]] follows from it, and the term stands on the purpose it was chosen for, distinguishing a clean win from a pyrrhic one. The full correction is in [[../archive/experiments/2026-08-04-flip-and-collapse]].

## The terminal family

Everything here keeps the signal at the end of the episode, which is what makes it honest: the objective is the thing being optimized rather than a proxy for it.

| Candidate | Form | What it adds | What it costs |
|---|---|---|---|
| Win or lose | $\pm 1$ | Nothing to exploit, it is exactly the objective | One bit per episode, and no gradient at all in a matchup the agent always loses |
| Own survival (implemented) | $\pm 1 + h_T / h_0$ | Distinguishes a clean win from a pyrrhic one, and a cheap loss from a rout | The weighting is a modelling claim about what an army is worth |
| Value-weighted survival (implemented 2026-08-05, owner-directed) | $\pm 1 + \sum_i c_i n_i^T / \sum_i c_i n_i^0$ with $c_i$ the engine's creature strength | Weights each creature by what it is worth rather than what its hit points weigh, so losing a Champion costs a Champion; the same pricing the budget sampler and difficulty weight use | Engine-computed per side in the terminal record; opt-in as `reward_margin="strength"`, hit-points margin stays the default; training effect unmeasured, and its live consumer is search rollout scoring, where it makes root-PUCT prefer minimum-value-loss wins |
| Speed bonus | $\pm 1 + \alpha(1 - T / T_{\max})$ | Rewards finishing quickly, which matters because a battle is one episode in a longer game | Encourages risk-taking that a campaign would not want, and $\alpha$ is arbitrary |
| Opponent-relative | $\pm 1 + (\text{own survival} - \text{foe survival})$ | Rewards damage inflicted as well as damage avoided | Reintroduces the degeneracy above, since the foe's survival is almost always zero |
| Two-sided piecewise (implemented 2026-08-07, owner-directed) | win: $+1 + \text{own kept}$; loss: $-1 + \text{foe destroyed}$, strength-priced | The owner's form: losses graded by the damage they dealt, since surviving a loss often means having fled while damage dealt measures having fought; dodges both recorded objections, the win branch never reads the foe and the credit is terminal-only | Opt-in as `reward_margin="two_sided"`; composes with difficulty weighting, sign split intact since every loss stays at or below zero; training effect unmeasured |
| Balanced margin (implemented 2026-08-09, owner-directed) | win: $f^{\text{own}}$; loss: $-f^{\text{foe}}$, strength-priced | Exactly the two-sided form with the flat win bonus removed, so it is the strength margin $f^{\text{own}} - f^{\text{foe}}$; the outcome bit falls from 95 to 82 percent of reward variance, a 4.8-fold cut in its squared weight, and the two chairs become exactly zero sum on any decided battle | Opt-in as `reward_margin="balanced"`; the outcome branch still comes from `_side_won` so the stalemate and `round_limit` resolutions are unchanged; training effect measured by a paired three-seed run, see [[#The win bonus, and what removing it costs, 2026-08-09]] |

Value-weighted survival is the one worth measuring next. Hit points are a poor proxy for what a stack is worth, because a Peasant has one hit point and a Master Swordsman thirty, so the current reward already weights by hit points implicitly, and creature cost is the game's own answer to the same question.

## The dense family

Everything here pays out during the episode. The reason to want it is credit assignment: with a terminal-only signal, a twenty-decision battle attributes one number to twenty decisions, and the leave-one-out baseline in [[rlhf-transfer]] makes that worse by spreading one advantage evenly across all of them.

The reason to distrust it is that it teaches the proxy. A reward for damage dealt will trade a stack to deal damage when retreating was correct, and no amount of tuning removes that, because the agent is correctly optimizing what it was given.

The owner's 2026-08-06 critique states the credit-assignment cost of the terminal family exactly: a trajectory-level advantage, which is what the group-relative estimators spread over every decision, upweights a bad action that later good actions redeemed and downweights a good action inside a lost game, so the policy never receives the per-decision signal that would correct the bad action itself. [[../research/works/gae]] is the canonical dial for this trade, and its low-bias end pays in exactly the currency our critic measurement says is counterfeit off-distribution.

Rather than deciding from argument, `credit_assignment.py` measured the two error rates the critique names, using root-PUCT rollout values as per-decision ground truth on mid-band matchups. The asymmetry is loud: among decisions trajectory credit reinforces, only 1.4 percent are search-bad (1 of 74), while among decisions it punishes, 89.7 percent are search-good (61 of 68), actions within a tenth of the state's best. In this domain, at this policy, the dominant mis-signing is good play blamed for doomed episodes, not bad play redeemed, and a group baseline does not remove it, since losing episodes of a mixed matchup still carry negative advantage onto near-optimal moves. Small print: the punished sample comes essentially from one matchup whose outcomes varied, and 32-simulation values carry their own noise.

What follows from the measurement: per-decision labels (search distillation, or one-step advantage weighting on teacher states per [[../research/works/one-step-offline-rl]] and [[../research/works/alphastar-unplugged]]) fix the failure that actually occurs here. And hopeless matchups are poison for trajectory credit specifically because they consist entirely of punished-but-good decisions, which the struggle-targeted round's harm already showed behaviorally.

| Candidate | Form | Failure it invites |
|---|---|---|
| Damage dealt | $+\lambda \cdot \text{damage inflicted}$ | Suicidal attacks, since damage inflicted is paid immediately and the loss arrives at the end |
| Damage differential | $\lambda(\text{dealt} - \text{taken})$ | Better, but still rewards trading evenly when disengaging wins |
| Army-strength delta | $\lambda(\Phi_t - \Phi_{t-1})$ for some strength estimate | This is potential-based if written as a difference, which moves it to the next section |
| Kill bonus | $+\lambda$ per stack destroyed | Focus-firing a nearly dead stack over a more dangerous target |
| Positional | proximity, threat coverage, being shot at | Encodes a tactical theory that may be wrong, and is the hardest to audit |

The pattern is that every dense candidate is a hypothesis about what good play looks like, and the agent will exploit the gap between the hypothesis and the truth. That is why [[../decisions/0005-training-and-reward]] puts potential-based shaping ahead of all of them.

## Potential-based shaping, which is the safe way to be dense

A shaping term of the restricted form

$$F(s, a, s') = \gamma\,\Phi(s') - \Phi(s)$$

telescopes along any trajectory, so it adds a constant depending only on the start state and cannot change which policy is optimal (Ng, Harada and Russell, 1999). [[rl-methods]] carries the derivation. This buys density with a proof attached, which no other dense candidate has.

What it does not buy is usefulness. A poor $\Phi$ is harmless and worthless: correctness is guaranteed, informativeness is not. So the design question is entirely about which potential to use.

| Potential | Rationale | Concern |
|---|---|---|
| $\Phi = $ own hit points minus foe hit points | The simplest strength estimate, and directly available | Ignores position entirely, so it says nothing about the half of the game that is movement |
| $\Phi = $ value-weighted own force minus foe force | As above with creature costs | Same, plus the cost table question |
| $\Phi = $ the engine's own `engine_strength` | The built-in AI's own evaluator, already computed | Only in the `full_v1` profile, so a policy shaped by it cannot be deployed on `observable_v1`, which is the asymmetry [[../decisions/0001-observation-profiles]] warns about |
| $\Phi = V^{\pi^{*}}$, the fitted teacher value | Principled: shaping by a value function is the ideal case, since it makes the advantage the true one | The first fixture-era fit explained 0.835 of held-out return variance on an episode split, a number later measurement retracted in generality: on the current corpus the same construction reads 0.302 on teacher play and $-0.131$ on student-visited states ([[training-design#The behavior value, measured where it would be spent]]), so the remaining question is whether shaping by it beats using it as a baseline |

The last row is the interesting one. Potential-based shaping with $\Phi$ equal to the true value function makes every step's shaped reward equal to the advantage, which is the densest possible correct signal. Pre-fitting a critic on teacher play, which [[training-design]] proposed for a different reason, is most of the work, and as of 2026-08-03 that fit exists and explains 0.835 of held-out return variance.

What the fit does not settle is whether shaping by it is worth anything over using it as a baseline, which is what the critic already does. Both subtract the same quantity; shaping moves it inside the reward while a baseline keeps it outside. The measurement that would separate them has not been run.

## The difficulty-weighted candidate, owner-directed 2026-08-05

The owner directed a candidate the terminal family lacked: weight the reward by fight difficulty, the opponent-to-own strength ratio, so an easy victory stops paying full price and a lopsided loss stops costing one. The implemented form (`difficulty_weight` and `apply_difficulty` in `python/fheroes2_agent/env.py`, opt-in through `reward_weighting="difficulty"`) prices both starting armies by the engine's own creature strength, the same pricing the value-budget sampler uses, takes $d$ as the enemy-to-own ratio clipped into $[1/4, 4]$, and sets $w = d^{1/2}$, so $w$ spans $[0.5, 2]$. Wins multiply by $w$ and losses by $1/w$: a hard fight pays double for winning and forgives half of losing, an easy fight pays half and punishes double. The sign split is well defined because the margin-weighted reward never lands strictly between 0 and 1. Commander stat bonuses are not priced into $d$, a known simplification, since strength tables are per-creature.

Where the weighting can actually bite is narrower than it looks, and the analysis matters more than the form. Within one matchup, $w$ is a constant, so a group-relative baseline subtracts most of it and GRPO's standard-deviation normalization cancels a pure scaling entirely; a critic likewise absorbs any per-matchup constant into $V(s_0)$. What survives normalization is the asymmetry: in a mixed group the win-loss gap becomes $w + 1/w \ge 2$ rather than 2, reshaped toward whichever outcome the difficulty makes informative, and across matchups in a plain PPO batch the weighting reallocates gradient from easy to hard fights directly. So the candidate is a hypothesis about cross-matchup credit allocation, not about within-matchup learning, and a null result under group baselines would itself be informative.

Both arms of the measurement share every other choice; the run and its numbers live in [[../archive/experiments/2026-08-05-planner-reward-battlefields]], and the criteria in [[../decisions/0005-training-and-reward]] still decide adoption.

## The lexicographic alternative, from the owner-supplied guide

Every candidate above scalarizes winning and losses into one number, and the guide vendored beside [[../research/works/generalized-battle-agent-guide]] argues the scalarization itself is the hazard: with $R = \mathbf{1}\{\text{win}\} - \lambda \cdot \text{losses}$, a large $\lambda$ prefers a low-casualty defeat to a costly necessary win, and no single $\lambda$ is right across matchups. Its alternative is lexicographic, maximize win probability first, then minimize conditional losses among actions within a tolerance of the best, served by two value heads, $V_{\text{win}}$ and $V_{\text{loss}}$, with the tolerance calibrated on held-out games.

The margin-weighted reward this project runs is a mild scalarization, the $\pm 1$ dominating a survival term bounded by one, so a pyrrhic win at 1.1 always beats a cheap loss at $-0.4$ and the guide's pathology cannot invert an outcome here. What the lexicographic form would add is cleaner priority once losses matter strategically, at the price of a second head and an inference-time rule. It joins the candidate table as documented rather than measured; the criteria in [[../decisions/0005-training-and-reward]] still decide.

## Two things that are not reward shape but are often mistaken for it

The discount. $\gamma$ belongs to the objective alongside the reward, and with a terminal-only signal it controls how much a long battle is penalized relative to a short one. At $\gamma = 1$ a win is worth the same however long it took. This is a modelling choice being made implicitly by a default.

The scenario distribution. A reward can only carry information where the outcome varies, so a matchup the agent always loses is degenerate under every candidate above. [[scenario-distribution]] measured this and it dominates the choice of reward: no shaping fixes a scenario with no outcome variance, and difficulty filtering is not an alternative to reward design but a precondition for it.

## How a candidate would be chosen

Against the criteria in [[../decisions/0005-training-and-reward]], and with the measurement [[rlhf-transfer]] describes.

Report the gold objective and the proxy together, plotted against divergence from the cloned checkpoint rather than against training steps. The failure being watched for is the two curves separating, which is what teaching the proxy looks like from the outside. Without the shared axis a shaped reward that has started optimizing the wrong thing looks like a run that is training well.

Prefer the terminal candidates until learning demonstrably stalls, which has not happened: the runs recorded in [[training-design]] converge in tens of iterations on the matchups where the signal is non-degenerate. The case for density is not yet made, which is the strongest argument for leaving the record's decision where it is.

## Related

- [[../decisions/0005-training-and-reward]], the record this informs.
- [[rl-methods]], for the shaping theorem and the estimators the reward feeds.
- [[scenario-distribution]], for why the scenario decides more than the reward does.
- [[rlhf-transfer]], for over-optimization and the proxy-against-gold protocol.


## The commander the pricing could not see, 2026-08-09

Every strength figure this page uses came from `Monster::GetMonsterStrength()` at base creature stats, and the owner asked whether that accounts for hero stats. It does not, and the consequence lands on the reward rather than on the observation. The observation is fine: the worker records each unit's `GetAttack()` and `GetDefense()`, which reach `ArmyTroop::GetAttack()` and already add the commander's, so a policy sees the commander's entire combat contribution on every stack of both sides, which is exactly where it acts. The reward did not: the terminal record's per-side strength summed base-stat pricing, so two identical armies priced identically even when one was led.

The size of that hole, measured on identical armies with a ten-attack ten-defense commander on one side: base pricing calls both sides 387.9, and pricing each stack at its effective attack and defense calls the commanded side 708.0 against 387.9. The battle agrees with the second number, the engine winning every one of twenty-four such fights from the commanded side.

Where that hole does and does not reach is the part worth stating carefully, because the first version of this section overstated it. The two-sided reward is a ratio inside one side, own strength kept over own starting strength on a win and enemy strength destroyed over enemy starting strength on a loss, so a multiplier applied to a whole side appears in both numerator and denominator and cancels. Measured over 320 side-episodes on mixed armies with a large commander on one side and a small one on the other, half of them ending in partial survival, which is the only case where a per-creature multiplier can fail to cancel because the surviving mix differs from the starting mix, the two pricings differ by at most 0.004 and by 0.002 on average. The reward was not obscured.

What the commander does obscure is every cross-side comparison, where no cancellation happens: the budget the sampler matches the two armies with, which is why an evenly budgeted matchup could be decided before the first turn, and the difficulty weighting, which prices an enemy-to-own ratio the same way. The first is fixed in [[scenario-distribution#The commander is not priced, and it decides identical armies]]; the second is still owed.

The runner now emits `strength_commanded` and `initial_strength_commanded` beside the base figures, and `reward_margin="two_sided_commanded"` prices the margin in them. It is offered rather than defaulted, on this project's usual staging rule: every reward, report and checkpoint before 2026-08-09 is denominated in the base pricing, and a silent switch would make the new numbers incomparable with all of them without saying so. What is owed next is the measurement that decides adoption, a paired run under each pricing on matchups that carry commanders, and the same question asked of the difficulty weighting, which prices its enemy-to-own ratio the same way.

## The win bonus, and what removing it costs, 2026-08-09

The owner asked whether $1 + \text{fraction remaining}$ overweights winning, on the reasoning that a side with force remaining has already won, so the outcome is carried by the surviving fraction and the extra one only restates it. The premise is very nearly true. Forty sampled battles produced forty cases of surviving and winning and none of surviving and losing, and the exceptions are the two unfinished terminations rather than anything in ordinary play, at a measured 0.473 percent of training episodes.

The algebra says the observation is exactly right. The two-sided loss branch is $-1 + (1 - f^{\text{foe}})$, which is already $-f^{\text{foe}}$, so only the win branch ever carried the bonus. Removing it gives $r_T = f^{\text{own}} - f^{\text{foe}}$, the strength margin, and the identity

$$r^{\text{balanced}}_T = r^{\text{two-sided}}_T - \mathbb{1}[\text{won}]$$

holds on every termination. `terminal_reward_balanced` in `python/fheroes2_agent/env.py` implements it, selected by `reward_margin="balanced"`.

Two properties follow. The balanced form is exactly zero sum on any decided battle, where the two chairs sum to $0$ rather than to $+1$, which is worth something to self-play because the learner and its frozen opponent can no longer both be scored as gaining. And the weight on the outcome shifts a long way. Decomposing the reward's variance by the law of total variance over six battery suites, the win-loss bit carries 95 percent of it under the current form and 82 percent under the balanced one, a 4.8-fold cut in its squared weight, because the between-branch gap falls from $2 + f^{\text{win}} - f^{\text{loss}}$ to $1 + f^{\text{win}} - f^{\text{loss}}$. The graded terms that were designed to say how a battle was won or lost get roughly four times the say they had.

The outcome branch is still decided by `_side_won` rather than by the sign of the margin, which is what keeps the unfinished terminations resolved as they were. A bare margin reads both off material and gets both wrong: at the forty-deathless-round stalemate it would pay an attacker ahead on material for refusing to engage, exactly the exploit the flat $-1$ exists to close, and at the hundred-round `round_limit`, scored a loss for both sides because truncation is an artifact rather than a result, it would hand the leader a positive score. Deciding the branch first and dropping the bonus second is what makes the identity above hold everywhere instead of only on decided battles.

Whether the shift helps a trained policy is a separate question from whether the form is better posed, and it is measured rather than assumed. A paired run trains three seeds under each objective from one anchor on one matchup set with the leash at $\beta = 0.5$, scored on the full battery.

## What search maximizes is a separate choice, 2026-08-09

Root search does not consult the training reward by convention; it consults it by construction. `rollout` in `python/fheroes2_agent/search.py` returns the side environment's terminal reward, so whichever `reward_margin` that environment was built with is the quantity search maximizes. That was never chosen. The battery passes `two_sided` and `capture_replay.py` passed nothing and inherited the `hit_points` default, which is how the same checkpoint on the same armies and the same battlefield produced 0.00 from one loop and 0.50 from the other, ten episodes each, paired on the rollout seed.

Varying only the side environment's margin on that cell separates the families cleanly. The two that grade a lost battle by own survival, `hit_points` and `strength`, both read 0.90; the two that grade it by enemy destruction, `two_sided` and `balanced`, both read 0.00. The win branch does not distinguish them and the loss branch does, which is what one would expect from a search whose rollouts mostly end in defeat: that branch is most of the signal the tree sees, and the two families rank candidate moves by different things.

The full run settled it. `agent_play/experiments/search_objective.py` plays every margin over the mirror suite from both chairs across four battlefields with the rollout seed reset per cell, 48 cells each, so the objective is the only thing that varies and the comparison is paired. Against `two_sided` at 0.764, the two survival-graded margins gain, `hit_points` by $+0.191$ and `strength` by $+0.174$, both against a paired standard error of 0.044 and both with roughly twenty cells better against two or three worse. The balanced margin loses $-0.045$, which is the expected result rather than a surprise, since it inherits the same loss branch and differs from `two_sided` only in the win bonus that a rollout ending in defeat never sees.

The single cell that exposed the problem read 0.00 against 0.90 and so overstated the size by a wide margin. The direction held.

A second defect surfaced from the same investigation, and chasing it down produced the more important finding. `rollout` replays the action prefix in the side environment, and that replay reproduces the live state only when both sit on the same world seed, since the obstacle layout derives from it. Every harness built its side environment with the live environment's `seeds`, and because `reset` respawns the worker whenever a decision is pending, the side environment returned to variant zero on every rollout while the live episode rotated over four. The prefix guarantee the method rests on therefore did not hold, and `BattleEnv.current_battlefield` plus `sync_side_environment` now pin the side environment to the live variant.

Pinning the variant fixes the terrain and buys something else that was not intended. `agent_battle_runner.cpp` derives the combat seed from the tile index, the map seed and the two armies, all of which the world seed fixes, so a pinned side environment inherits the live battle's exact random stream. Search then evaluates candidates under the dice that will actually be rolled rather than under the distribution they are drawn from, which is oracle access rather than planning. `combatSeedOffset` was added to the scenario, defaulting to zero and leaving every golden digest bit-identical, purely so the two can be separated.

`agent_play/experiments/search_leakage.py` separates them on the mirror suite from both chairs, 96 episodes an arm, paired per cell. With the live dice the searching agent reads 0.927. With the same battlefield and independent dice it reads 0.604, a paired $-0.323$ at a standard error of 0.047 with eleven of twelve cells down. With neither, the original defect, it reads 0.573. So the live dice are worth $+0.323$ and correct terrain is worth $+0.031$.

That reverses an earlier attribution and the earlier claim is withdrawn. `search_sync.py` measured the desync costing between 0.12 and 0.62 and credited it to terrain; it was varying the dice at the same time and was mostly measuring them. The terrain fix remains correct, because the prefix replay genuinely requires it, but it is nearly free rather than decisive.

It also means the nine-suite sweep run earlier that day, which put the searching agent at or above the built-in AI everywhere, was measured with the live dice. Which offset a harness passes now decides what its numbers mean, so neither value is a default anywhere: `sync_side_environment` and `search_agent_battery` both take it explicitly and stamp it into the report.

Re-measured with the offset set and a paired no-search control, the picture is a tie rather than a win. Six of nine suites land above the engine and none is separated from it by more than two standard errors, with held-out at 0.619 against 0.660 and the mirrors at 0.500 and 0.583 against 0.361 and 0.639. What survives intact is the search's own contribution, $+0.132$ averaged over the suites against the same weights without it, separated at four of them and reaching $+0.354$ on the mirrors from the attacking chair. The reading is a network well below the engine, 0.512 on held-out, that search lifts to roughly engine level (`honest_sweep.json`).

None of this settles the training reward. What a policy is trained on and what its search maximizes are separate choices, and the point of measuring was to keep them separate on purpose rather than by accident. The same now goes for what kind of model the search is given: a perfect one including the dice is a legitimate upper bound and a useful diagnostic, but it is not the number to quote against a hand-written opponent that gets no model at all.

## Stalls, and the evasion exploit the owner predicted

A battle nobody finishes is a possibility the reward has to price, because a policy that cannot win can always try not to lose. The owner raised the flying version, a fast unit that simply keeps its distance forever; the measured facts as of 2026-08-06 were that `simple_v1` rejected flying movement outright, so the flying exploit could not be fielded, and `flying_v1` opened it on 2026-08-10 and the exploit was then fielded and priced: an evading Sprite ends in `stalemate` at exactly round forty like the walking version, defender $+2.00$ and attacker $-1.00$ flat, none looping, that the runner already stops any battle after forty consecutive no-death rounds with a `stalemate` termination (61 of 16,060 recorded episodes) and hard-caps at 100 rounds, and that the walking version of the exploit works today: `evasion_stalemate.py` commands fast Rogues to always move to the cell farthest from slow Zombies, and every episode ends in `stalemate` at exactly round 40, none looping.

What a stall pays is grounded in the engine rather than chosen. The built-in AI's own stalemate breaker forces the attacking hero to retreat after fifty deathless turns, and a forced retreat loses the attacker the battle and the army. `_side_won` in `env.py` scores our earlier termination the same way: the defender who outlasted the attacker wins, +1 plus survival, and the attacker who failed to force an engagement loses with the survival term zeroed, -1.0 flat. The zeroing matters: the first run of the demo showed an evading attacker banking 0.0 through full survival, which still beat fighting and losing at -0.4, exactly the preference the exploit needs; -1.0 flat makes any fight with survivors strictly better than any stall, for the side whose job is to engage.

The incidence is now measured for trained policies, not only for the teacher, per the owner's 2026-08-08 double-check. `rounds_probe.py` played the gen1 anchor and all six round-two checkpoints over the training and held-out matchup slices, 616 episodes with exact terminal round counts: every episode ended in victory or defeat, none in `stalemate` or `round_limit`, means sit at 8 to 10 engine rounds per policy, and the longest battle observed anywhere was 34 rounds, against a window that needs forty consecutive deathless ones. The teacher-corpus census stands beside it at 61 stalemates in 16,060 episodes. 

Training-side incidence is now measured rather than merely recorded: twelve vendored heartbeats carrying per-iteration termination counts cover 384,000 training episodes and read 1,815 stalemates, 0.473 percent, with no round-limit truncations at all, against the teacher corpus's 0.38 percent. Reinforcement therefore stalls about a quarter more often than the teacher does and still rarely, and the pricing holds without a policy ever having found the exploit (`rounds_probe.json` and the run heartbeats, 2026-08-08 run reports).

<!-- verify
grep    python/fheroes2_agent/env.py :: def terminal_reward_balanced
grep    python/fheroes2_agent/env.py :: def reward_from_record
grep    python/fheroes2_agent/env.py :: REWARD_MARGINS
exists  agent_play/experiments/search_objective.py
exists  agent_play/experiments/search_sync.py
grep    python/fheroes2_agent/env.py :: def current_battlefield
grep    python/fheroes2_agent/search.py :: def sync_side_environment
grep    python/fheroes2_agent/search.py :: return step.reward
exists  agent_play/docs/archive/experiments/files/2026-08-08-run-reports/rounds_probe.json
exists  agent_play/experiments/rounds_probe.py
grep    python/fheroes2_agent/train_ppo.py :: terminations
-->

Evaluation win rates are deliberately not changed by any of this: `measure` still counts only outright destruction as a win, so a stalling policy cannot inflate a battery column, and the reward's game-faithful semantics live only where behavior is optimized. The demo has now rerun with Sprites and the termination held unchanged, at exactly round forty with none looping. The owner then asked the question the termination does not answer, and it is the right one: a terminated exploit is not a disincentivized one.

Measured, evasion is not merely available to the defender, it is dominant. A stalemate pays the defender $1 + 1.0 = +2.00$, which is the largest reward the objective can produce at all, while the trained policy fighting the same matchup scores $-1.000$ with a flying defender and $-0.958$ with a walking one. Refusing to fight therefore beats fighting by $+3.000$, the whole range, and it beats a perfect fought win too, since that also caps at $+2.00$ and no real win keeps everything. Any defender that can kite has a strictly dominant strategy that never engages.

Two things follow, and the second is the uncomfortable one. Flying did not create this: fast walking Rogues reach the same $+2.958$, so the original demo was already displaying a dominant strategy and was read only as a termination check. What flying changes is reachability, because a flier escapes board geometries a walker cannot. And the game-faithfulness argument, that the engine's own breaker forfeits the battle to the defender, justifies the sign of the payoff and says nothing about its magnitude.

Why no policy has found it is a statement about exploration rather than about the objective: stalemates are 0.473 percent of training episodes and a deliberate forty-round evasion is not something exploration stumbles into. Candidate corrections are recorded rather than applied, because the reward is the owner's choice under [[../decisions/0005-training-and-reward]]. A stalemate defender win could pay $+1.0$ flat, mirroring the attacker's flat $-1.0$, which makes any fought win with surviving force strictly better than refusing. It could be scored by damage dealt rather than force kept, which pays pure evasion nothing. Or the reward could stand and the sampler screen out matchups where evasion is free.
