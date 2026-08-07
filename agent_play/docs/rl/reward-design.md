---
title: Reward design, the space and what is known about it
type: design
updated: 2026-08-03
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


## Stalls, and the evasion exploit the owner predicted

A battle nobody finishes is a possibility the reward has to price, because a policy that cannot win can always try not to lose. The owner raised the flying version, a fast unit that simply keeps its distance forever; the measured facts as of 2026-08-06 are that `simple_v1` rejects flying movement outright (deferred to Phase 1b), so the flying exploit cannot be fielded yet, that the runner already stops any battle after forty consecutive no-death rounds with a `stalemate` termination (61 of 16,060 recorded episodes) and hard-caps at 100 rounds, and that the walking version of the exploit works today: `evasion_stalemate.py` commands fast Rogues to always move to the cell farthest from slow Zombies, and every episode ends in `stalemate` at exactly round 40, none looping.

What a stall pays is grounded in the engine rather than chosen. The built-in AI's own stalemate breaker forces the attacking hero to retreat after fifty deathless turns, and a forced retreat loses the attacker the battle and the army. `_side_won` in `env.py` scores our earlier termination the same way: the defender who outlasted the attacker wins, +1 plus survival, and the attacker who failed to force an engagement loses with the survival term zeroed, -1.0 flat. The zeroing matters: the first run of the demo showed an evading attacker banking 0.0 through full survival, which still beat fighting and losing at -0.4, exactly the preference the exploit needs; -1.0 flat makes any fight with survivors strictly better than any stall, for the side whose job is to engage.

Evaluation win rates are deliberately not changed by any of this: `measure` still counts only outright destruction as a win, so a stalling policy cannot inflate a battery column, and the reward's game-faithful semantics live only where behavior is optimized. If flying support arrives with Phase 1b, the demo reruns with Sprites, and the defender-side incentive, evasion as a legitimate +2.0 when the attacker cannot force a fight, becomes a real strategy the training distribution has to expect.
