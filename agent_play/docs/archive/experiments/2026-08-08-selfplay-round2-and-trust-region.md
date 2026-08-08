---
title: "Self-play round two at settled budgets, and the trust-region rematch, 2026-08-08"
type: experiment-log
updated: 2026-08-08
tags: [agent-env, archive, experiment, self-play, trust-region]
---

# Self-play round two at settled budgets, and the trust-region rematch, 2026-08-08

The night's plan followed directly from the convergence verdicts: the 400-iteration self-play run had stopped mid-climb, so the continuation round ran the same recipe at 1000 iterations across three seeds, with a matched control trio through the identical `SelfPlayEnv` code path whose pool is the built-in AI alone, so the arms differ in nothing but who answers the other side. Every run trains from the gen1 anchor on the standard pool's first twelve matchups under the two-sided reward with warmup and floor, and every artifact carries the trust-region stamp introduced the same evening. Judgment is the full apparatus by standing rule: battery with quality columns and rungs, symmetry gauge, convergence report, and duels at forty episodes, clear of the measured $\pm 0.06$ noise band.

## Round two: converged, better where it trained, worse where it did not

The budget question closed first: all six runs read converged on the trained reward, the win and loss reward decompositions, the rate, and the value loss, so nothing further was left in this recipe at this distribution (`convergence_round2.json`).

On the training distribution the improvement is large and the forty-episode duels put it outside noise. Against the built-in AI the self-play trio reads 0.950, 0.975 and 0.950 where the anchor reads 0.650, and the control trio 0.850 to 0.900; against share2 and clone v4 the ordering is the same with self-play ahead of control on five of six cells (`duels_round2.json`).

The battery shows what paid for it. Held-out falls from the anchor's 0.506 to 0.385, 0.419 and 0.369 for self-play and 0.346, 0.298 and 0.342 for the control; the Thunk ladder mean falls from 0.885 into the 0.6 to 0.8 band for both arms; defender play and hordes erode everywhere; commanders hold at 0.95 to 0.99 for everyone. One suite separates the arms cleanly: real maps, where self-play sits at 0.557 to 0.568 against the anchor's 0.564, fully preserved, while the control drops to 0.516 to 0.550. Self-play pays measurably less erosion than fixed-opponent training at the same budget, replicated across seeds (`battery_round2.json`).

The symmetry gauge, read against the engine's own $+0.071$ attacker gap, moves two of three self-play seeds strongly attacker-leaning (gaps $+0.171$ and $+0.212$, the third flat) and one of three control seeds, with seed errors near 0.08, so the 400-iteration finding that self-play reshapes side behavior toward the game's true asymmetry holds directionally, with honest noise (`symmetry_round2.json`).

The reading offered as hypothesis rather than verdict: the training pool was twelve matchups, gen1's breadth came from tens of thousands of generator samples, and the pattern, training-set mastery beside held-out decay with commanders and real maps least affected, is what distribution narrowness looks like. The next round's single highest-leverage change is self-play over the full matchup generator at the same budget, and no conclusion about self-play's ceiling is licensed before that runs.

## The rounds double-check: what the caps actually are, and whether anything trips them

The owner asked what a round means in the forty that guards stalls, and the code answer is the engine's own unit: one `arena.Turns()` call per round, every unit on both sides acting if it can, counted in `agent_battle_runner.cpp`. The forty is not a cap but a sliding window, forty consecutive rounds without a death tripping the `stalemate` termination, while the hard truncation sits at `maxRounds` 100 as `round_limit`; the battery's length column is a third unit entirely, learner decisions.

`rounds_probe.py` then measured whether validation play ever approaches either bound: the anchor and all six round-two checkpoints over both matchup slices, 616 episodes with exact terminal round counts. Every episode ended in victory or defeat, no stalemates and no round-limit truncations; per-policy means sit at 8 to 10 rounds on both sets, the reinforcement-trained policies play slightly shorter battles than the anchor, and the longest battle observed anywhere was 34 rounds, against a window that needs forty deathless ones in a row. The teacher-corpus census stands beside it, 61 stalemates in 16,060 episodes. Whether the round-two training episodes themselves ever stalled is unreconstructable, heartbeats carried no termination counts, so `train_ppo.py` now emits a per-iteration termination dict and every future run answers this from its own record (`rounds_probe.json`).

## The trust-region rematch

The nine divergence-gated runs, the paper's binary lower bound at threshold 0.05 and the exact total variation at 0.05 and 0.20, three seeds each at the same budget, anchor, opponent path and reward as the control trio, ran overnight with the chained evaluation behind them; this section takes their verdict when the reports land.
