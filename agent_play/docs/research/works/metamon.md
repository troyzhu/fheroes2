---
title: "Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers (Metamon)"
type: paper
authors: Grigsby, Xie, Sasek, Zheng, Zhu
year: 2025
arxiv: "2504.04395"
quality: primary
urls:
  - https://arxiv.org/abs/2504.04395
  - https://github.com/UT-Austin-RPL/metamon
tags: [reference, turn-based, offline-rl, imitation, self-play, prior-art]
local: ["files/arxiv-2504.04395.pdf"]
---

# Metamon, offline RL on a turn-based battle ladder

Read on 2026-08-04 while surveying turn-based battle games beyond the wargame corpus. Pokémon Showdown is the closest thing turn-based battling has to a competitive benchmark, and this is its strongest documented result: agents that climbed into the top 10 percent of active human ladder players with no search of any kind, from UT Austin, published at RLC 2025 and later the winning baseline of the NeurIPS 2025 PokéAgent challenge.

## What they did

Three stages, which read as this project's staging generalized. They reconstruct the first-person view of an agent from a decade of third-person spectator logs, over five million human battles, and imitate them. They then improve past the demonstrators with offline reinforcement learning on the same data, meaning value-weighted learning on fixed trajectories rather than fresh rollouts. Finally they fine-tune offline on self-play data generated between their own agents, tens of millions of battles, with large sequence models throughout because the game is heavily partially observed and team information must be inferred across turns.

Evaluation is against a standardized suite of teams and opponent policies spanning beginner to expert, and then against live humans on the ranked ladder.

## What transfers here

The replay-reconstruction move is one this project gets for free and has barely used. Their expensive step, rebuilding an agent's view from spectator logs, is exactly what the passive recorder already produces at 4,500 episodes a second with perfect information. The lesson is what they do with it: imitation is the floor, not the ceiling, and offline RL on the same recorded data improved past the demonstrator before any online rollout. Stage 2b's critic fit is a first step of that shape; offline advantage-weighted training on teacher episodes would be the full step, and it needs no new environment work.

The standardized opponent suite is the sharper import. [[../../decisions/0005-training-and-reward]] requires training against a mixture of engine configurations, and nothing has been built for it; every run so far faces one teacher at one difficulty. Their beginner-to-expert opponent ladder is what made their evaluation mean something, and the engine's difficulty settings plus recorded checkpoints of our own past policies would supply the same axis cheaply.

Their sequence models earn their size from partial observability. This project's `full_v1` profile sees everything, so nothing here argues for a transformer yet, but `observable_v1` is pending, and their result says the cost of partial observability is paid in architecture rather than being avoidable.

## What does not transfer

Pokémon has no board. The whole spatial half of this project's action space, movement and positioning on 99 cells, has no analogue there, so their flat move-choice action space says nothing about ours. Their scale, 200 million parameters and millions of battles, is calibrated to a decade of human data and a fanbase-sized opponent pool; nothing about this project's single-machine budget follows from it.

## Related

- [[vcmi-gym]], the closest prior art on the wargame side; Metamon is the closest on the duel side.
- [[../../rl/training-design#Pre-fitting the critic on teacher play]], the stage their offline-RL result argues to extend.
- [[../../rl/rlhf-transfer]], for the estimator vocabulary shared across settings.
