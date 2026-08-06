---
title: "AlphaZero (2017)"
type: reference
source: https://arxiv.org/abs/1712.01815
read: 2026-08-05
runs: []
tags: [reference, mcts, self-play, distillation, puct]
local: ["files/arxiv-1712.01815.pdf"]
---

# AlphaZero (2017)

Silver et al., the canonical search-as-teacher loop: PUCT search over a learned policy prior and value function produces visit counts, the visit distribution becomes the policy's training target, and the game outcome becomes the value's target, iterated under self-play. Search is the improvement operator and distillation closes the loop; chess, shogi and Go fall to the same recipe with no domain features beyond the rules.

Two transfer caveats for this project, both from the 2026-08-05 review. AlphaZero's value function is retrained every iteration on the improved policy's own games, which is exactly the re-grounding that [[alphastar-unplugged]] shows is load-bearing: iterating search against a frozen behavior value degrades instead. And AlphaZero's simulator is a perfect copyable model, where ours is a process singleton reached by prefix replay, cheap at about five milliseconds a rollout but serial per worker.

Where we use it: the PUCT form in [[../../rl/rl-methods#Search as an improvement operator]] and the design frame for search-as-teacher through the existing relabeling pipeline, with the one-step discipline the Unplugged evidence imposes.

Related: [[uct]], [[alphastar-unplugged]], [[stratega]]
