---
title: "Expert Iteration (NeurIPS 2017)"
type: reference
source: https://arxiv.org/abs/1705.08439
read: 2026-08-06
runs: []
tags: [reference, search, distillation, imitation]
local: ["files/arxiv-1705.08439.pdf"]
---

# Expert Iteration (NeurIPS 2017)

Anthony, Tian and Barber name the loop this project arrived at independently: planning and generalization split into separate jobs, tree search planning new policies and a network generalizing them, each improving the other, with the network's priors making the next search cheaper. Their own framing is the one that matters here, quoted from the paper: expert iteration extends imitation learning to domains where the best available expert cannot play well enough, by re-solving the imitation problem against an expert that search keeps improving.

That is precisely this project's position. The demonstrator is the engine's scripted planner, the measured ceiling is that the student converges to it and stops ([[../../archive/experiments/2026-08-05-dagger-and-battlefield-transfer]]), and search is the only operator measured above it. So search-as-teacher is not an invention here but the textbook answer to a ceiling, and the literature's name for it is expert iteration.

Where we use it: the naming and the framing of the search-teaching generations, and the argument that further supervised rounds against a fixed expert cannot pass the expert while iterated search can.

Related: [[alphazero]], [[uct]], [[alphastar-unplugged]], [[mcts-regularized-policy-optimization]]
