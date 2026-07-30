---
title: "Additional pipeline sources (fetched; peripheral or unverified)"
type: collection
quality: mixed
runs: [rl-approaches, minimap-observations]
tags: [reference, peripheral]
local:
  - "files/sample-factory-README.md"
  - "files/lux-ai-2021-winner-README.md"
  - "files/arxiv-2410.17647.pdf"
  - "files/arxiv-2104.03113.pdf"
  - "files/arxiv-2305.19240.pdf"
  - "files/arxiv-2607.06514.pdf"
---

# Additional pipeline sources

Sources the research pipeline fetched whose claims informed search/extraction but contributed
few or no claims to the verified top-25. Kept locally for completeness; summaries below are
one-liners from their own titles/READMEs — **not** verified findings.

- **Sample Factory** (github.com/alex-petrenko/sample-factory) — high-throughput single-machine
  RL (APPO); candidate training infrastructure if CleanRL-style scripts ever need more
  throughput.
- **Kaggle Lux AI 2021 winner** (github.com/IsaiahPressman/Kaggle_Lux_AI_2021) — practice
  example of board-plane encodings + CNN policies for a grid strategy competition.
- **Entity-based Reinforcement Learning for Autonomous Cyber Defence** (arXiv:2410.17647) —
  entity-observation RL outside games; appeared in both runs' fetch sets.
- **Scaling Scaling Laws with Board Games** (arXiv:2104.03113) — compute/performance scaling on
  Hex via AlphaZero-style training; relevant to the planning-methods door
  ([[ref-stratega]]).
- **NetHack is Hard to Hack** (arXiv:2305.19240) — follow-up analysis of NLE agents
  ([[ref-nle]]).
- **FootsiesGym** (arXiv:2607.06514) — two-player zero-sum fighting-game benchmark; two-sided
  evaluation/league context.

Related: [[index]]
