---
title: "STRATEGA — a general strategy games framework"
type: project
authors: Dockhorn, Grueso, Jeurissen, Perez-Liebana
year: 2020
arxiv: "2009.05643"
quality: primary
urls:
  - https://gaigresearch.github.io/2020/06/15/dockhorn2020stratega/
  - https://arxiv.org/abs/2009.05643
  - https://github.com/GAIGResearch/Stratega
runs: [rl-approaches]
tags: [reference, strategy-games, forward-model, planning]
local: ["files/stratega-page.html", "files/arxiv-2009.05643.pdf"]
---

# Stratega

General n-player strategy-games framework (turn-based as of the AIIDE-20 paper, YAML-configured games) whose agent API is simulator-centric rather than gym-style: agents implement `computeAction(GameState, ForwardModel&, Timer)` and use a copyable, steppable forward model (~100k calls/s for turn-based games) for statistical forward planning (bundled MCTS/RHEA; zero gym references in the repo).

Verified claims (3-0, two merged): the YAML-driven framework description and the forward-model-centric agent interface, both code-confirmed (`Agent.h`).

Where we use it: the reminder that our deterministic ~4,600 eps/s battle core is a first-class planning asset, keep a copyable-state/forward-model door open for MCTS/MuZero-style methods after the process-parallel worker phase ([[report-rl-approaches]] §4).

Related: [[ref-vcmi-gym]], [[ref-misc-pipeline-sources]] (Scaling Scaling Laws with Board Games)
