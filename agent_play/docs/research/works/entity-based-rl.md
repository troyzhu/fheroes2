---
title: "Entity-based RL — entity-gym, RogueNet, and related evidence"
type: project
authors: Winter (entity-gym/RogueNet); Jankovics, Garcia Ortiz, Alonso (arXiv:2206.02855)
year: 2022-2023
quality: primary
urls:
  - https://github.com/entity-neural-network/entity-gym
  - https://github.com/entity-neural-network/rogue-net
  - https://clemenswinter.com/2023/04/14/entity-based-reinforcement-learning/
  - https://arxiv.org/abs/2206.02855
runs: [rl-approaches, minimap-observations]
tags: [reference, entity-lists, transformers, observation-design]
local: ["files/entity-gym-README.md", "files/rogue-net-README.md", "files/entity-based-rl-blog.html", "files/arxiv-2206.02855.pdf"]
---

# Entity-based RL (entity-gym / RogueNet / Jankovics et al.)

The entity-list end of the observation-design spectrum: environments expose dynamically-sized lists of typed entities instead of fixed tensors, consumed by ragged-batch transformers (RogueNet) or slot-attention/GNN encoders (arXiv:2206.02855, which trains shared-encoder actor-critic with PPO).

Verification status, handle with care: these sources were fetched in both research runs and several extracted claims (e.g., RogueNet matching the pixel-based IMPALA CNN on Procgen with ~50× fewer parameters; transformer-over-entities beating fixed-input MLPs on varying-topology tasks) appeared in extraction, but did not reach the verified top-25 of either run, cite the underlying sources directly, not the research reports, for these numbers.

Where we use it: background for the padding-vs-tokenization decision (padded slots won for v1 on vcmi-gym precedent; entity-transformers remain the upgrade path), and for the deferred policy-head ablation at the training milestone.

Related: [[alphastar]], [[vcmi-gym]], [[misc-pipeline-sources]]
