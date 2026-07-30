---
title: "ARLinBfW — Adversarial Resilience Learning in Battle for Wesnoth"
type: codebase
authors: Stelter
year: 2019
quality: primary
urls:
  - https://github.com/DStelter94/ARLinBfW
  - https://arxiv.org/abs/1811.06447
runs: [rl-approaches]
tags: [reference, wesnoth, headless-engine, text-protocol, prior-art]
local: ["files/arlinbfw-README.md", "files/arxiv-1811.06447.pdf"]
---

# ARLinBfW (Battle for Wesnoth RL environment)

Existence proof for our architecture family: an OpenAI-Gym-like environment wrapping the real C++ Battle for Wesnoth engine, run headless (`--nogui --nosound --multiplayer-repeat`), bridged to Python via an out-of-process text channel (Lua prints observations to the subprocess's stdout; actions are passed back through a polled file). Used to test multi-agent RL / Adversarial Resilience Learning (arXiv:1811.06447).

Verified claims (3-0, two merged): the headless invocation, the stdout-observations + file-polled-actions bridge, and the real-engine C++ patch basis.

Where we use it: validates the headless-C++-engine + text-protocol pattern; our bidirectional JSONL-over-stdio is the strictly cleaner version. Tiny and stale (2 commits, 2019), precedent, not reusable code.

Related: [[vcmi-gym]]
