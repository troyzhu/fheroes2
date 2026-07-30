---
title: fheroes2 agent environment — documentation index
type: moc
updated: 2026-07-30
tags: [agent-env, index, entry-point]
---

# fheroes2 agent environment

A deterministic, headless, structured environment for fheroes2 battles, built so a policy can be trained on them. The environment reads true engine state and selects from engine-generated legal actions. It renders nothing and reads no pixels.

This tree lives under `agent_play/` rather than under the repository's `docs/` directory, because `docs/` is the source of the project's published website (`.github/workflows/pages.yml` builds Jekyll from it) and this material is internal working documentation.

## Where to start

| If you want to | Read, in order |
|---|---|
| Understand the problem, with no background in reinforcement learning or in this game | [[rl-and-the-battle-domain]], then [[README]] |
| Build it and see the current state | [[README]] |
| Understand the research and the evidence | [[research/findings]], then [[research/prior-art]] for the codebases, then [[research/README]] to look up a source |
| Understand what is implemented and how | [[implementation/inventory]] for what exists, then [[implementation/README]] for how each mechanism works |
| Understand why an interface is the way it is | [[decisions/README]] |
| Trace a number back to the run that produced it | [[archive/README]] |

## The tree

```
agent_play/docs/
├── INDEX.md                     this file
├── README.md                    orientation, notation, scope, build, current state
├── rl-and-the-battle-domain.md  RL vocabulary, the battle domain, comparison to other games
├── research/                    the literature and what it establishes
├── implementation/              how the built mechanisms work
├── decisions/                   accepted decision records
└── archive/                     dated and pinned records; not a reading path
```

Documents live in the first four locations. Everything under `archive/` is provenance, kept so a claim can be traced to its run, and every file there carries a date or a commit in its name because every file there goes stale.

## Related code

| Path | What it is |
|---|---|
| `src/fheroes2/battle/battle_seed.{h,cpp}` | Shared combat-seed helper, used by the engine and by the environment |
| `src/fheroes2/battle/battle_action_validation.{h,cpp}` | Command legality, lifted verbatim from the engine so both execution and enumeration use one implementation |
| `src/fheroes2/battle/battle_decision_controller.h` | The optional decision hook |
| `src/fheroes2/agent/` | The environment library, compiled into the normal executable, with no entry point |
| `src/agent_worker/` | The worker entry point, outside both build systems' source globs |
| `python/fheroes2_agent/data/` | The generated monster capability audit that defines the `simple_v1` allowlist |
| `agent_play/spike/`, `agent_play/tests/`, `agent_play/verify_m*.sh` | The Phase 0 spike, unit tests, and the milestone verification gates |
