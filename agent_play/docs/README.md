---
title: fheroes2 agent environment, documentation index
type: moc
updated: 2026-08-03
tags: [agent-env, index, entry-point]
---

# fheroes2 agent environment

A deterministic, headless, structured environment for fheroes2 battles, built so a policy can be trained on them. The environment reads true engine state and selects from engine-generated legal actions. It renders nothing and reads no pixels.

This tree lives under `agent_play/` rather than under the repository's `docs/` directory, because `docs/` is the source of the project's published website (`.github/workflows/pages.yml` builds Jekyll from it) and this material is internal working documentation.

## The two documents to start from

[[overview]] says what the project is, fixes both vocabularies, and states where it stands. [[roadmap]] says where it is going and what has to be answered before each next step. Everything else hangs off those two, and [[overview#Where everything is]] is the full map with a purpose per directory.

## Where to start

| If you want to | Read, in order |
|---|---|
| Understand the problem, with no background in reinforcement learning or in this game | [[rl/rl-and-the-battle-domain]], then [[overview]] |
| Build it and see the current state | [[overview]] |
| Look up a symbol or a project term | [[overview#Notation]] |
| Know how a policy will be trained and what it is rewarded for | [[decisions/0005-training-and-reward]] for the decisions, then [[rl/README]] for the mechanics |
| Understand what is implemented and how | [[implementation/inventory]] for what exists, then [[implementation/README]] for how each mechanism works |
| Understand why an interface is the way it is | [[decisions/README]] |
| Understand the research and the evidence | [[research/findings]], then [[research/prior-art]] for the codebases, then [[research/README]] to look up a source |
| Know what is planned beyond battles, including the navigation agent | [[roadmap]] |
| Trace a number back to the run that produced it | [[archive/README]] |

## The tree

```
agent_play/docs/
├── README.md          this file, a routing index
├── overview.md        the problem, both vocabularies, scope, state, build, and the full map
├── roadmap.md         where the project is aimed, and what each phase needs answered first
├── rl/                the learning side, five documents from the domain to training design
├── implementation/    the environment side, one primer per built mechanism
├── decisions/         accepted decision records, each with its options and trade-offs
├── research/          the literature and what it establishes
└── archive/           dated and pinned records, provenance rather than a reading path
```

The split that matters is `rl/` against `implementation/`. The environment is built and verified, and nothing under `rl/` is implemented yet. Keeping them apart means a reader always knows which side of that line a document sits on.

Everything under `archive/` is provenance, kept so a claim can be traced to its run, and every file there carries a date or a commit in its name because every file there goes stale.

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
| `agent_play/lint_docs.sh` | The documentation gate, the style contract plus wikilink resolution |
