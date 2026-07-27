# CLAUDE.md

This is a personal fork of [fheroes2](https://github.com/ihhub/fheroes2). Two independent side
projects live on their own branches; the branch you are on determines which applies.

## Branch `agent-env` — headless battle environment for training

**Read `docs/agent/START_HERE.md` first, in full, before doing anything else.** It is the single
entry point: project goal, current status, findings that overturned the original plan, build and
verification commands, ordered next steps, and the decisions not to relitigate.

Quick orientation: Phase 0 is complete and passing (7/7); nothing beyond it is implemented. The
branch modifies no engine source. Verify the build with:

```bash
make -C src/dist -j"$(sysctl -n hw.ncpu)" && ./agent_play/spike/build_spike.sh && ./agent_play/spike/verify_phase0.sh
```

## Branch `play-harness` — Claude plays the game through the real UI

A separate experiment: an opt-in engine patch that dumps rendered frames to `$FHEROES2_HARNESS/frame.bmp`
and reads input commands from a FIFO, so an agent can play with no macOS Screen Recording or
Accessibility permissions. Inert unless `FHEROES2_HARNESS` is set. Not intended for upstream, and
deliberately not merged into `agent-env`.

## Build notes (both branches)

- This repo has **two** build systems: CMake, and a plain Makefile under `src/dist`. The Makefile
  path is the one in regular use here (`make -C src/dist -j10`).
- **Run `make -C src/dist clean` after any upstream sync.** The `-MD` depfiles hard-code header
  paths, so an upstream header rename breaks incremental builds with `No rule to make target`.
- `origin` is the fork. An `upstream` remote points at `ihhub/fheroes2` with its push URL
  deliberately disabled — never push there.
