# Mac mini handoff

Everything needed to continue the agent-environment work on the target Mac mini M2.

## What transfers

**Only the git branch.** Nothing else needs copying — that is a direct consequence of the Phase 0
finding that headless battles require no game assets.

| Item | How it travels | Notes |
|---|---|---|
| Spec, audit, spike source, scripts | `agent-env` branch on the fork | committed |
| Built binaries (`smoke_battle`, `fheroes2`) | **not transferred** | rebuild on the mini; excluded from git |
| HoMM2 / demo game data | **not needed** | Phase 0 runs with zero game data |
| `~/.fh2play` sandbox | **not needed** | belongs to the separate play-harness project |

Explicitly *not* included: the `play-harness` branch's engine patch (frame dumping + input FIFO).
That is a different project, and the agent environment is kept clean of it on purpose. See
`local_source_audit.md` for why, plus the digest evidence that it made no difference.

## Setup on the mini

```bash
# 1. Toolchain
xcode-select --install                       # if not already present
brew bundle --file script/macos/Brewfile     # gettext, sdl2, sdl2_mixer, sdl2_image
brew install cmake                           # NOT in the Brewfile; needed only for the CMake path

# 2. Source
git clone git@github.com:troyzhu/fheroes2.git
cd fheroes2
git switch agent-env

# 3. Build the game objects, then the spike
make -C src/dist -j"$(sysctl -n hw.ncpu)"
./agent_play/spike/build_spike.sh

# 4. Verify
./agent_play/spike/verify_phase0.sh
```

Expected: **7 passed, 0 failed.** The map seed should be `2227197244` and the melee digest
`2cfd42cb104aa5e7` — these are machine-independent, so a mismatch on the mini is a real finding and
should be investigated before building anything on top.

## Read in this order

1. `agent_play/fheroes2_agent_system_spec_v0.3.md` §0.1 — what changed after validation
2. `docs/agent/local_source_audit.md` — the Phase 0 report and the assumption table
3. `agent_play/spike/README.md` — how to drive the spike

## Known gotchas

- **Always `make -C src/dist clean` after syncing upstream.** The `-MD` depfiles hard-code header
  paths, so an upstream header rename breaks incremental builds with
  `No rule to make target '.../<header>.h'`.
- **`xgettext` must be the real one.** On the dev machine it resolved to a pyenv shim
  (`~/.pyenv/shims/xgettext`) and happened to work. If the Makefile build fails in the `.pot` step,
  put Homebrew's `gettext` ahead of pyenv on `PATH`.
- **CMake is optional for Phase 0** but required to reproduce the spec's §2.2 build and the
  normal-game regression. The Brewfile does not install it.

## What to do first on the mini

The four items Phase 0 could not close, roughly in value order:

1. **Debug-build assertion run.** The highest-value gap. Release compiles out the asserts in
   `ApplyAction` and the `assert( _interface != nullptr )` on the non-AI path, so this is the real
   test that the action pipeline is sound:
   ```bash
   make -C src/dist clean
   FHEROES2_WITH_DEBUG=1 make -C src/dist -j"$(sysctl -n hw.ncpu)"
   ./agent_play/spike/build_spike.sh && ./agent_play/spike/verify_phase0.sh
   ```
   Note this overwrites the Release objects; rebuild without the flag afterwards.
2. **Re-measure throughput and RSS** on the mini and record them in `benchmark_m2.md`. The M3 numbers
   in the audit (~5000 episodes/s, 12.9 MB) are indicative only and must not be quoted as a baseline.
3. **CMake normal-game regression** plus a real Battle Only run through the UI.
4. **Sanitizers** (ASan/UBSan) on a representative battle.

After that, Milestone 1 per the spec — noting that it no longer requires an engine patch, since the
world seed is controlled by reseeding the thread-local RNG (§7.2, Option 1).
