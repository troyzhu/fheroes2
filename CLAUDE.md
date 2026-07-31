# CLAUDE.md

This is a personal fork of [fheroes2](https://github.com/ihhub/fheroes2). Two independent side
projects live on their own branches; the branch you are on determines which applies.

## Branch `agent-env` — headless battle environment for training

**Read `agent_play/docs/README.md` first, then follow its routing table.** The documentation lives
under `agent_play/`, not under `docs/`, because `docs/` is the source of the project's published
website (`.github/workflows/pages.yml` builds Jekyll from it) and this material is internal.

The tree separates reading material from records. `README.md` routes; `overview.md` carries
orientation, scope, build and current state; `notation.md` fixes the mathematical symbols;
`rl-and-the-battle-domain.md` teaches the reinforcement-learning vocabulary and the battle domain
from scratch; `research/` holds the literature; `implementation/` holds one primer per built
mechanism; `decisions/` holds the accepted records; and `archive/` holds dated logs, benchmarks,
and raw research runs, which are provenance rather than a reading path.

**Notation is not a free choice.** All mathematics matches the repository owner's personal RL wiki,
an Obsidian vault under Dropbox at `Papers/wiki` (on this machine,
`/Volumes/External Drive/Dropbox/Papers/wiki`). That vault holds ~6,000 concept notes and already
defines nearly every technique used here, so this documentation is meant to extend it rather than
run a parallel vocabulary. Only its core RL notes are in scope; the RLHF, GRPO, and LLM material is
a different problem. `agent_play/docs/notation.md` is the contract: the shared symbols with the
wiki note that defines each, the symbols this project adds, a coverage map marking every topic as
recap or new, and a translation table for anyone reading Zhao's *Mathematical Foundations of RL*,
which uses different symbols. Read it before editing any document that carries equations.

**That vault is read-only. Never write to it, and never edit, create, or reorganize a file under
`Papers/wiki`.** Read it to check a symbol or to see whether a concept is already covered, and make
every change on the fheroes2 side.

Concretely the wiki convention is `V^\pi(s)`, `Q^\pi(s,a)`, `A^\pi(s,a)`, `\pi_\theta(a \mid s)`,
`P(s' \mid s,a)`, `R(s,a,s')`, `\rho_0`, `V_\phi` for the critic, `r_t(\theta)` for the PPO ratio,
and `\log` rather than `\ln`. Do not "correct" these toward Zhao's `v_\pi`/`q_\pi`/`\delta_\pi`/
`\pi(a \mid s,\theta)`/`d_0`/`\ln`; that was tried and reverted, because the owner's actual notes
use the Sutton-Barto convention regardless of which textbook they cite.

**Run `./agent_play/lint_docs.sh` before claiming any documentation change is done.** It enforces
the writing contract at `~/.claude/plugins/marketplaces/troyzhu/docs/WRITING_STYLE.md` (em-dash and
bold budgets, banned constructions, no question headings, paragraphs under 160 words, and no
hard-wrapped prose, since this vault is one paragraph per line) and resolves every wikilink, which
is what catches the silent breakage that a file rename causes.

Scope note: everything built so far is the **battle** environment. `agent_play/docs/roadmap.md`
records the wider goal, including the adventure-map agent covering movement, recruitment, and town
management, and the research owed before any of it is designed. `decisions/0005-training-and-reward.md`
records how a policy will be trained and what it will be rewarded for, and
`agent_play/docs/training-design.md` carries the mechanics behind it (network architecture,
losses, hyperparameter tables, alternatives at each choice), and `agent_play/docs/rl-methods.md`
defines every RL technique the documentation names, deriving the chain from the policy gradient
through PPO and giving a verdict on each alternative.

Quick orientation: Phase 0 and Milestones 1 through 3 are complete and verified on the target
Apple M2 Mac mini. Engine changes are deliberately small: two verbatim lifts
(`battle_seed.{h,cpp}`, `battle_action_validation.{h,cpp}`), one optional hook
(`battle_decision_controller.h`), and the entry-point-free library under `src/fheroes2/agent/`
that both build systems compile into the normal executable without behavior change. Verify with:

```bash
make -C src/dist -j"$(sysctl -n hw.ncpu)" && ./agent_play/spike/build_spike.sh \
  && ./agent_play/spike/verify_phase0.sh && ./agent_play/verify_m1.sh \
  && ./agent_play/verify_m2.sh && ./agent_play/verify_m3.sh \
  && ./agent_play/lint_docs.sh
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
