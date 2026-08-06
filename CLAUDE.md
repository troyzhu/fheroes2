# CLAUDE.md

This is a personal fork of [fheroes2](https://github.com/ihhub/fheroes2). Two independent side
projects live on their own branches; the branch you are on determines which applies.

## Branch `agent-env` — headless battle environment for training

**Read `agent_play/docs/README.md` first, then follow its routing table.** The documentation lives
under `agent_play/`, not under `docs/`, because `docs/` is the source of the project's published
website (`.github/workflows/pages.yml` builds Jekyll from it) and this material is internal.

Two documents sit at the top and everything else lives in a directory that says what it is.
`README.md` routes. `overview.md` carries the problem, both vocabularies including the notation
contract, scope, build, current state, and the full map. `roadmap.md` says where the project is
aimed and what each phase is waiting on. Then `rl/` holds the learning side (domain, methods,
training design, scenario distribution, RLHF transfer), `implementation/` holds the environment
side with one primer per built mechanism, `decisions/` holds the accepted records, `research/`
holds the literature, and `archive/` holds dated logs and raw runs, which are provenance rather
than a reading path.

**`rl/` used to be entirely unimplemented and no longer is, so check before claiming either way.**
As of 2026-08-05 all four stages exist: stage 1 cloning, stage 2 DAgger (first round measured,
`dagger_iteration.py`), stage 2b critic pre-fitting and stage 3 reinforcement learning, built
under `python/fheroes2_agent/` plus `agent_play/experiments/` and gated by `verify_agent.sh`,
along with the calibrated scenario generator. Still unbuilt: every reward candidate except the
margin-weighted terminal one and the opt-in difficulty weighting, the `observable_v1` profile
and `planes_v1`. Each page under `rl/` says which of its own content exists;
`implementation/inventory.md` is the per-component list.

**Notation is not a free choice.** `agent_play/docs/overview.md`'s Notation section is the contract, and the tree is
self-contained: no page depends on a file outside this repository. The symbols match the owner's
own RL study notes so this material reads as a continuation of them. Concretely: `V^\pi(s)`,
`Q^\pi(s,a)`, `A^\pi(s,a)`, `\pi_\theta(a \mid s)`, `P(s' \mid s,a)`, `R(s,a,s')`, `\rho_0`,
`V_\phi` for the critic, `\rho_t(\theta)` for the PPO ratio, `\varepsilon` for the clip half-width,
`\epsilon` for the DAgger per-decision error rate, `\Psi_t` for the policy-gradient scoring term,
and `\log` rather than `\ln`.

Do NOT "correct" these toward Zhao's `v_\pi`/`q_\pi`/`\delta_\pi`/`\pi(a \mid s,\theta)`/`d_0`/`\ln`.
That was tried once and fully reverted. The owner cites Zhao for the Bellman-contraction material
but writes in the Sutton-Barto convention throughout, verified by counting symbols across their
notes. That section keeps a Zhao translation table for reading the book alongside.

**The owner's notes live at `/Volumes/External Drive/Dropbox/Papers/`, and that tree is READ-ONLY.**
Never create, edit, move, or reorganize anything under it. The relevant parts are
`Papers/study/problems/reinforcement-learning/` (74 problem cards, cited in the overview's Notation section by id as
`rl-014`), `Papers/study/dive/rlhf-book-lambert/`, and `Papers/wiki/concepts/`. Read them to check
a convention, then make every change on the fheroes2 side. **The fork is public**, so their notes
must never be copied in verbatim; write original pages citing card ids as provenance instead.

**Run `./agent_play/lint_docs.sh` before claiming any documentation change is done.** It enforces
the writing contract at `~/.claude/plugins/marketplaces/troyzhu/docs/WRITING_STYLE.md` (em-dash and
bold budgets, banned constructions, no question headings, paragraphs under 160 words, and no
hard-wrapped prose, since this vault is one paragraph per line) and resolves every wikilink
including its heading anchor, which is what catches the silent breakage a file rename or a section
retitle causes. In whole-tree mode it also checks facts, because status prose rotted silently
three times in one day: moc READMEs must index every sibling page, `<!-- verify -->` blocks
(same grammar as `verify_memory.sh`) are checked so status sentences can declare their
invalidators, backticked `src/` and `python/` paths must exist outside `archive/`, and every
engine file changed relative to `master` must be named in `inventory.md`'s engine-source ledger.
When you write a sentence about what is or is not built, add its invalidator to the nearest
verify block; never write a document count in prose.

Scope note: everything built so far is the **battle** environment. `agent_play/docs/roadmap.md`
records the wider goal, including the adventure-map agent covering movement, recruitment, and town
management, and the research owed before any of it is designed. `decisions/0005-training-and-reward.md`
records how a policy will be trained and what it will be rewarded for, and
`agent_play/docs/rl/training-design.md` carries the mechanics behind it (network architecture,
losses, hyperparameter tables, alternatives at each choice), and `agent_play/docs/rl/rl-methods.md`
defines every RL technique the documentation names, deriving the chain from the policy gradient
through PPO and giving a verdict on each alternative.

Quick orientation: Phase 0 and Milestones 1 through 3 are complete and verified on the target
Apple M2 Mac mini, and the training work ran ahead of Milestones 4 through 6 rather than after
them. Engine changes are deliberately small: two verbatim lifts (`battle_seed.{h,cpp}`,
`battle_action_validation.{h,cpp}`), one optional hook (`battle_decision_controller.h`), one
opt-in render seam for replay videos (a null-by-default render observer on the display, a
defaulted `showInterface` on the runner, a `Race::NONE` art case in `battle_interface.cpp`; see
`agent_play/docs/implementation/replay-rendering.md`), and the entry-point-free library under
`src/fheroes2/agent/` that both build systems compile into the normal executable without
behavior change. Verify with:

```bash
make -C src/dist -j"$(sysctl -n hw.ncpu)" && ./agent_play/spike/build_spike.sh \
  && ./agent_play/spike/verify_phase0.sh && ./agent_play/verify_m1.sh \
  && ./agent_play/verify_m2.sh && ./agent_play/verify_m3.sh \
  && ./agent_play/verify_agent.sh \
  && ./agent_play/lint_docs.sh && ./agent_play/verify_memory.sh
```

`agent_play/experiments/` holds measurements too slow for a gate, with results in
`agent_play/docs/archive/experiments/`. **Do not run a verification gate while one is in flight**:
the gates relink the worker binary and an experiment spawns it once per episode, which killed a
sixty-seed sweep at seed 34. Pass a copy of the binary instead, which every script accepts as an
argument for exactly this reason.

`verify_memory.sh` checks that this project's agent memory still describes reality. Memory files
live outside the repository and nothing fails when a claim in one stops being true, which is how a
line once ended up telling a future session to "correct" the notation into the wrong convention.
Each memory file declares what makes it true in an HTML comment, and every repository path it names
in prose is checked as well. **The single milestone table is `overview.md`'s "Where the project
stands"; do not add a second one anywhere.**

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
