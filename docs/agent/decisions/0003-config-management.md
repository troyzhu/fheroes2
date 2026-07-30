# ADR 0003 — Versioned config files govern every tunable; artifacts embed their resolved config

- Status: accepted 2026-07-27 (binding from Milestone 4 onward; C++ scenario JSON per spec §11)
- Context: user requirement 2026-07-27; spec §11 (scenario schema), §15 (trajectory metadata), §16.4 (determinism metadata); `research_rl_approaches.md` §4

## Context

The project is about to accumulate many configuration axes: scenario suites, observation profile (ADR 0001), action-schema version (ADR 0002), RL algorithm variant (masked PPO first; PPG/PPO-DNA/GRPO-style variants are an open axis), network architecture, hyperparameters, opponent mixtures, evaluation suites, seeds, worker counts. Scattering these across code constants, CLI flags and session memory does not survive multi-session work, neither for humans nor for a coding agent picking the project up cold. vcmi-gym (the shipped precedent) runs YAML-configured training with W&B tracking and PBT; our own determinism discipline already hashes scenarios (§11.3).

## Decision

1. One `configs/` tree at the repo root, YAML, split by concern: `configs/env/` (scenario suites, observation profile, action schema version),

   `configs/train/` (algorithm, model, hyperparameters, opponent mix), `configs/eval/` (fixed pools, seeds, league settings).

2. Strict schemas, no silent defaults. Python side validates with pydantic (unknown fields rejected, mirroring §11.1); every default lives in the schema definition, documented there, never as a bare constant in training code. The C++ worker keeps the spec §11 strict JSON scenario schema; YAML configs *reference* scenario files, they do not replace them.
3. Explicit layering only: a config may declare `extends: <relative path>` (deep merge, overlay wins). No CLI-only overrides that bypass the file record, a CLI override must be written into the resolved config that gets stamped (below).
4.

   Reproducibility rule (the load-bearing part): every produced artifact, trajectory header, benchmark report, checkpoint metadata, W&B run, embeds `{resolved_config, sha256(canonical_config), git_commit, schema_versions}`. Canonicalization follows §11.3 (sorted keys, no insignificant whitespace). An artifact whose config hash cannot be recomputed is treated as corrupted.
5. Tooling stance: plain YAML + pydantic now; no Hydra initially (composition magic vs traceability). Revisit only if sweep orchestration outgrows PBT/W&B, and record the change here.
6. Discoverability for future sessions: `configs/README.md` lists every axis with its schema location; this ADR is the pointer of record from `START_HERE.md`.

## Consequences

- Milestone 4's worker gains nothing new (scenario JSON already spec'd); Milestone 5's Python package ships the pydantic schemas, `configs/` seeds, and the stamping helper.
- Experiments become diffable files; "what did run X use?" is answered by the artifact itself, not by conversation history.
- Algorithm-variant choice (e.g. PPO vs GRPO-style) is a config enum plus a schema-versioned code path, never an untracked edit.
