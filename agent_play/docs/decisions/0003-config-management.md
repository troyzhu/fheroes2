---
title: "ADR 0003, versioned configuration governs every tunable"
type: adr
status: accepted
updated: 2026-08-03
related_concepts: ["[[../rl/training-design]]", "[[../overview]]"]
tags: [adr, config, reproducibility, agent-env]
---

# ADR 0003 — Versioned config files govern every tunable; artifacts embed their resolved config

- Status: accepted 2026-07-27, binding from Milestone 4 onward
- Implementation: not built. No `configs/` tree exists at the repository root as of 2026-07-31, which is consistent with the record binding from Milestone 4 rather than a lapse.
- Evidence: user requirement 2026-07-27; spec §11 (scenario schema), §15 (trajectory metadata), §16.4 (determinism metadata); [[../archive/research-runs/2026-07-27-rl-approaches]] §4; [[../research/works/vcmi-gym]] as the shipped precedent
- Hyperparameters this governs: [[../rl/training-design]]

## Table of contents
- [[#Context]]
- [[#The sub-problem]]
- [[#Options considered]]
- [[#Why this one, and what it cost]]
- [[#Decision]]
- [[#Consequences]]

## Context

The project is about to accumulate many configuration axes: scenario suites, observation profile (ADR 0001), action-schema version (ADR 0002), the reinforcement-learning algorithm variant (masked PPO first, with phasic policy gradient, PPO-DNA, and group-relative variants as an open axis, all defined in [[../rl/rl-methods]] and [[../rl/rlhf-transfer]]), network architecture, hyperparameters, the opponent mixture meaning which configurations of the built-in AI the agent trains against, evaluation suites, seeds, and worker counts. Scattering these across code constants, CLI flags and session memory does not survive multi-session work, neither for humans nor for a coding agent picking the project up cold. [[../research/works/vcmi-gym|vcmi-gym]], the shipped precedent, runs YAML-configured training with experiment tracking and population-based training, meaning a population of runs whose hyperparameters are periodically copied from better performers and perturbed. Our own determinism discipline already hashes scenarios (§11.3).

## The sub-problem

Where does a tunable live, and how does a finished artifact prove which values produced it?

The project is about to acquire many independent axes: scenario suites, the observation profile from [[0001-observation-profiles]], the modality from [[0004-spatial-observation-modality]], the algorithm and its hyperparameters from [[0005-training-and-reward]] and [[../rl/training-design]], opponent mixtures, evaluation pools, seeds. The failure this guards against is specific and was named by the owner. A coding agent picking the project up cold, or a human six months later, cannot tell what a number came from if the answer lives in conversation history.

## Options considered

| Option | What it is | For | Against |
|---|---|---|---|
| Constants in code | Values live where they are used | Nothing to build | A change is a code change, and a past run is unrecoverable without archaeology through the diff |
| Command-line flags | Values passed at launch | Easy to sweep | The record of a run lives in a shell history that is not kept. Silent divergence between what was intended and what ran |
| Plain YAML plus a strict schema (chosen) | Files under `configs/`, validated, with every artifact stamping its resolved config and hash | Diffable, reviewable, and an artifact answers for itself | Requires schema maintenance, and a stamping helper that every producer must call |
| Hydra or a similar composition framework | Structured composition, sweeps, overrides | Powerful sweep orchestration | Composition indirection makes it harder to answer what a run actually used, which is the one property this record exists to guarantee |
| Experiment-tracker as source of truth | Let the tracker record parameters | Zero local machinery | Values become unreconstructible if the tracker is unavailable, and the tracker records what it was told, not what ran |

## Why this one, and what it cost

The load-bearing requirement is not configuration but reproducibility, and that inverts how the options rank. Every option can hold values; only one makes an artifact self-describing. Requiring each artifact to embed its resolved configuration, that configuration's hash, the commit, and the schema versions means the question of what produced a number is answered by the number's own file. An artifact whose hash cannot be recomputed is treated as corrupted, which is a stronger contract than a convention.

Hydra was rejected for the reason it is usually adopted. Its composition is convenient and it makes the resolved configuration harder to state plainly, and this record's whole purpose is that the resolved configuration is stated plainly. That is a defensible trade to revisit if sweep orchestration outgrows what population-based training and an experiment tracker provide, and revisiting it means amending this record.

The cost is discipline that nothing enforces yet. A command-line override that bypasses the file record would break the guarantee silently, which is why the record forbids overrides that are not written back into the stamped configuration. Until Milestone 5 ships the stamping helper, that rule is a convention rather than a mechanism.

## Decision

1. One `configs/` tree at the repo root, YAML, split by concern: `configs/env/` (scenario suites, observation profile, action schema version),

   `configs/train/` (algorithm, model, hyperparameters, opponent mix), `configs/eval/` (fixed pools, seeds, league settings).

2. Strict schemas, no silent defaults. The Python side validates with pydantic, a library that turns a typed class declaration into a validator and rejects unknown fields, mirroring §11.1. Every default lives in the schema definition, documented there, never as a bare constant in training code. The C++ worker keeps the spec §11 strict JSON scenario schema; YAML configs *reference* scenario files, they do not replace them.
3. Explicit layering only: a config may declare `extends: <relative path>` (deep merge, overlay wins). No CLI-only overrides that bypass the file record, a CLI override must be written into the resolved config that gets stamped (below).
4.

   Reproducibility rule (the load-bearing part): every produced artifact, trajectory header, benchmark report, checkpoint metadata, W&B run, embeds `{resolved_config, sha256(canonical_config), git_commit, schema_versions}`. Canonicalization follows §11.3 (sorted keys, no insignificant whitespace). An artifact whose config hash cannot be recomputed is treated as corrupted.
5. Tooling stance: plain YAML + pydantic now; no Hydra initially (composition magic vs traceability). Revisit only if sweep orchestration outgrows PBT/W&B, and record the change here.
6. Discoverability for future sessions: `configs/README.md` lists every axis with its schema location; this ADR is the pointer of record from [[../overview]].

## Consequences

- Milestone 4's worker gains nothing new (scenario JSON already spec'd); Milestone 5's Python package ships the pydantic schemas, `configs/` seeds, and the stamping helper.
- Experiments become diffable files; "what did run X use?" is answered by the artifact itself, not by conversation history.
- Algorithm-variant choice (e.g. PPO vs GRPO-style) is a config enum plus a schema-versioned code path, never an untracked edit.
