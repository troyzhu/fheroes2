---
title: "Consolidated reference summary and analysis"
tags: [reference, synthesis]
updated: 2026-07-29
---

# Consolidated reference summary and analysis

What the whole corpus (two verified research runs, ~35 unique works, 43 local files)
collectively establishes for the fheroes2 battle agent — with per-claim provenance living in
the two full reports ([[../research_rl_approaches]], [[../research_minimap_observations]]) and
per-work detail in the [[index|vault notes]]. Verification legend: claims cited here survived
3-vote adversarial verification unless marked *(unverified)*.

## 1. The pipeline we are building is proven end-to-end

[[ref-vcmi-gym]] is the existence proof at genre-distance zero: a Gymnasium env over an
open-source HoMM engine, masked CleanRL-style PPO, padded-slot + per-hex observations — and its
models ship inside VCMI 1.7.0 as a selectable battle AI. [[ref-gym-microrts]] is the hardware
proof: SOTA vs all past competition bots in ~60 h on one 16 GB machine. [[ref-arlinbfw]] proved
a headless C++ engine + out-of-process text protocol works at all. Nothing in the corpus
contradicts our architecture; two things amended it (below).

## 2. Observations: structured always; spatial planes as a modality; pixels never

Every successful system consumes semantic state: SC2's typed feature layers ([[ref-sc2le]],
[[ref-pysc2]]), vcmi-gym's 12,685-float padded encoding, NLE's symbolic grids, OpenAI Five's
arrays *(rationale unverified)*. Where a "minimap" exists it is a **synthetic rasterization of
game state**, never captured render output — and [[ref-griddly]] shows semantic planes match
pixel observers on RL performance at ~14× the throughput. Hybrids are the production norm:
entity list + spatial planes + scalar vector, concatenated ([[ref-alphastar]], [[ref-nle]]),
with scatter connections as the entity→spatial fusion. Hence ADR 0001 (profiles), ADR 0004
(planes_v1 modality, pixels rejected). Known gaps: no published RGB-vs-feature comparison; no
head ablations at 11×9 scale (we will run our own).

## 3. Actions: fixed canonical space + engine-computed mask

Masking is provably a valid policy gradient and empirically decisive (0.0 → 0.82–0.91
cumulative win rate; penalties collapse) — [[ref-invalid-action-masking]],
[[ref-gym-microrts]]. Every shipped system uses a fixed space + boolean mask; vcmi-gym's
factorized variant failed to converge; variable-length candidate lists appear nowhere. Hence
ADR 0002: one engine enumeration feeds both the canonical mask and the semantic candidate list,
keeping AlphaStar-style pointer heads compatible.

## 4. Training: BC from the scripted teacher, then masked PPO vs diverse opponents

[[ref-alphastar]]'s supervised stage (87 % vs Elite bot before any RL) validates
demonstrations-first; our Milestone 2 passive teacher logs are exactly that substrate.
The shipped stack precedent is single-file CleanRL-style masked PPO with PBT tuning
([[ref-vcmi-gym]]); opponent diversity at train time is load-bearing ([[ref-gym-microrts]]).
Gaps: no verified small-scale BC→RL transition recipe; no Apple-silicon MPS benchmarks; no
verified self-play league scheduling at our scale — start with scripted-opponent mixes.
The forward-model/planning door stays open ([[ref-stratega]]) thanks to the deterministic core.

## 5. Evaluation: seeded fixed pools + TrueSkill with uncertainty stopping

Fixed scripted-opponent pools × N seeds × 100 games under a step cap, plus a TrueSkill league
that stops when rating sigma converges ([[ref-microrts-py]], [[ref-gym-microrts]]). Calibration
from the only shipped HoMM system: ~45 % → ~65 % vs the strong scripted bot took many
iterations *(self-reported)* — parity with fheroes2's AI is a multi-iteration goal. Our
SHA-256 state/decision digests exceed anything verified in the corpus; keep them.

## 6. Partial observability: an option kept cheap, not a requirement

Creature-only HoMM2 battles are informationally symmetric (engine-verified `WAR_INFO` fact);
hidden information is RNG, not fog. The asymmetric/privileged-critic literature
([[ref-asymmetric-actor-critic]]) motivates keeping `full_v1` as an oracle profile at
near-zero cost, but **no claim on those setups survived verification** — a marked gap, not a
plan. Real PO arrives with hero mana / adventure-map fog; the schema seam is already in place.

## Standing cautions (from the verification record)

- vcmi-gym constants are version-drifted (docs "Aug 2024"; v13+ envs exist) — copy patterns,
  not numbers; its win rates are self-reported.
- Refuted, never cite: "MMAI never shipped" (it did); "masking added because SB3 lacked it";
  the NLE-vs-ALE 14.4K/0.9K throughput figure.
- Source arithmetic sloppiness is reproduced faithfully in the reports (vcmi "165×12=1320";
  µRTS "301 logits"; AlphaStar pointer bar 36 % not 38 %).
- [[ref-microrts-py]] is deprecated (Aug 2025) — frozen reference.
- AlphaStar component-level ablation magnitudes sit behind the Nature paywall.
