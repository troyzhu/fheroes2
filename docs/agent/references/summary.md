---
title: "What the literature establishes — consolidated analysis"
type: synthesis
status: current
question: "How should an RL environment and agent for turn-based tactical battles be built, and what does that mean for fheroes2?"
sources: 35
verified_claims: "47 of 50 across two runs (3-vote adversarial verification)"
last_updated: 2026-07-30
related_concepts: ["[[../concepts/legal-actions-and-masking]]", "[[../concepts/observation-design]]", "[[../concepts/teacher-coverage-and-behavior-cloning]]"]
tags: [synthesis, rl, literature, agent-env]
---

> **What this note is.** The consolidated *findings* of two adversarially verified literature
> runs, written to be read as an argument rather than a log: what the evidence establishes, how
> confident we are, and what it implies for our design. Per-work detail lives in [[index|the
> vault notes]]; the full claim-by-claim reports with votes are
> `../research_rl_approaches.md` and `../research_minimap_observations.md`; how the sweeps were
> run is a method note at the [[#Provenance (method note)|end]], deliberately out of the way.
>
> **Confidence convention.** Claims below are *verified* (survived 3-vote adversarial
> verification) unless marked **[unverified]** (fetched and on-topic but never reached a verified
> claim set — cite the source directly, not us) or **[gap]** (no evidence survived; we are
> reasoning from absence).

## Table of contents
- [[#The answer in one paragraph]]
- [[#Key terms]]
- [[#1. The pipeline is proven end to end]]
- [[#2. Observations: structured always, planes optionally, pixels never]]
- [[#3. Actions: a fixed space with an engine-computed mask]]
- [[#4. Training: clone the teacher first, then reinforce]]
- [[#5. Evaluation: seeded pools plus a rating league]]
- [[#6. Partial observability: an option kept cheap]]
- [[#What the evidence does not cover]]
- [[#Standing cautions]]
- [[#Provenance (method note)]]

## The answer in one paragraph

Everything we needed to decide has a shipped precedent. A Gymnasium environment over an
open-source Heroes engine, trained with masked PPO on padded-entity + per-hex observations,
produced models that ship *inside the game* — so the architecture is not speculative. Legal-action
masking over a **fixed** action space is both provably correct and empirically decisive (0.0
versus 0.82–0.91 win rate in controlled ablations), which is the single strongest result in the
corpus and the one that amended our spec. Observations should be structured; where a "minimap"
appears in successful systems it is a *synthetic semantic rasterization*, never rendered pixels,
and the non-pixel path measured ~14× cheaper at equal performance. Behavior cloning before RL is
validated at the highest available scale. And a competitive agent in this genre was trained in
~60 hours on a single 16 GB machine — the hardware plan is realistic.

## Key terms

Terms used below that carry precise meanings in this literature.

| Term | Meaning |
|---|---|
| **Feature layer** | A coarse spatial plane rendered *from game state* by a synthetic top-down camera — the SC2 "minimap". Not a screenshot. |
| **Entity list** | Per-unit records consumed as a set, usually by attention; the alternative to per-cell planes. |
| **Scatter connection** | Injecting per-entity embeddings into spatial planes at each unit's location (AlphaStar) — entity→spatial fusion. |
| **Action masking** | Setting illegal logits to $-\infty$ before the softmax; provably a valid policy gradient. |
| **Factorized (composed) action** | Decomposing one action into independent discrete components, each with its own softmax, to avoid a combinatorial flat space. |
| **Observer (Griddly)** | A configurable view over one game state — VECTOR (semantic) or one of several pixel renderers. |
| **TrueSkill league** | Rating-based evaluation that schedules matches until rating uncertainty converges, rather than a fixed match count. |
| **PBT** | Population-based training: hyperparameter search that mutates a live population of runs. |
| **Tier A / B** | Source credibility grades used in the reports (peer-reviewed/official vs reputable press/self-reported). |

## 1. The pipeline is proven end to end

**vcmi-gym → MMAI** is the existence proof at genre-distance zero: an RL environment over the
open-source VCMI Heroes III engine whose trained models shipped in VCMI 1.7.0 as a selectable
battle AI. No fheroes2/HoMM2 RL environment exists — we would be the first.

**Gym-µRTS** is the hardware proof: state of the art against all thirteen tested past-competition
bots (91 % cumulative win rate) in ~60–63 hours on one machine with a single GPU and 16 GB RAM.
*(Caveat: single-map, CUDA GPU; no Apple-silicon MPS benchmark exists anywhere — **[gap]**.)*

**ARLinBfW** shows the integration shape works: a headless C++ game engine driven out-of-process
over a text channel. Our bidirectional JSONL-over-stdio is a strictly cleaner version of it.

**What this implies for us.** Nothing in the corpus contradicts our architecture. Two things
amended it — the action interface (§3) and the addition of a plane modality (§2).

## 2. Observations: structured always, planes optionally, pixels never

Every successful system in this genre consumes semantic state. SC2's feature layers are typed
categorical/scalar planes drawn from game state, and DeepMind's stated rationale is that agents
should not spend capacity learning to read numbers off a screen. vcmi-gym encodes a flat
12,685-float observation: 20 padded stack slots × 98 floats plus 165 hexes × 65 floats, using
one-hot categories with an explicit NULL for empty slots. MicroRTS uses per-cell one-hot planes
with no entity list at all. OpenAI Five used ~16,000 semantic values because rendering would have
multiplied compute **[unverified]**.

**Planes and entities are complementary, not rivals.** AlphaStar feeds an entity transformer, a
semantic-plane CNN, and a scalar vector into one core — and additionally *scatters* entity
embeddings into the planes. NetHack's baseline does the same at small scale, including a 9×9
egocentric-crop CNN, which is practice evidence for CNNs at board sizes like our 11×9.

**Pixels lose on cost with no compensating benefit.** Griddly runs the same games under a
non-pixel VECTOR observer and three real pixel renderers: RL performance was consistent across
representations over 150 experiments, while the vector observer ran ~72,800 FPS against ~5,000
rendered. For us the argument is stronger still — rendering would reintroduce the display and
asset dependencies our headless core exists to avoid.

**The Markov discipline.** vcmi-gym's hard-won lesson across 12+ observation iterations: an
attribute that influences dynamics must be *in the observation* or *out of the dynamics*. They
chose removal (deleting morale/luck/terrain effects). **We chose exposure** — mechanics stay live
and their fields appear in both profiles. See [[../concepts/observation-design]].

→ **Our decisions:** [[../decisions/0001-observation-profiles|ADR 0001]] (profiles),
[[../decisions/0004-spatial-observation-modality|ADR 0004]] (`planes_v1`, pixels rejected).

## 3. Actions: a fixed space with an engine-computed mask

This is the corpus's sharpest result and the one that changed our design.

**The theory.** Masking is a state-dependent differentiable transform of the logits, so the masked
update remains a valid policy gradient (Huang & Ontañón, FLAIRS 2022). Implement it by replacing
invalid logits with a large negative constant before the softmax — gradients on masked entries
become exactly zero — and apply the mask at *both* sampling and gradient time; sample-only masking
destabilizes PPO's KL.

**The evidence.** Full-game microRTS: PPO with no mask scored **0.0** cumulative win rate; masking
only the action *type* (the PySC2/SMAC style) reached 0.32; full per-component masking reached
0.82–0.91. In scaling ablations, penalty-based legality collapses on larger maps while masking
stays flat. **Never use penalties for legality.**

**The interface everyone ships** is a *fixed* discrete space plus a boolean mask — vcmi-gym's flat
`Discrete(2312)`, MicroRTS's per-cell factorized components. No verified codebase consumes
variable-length candidate lists, and vcmi-gym's factorized multi-head variant *failed to
converge*. Pointer/attention selection over enumerated candidates (AlphaStar) is the heavyweight
option, viable later.

**How this amended our spec.** The spec's per-decision ephemeral `action_id` list conflicted with
all of it. The fix keeps both properties: one engine enumeration now produces a fixed canonical
index/mask *and* the semantic candidate list. See [[../concepts/legal-actions-and-masking]].

→ **Our decision:** [[../decisions/0002-action-space|ADR 0002]].

## 4. Training: clone the teacher first, then reinforce

**Behavior cloning first is validated at the strongest available scale.** AlphaStar's purely
supervised stage reached 87 % win rate against the built-in Elite bot *before any RL*, with an
ablation ladder (0 → 7 → 36 → 71 → 87 %) attributing most of it to the pointer network,
transformer, and scatter connections. Our Milestone 2 passive teacher logs are exactly this
dataset.

**The stack that shipped** is single-file CleanRL-style masked implementations (MPPO, MPPG,
MPPO-DNA, MQRDQN) with Ray PBT and W&B; Stable-Baselines3 was prototyped and deliberately
dropped. For one Apple-silicon machine, a single-file masked PPO gives full control of device
placement, with sb3-contrib MaskablePPO as the off-the-shelf fallback.

**Opponent diversity at train time is load-bearing.** The microRTS SOTA trained against a mixture
(18 CoacAI + 2 each of three weaker bots across 24 parallel envs); single-opponent agents lose to
simple rushes.

**Keep the planning door open.** Stratega's agent API is forward-model-centric (~100k
simulations/s) precisely to serve MCTS/RHEA; vcmi-gym has unpublished MuZero experiments. Our
deterministic ~4,600 eps/s core is a first-class planning asset once the one-arena-per-process
constraint is addressed.

**[gap]** No verified small-scale BC→RL transition recipe (DAgger, KL-to-teacher, offline
warm-start) survived; vcmi-gym appears to have trained from scratch. Expect iteration.

## 5. Evaluation: seeded pools plus a rating league

Two protocols, both from microRTS practice, both adoptable directly:

- **Fixed-pool win rate** — N seeds per configuration, best seed evaluated over 100 games per
  scripted opponent under a step cap, reporting cumulative win rate, model size, and wall-clock.
- **TrueSkill league** — mixes scripted bots and checkpoints, scheduling matches *until rating
  uncertainty converges* (`while sigma > 1.4`), ranking by `mu − 3·sigma`.

**Calibration.** From the only shipped comparable system: ~75 % versus the weak scripted bot and
~45 % versus the strong one on the first working model; a much later iteration averaged ~65 %
versus the strong bot *(self-reported)*. **Parity with the engine's AI is a multi-iteration
goal.**

Our seeded-deterministic suites and SHA-256 digests *exceed* anything found in the corpus —
nothing comparable exists in the verified set **[gap]**, in our favor. Keep them.

## 6. Partial observability: an option kept cheap

Creature-only fheroes2 battles are informationally symmetric — the battle UI reveals any unit's
full stat sheet with no ownership gating — so hidden information is *randomness*, not fog. The
asymmetric/privileged-critic literature motivates keeping a full-state profile available at
near-zero cost, but **no claim on those setups survived verification [gap]**: that is an option,
not a plan. Real partial observability arrives with hero mana and adventure-map fog; the schema
seam is already in place.

## What the evidence does not cover

Stated plainly, because these are where confident-sounding claims would be unfounded:

1. **No published RGB-vs-feature-layer comparison** for SC2 — the promised study never appeared in
   verifiable form.
2. **No architecture ablation at ~11×9 board scale** — whether a CNN over planes beats an
   entity-transformer or an MLP here is unknown. Cheaply answerable in-house by swapping heads
   over the same planes.
3. **No small-scale BC→RL recipe**, and **no Apple-silicon MPS throughput data** at relevant model
   sizes.
4. **No hex-rasterization convention evidence** — we standardize on the engine's own 11×9 cell
   indexing and document it.
5. **No self-play league scheduling evidence at our scale** — start with scripted-opponent
   mixtures.

## Standing cautions

- **Version drift.** vcmi-gym's concrete numbers come from documentation of the v3 environment
  "as of Aug 2024"; v13+ exists in code. Copy design *patterns*, never constants.
- **Self-reported results.** All vcmi-gym win rates are the author's, never independently
  replicated.
- **Refuted claims — never cite:** that MMAI never shipped (it did, VCMI 1.7.0); that masking was
  added because SB3 lacked it (the author migrated deliberately); the NLE-vs-ALE throughput figure
  (14.4K vs 0.90K steps/s, refuted 0-3).
- **Source arithmetic is sloppy in places** and is reproduced faithfully in the reports: vcmi's
  "165×12=1320" should be 1980; µRTS's "301 vs 50 million" logits do not recompute exactly;
  AlphaStar's pointer-ablation bar is 36 %, not ~38 %.
- **MicroRTS-Py was deprecated by Farama in Aug 2025** — a canonical but frozen reference.
- **AlphaStar's component-level ablation magnitudes sit behind the Nature paywall**; only the fact
  that those components were ablated is openly quotable.

## Provenance (method note)

Two runs of the same pipeline, both in 2026:

| Run | Date | Question | Sources | Claims extracted | Verified | Outcome |
|---|---|---|---|---|---|---|
| RL approaches | 07-27 | Environment + agent state of the art | 23 | 115 | 25 → 23 confirmed, 2 refuted | ADR 0001, ADR 0002 |
| Minimap/hybrid observations | 07-29 | Should the agent get a coarse spatial view? | 20 | 96 | 25 → 24 confirmed, 1 refuted | ADR 0004 |

Method: 5 search angles fanned out per run → sources fetched and read → falsifiable claims
extracted → the top 25 by relevance verified by **three independent adversarial votes each**
(2 of 3 refutations kill a claim) → survivors merged into findings. Claims that never reached
verification are marked **[unverified]** above; topics where nothing survived are marked
**[gap]**. Local copies of every source (43 files) live in `files/`, with `manifest.tsv`
recording URL, status, size, and title; `fetch_references.sh` re-fetches reproducibly.

Full claim-level detail with votes and verbatim evidence: `../research_rl_approaches.md`,
`../research_minimap_observations.md`. Per-work notes: [[index]].
