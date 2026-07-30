---
title: What the literature establishes — consolidated analysis
type: synthesis
status: current
question: "How should an RL environment and agent for turn-based tactical battles be built, and what does that mean for fheroes2?"
updated: 2026-07-30
related_concepts: ["[[../implementation/legal-actions-and-masking]]", "[[../implementation/observation-design]]", "[[../implementation/teacher-coverage-and-behavior-cloning]]"]
tags: [synthesis, rl, literature, agent-env]
---

> **What this note is.** The consolidated findings of two adversarially verified literature runs, written as an argument rather than a log: what the evidence establishes, how confident we are, and what it implies for the design. Per-work detail is in [[README|the vault index]], the claim-by-claim reports with votes are [[../archive/research-runs/2026-07-27-rl-approaches]] and [[../archive/research-runs/2026-07-29-spatial-observations]], and the sweep mechanics are a method note at the end.
>
> **Confidence convention.** A claim is verified unless marked (unverified), meaning it was fetched and on-topic but never reached a verified claim set, so cite the source rather than this note, or (gap), meaning no evidence survived and the reasoning is from absence.

## Table of contents
- [[#Key terms]]
- [[#The answer in one paragraph]]
- [[#1. The pipeline is proven end to end]]
- [[#2. Observations: structured always, planes optionally, pixels never]]
- [[#3. Actions: a fixed space with an engine-computed mask]]
- [[#4. Training: clone the teacher first, then reinforce]]
- [[#5. Evaluation: seeded pools plus a rating league]]
- [[#6. Partial observability: an option kept cheap]]
- [[#What the evidence does not cover]]
- [[#Standing cautions]]
- [[#Provenance (method note)]]

## Key terms

| Term | Meaning |
|---|---|
| Feature layer | A coarse spatial plane rendered from game state by a synthetic top-down camera, the SC2 "minimap". Not a screenshot. |
| Entity list | Per-unit records consumed as a set, usually by attention; the alternative to per-cell planes. |
| Scatter connection | Injecting per-entity embeddings into spatial planes at each unit's location, the AlphaStar entity-to-spatial fusion. |
| Action masking | Setting illegal logits to a large negative constant before the softmax; provably a valid policy gradient. |
| Factorized action | One action split into independent discrete components, each with its own softmax, avoiding a combinatorial flat space. |
| Observer (Griddly) | A configurable view over one game state, either the semantic VECTOR observer or one of several pixel renderers. |
| TrueSkill league | Rating-based evaluation that schedules matches until rating uncertainty converges rather than to a fixed match count. |
| PBT | Population-based training: hyperparameter search that mutates a live population of runs. |
| DAgger | Dataset Aggregation: iterative imitation learning that labels the states the student actually visits, correcting distribution shift. |
| RHEA | Rolling-horizon evolutionary algorithm, a planning method rather than a learner. |
| MPPO family | vcmi-gym's four single-file masked learners, built on CleanRL implementations. |
| Weights and Biases | The experiment-tracking service those runs logged to. |
| Tier A and Tier B | Source credibility grades in the underlying reports: peer-reviewed or official against reputable press or self-reported. |

## The answer in one paragraph

Everything we needed to decide has a shipped precedent. A Gymnasium environment over an open-source Heroes engine, trained with masked PPO on padded-entity and per-hex observations, produced models that ship inside the game, so the architecture is not speculative. Legal-action masking over a fixed action space yields an unbiased gradient of the masked policy's objective, and in the one controlled ablation available it moved a cumulative win rate from 0% to between 82% and 91%. That finding amended our specification.

It rests on a single environment and has not been replicated elsewhere. Observations should be structured, and where a minimap appears in successful systems it is a synthetic semantic rasterization rather than rendered pixels, with the non-pixel path measuring roughly 14 times cheaper at equal performance. Behavior cloning before reinforcement learning is validated at the highest available scale. A competitive agent in this genre was trained in about 60 hours on a single 16 GB machine, so the hardware plan is realistic.

## 1. The pipeline is proven end to end

[[works/vcmi-gym|vcmi-gym]] and its shipped model, MMAI, are the existence proof at genre-distance zero: an RL environment over the open-source VCMI Heroes III engine whose trained models shipped in VCMI 1.7.0 as a selectable battle AI. No fheroes2 or HoMM2 RL environment exists, so this project would be the first.

[[works/gym-microrts|Gym-µRTS]] is the hardware proof, reaching state of the art against all thirteen tested past-competition bots at a 91% cumulative win rate, in about 60 to 63 hours on one machine with a single GPU and 16 GB of memory. The caveat is that the result is single-map and used a CUDA GPU, and no Apple-silicon benchmark exists anywhere (gap).

[[works/arlinbfw|ARLinBfW]] shows the integration shape works, driving a headless C++ game engine out-of-process over a text channel. Our bidirectional JSONL-over-stdio is a cleaner version of the same idea.

Nothing in the corpus, which we assembled ourselves, contradicts our architecture. Two findings amended it: the action interface in section 3, and the addition of a plane modality in section 2.

## 2. Observations: structured always, planes optionally, pixels never

Every system in the corpus consumes semantic state. [[works/sc2le|SC2's feature layers]] are typed categorical and scalar planes drawn from game state, on DeepMind's stated rationale that agents should not spend capacity learning to read numbers off a screen. vcmi-gym encodes a flat 12,685-float observation, being 20 padded stack slots of 98 floats plus 165 hexes of 65 floats, with one-hot categories and an explicit NULL for empty slots. [[works/microrts-py|MicroRTS]] uses per-cell one-hot planes and no entity list at all. [[works/openai-five|OpenAI Five]] used about 16,000 semantic values because rendering would have multiplied compute (unverified).

Planes and entities are complementary rather than rival. [[works/alphastar|AlphaStar]] feeds an entity transformer, a plane convolutional network, and a scalar vector into one core, and additionally scatters entity embeddings into the planes. [[works/nle|NetHack's baseline]] does the same at small scale, including a 9 by 9 egocentric-crop convolutional network, which is practice evidence for convolutional policies at board sizes like our 11 by 9.

Pixels lose on cost with no compensating benefit. [[works/griddly|Griddly]] runs the same games under a non-pixel VECTOR observer and three real pixel renderers, and task performance was consistent across representations over 150 experiments while the vector observer ran at about 72,800 frames per second against about 5,000 rendered. For us the argument is stronger still, since rendering would reintroduce the display and asset dependencies the headless core exists to avoid.

One discipline underlies all of it. vcmi-gym's lesson across more than twelve observation iterations is that an attribute influencing the dynamics must be either in the observation or out of the dynamics. They chose removal, deleting morale, luck, and terrain effects. We chose exposure, so those mechanics stay live and their fields appear in both profiles. See [[../implementation/observation-design]].

Our decisions: [[../decisions/0001-observation-profiles|ADR 0001]] for profiles, and [[../decisions/0004-spatial-observation-modality|ADR 0004]] for `planes_v1` with pixels rejected.

## 3. Actions: a fixed space with an engine-computed mask

This finding changed our design.

The theory holds that masking is a state-dependent differentiable transform of the logits, so the masked update remains a valid policy gradient ([[works/invalid-action-masking|Huang and Ontañón, FLAIRS 2022]]). Implement it by replacing invalid logits with a large negative constant before the softmax, written as $-\infty$ in the math and as $-10^8$ in code, and apply the mask at both sampling and gradient time. Masking only at sampling leaves the stored behavior log-probabilities and the recomputed ones drawn from different distributions, so the PPO ratio compares two distributions and the KL estimate diverges.

The evidence is a controlled ablation in full-game microRTS. PPO with no mask scored a 0% cumulative win rate, masking only the action type reached 32%, and full per-component masking reached 82% to 91%. In scaling ablations, penalty-based legality collapses on larger maps while masking holds roughly constant.

The interface everyone ships is a fixed discrete space plus a boolean mask, whether vcmi-gym's flat 2,312-slot space or MicroRTS's per-cell factorized components. No verified codebase consumes variable-length candidate lists, and vcmi-gym's factorized multi-head variant failed to converge. Pointer selection over enumerated candidates, as in AlphaStar, is the heavyweight option and stays available later.

This is what amended our specification. The original per-decision ephemeral action list conflicted with all of the above, and the fix keeps both properties, since one engine enumeration now produces a fixed canonical index with its mask and the semantic candidate list. See [[../implementation/legal-actions-and-masking]] and [[../decisions/0002-action-space|ADR 0002]].

## 4. Training: clone the teacher first, then reinforce

Behavior cloning first is validated at the strongest available scale. [[works/alphastar|AlphaStar's]] purely supervised stage reached an 87% win rate against the built-in Elite bot before any reinforcement learning, with the openly published ablation attributing most of the gain to the pointer network, the transformer, and scatter connections. Our Milestone 2 passive teacher logs are exactly this dataset; see [[../implementation/teacher-coverage-and-behavior-cloning]].

The stack that shipped is a set of single-file masked learners built on CleanRL, tuned with population-based training and tracked in Weights and Biases; Stable-Baselines3 was prototyped and then deliberately dropped. For one Apple-silicon machine, a single-file masked PPO gives full control of device placement, with sb3-contrib's MaskablePPO as the off-the-shelf fallback (see [[prior-art]]).

Train against a mixture of opponents. The microRTS result trained against a mixture across 24 parallel environments, and single-opponent agents lose to simple rushes.

The planning door stays open. [[works/stratega|Stratega's]] agent API is forward-model-centric at roughly 100,000 simulations per second precisely to serve MCTS and RHEA, and vcmi-gym has unpublished MuZero experiments. Our deterministic core at about 4,600 episodes/s is a first-class planning asset once the one-arena-per-process constraint is addressed.

No verified small-scale recipe survived for the transition from cloning to reinforcement learning, covering DAgger, KL-to-teacher regularization, or offline warm-starts, and vcmi-gym appears to have trained from scratch (gap). Expect iteration.

## 5. Evaluation: seeded pools plus a rating league

Two protocols come from [[works/microrts-py|microRTS]] practice and are adoptable directly. A fixed-pool win rate uses N seeds per configuration, evaluates the best seed over 100 games per scripted opponent under a step cap, and reports cumulative win rate, model size, and wall-clock time. A TrueSkill league mixes scripted bots and checkpoints, scheduling matches until rating uncertainty converges below a threshold and ranking by $\mu - 3\sigma$.

Calibration comes from the only shipped comparable system, which reported about 75% against the weak scripted bot and 45% against the strong one on its first working model, and about 65% against the strong bot on a much later iteration, all self-reported. Parity with the engine's AI is a multi-iteration goal.

No source in the corpus reports episode-level state digests or seeded regression suites, so there is no precedent to calibrate ours against (gap); ours are described in [[../implementation/determinism-seeds-and-digests]]. Treat them as cheap and locally validated rather than as a demonstrated edge.

## 6. Partial observability: an option kept cheap

Creature-only fheroes2 battles are informationally symmetric, since the battle interface reveals any unit's full stat sheet with no ownership gating, so the hidden information is randomness rather than fog. The asymmetric and privileged-critic literature motivates keeping a full-state profile available at near-zero cost, but no claim about those setups survived verification (gap), which makes it an option rather than a plan. Genuine partial observability arrives with hero mana and adventure-map fog, and the schema seam is already in place.

## What the evidence does not cover

Stated plainly, because these are where confident-sounding claims would be unfounded.

No published comparison of RGB against feature layers exists for SC2, since the promised study never appeared in verifiable form. No architecture ablation exists at an 11 by 9 board scale, so whether a convolutional network over planes beats an entity transformer or a plain multilayer perceptron here is unknown, and it is cheaply answerable in-house by swapping heads over the same planes. No small-scale cloning-to-reinforcement recipe and no Apple-silicon throughput data exist at relevant model sizes. No evidence favors any hex-rasterization convention, so we standardize on the engine's own cell indexing and document it. No self-play league scheduling evidence exists at our scale, so scripted-opponent mixtures come first.

## Standing cautions

vcmi-gym's concrete numbers describe the environment version documented as of August 2024, while later versions exist in code, so copy design patterns rather than constants. All of its win rates are the author's own and were never independently replicated.

Three claims were refuted during verification and must never be cited: that MMAI never shipped, when it shipped in VCMI 1.7.0; that masking was added because Stable-Baselines3 lacked it, when the author migrated deliberately; and a NetHack-against-Atari throughput comparison that lost 0 to 3.

Source arithmetic is sloppy in places and the underlying reports reproduce it faithfully. The vcmi-gym blog's melee-action count states 165 times 12 as 1,320 where the sum requires 1,980; the Gym-µRTS paper's claim that factorization cuts roughly 50 million joint actions to 301 logits does not recompute exactly from its own tables; and AlphaStar's pointer-network ablation bar reads 36% rather than the 38% sometimes quoted.

[[works/microrts-py|MicroRTS-Py]] was deprecated by Farama in August 2025, making it a canonical but frozen reference. AlphaStar's component-level ablation magnitudes sit behind the Nature paywall, so only the fact that those components were ablated is openly quotable.

## Provenance (method note)

Two runs of the same pipeline, both in 2026.

| Run | Date | Question | Sources | Claims extracted | Verified | Outcome |
|---|---|---|---|---|---|---|
| Environment and agent design | 07-27 | State of the art for this genre | 23 | 115 | 25 checked, 23 confirmed, 2 refuted | ADR 0001, ADR 0002 |
| Spatial observations | 07-29 | Should the agent get a coarse spatial view | 20 | 96 | 25 checked, 24 confirmed, 1 refuted | ADR 0004 |

Counting the overlap between runs, the corpus covers roughly 35 distinct works held as 43 fetched source files with 15 per-work notes. Each run fanned out five search angles, fetched and read the sources, extracted falsifiable claims, then put the 25 most relevant through three independent adversarial votes each, where two refutations of three kill a claim. Survivors were merged into the findings above.

Local copies live in `files/`, with `manifest.tsv` recording the URL, status, size, and title of every file, and `fetch_references.sh` re-fetching them reproducibly.

## Related

- [[README]], the scannable source catalogue with quality grades.
- [[prior-art]], what each open-source codebase contains and where to look inside it.
- [[../archive/research-runs/2026-07-27-rl-approaches]] and [[../archive/research-runs/2026-07-29-spatial-observations]], the claim-by-claim reports with verification votes.
- [[../rl-and-the-battle-domain]], the RL vocabulary and the battle domain these findings apply to.
- [[../README]], the system as it stands.
