---
title: Reference vault — index
type: moc
updated: 2026-07-30
related_concepts: ["[[../implementation/legal-actions-and-masking]]", "[[../implementation/observation-design]]"]
tags: [reference, index, moc, agent-env]
---

> **What this note is.** The scannable catalogue of every work behind the two verified literature runs. Read [[findings]] first for what the corpus establishes; come here to find a specific source, its quality, and where it is used. Provenance and vault mechanics are at the bottom.

For what each codebase actually contains and where to look inside it, read [[prior-art|the repository orientation]] first. Reading order for someone new to the corpus: [[works/vcmi-gym]] for the closest prior art, then [[works/gym-microrts]] for the masking evidence and single-machine feasibility, then [[works/sc2le]] for observation design.

## The corpus

| Source | Type | Year | Quality | What it establishes | Feeds |
|---|---|---|---|---|---|
| [[works/vcmi-gym]] | project | 2023–26 | primary | Heroes III battle RL shipped as an in-game AI; padded-entity plus per-hex encoding; flat masked action space; CleanRL-style stack | ADR 0002, ADR 0003 |
| [[works/botbowl-competitions]] | project | 2019–23 | primary | Four Blood Bowl bot competitions where strictly-from-scratch RL never beat a scripted bot; the one ML win required imitation bootstrap | tabula-rasa verdict, [[../rl/program-review]] |
| [[works/pokemon-selfplay-from-scratch]] | paper | 2019, 2025 | primary | Pure self-play PPO beats a tree-search bot for ninety-one dollars in 1v1; the same recipe loses to a simple heuristic in 6v6 until imitation-initialized | tabula-rasa verdict, self-play budgets |
| [[works/lux-ai-competitions]] | project | 2021-23 | primary | From-scratch self-play wins season one with shaped-then-sparse annealing and a frozen-teacher KL bridge; a rule-based bot wins season two over every learned entry | tabula-rasa verdict, annealing pattern |
| [[works/gym-microrts]] | paper | 2021 | primary | Masking ablations (0% unmasked to 82–91% fully masked); factorized heads; state of the art in ~60 h on one 16 GB machine | ADR 0002, hardware plan |
| [[works/invalid-action-masking]] | paper | 2022 | primary | Masking is a valid policy gradient; the $-10^8$ implementation; penalties collapse as the illegal space grows | ADR 0002 |
| [[works/dppo-trust-region]] | paper | 2026 | primary | PPO's clip constrains a one-sample estimate of TV divergence, not the divergence; over-penalizes low-probability actions and under-penalizes high ones. The exact divergence is affordable at 793 slots | [[../rl/rlhf-transfer]] |
| [[works/group-std-identity]] | paper | 2026 | primary | Mean centering is leave-one-out up to $G/(G-1)$, so Dr. GRPO and RLOO are one arm; the group reward std is the update size itself; DAPO's dynamic sampling is the silent-group drop | [[../rl/rlhf-transfer]], `objectives.py` |
| [[works/metamon]] | paper | 2025 | primary | Top-10% human ladder play in Pokémon Showdown with no search: imitation from reconstructed replays, then offline RL past the demonstrator, then offline self-play fine-tuning; standardized beginner-to-expert opponent suite | [[../rl/training-design]], opponent mixture |
| [[works/generalized-battle-agent-guide]] | report | 2026 | primary | Owner-supplied design for this exact problem: value-budget sampling, structured ability records, candidate effect summaries, lexicographic objective, scenario-family splits | sampler adopted, abilities tracked |
| [[works/sc2le]] | paper | 2017 | primary | Feature layers as synthetic semantic rasterizations, never RGB; structured tensors alongside planes | ADR 0004 |
| [[works/pysc2]] | project | 2017 | primary | The shipped observer for SC2LE: independently toggleable observation modalities behind one API | ADR 0001, ADR 0004 |
| [[works/alphastar-unplugged]] | paper | 2023 | primary | One-step improvement against the behavior value is the recipe that works offline at scale; MCTS at training time collapses, at inference improves | training program after the 2026-08-05 review |
| [[works/uct]] | paper | 2006 | primary | The UCB variant of Monte-Carlo planning: bandit selection per node, rollout values, anytime convergence | search probe |
| [[works/alphazero]] | paper | 2017 | primary | PUCT search as the improvement operator, distilled back each iteration with a re-grounded value | search-as-teacher design |
| [[works/one-step-offline-rl]] | paper | 2021 | primary | One improvement step against the behavior value beats iterating, because iteration queries the value off-data | improvement-step discipline |
| [[works/bcq-extrapolation]] | paper | 2019 | primary | Extrapolation error: value estimates queried off-data, then maximized into | the critic calibration result |
| [[works/double-q-overestimation]] | paper | 2016 | primary | Max-operator overestimation in bootstrapped values, absent from our MC-fitted critic, acquired the moment anything bootstraps | value-estimation boundary |
| [[works/expert-iteration]] | paper | 2017 | primary | Planning and generalization split into two jobs; imitation extended to domains where the best expert is not good enough | the search-teaching loop's name and framing |
| [[works/mcts-regularized-policy-optimization]] | paper | 2020 | primary | Visit counts approximate a regularized policy-optimization solution, poorly at small simulation budgets | the value-leaf probe and the distillation target |
| [[works/muzero]] | paper | 2020 | primary | Value trained on search returns and jointly with the policy; the learned model is what we do not need | the requirements for a usable leaf evaluator |
| [[works/cql]] | paper | 2020 | primary | Pessimistic Q lower-bounds the policy's value, so improvement cannot exploit off-data guesses | remedies survey, pessimism family |
| [[works/iql]] | paper | 2022 | primary | Expectile regression never evaluates unseen actions; policy extracted by advantage weighting | remedies survey, avoid-the-query family |
| [[works/awr]] | paper | 2019 | primary | Improvement as supervised regression on taken actions weighted by exponentiated advantage | the first arm this project runs |
| [[works/td3-bc]] | paper | 2021 | primary | A cloning term inside the policy loss matches elaborate offline machinery | confirms our stage-3 anchor |
| [[works/edac-ensembles]] | paper | 2021 | primary | Ensemble disagreement penalizes exactly where data is absent | remedies survey, uncertainty family |
| [[works/gae]] | paper | 2016 | primary | The bias-variance dial of credit assignment; its low-bias end pays in value accuracy our critic does not have off-distribution | credit-assignment measurement |
| [[works/alphastar]] | paper | 2019 | primary | Entity transformer, semantic minimap, scatter connections; supervised stage reaching 87% before any RL | ADR 0004, BC staging |
| [[works/griddly]] | project | 2020–23 | primary | Multiple observers over one state; semantic planes match pixel observers at roughly 14× the throughput | ADR 0004 |
| [[works/microrts-py]] | codebase | 2021–25 | primary | `CategoricalMasked` reference implementation; `partial_obs` flag; TrueSkill league evaluation | ADR 0001, evaluation |
| [[works/nle]] | project | 2020 | primary | Symbolic multi-component observations; CNN over embedded per-cell glyphs at small board scale | ADR 0004 |
| [[works/stratega]] | project | 2020 | primary | Forward-model-centric agent API for planning methods | planning option |
| [[works/arlinbfw]] | codebase | 2019 | primary | Headless C++ game engine driven out-of-process over a text channel | worker architecture |
| [[works/entity-based-rl]] | project | 2022–23 | mixed | Entity-list APIs and ragged-batch transformers | upgrade path |
| [[works/openai-five]] | paper | 2019 | primary | Structured arrays over pixels at scale | pixel-cost rationale |
| [[works/asymmetric-actor-critic]] | paper-group | 2017–22 | primary | Privileged-critic and recurrent POMDP baselines | ADR 0001 (option only) |
| [[works/misc-pipeline-sources]] | collection | mixed | mixed | Sample Factory, Lux AI winner, board-game scaling laws, NetHack follow-up, FootsiesGym | background |

Quality reads as the source class, not as our endorsement. Two entries carry caveats that matter at the point of use: [[works/entity-based-rl]] and [[works/openai-five]] contributed claims that never reached a verified claim set, so cite those sources directly rather than citing us.

## How the corpus maps to our decisions

| Decision | Anchored by |
|---|---|
| ADR 0001, observability profiles | [[works/microrts-py]], [[works/asymmetric-actor-critic]], plus the engine's own `WAR_INFO` behavior |
| ADR 0002, fixed action space with mask | [[works/invalid-action-masking]], [[works/gym-microrts]], [[works/vcmi-gym]], [[works/alphastar]] |
| ADR 0003, versioned configuration | [[works/vcmi-gym]] |
| ADR 0004, semantic planes and no pixels | [[works/sc2le]], [[works/griddly]], [[works/alphastar]], [[works/nle]] |

## Provenance

Two runs of the same pipeline produced this corpus: literature on environment and agent design (2026-07-27, 23 sources) and coarse-spatial observation design (2026-07-29, 20 sources). Counting the overlap, that is roughly 35 distinct works, 43 fetched source files, and 15 per-work notes.

Local copies live in `files/`, with `manifest.tsv` recording the URL, fetch status, byte size, and title of every file. `fetch_references.sh` re-fetches them reproducibly. One repository ships no README upstream, which the manifest records as a failed fetch rather than hiding.

Opening `agent_play/docs/` or the repository root as an Obsidian vault resolves every wikilink here.

## Related

- [[findings]], what the corpus establishes, with confidence markers.
- [[../archive/research-runs/2026-07-27-rl-approaches]] and [[../archive/research-runs/2026-07-29-spatial-observations]], the claim-by-claim reports with verification votes.
- [[prior-art]], an orientation to the open-source codebases behind it.
- [[../implementation/README|Concept primers]], the teaching layer these findings feed.
- [[../overview]], the system as it stands.
