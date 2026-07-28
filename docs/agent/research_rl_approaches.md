# RL approaches for the fheroes2 battle environment — verified literature consolidation

> Produced 2026-07-27 by a fan-out research pipeline: 5 search angles → 23 primary sources fetched
> → 115 falsifiable claims extracted → top 25 adversarially verified (3 independent votes each,
> 2/3 refutations kill) → 23 confirmed, 2 refuted → merged into the 15 findings below. Verdicts,
> votes and verbatim evidence quotes were preserved; **[gap]** marks topics where no claim survived
> verification and conclusions are weaker. Read alongside
> `agent_play/fheroes2_agent_system_spec_v0.3.md`; design decisions extracted from this report
> live in `docs/agent/decisions/`.

## The one-paragraph answer

The verified state of the art converges on one recipe for exactly our genre and hardware class:
encode structured (non-pixel) state as a **fixed-slot padded entity list plus per-cell board
features**; expose a **fixed canonical discrete action space with an engine-computed legal-action
mask** (provably a valid policy gradient, and empirically the difference between 0 % and ~82–91 %
win rate); train **CleanRL-style masked PPO against a mixture of scripted opponents**; evaluate by
**seeded fixed-pool win rates plus a TrueSkill league with uncertainty-based stopping**. The
pipeline is proven end-to-end: vcmi-gym's trained models ship inside VCMI 1.7.0 as the MMAI battle
AI for HoMM3, and Gym-microRTS reached state of the art in ~60 h on a single 16 GB machine. One
element of our spec conflicts with all verified practice — the *variable-length* `action_id`
candidate list (see "Design deltas" and ADR 0002).

## 1. Observation representation (structured, never pixels)

- **Closest template — vcmi-gym (HoMM3, 165-hex board)** encodes battle state as a flat 12,685-float
  Box: a fixed-slot padded entity list (20 stack slots × 98 floats — id, coordinates, side,
  quantity, attack/defense, HP, speed, morale, luck, abilities) plus per-hex features (165 × 65,
  including passability and per-hex action-mask bits), using one-hot "categorical explicit with a
  dedicated NULL category" for empty slots, plus binary and normalized encodings. Verified 3-0
  against `doc/env_info.md`. For our 99-cell board with ≤5 stacks/side: **per-cell features + 10
  padded stack slots with explicit NULL categories is the proven baseline**; tokenization and GNNs
  are optimizations, not prerequisites. *(Caveat: numbers are the documented VCMI-v3 env "as of
  Aug 2024"; docs flagged outdated — copy the pattern, not the constants.)*
- **The other proven pattern — per-cell one-hot planes** (Gym-microRTS / MicroRTS-Py): an
  `(h, w, 29)` binary tensor (HP 5, resources 5, owner 3, unit type 8, action 6, terrain 2), no
  entity list at all. Verified 3-0 (paper arXiv:2105.13807 + repo).
- **High-capacity end — AlphaStar** (Nature 2019): variable-length entity list (≤512) through
  transformer self-attention, spatial merge via scatter connections, deep LSTM for partial
  observability, autoregressive action heads with a pointer network over the entity set. The
  supervised ablation credits these components with most performance: 0 → 7 → 36 (+pointer) → 71
  (+transformer) → 87 % (+scatter) win rate vs the Elite bot — **before any RL**. Verified 3-0.
- **Markov discipline (transferable lesson from vcmi-gym's 12+ observation iterations)**: any
  attribute that influences battle dynamics must be *in* the observation or *out of the dynamics*
  — vcmi-gym deliberately **removed** terrain/morale/luck effects rather than half-observe them
  ("a cardinal sin of RL environment design is to ignore the Markov property"). Its v12+ models
  moved to a heterogeneous graph (165 HEX nodes, 7 edge types, GENConv) that "seemed to perform
  best" (~65 % vs BattleAI). Verified 3-0. **Our stance differs**: we keep morale/luck mechanics
  but expose their state fields in every observation profile (spec §12.3 already does), which
  preserves Markov structure with stochastic transitions; see ADR 0001.
- Structured-state vs pixels: no pixel path was ever competitive in the verified set (OpenAI Five's
  structured-arrays-because-rendering-multiplies-compute rationale and the entity-based RogueNet
  50×-fewer-params result appeared in extraction but did not reach the verified top-25 — **[gap]**,
  directionally consistent, cite with care).

## 2. Full vs partial observability — dual modes

- **Precedent for dual modes in one env**: MicroRTS-Py exposes full observability by default and a
  partial mode via a constructor flag (`partial_obs=True`) that **appends two visibility planes to
  the same tensor** (29 → 31 channels) instead of defining a second schema. Verified 3-0. This is
  the pattern ADR 0001 adopts (a profile field, one schema).
- **Engine fact (verified locally, not by the pipeline)**: fheroes2's battle UI shows the full unit
  info sheet for *any* clicked unit with no ownership gating (`battle_interface.cpp`
  `Cursor::WAR_INFO` → `Dialog::ArmyInfo`), so creature-only battles are informationally symmetric.
  The only "privileged" fields in our full state are engine-internal values (e.g.
  `engine_strength`) and RNG internals.
- **Asymmetric/oracle-critic setups**: the angle was searched (Baisero & Amato's unbiased
  asymmetric actor-critic, arXiv:2105.11674, and related sources were fetched) but **no claim on
  this topic survived to verification — [gap]**. Open question recorded: whether POMDP machinery is
  needed at all for HoMM2 battles, where hidden information is mostly RNG rather than fog of war.
  Keeping a full-state profile available (ADR 0001) preserves the option at zero cost.

## 3. Action space — the load-bearing findings

- **Masking is theoretically sound**: Huang & Ontañón (FLAIRS 2022, arXiv:2006.14171) prove the
  masked policy's update is a valid policy gradient — masking is a state-dependent differentiable
  transform of the logits; implemented by setting invalid logits to a large negative constant
  before softmax (gradients of invalid logits become exactly zero), applied at **both** sampling
  and gradient time (sample-only masking destabilizes PPO's KL). Canonical code:
  MicroRTS-Py `CategoricalMasked` (`torch.where(mask, logits, -1e8)`); sb3-contrib MaskablePPO
  cites the same paper. Verified 3-0 (three merged claims).
- **Masking is empirically essential and dominates penalties**: full-game microRTS PPO with no
  mask: **0.0** cumulative win rate; partial mask (action type only, the PySC2/SMAC style): 0.32;
  full per-component mask: 0.82–0.91. In scaling ablations, penalty agents (r = 0/−0.01/−0.1/−1)
  collapse on ≥10×10 maps while masking stays roughly constant. **Mask every component; never use
  penalties for legality.** Verified 3-0 (two merged claims).
- **Fixed space + mask is what everyone ships**: vcmi-gym uses a flat fixed `Discrete(2312)`
  (2 global + per-hex move/shoot/melee-by-direction over 165 hexes) with a boolean mask; the
  author's factorized multi-head variant **failed to converge**. No verified codebase consumes
  variable-length candidate lists directly. Verified 2-1 (medium confidence; a sibling claim
  pinning exact API constants was refuted — patterns transfer, constants don't).
- **Factorized ("composed") heads** are the proven alternative when the space grows
  combinatorially (microRTS: 8 independent softmax components, ~300 logits instead of ~10⁷ joint
  actions; per-cell "gridnet" variant). For our ~10³-scale simple_v1 space, flat-masked is
  simplest; factorize only when spells × targets × parameters arrive. Verified 3-0.
- **Pointer/attention over enumerated candidates** (AlphaStar) is the architectural home of our
  candidate-list design and remains compatible later — but it is the heavyweight option, not the
  starting point. Verified 3-0.

**Consequence for our spec (§10.4)**: keep the engine-enumerated candidates as the single source
of legality, but define a **fixed canonical action indexing** over the 11×9 board and derive both
the boolean mask and the candidate list from the same enumeration → ADR 0002.

## 4. Training pipeline staging

- **Behavior cloning first is validated** at the strongest available scale: AlphaStar's purely
  supervised stage reached 87 % vs the Elite bot before any RL. Our M2 decision-hook milestone
  (passive built-in-AI trajectory logging) is exactly the demonstration-collection prerequisite.
  Verified 3-0. **[gap]**: no small-scale BC→RL transition recipe (DAgger, KL-to-teacher,
  offline-RL warm-start) survived verification; vcmi-gym apparently trained from scratch — treat
  the §21 BC→DAgger→RL ladder as sensible but locally unproven.
- **The stack that shipped**: vcmi-gym's production models came from CleanRL-inspired single-file
  masked implementations (MPPO, MPPG, MPPO-DNA, MQRDQN), tuned with Ray Population Based Training
  + W&B; SB3 was prototyped then deliberately dropped; MuZero/mctx and IMPALA/DreamerV3
  experiments exist unpublished. For one Apple-silicon machine: **a single-file CleanRL-style
  masked PPO (full control of device placement, MPS/CPU-friendly) is the best-precedented choice**,
  with sb3-contrib MaskablePPO as the off-the-shelf fallback. Verified 3-0. (Refuted 0-3: the
  narrative that masking was added because SB3 lacked it.)
- **Single-machine feasibility**: microRTS SOTA (91 % cumulative vs all past competition bots,
  incl. champion CoacAI) took ~60–63 h on 1 GPU / 3 vCPU / 16 GB, single map. With our env at
  ~4,600 eps/s, the learner (MPS/CPU), not the env, will be the bottleneck. Verified 3-0.
  *(Caveat: CUDA GPU; no verified Apple-silicon MPS benchmark exists — open question.)*
- **Opponent diversity matters at train time**: the microRTS SOTA trained against a mix across its
  24 parallel envs (18 CoacAI + 2 each of three weaker bots); single-opponent agents lose to
  simple rushes. Map to our engine's AI at multiple difficulty/personality configurations.
  Verified 3-0. **[gap]**: self-play league *scheduling* (AlphaStar league) did not survive
  verification at our scale — start with scripted-opponent mixtures.
- **Keep the planning door open**: Stratega's agent API is forward-model-centric
  (`computeAction(GameState, ForwardModel&, Timer)`, ~100k calls/s) precisely to serve
  MCTS/RHEA-style methods; vcmi-gym has MuZero experiments. Our deterministic ~4,600 eps/s core is
  a first-class asset — expose a copyable-state/forward-model mode eventually, after the
  one-arena-per-process constraint is addressed (worker processes are today's answer). Verified
  3-0 (two merged claims).

## 5. Evaluation protocols

- **Adopt microRTS's two-protocol practice** (verified 3-0, two merged claims):
  (a) *fixed-pool win rate*: N seeds per config, best-seed evaluation of 100 games per scripted
  opponent under a step cap, reporting cumulative win rate + model size + wall-clock;
  (b) *TrueSkill league* mixing scripted bots and checkpoints, scheduling matches until rating
  uncertainty converges (`while sigma > 1.4`), leaderboard by `mu − 3·sigma`.
- **Calibration from the only shipped HoMM system** (verified 2-1, self-reported): first working
  vcmi-gym model ~75 % vs StupidAI / ~45 % vs BattleAI; the v12 GNN averaged 65 % vs BattleAI.
  **Beating the strong scripted AI took many iterations with a working pipeline — treat parity
  with fheroes2's AI as a multi-iteration goal.**
- Our seeded-deterministic suite + SHA-256 digests *exceed* verified common practice (nothing
  comparable found — **[gap]**, in our favor); keep them.

## 6. Prior art map

| Project | What it proves for us | Status |
|---|---|---|
| **vcmi-gym / MMAI** (HoMM3) | The whole pipeline ships: masked PPO on padded+per-hex encoding → in-game AI in VCMI 1.7.0 (2025-12-24, `vcmi-mods/mmai`). No fheroes2/HoMM2 RL env exists — we'd be the first. | Active; study first (3-0) |
| **Gym-microRTS / MicroRTS-Py** | Masking theory-in-practice, factorized heads, per-cell planes, dual obs modes, eval protocols, 60 h/16 GB SOTA | Canonical but **deprecated by Farama Aug 2025** — frozen reference |
| **AlphaStar** | Entity-transformer + pointer + LSTM; BC-before-RL (87 %) | Architecture ceiling, incomparable scale |
| **Stratega** | Forward-model-centric API for planning agents | Active precedent |
| **ARLinBfW** (Wesnoth) | Headless C++ engine + out-of-process text protocol works; our JSONL-over-stdio is a strictly cleaner version | Existence proof only (2 commits, 2019) |
| **entity-gym / RogueNet, OpenAI Five** | Entity-list APIs; structured-vs-pixel economics | Fetched, not in verified top-25 — [gap] |

## Design deltas for our spec (the "conflicts" question)

1. **Action space (real conflict)** — spec §10's variable-length, per-decision-ephemeral
   `action_id` list vs universal fixed-space+mask practice → **amended by ADR 0002**: one engine
   enumeration feeds both a fixed canonical index/mask (for standard masked-PPO tooling) and the
   semantic candidate list (for protocol, teacher matching, debugging, future pointer heads).
2. **SHA-256 replay digests** — no conflict; beyond common practice. Keep.
3. **4 worker processes** — no conflict now; microRTS used 24 parallel envs, so plan to scale
   workers when the learner stops being saturated (open question: MPS throughput crossover).
4. **Observation profiles** — not a conflict but a gap in our spec (§12 exposes only full state);
   **ADR 0001** adds `full_v1`/`observable_v1` following the MicroRTS one-schema pattern.

## Open questions carried forward

1. Is POMDP machinery needed at all for HoMM2 battles (hidden info ≈ RNG, not fog)? No verified
   asymmetric-critic evidence for near-fully-observable tactical games.
2. Which BC→online-RL transition recipe works at small scale for turn-based battles?
3. Masked-PPO throughput on M2/MPS at our model sizes, and the env-worker crossover point.
4. How far vcmi-gym's current v13+ encodings and unpublished MuZero/IMPALA/Dreamer experiments
   diverge from its documented v3 layout (docs lag code).

## Verification caveats (verbatim-faithful)

vcmi-gym constants are version-drifted (docs "as of Aug 2024"; v13/v14/v15 exist) and its win
rates are the author's self-reports; two findings rest on 2-1 split votes (flat-2312
characterization; eval numbers). Source arithmetic sloppiness reproduced faithfully: the vcmi
blog's "165×12=1320" should be 1980; microRTS's "301 vs 50 million" logits recompute to ~334–341
vs ~10⁸; AlphaStar's pointer ablation is 36 %, not ~38 %. The 91 %/60 h result is single-map on a
CUDA GPU. Refuted (0-3, do not repeat): "MMAI never shipped" (it did, in VCMI 1.7.0) and "masking
was added because SB3 lacked it" (author deliberately migrated to CleanRL-style code).
