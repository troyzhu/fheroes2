---
title: "The system, end to end: a code tour for a new engineer"
type: guide
updated: 2026-08-12
related_concepts: ["[[inventory]]", "[[inference-walkthrough]]", "[[observation-design]]", "[[legal-actions-and-masking]]", "[[../rl/the-policy-network]]", "[[../rl/program-review]]"]
tags: [agent-env, implementation, guide, onboarding]
---

# The system, end to end: a code tour for a new engineer

This is the page to read first if you want to understand the code rather than the experiments. It walks the pipeline once, engine to training, with a short snippet or an exact file pointer at every stage. Each section ends with where to go deeper. The experimental verdicts live in [[../rl/program-review]]; this page is about how the machine works.

## Table of contents

- [[#The pipeline in one view]]
- [[#Stage 1: the engine plays a battle and asks for decisions]]
- [[#Stage 2: the worker turns that into a line protocol]]
- [[#Stage 3: the Python environment wraps it as reset and step]]
- [[#Stage 4: observations and actions become tensors]]
- [[#Stage 5: the policy network]]
- [[#Stage 6: root search wraps the policy]]
- [[#Stage 7: training, four stages on one network]]
- [[#Stage 8: evaluation and the gates]]
- [[#Where the code lives, by question]]

## The pipeline in one view

| Stage | What happens | Where |
|---|---|---|
| Engine | The real game simulates a battle, deterministically | `src/fheroes2/battle/`, behavior-preserving seams only, digest-proven ([[inventory]] ledger) |
| Agent library | Scenario setup, decision hook, observation serializer, digests | `src/fheroes2/agent/`, additive, no game file modified |
| Worker | One process, one battle at a time, JSONL on stdin/stdout | `src/agent_worker/main.cpp` |
| Environment | `reset()`/`step()` over the worker process | `python/fheroes2_agent/env.py` |
| Encoding | Observation dict to a 634-float vector, action to an index in 793 | `python/fheroes2_agent/encoding.py` |
| Policy | Small MLP, masked logits over the 793 actions, value head | `python/fheroes2_agent/policy.py` |
| Search | One-ply PUCT with full-battle rollouts in a side environment | `python/fheroes2_agent/search.py` |
| Training | Cloning, DAgger, distillation, anchored PPO, self-play | `python/fheroes2_agent/` plus `agent_play/experiments/` |
| Evaluation | Ten suites, thirteen columns, seven verification gates | `python/fheroes2_agent/suites.py`, `agent_play/verify_*.sh` |

## Stage 1: the engine plays a battle and asks for decisions

The engine owns the call stack. Nothing steps the game from outside; `Arena::Turns()` advances a whole round, and inside it each unit's turn asks a controller for a decision. The agent library adds exactly that seam: a `DecisionController` whose `chooseActions` is called by the arena and blocks until an answer arrives (`src/fheroes2/agent/agent_external_controller.h`). Everything else in `src/fheroes2/agent/` supports that seam: scenario construction and validation (`agent_scenario.cpp`), the observation serializer (`agent_observation.cpp`), canonical digests for determinism checks (`agent_digest.cpp`), and the episode runner (`agent_battle_runner.cpp`). Determinism is load-bearing: the same scenario and seed reproduce a battle bit for bit, which is what makes replay, search prefix replay, and the golden-digest gates possible. Deeper: [[battle-turn-dispatch]], [[determinism-seeds-and-digests]].

## Stage 2: the worker turns that into a line protocol

`fheroes2_agent_worker --protocol` runs one episode at a time and speaks JSONL. At every decision it prints one line and blocks on stdin for the reply:

```text
worker -> {"record":"decision","observation":{...},"legal_actions":[14,113,...]}
caller -> 113
worker -> {"record":"terminal","termination":"victory","attacker":{"strength":...},...}
```

The terminal record carries per-side strength, hit points, commanded strengths, round count and two digests, so a caller can score the battle any way it likes: the environment defines no reward (ADR 0003, ADR 0005). One process runs one episode because the arena is a file-static singleton; parallelism means several worker processes. Deeper: [[command-encoding-and-snapshots]], and `src/agent_worker/main.cpp` reads top to bottom in one sitting.

## Stage 3: the Python environment wraps it as reset and step

`BattleEnv` spawns the worker and re-presents the blocking protocol as the usual loop. The whole step is a write, a readline, and a dispatch on the record type:

```python
def step(self, action: int) -> Step:
    self._proc.stdin.write(f"{int(action)}\n")
    self._proc.stdin.flush()
    record = self._readline()
    if record["record"] == "decision":
        return Step(encode_observation(record["observation"]),
                    encode_mask(record["legal_actions"]), 0.0, False)
    ...  # terminal: compute the configured reward margin, return done=True with info
```

Scenario knobs are constructor arguments that become worker flags: armies, commanders, `allow_wide`, `allow_flying`, `seed_offset` (which battlefield), `combat_seed_offset` (independent dice for search). `reward_margin` selects the terminal reward from `REWARD_MARGINS` and never reaches the worker, which defines no reward. Deeper: `python/fheroes2_agent/env.py`, whose docstrings carry the design arguments inline.

## Stage 4: observations and actions become tensors

One observation becomes a float vector of length 634: ten unit slots of 63 named features each, plus four globals (`FEATURE_NAMES` in `python/fheroes2_agent/encoding.py` is the single authoritative layout). One action is an index into a fixed space of 793:

```python
ACTION_SPACE_SIZE = 1 + 99 + 99 + 99 * 6  # 793, ADR 0002
```

That is one skip, 99 move targets and 99 ranged targets for the 11 by 9 board, and 594 melee cell-direction pairs. Legality arrives per decision as a mask over the 793. The optional `planes_v1` modality adds a channels-first $(7, 9, 11)$ tensor beside the vector (`encode_planes`); the worker emits only the obstacle layer, the other six channels derive from the units. Deeper: [[observation-design]], [[legal-actions-and-masking]], [[../decisions/0002-action-space]].

## Stage 5: the policy network

`BattlePolicy` is deliberately small, about 397k parameters: a shared two-layer slot encoder, concatenation pooling, a two-layer trunk, a 793-logit policy head and a scalar value head. The masking is the load-bearing detail:

```python
def forward(self, observations, masks, planes=None):
    hidden = self.features(observations, planes)
    logits = self.policy_head(hidden)
    logits = logits.masked_fill(~masks, MASK_FILL)   # MASK_FILL = -1e8
    return logits, self.value_head(hidden).squeeze(-1)
```

Illegal actions carry probability exactly zero in float32 and receive no gradient, which every loss in the repo relies on. Checkpoints are loaded only through `load_policy`, which reads the architecture from the state dict itself. Deeper: [[../rl/the-policy-network]].

## Stage 6: root search wraps the policy

Search is one ply with Monte Carlo playouts, not a tree. At a decision, candidates are scored by PUCT, and each playout replays the action prefix in a persistent side environment, applies the candidate, then lets the policy finish the battle from its own chair while the engine's built-in AI answers for the opponent, so rollout returns are values against the real opponent model rather than self-play values:

```python
def rollout(sim, model, prefix, first):
    observation, mask = sim.reset()
    for action in prefix:
        step = sim.step(action)          # exact replay on shared dice; resampled under an offset
    step = sim.step(first)               # the candidate under test
    while not step.done:
        step = sim.step(policy_action(model, step.observation, step.mask, env=sim))
    return step.reward   # the side env's terminal reward; the built-in AI answered the other chair
```

Two configuration choices decide what the numbers mean, both stamped on every report: the side environment's `reward_margin` is what search maximizes, and its `combat_seed_offset` must be nonzero for honest numbers, because a zero offset hands search the live battle's dice ([[../decisions/0008-search-configuration]]). A Sequential Halving allocator exists beside PUCT for simple-regret experiments; PUCT is the measured default. Deeper: `python/fheroes2_agent/search.py`, [[inference-walkthrough]] for one decision under the microscope.

## Stage 7: training, four stages on one network

All four stages exist and are gated. Behavior cloning fits the policy to `AI::BattlePlanner`'s recorded choices (`python/fheroes2_agent/train_bc.py`, data via `agent_play/experiments/record_diverse.py`). DAgger relabels student-visited states through the planner probe, the engine seam that answers at arbitrary states without perturbing the battle (`agent_play/experiments/dagger_iteration.py`). Search distillation trains on search-labeled corpora, hard argmax or soft targets (`agent_play/experiments/soft_distill.py`). Reinforcement is masked PPO with an optional KL leash to the frozen supervised anchor, the one configuration measured to train without eroding it (`python/fheroes2_agent/train_ppo.py`, [[../decisions/0007-anchored-ppo]]); self-play runs it against a pool of frozen checkpoints (`python/fheroes2_agent/selfplay.py`). Rewards are terminal-only and chosen per run from `REWARD_MARGINS` in `env.py`. Deeper: [[../rl/training-design]] for the losses and hyperparameters, [[teacher-coverage-and-behavior-cloning]] for the data side.

## Stage 8: evaluation and the gates

Evaluation suites live in `python/fheroes2_agent/suites.py`, ten of them; the five that can separate players carry the scoreboard, and every harness reports the same thirteen-column block, rate, reward split by outcome, strength margin, both entropy forms, effective actions, rounds, built by `scenarios.measure` and the battery's quality dict (`agent_play/experiments/validation_battery.py`), so a rate can never hide how it was earned. In the search battery two near-identical flags matter: `--reward-margin` sets what search maximizes and `--report-margin` sets what the reward column measures, stamped as `search_objective` and `reward_margin` in every report. Verification gates are the fast contract: Phase 0 invariants, M1 determinism goldens, M2 hook inertness, M3 legal-action coverage, `verify_agent.sh` for the Python stack, `lint_docs.sh` for these documents, `verify_memory.sh` for the assistant's memory. Run them all with the block in [[../overview#Build and verify]]. Slow measurements are experiment scripts under `agent_play/experiments/`, one question each, indexed in its `README.md` with runtimes.

## Where the code lives, by question

| Question | Start at |
|---|---|
| What did the agent change in the engine | [[inventory]], the engine-source ledger, then `git diff master --stat -- src/` |
| How does one decision flow end to end | [[inference-walkthrough]] |
| What exactly does the policy see | [[observation-design]], then `FEATURE_NAMES` in `python/fheroes2_agent/encoding.py` |
| Why is an action legal or not | [[legal-actions-and-masking]] |
| What is the reward | `REWARD_MARGINS` and the `terminal_reward_*` functions in `python/fheroes2_agent/env.py`, [[../rl/reward-design]] |
| Why was a design chosen | `../decisions/`, one accepted record per choice |
| What has been measured | [[../rl/program-review]], verdict-first |
| Reproduce the headline 0.700 | `search_agent_battery.py` at 16 playouts, offset 987631, seeds 11 to 13, per `files/2026-08-10-run-reports/`; the checkpoint is vendored at `agent_play/docs/archive/experiments/files/2026-08-09-checkpoints/band_soft_s0.pt` |
