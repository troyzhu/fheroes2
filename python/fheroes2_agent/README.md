# `fheroes2_agent`, module map and the two training families

The Python side of the battle environment. Everything here consumes the worker's line protocol and produces checkpoints; the environment itself, the action space and the observation schema live in C++ under `src/fheroes2/agent/`.

This file exists for one reason beyond orientation: two families of training method share this package, they are easy to confuse because both end in a `BattlePolicy` checkpoint, and mixing them silently produces results that look fine and mean nothing. The separation below is the contract.

## Modules

| Module | What it owns |
|---|---|
| `encoding.py` | The observation tensor and the legality mask. `FEATURE_NAMES` is the authoritative layout; `ENCODING_VERSION` is stamped into every checkpoint |
| `env.py` | `BattleEnv` (one battle per reset over the protocol), `MatchupPool` (rotation with group holding), the terminal rewards and their optional weightings |
| `policy.py` | `BattlePolicy`, the shared-trunk actor with a value head, plus `load_policy` which reconstructs the architecture a checkpoint's state dict describes |
| `dataset.py` | Recorded episodes to supervised arrays, split by episode or by matchup, with discounted returns for critic fitting |
| `scenarios.py` | Matchups, the samplers, calibration against a policy, and `measure` |
| `objectives.py` | Advantage estimators and trust regions for both families |
| `train_bc.py` | Family A, imitation: cloning and every distillation round |
| `train_critic.py` | Family A, the value head fitted on recorded returns |
| `train_ppo.py` | Family B with a critic: masked PPO with GAE, ratio clip or DPPO divergence gates |
| `selfplay.py` | `OpponentPool` (checkpoints plus the built-in AI as anchor) and `SelfPlayEnv`, the learner's reset/step view of a both-sides battle |
| `train_rloo.py`, `train_group.py` | Family B without a critic: leave-one-out, GRPO and Dr. GRPO group baselines |
| `render.py`, `watch.py` | Human-readable boards and action descriptions |

## Family A, value-based

Anything that estimates a value function and uses it: the critic fitted on recorded returns, PPO's GAE baseline, and the search leaf evaluators the 2026-08-06 work measured. Its failure modes are value failure modes, so it is judged by explained variance on the distribution the value will actually be queried on, and never by a fit on the training distribution alone. `agent_play/docs/rl/training-design.md` carries what is measured about the critic, including that the frozen imitation trunk caps it and a dedicated network reaches 0.61 where the shared probe reaches 0.09.

## Family B, value-free

Group-relative methods that replace the value with a baseline computed from siblings of the same start: leave-one-out, GRPO, Dr. GRPO. These require several episodes of the identical scenario in one group, which `MatchupPool(..., hold_within_group=True)` provides and which rotating per episode silently breaks. Their failure modes are baseline failure modes: a degenerate group, an advantage collapse, a matchup with no outcome variance. `agent_play/docs/rl/rlhf-transfer.md` carries the derivations and the identity between the variants.

## The rules that keep them apart

A checkpoint records the family that produced it, through `encoding_version` plus the trainer's own report; a report is the provenance, not the file name. A value-free run must not be given a fitted critic as a baseline and then reported as value-free, and a value-based run must not borrow a group baseline; the two answer different questions and their error bars are not comparable.

Shared code stays in `objectives.py` and `env.py` deliberately, so a change to the reward or the advantage floor lands on both families at once and neither drifts. Anything new that only one family needs belongs in that family's trainer, not in the shared modules. A dedicated value network, when it lands, is Family A infrastructure and gets its own module rather than another head on `BattlePolicy`, because the policy's trunk is frozen during value fitting for a reason, recorded in `agent_play/docs/rl/training-design.md`.
