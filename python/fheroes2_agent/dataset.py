"""Load recorded teacher episodes into behaviour-cloning arrays.

The worker writes one JSONL file per episode under `--audit-coverage --trajectory-dir`. Each
decision record carries the observation, the legal action set and the teacher's chosen index,
which is exactly one supervised sample.

Splitting is by episode rather than by decision, deliberately. Decisions inside one battle are
strongly correlated, so a decision-level split leaks: the same board one step apart would appear
on both sides and held-out agreement would read far too high.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

from .encoding import ACTION_SPACE_SIZE, ENCODING_VERSION, encode_mask, encode_observation


@dataclass
class Samples:
    """Encoded decisions. Rows line up across all four arrays."""

    observations: np.ndarray  # (n, OBSERVATION_SIZE) float32
    masks: np.ndarray  # (n, 793) bool
    actions: np.ndarray  # (n,) int64, the teacher's canonical index
    episode_ids: np.ndarray  # (n,) int64, which episode each row came from
    encoding_version: str = ENCODING_VERSION

    def __len__(self) -> int:
        return len(self.actions)

    def subset(self, rows: np.ndarray) -> "Samples":
        return Samples(self.observations[rows], self.masks[rows], self.actions[rows], self.episode_ids[rows])


def load_episode(path: pathlib.Path) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []

    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("record") != "decision":
            continue
        # A decision recorded without --audit-coverage has no observation and no label, so it is
        # not a sample. Skipping quietly would hide a mis-run collection, hence the check below
        # in load_dir that at least one sample was produced.
        if "observation" not in record or not record.get("teacher_resolved"):
            continue

        observations.append(encode_observation(record["observation"]))
        masks.append(encode_mask(record["legal_actions"]))
        actions.append(int(record["teacher_action"]))

    return observations, masks, actions


def load_dir(root: str | pathlib.Path) -> Samples:
    root = pathlib.Path(root)
    files = sorted(root.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no .jsonl episodes under {root}")

    all_obs: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_actions: list[int] = []
    all_episodes: list[int] = []

    for episode_id, path in enumerate(files):
        observations, masks, actions = load_episode(path)
        all_obs.extend(observations)
        all_masks.extend(masks)
        all_actions.extend(actions)
        all_episodes.extend([episode_id] * len(actions))

    if not all_actions:
        raise ValueError(f"{len(files)} files under {root} produced no samples; was --audit-coverage set when recording?")

    samples = Samples(
        observations=np.stack(all_obs),
        masks=np.stack(all_masks),
        actions=np.asarray(all_actions, dtype=np.int64),
        episode_ids=np.asarray(all_episodes, dtype=np.int64),
    )

    # The teacher's action must be legal. Milestone 3 already asserts this engine-side over the
    # candidate list; re-asserting it here catches an encoding or index-base mistake, which would
    # otherwise show up much later as a policy that cannot reach the actions it was taught.
    illegal = ~samples.masks[np.arange(len(samples)), samples.actions]
    if illegal.any():
        raise ValueError(f"{int(illegal.sum())} of {len(samples)} teacher actions are outside their legal mask")

    return samples


def split_by_episode(samples: Samples, holdout_fraction: float = 0.2, seed: int = 0) -> tuple[Samples, Samples]:
    """Split whole episodes, never individual decisions."""
    episodes = np.unique(samples.episode_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episodes)
    n_holdout = max(1, int(round(len(episodes) * holdout_fraction)))
    holdout_episodes = set(shuffled[:n_holdout].tolist())

    is_holdout = np.array([e in holdout_episodes for e in samples.episode_ids])
    return samples.subset(~is_holdout), samples.subset(is_holdout)


def summarize(samples: Samples) -> str:
    legal_per_decision = samples.masks.sum(axis=1)
    return (
        f"{len(samples)} samples from {len(np.unique(samples.episode_ids))} episodes, "
        f"encoding {samples.encoding_version}, observation width {samples.observations.shape[1]}, "
        f"action space {ACTION_SPACE_SIZE}, "
        f"legal actions per decision min {legal_per_decision.min()} median {int(np.median(legal_per_decision))} max {legal_per_decision.max()}, "
        f"distinct teacher actions {len(np.unique(samples.actions))}"
    )
