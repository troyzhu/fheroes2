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


def episode_returns(records: list[dict], gamma: float = 0.99) -> list[float]:
    """The discounted return at each decision of one recorded teacher episode.

    Rewards are terminal only, so the return at decision t is the terminal reward discounted by
    the number of decisions still to come. The teacher plays both sides, so the sign depends on
    whose turn it was: a decision taken by the side that went on to lose earns the loser's
    reward. That is what makes one episode supply both positive and negative targets, and it is
    the whole reason a critic fitted on teacher play is worth anything.
    """
    from .env import terminal_reward

    terminal = next(r for r in records if r.get("record") == "terminal")
    decisions = [r for r in records if r.get("record") == "decision" and "observation" in r]

    # Own starting hit points per side, read from the first observation, which is before damage.
    first = decisions[0]["observation"] if decisions else None
    start = {"attacker": 0.0, "defender": 0.0}
    if first is not None:
        for unit in first["units"]:
            start[unit["side"]] += unit["hit_points"]

    rewards = {side: terminal_reward(terminal, side, start[side] or 1.0) for side in ("attacker", "defender")}

    out = []
    total = len(decisions)
    for index, record in enumerate(decisions):
        side = "attacker" if record["observation"]["active_is_attacker"] else "defender"
        # Steps remaining for this decision, counted in decisions rather than rounds.
        out.append(rewards[side] * (gamma ** (total - 1 - index)))
    return out


@dataclass
class Samples:
    """Encoded decisions. Rows line up across all four arrays."""

    observations: np.ndarray  # (n, OBSERVATION_SIZE) float32
    masks: np.ndarray  # (n, 793) bool
    actions: np.ndarray  # (n,) int64, the teacher's canonical index
    episode_ids: np.ndarray  # (n,) int64, which episode each row came from
    returns: np.ndarray | None = None  # (n,) float32, discounted return, for critic fitting
    encoding_version: str = ENCODING_VERSION

    def __len__(self) -> int:
        return len(self.actions)

    def subset(self, rows: np.ndarray) -> "Samples":
        return Samples(self.observations[rows], self.masks[rows], self.actions[rows], self.episode_ids[rows],
                       None if self.returns is None else self.returns[rows])


def load_episode(path: pathlib.Path, gamma: float = 0.99) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[float]]:
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []

    records = [json.loads(line) for line in path.read_text().splitlines()]
    try:
        returns = episode_returns(records, gamma)
    except (StopIteration, KeyError, IndexError):
        returns = []

    for record in records:
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

    # Returns are computed over decisions carrying an observation, which is the same filter
    # applied above, so the two line up. A mismatch means the episode was recorded without
    # --audit-coverage and its returns are dropped rather than misaligned.
    if len(returns) != len(actions):
        returns = []
    return observations, masks, actions, returns


def load_dir(root: str | pathlib.Path) -> Samples:
    root = pathlib.Path(root)
    files = sorted(root.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no .jsonl episodes under {root}")

    all_obs: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_actions: list[int] = []
    all_episodes: list[int] = []
    all_returns: list[float] = []

    for episode_id, path in enumerate(files):
        observations, masks, actions, returns = load_episode(path)
        all_obs.extend(observations)
        all_masks.extend(masks)
        all_actions.extend(actions)
        all_episodes.extend([episode_id] * len(actions))
        all_returns.extend(returns if returns else [float("nan")] * len(actions))

    if not all_actions:
        raise ValueError(f"{len(files)} files under {root} produced no samples; was --audit-coverage set when recording?")

    samples = Samples(
        observations=np.stack(all_obs),
        masks=np.stack(all_masks),
        actions=np.asarray(all_actions, dtype=np.int64),
        episode_ids=np.asarray(all_episodes, dtype=np.int64),
        returns=np.asarray(all_returns, dtype=np.float32),
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
