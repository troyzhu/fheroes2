"""Self-play training environments: the learner faces policies instead of the built-in AI.

The design the owner asked for, kept deliberately small because the env abstraction carries it.
`SelfPlayEnv` runs the worker with both sides externally controlled, answers the opponent side's
decisions internally with a frozen policy, and exposes only the learner's decisions through the
usual reset/step contract, so every existing trainer works unchanged. `OpponentPool` samples the
opponent per episode from a shelf of checkpoints, the league-lite remedy for the classic
self-play failure of overfitting the latest self, with the built-in AI includable as an anchor
opponent by sampling `None`.

Reward perspective is the learner's side throughout, and the both-chair semantics built this
week (stall asymmetry included) are what make that well defined. The side=both path itself is
trustworthy as of the 2026-08-07 divergence resolution: protocol and runEpisode are
byte-identical, decision digests prove it.
"""

from __future__ import annotations

import pathlib
import random

import numpy as np
import torch

from .env import BattleEnv, Step, terminal_reward_strength, terminal_reward_two_sided
from .policy import load_policy


class OpponentPool:
    """Checkpoints as opponents, sampled per episode; None entries mean the built-in AI."""

    def __init__(self, checkpoints: list[str | None], seed: int = 0, weights: list[float] | None = None):
        self._entries: list[tuple[str | None, object]] = []
        for path in checkpoints:
            if path is None:
                self._entries.append((None, None))
            else:
                model = load_policy(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
                model.eval()
                self._entries.append((str(pathlib.Path(path).name), model))
        self._rng = random.Random(seed)
        self._weights = weights
        self.current_name: str | None = None
        self.current_model = None

    def draw(self):
        name, model = self._rng.choices(self._entries, weights=self._weights, k=1)[0]
        self.current_name, self.current_model = name, model
        return model


class SelfPlayEnv:
    """The learner's reset/step view of a both-sides battle against a pooled opponent.

    The learner always owns `learner_side`; every decision belonging to the other side is
    answered internally by the opponent model (sampled, matching deployment) before control
    returns. When the pool draws the built-in AI, the worker is respawned in single-side mode,
    since the engine's own planner cannot be injected per decision from outside.
    """

    # Only the strength-priced margins are offered: their initial totals are engine-emitted in
    # the terminal record, where the hit-point margin needs an episode-start capture this
    # wrapper does not track.
    def __init__(self, worker: str, matchup_kwargs: dict | list, pool: OpponentPool,
                 learner_side: str = "attacker", reward_margin: str = "strength",
                 reward_weighting: str = "none", seeds: int = 1, rotation_seed: int = 0):
        self._worker = worker
        # A list rotates matchups across episodes, the same distributional hygiene MatchupPool
        # provides; a single dict pins one matchup.
        self._matchups = matchup_kwargs if isinstance(matchup_kwargs, list) else [matchup_kwargs]
        self._rotation = random.Random(rotation_seed)
        self._kwargs = dict(self._matchups[0])
        self._pool = pool
        # `learner_side="alternate"` draws the chair per episode, which is what the 2026-08-08
        # scoreboard called for: every round until then trained the attacker chair alone while
        # two of the three largest remaining gaps to the engine are on the defending side, and
        # the engine's own mirror split (0.639 to the defender on symmetric armies) says the
        # defending chair is not the harder one, only the untrained one.
        self._alternate = learner_side == "alternate"
        self._learner_side = "attacker" if self._alternate else learner_side
        self._reward_margin = reward_margin
        self._reward_weighting = reward_weighting
        self._seeds = seeds
        self._env: BattleEnv | None = None
        self._mode: str | None = None  # "both" or "single"
        self.side = self._learner_side
        self.opponent_name: str | None = None

    def _ensure(self, mode: str) -> None:
        if self._env is not None and self._mode == mode and len(self._matchups) == 1:
            return
        if self._env is not None:
            self._env.close()
        side = "both" if mode == "both" else self._learner_side
        self._env = BattleEnv(self._worker, side=side, seeds=self._seeds,
                              reward_margin=self._reward_margin, reward_weighting=self._reward_weighting,
                              planes=False, **self._kwargs)
        self._mode = mode

    def _learner_turn(self) -> bool:
        raw = self._env._pending["observation"]
        is_attacker_turn = bool(raw["active_is_attacker"])
        return is_attacker_turn == (self._learner_side == "attacker")

    def _opponent_decide(self, observation: np.ndarray, mask: np.ndarray) -> int:
        model = self._pool.current_model
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(observation).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0))
            return int(torch.distributions.Categorical(logits=logits).sample())

    def _advance_to_learner(self, observation, mask):
        """Answer opponent decisions until the learner acts or the battle ends."""
        while not self._learner_turn():
            step = self._env.step(self._opponent_decide(observation, mask))
            if step.done:
                return None, None, step
            observation, mask = step.observation, step.mask
        return observation, mask, None

    def reset(self):
        opponent = self._pool.draw()
        self.opponent_name = self._pool.current_name
        if len(self._matchups) > 1:
            self._kwargs = dict(self._rotation.choice(self._matchups))
        if self._alternate:
            # Drawn per episode rather than swapped every other episode, so a matchup rotation of
            # any period cannot phase-lock with the chair and hand one side a fixed subset.
            self._learner_side = self._rotation.choice(("attacker", "defender"))
            self.side = self._learner_side
            # The worker is spawned per chair in single mode, so a chair change forces a respawn.
            self._mode = None
        self._ensure("single" if opponent is None else "both")
        observation, mask = self._env.reset()
        if self._mode == "both":
            observation, mask, terminal = self._advance_to_learner(observation, mask)
            if terminal is not None:
                # The opponent ended the battle before the learner ever moved; extremely rare
                # (requires zero learner decisions), and a fresh reset is the honest recovery.
                return self.reset()
        return observation, mask

    def step(self, action: int) -> Step:
        step = self._env.step(action)
        if step.done or self._mode == "single":
            return self._reperspect(step)
        observation, mask, terminal = self._advance_to_learner(step.observation, step.mask)
        if terminal is not None:
            return self._reperspect(terminal)
        return Step(observation, mask, 0.0, False)

    def _reperspect(self, step: Step) -> Step:
        """Terminal rewards from the learner's chair, whatever side the env reported for."""
        if not step.done:
            return step
        record = step.info
        if self._reward_margin == "strength":
            reward = terminal_reward_strength(record, self._learner_side)
        else:
            reward = terminal_reward_two_sided(record, self._learner_side)
        return Step(step.observation, step.mask, reward, True, record)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
