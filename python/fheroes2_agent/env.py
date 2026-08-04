"""A stepping environment on top of the blocking worker.

The engine owns the call stack: nothing calls `env.step(a)`, because `Arena::Turns()` advances a
whole round rather than one decision. The worker blocks inside the decision hook until an action
arrives, and this class re-presents that as the usual reset/step loop, which is the trampoline
PySC2 uses and which `overview.md` describes.

One process runs one episode at a time, because the engine's arena is a file-static singleton, so
vectorization means several processes rather than several arenas.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

import numpy as np

from .encoding import encode_mask, encode_observation


@dataclass
class Step:
    observation: np.ndarray
    mask: np.ndarray
    reward: float
    done: bool
    # Set only on the final step. Carries the whole terminal record for reward shaping and for
    # reporting, since the environment itself defines no reward (ADR 0005).
    info: dict | None = None


class BattleEnv:
    """One battle per reset, driven through the worker's JSONL protocol."""

    def __init__(self, worker: str, fixture: str = "m1_tiny_melee", side: str = "attacker", seeds: int = 1, home: str = "/tmp",
                 attacker: str | None = None, defender: str | None = None):
        self._cmd = [worker, "--protocol", "--fixture", fixture, "--side", side, "--seeds", str(seeds)]
        # Army overrides, "monsterId:count,...". These are the difficulty control: a matchup is
        # only worth training on when the policy neither always wins nor always loses it.
        if attacker:
            self._cmd += ["--attacker", attacker]
        if defender:
            self._cmd += ["--defender", defender]
        self._env = dict(os.environ, HOME=home)
        self._proc: subprocess.Popen | None = None
        self._pending: dict | None = None
        # Own hit points at the first decision, which is before any damage has been dealt, so it
        # is the starting force. The terminal record carries no initial totals.
        self._own_initial_hp: float = 0.0
        self.side = side

    def _readline(self) -> dict | None:
        assert self._proc is not None
        line = self._proc.stdout.readline()
        return json.loads(line) if line else None

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.close()
        self._proc = subprocess.Popen(
            self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=self._env,
        )
        while True:
            record = self._readline()
            if record is None:
                raise RuntimeError("worker produced no decision before exiting")
            if record["record"] == "decision":
                self._pending = record
                mine = self.side == "attacker"
                self._own_initial_hp = float(
                    sum(u["hit_points"] for u in record["observation"]["units"] if (u["side"] == "attacker") == mine)
                )
                return encode_observation(record["observation"]), encode_mask(record["legal_actions"])

    def step(self, action: int) -> Step:
        assert self._proc is not None and self._pending is not None
        self._proc.stdin.write(f"{int(action)}\n")
        self._proc.stdin.flush()

        record = self._readline()
        if record is None:
            raise RuntimeError("worker closed the stream without a terminal record")

        if record["record"] == "decision":
            self._pending = record
            return Step(encode_observation(record["observation"]), encode_mask(record["legal_actions"]), 0.0, False)

        # Terminal. The reward is defined here rather than in the environment, per ADR 0005,
        # which keeps the objective a training-configuration choice rather than engine behaviour.
        return Step(
            observation=np.zeros_like(encode_observation(self._pending["observation"])),
            mask=np.zeros(encode_mask([]).shape, dtype=bool),
            reward=terminal_reward(record, self.side, self._own_initial_hp),
            done=True,
            info=record,
        )

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()
        self._proc = None
        self._pending = None

    def __del__(self):
        self.close()


def terminal_reward(record: dict, side: str, own_initial_hit_points: float) -> float:
    """The margin-weighted terminal reward, the leading candidate in ADR 0005.

    Plus or minus one for the outcome, plus the fraction of one's own starting force that
    survived. Terminal only, which keeps the objective the thing being optimized rather than a
    proxy for it.

    The survival fraction is measured against the starting force rather than against what the
    opponent has left. Relative margin looks natural and is useless here: a decided battle almost
    always ends with the loser wiped out, so (own - foe) / (own + foe) is 1.0 whether the winner
    finished with fifty hit points or five, and the term collapses to the win-loss signal it was
    meant to enrich. Measured against the start, a clean win scores 2.0, a pyrrhic win 1.1, a
    cheap loss -0.4 and a rout -1.0.

    That ordering is what makes a hopeless matchup still teach something, which
    scenario-distribution.md argues is the strongest reason to prefer this candidate over pure
    win-loss.
    """
    own = "attacker" if side == "attacker" else "defender"
    survived = record[own]["hit_points"] / own_initial_hit_points if own_initial_hit_points > 0 else 0.0
    won = record["termination"] == ("victory" if side == "attacker" else "defeat")
    return (1.0 if won else -1.0) + survived


class MatchupPool:
    """Rotate over several matchups, one battle at a time.

    Training on a single matchup measures that matchup, not the policy. Rotating means the
    gradient comes from a distribution, which is what makes a reported number a statement about
    the generator rather than about one army pair.
    """

    def __init__(self, worker: str, matchups, side: str = "attacker", seed: int = 0, home: str = "/tmp"):
        import random

        self._worker = worker
        self._matchups = list(matchups)
        self._side = side
        self._rng = random.Random(seed)
        self._home = home
        self._env: BattleEnv | None = None
        self.side = side
        self.current = None

    def reset(self):
        self.close()
        self.current = self._rng.choice(self._matchups)
        self._env = BattleEnv(self._worker, side=self._side, attacker=self.current.attacker,
                              defender=self.current.defender, home=self._home)
        return self._env.reset()

    def step(self, action: int) -> Step:
        assert self._env is not None
        return self._env.step(action)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    @property
    def _pending(self):
        return self._env._pending if self._env else None
