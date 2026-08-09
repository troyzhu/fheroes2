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
import pathlib
import subprocess
from dataclasses import dataclass

import numpy as np

from .encoding import encode_mask, encode_observation, encode_planes

_STRENGTH: dict[int, float] | None = None


def _monster_strength() -> dict[int, float]:
    """Engine strength per monster id, from the vendored capability audit."""
    global _STRENGTH
    if _STRENGTH is None:
        path = pathlib.Path(__file__).parent / "data" / "monster_capabilities_v1.json"
        _STRENGTH = {r["monster_id"]: float(r["strength"]) for r in json.loads(path.read_text())}
    return _STRENGTH


def difficulty_weight(observation: dict, side: str, exponent: float = 0.5, cap: float = 4.0) -> float:
    """Opponent-to-own strength ratio at this state, tempered into a reward weight.

    Counts are priced by the engine's own creature strength, the same pricing the value-budget
    sampler uses, so a Peasant and a Champion are not the same unit. The ratio is clipped to
    [1/cap, cap] before the exponent, because horde matchups reach order-of-magnitude ratios and
    an unbounded weight would let one scenario dominate a batch. Commander stat bonuses are not
    priced in, a known simplification recorded in reward-design.

    At the default exponent 0.5 and cap 4, the weight spans [0.5, 2].
    """
    strength = _monster_strength()
    mine = side == "attacker"
    own = sum(strength[u["monster_id"]] * u["count"] for u in observation["units"] if (u["side"] == "attacker") == mine)
    enemy = sum(strength[u["monster_id"]] * u["count"] for u in observation["units"] if (u["side"] == "attacker") != mine)
    if own <= 0.0 or enemy <= 0.0:
        return 1.0
    ratio = min(max(enemy / own, 1.0 / cap), cap)
    return ratio ** exponent


def apply_difficulty(reward: float, weight: float) -> float:
    """Difficulty-weighted terminal reward: wins scale by the weight, losses by its inverse.

    A hard fight (weight above one) pays more for winning and forgives losing; an easy fight
    pays less for winning and punishes losing harder. Both directions serve the same end, that
    the gradient stops over-rewarding easy victories and over-penalizing lopsided losses. The
    sign split is well defined because the margin-weighted terminal reward never lands strictly
    between 0 and 1: a win is at least 1 and everything else is at most 0.
    """
    return reward * (weight if reward > 0 else 1.0 / weight)


class ScenarioRejected(RuntimeError):
    """The worker refused the scenario, usually an army the allowlist or limits do not permit."""


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
                 attacker: str | None = None, defender: str | None = None,
                 attacker_hero: str | None = None, defender_hero: str | None = None,
                 allow_wide: bool = False, probe_teacher: bool = False,
                 reward_weighting: str = "none", reward_margin: str = "hit_points",
                 seed_offset: int = 0, planes: bool = False):
        if reward_weighting not in ("none", "difficulty"):
            raise ValueError(f"unknown reward_weighting {reward_weighting!r}")
        if reward_margin not in ("hit_points", "strength", "two_sided"):
            raise ValueError(f"unknown reward_margin {reward_margin!r}")
        self._reward_margin = reward_margin
        self._cmd = [worker, "--protocol", "--fixture", fixture, "--side", side, "--seeds", str(seeds)]
        # Which battlefield variant the seed cycle starts at. A search side-environment given
        # the same offset replays the live environment's battlefield, which is what makes
        # battlefield-varied search labels possible without a sync protocol.
        if seed_offset:
            self._cmd += ["--seed-offset", str(seed_offset)]
        # DAgger relabeling: each decision record then carries "teacher_action", the planner's
        # own choice at the same state, when it resolves inside simple_v1.
        if probe_teacher:
            self._cmd += ["--probe-teacher"]
        # ADR 0004's planes_v1: the worker appends the obstacle layer to every observation and
        # the env keeps the rasterized tensor of the latest state in `last_planes`, so callers
        # that feed a planes-built policy read it beside the flat vector without an API break.
        self._planes = planes
        if planes:
            self._cmd += ["--planes"]
        # Army overrides, "monsterId:count,...". These are the difficulty control: a matchup is
        # only worth training on when the policy neither always wins nor always loses it.
        if attacker:
            self._cmd += ["--attacker", attacker]
        if defender:
            self._cmd += ["--defender", defender]
        # Hero commanders, "attack:defense". Real maps always have one, and every unit's
        # effective stats include the commander's, so a faithful map fight needs these.
        if attacker_hero:
            self._cmd += ["--attacker-hero", attacker_hero]
        if defender_hero:
            self._cmd += ["--defender-hero", defender_hero]
        if allow_wide:
            self._cmd += ["--allow-wide"]
        self._env = dict(os.environ, HOME=home)
        self._attacker = attacker
        self._defender = defender
        self._proc: subprocess.Popen | None = None
        self._pending: dict | None = None
        # planes_v1 tensor of the latest presented state, None unless planes=True.
        self.last_planes = None
        # Own hit points at the first decision, which is before any damage has been dealt, so it
        # is the starting force. The terminal record carries no initial totals.
        self._own_initial_hp: float = 0.0
        self._reward_weighting = reward_weighting
        self._reward_margin = reward_margin
        self._difficulty = 1.0
        self.side = side
        # The scenario id of the episode in progress, from the worker's episode_start record.
        # With seeds > 1 this is how a caller tells the battlefield variants apart.
        self.scenario_id: str | None = None

    def _readline(self) -> dict | None:
        assert self._proc is not None
        line = self._proc.stdout.readline()
        return json.loads(line) if line else None

    def _spawn(self) -> None:
        self.close()
        self._proc = subprocess.Popen(
            self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=self._env,
        )

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        # Between episodes the worker is kept alive and advances to its next scenario, which is
        # what makes `seeds` a rotation over battlefields rather than dead configuration: the
        # worker's scenario list is the world-seed variants, one episode each, and a fresh
        # process restarts the cycle. A reset that abandons a battle mid-episode still needs a
        # fresh process, because the worker is blocked waiting for an action.
        spawned = False
        if self._proc is None or self._pending is not None:
            self._spawn()
            spawned = True
        self._pending = None
        while True:
            record = self._readline()
            if record is None:
                if spawned:
                    # Died right after spawning, which is the worker refusing the scenario.
                    # Surface its own reason rather than an unexplained crash mid-sweep.
                    detail = (self._proc.stderr.read() or "").strip().splitlines()
                    reason = detail[-1] if detail else "no diagnostic on stderr"
                    raise ScenarioRejected(f"{reason} (attacker={self._attacker}, defender={self._defender})")
                # Scenario list exhausted: start the next cycle of battlefields.
                self._spawn()
                spawned = True
                continue
            if record["record"] == "episode_start":
                self.scenario_id = record.get("scenario_id")
            elif record["record"] == "decision":
                self._pending = record
                self.last_planes = encode_planes(record["observation"]) if self._planes else None
                mine = self.side == "attacker"
                self._own_initial_hp = float(
                    sum(u["hit_points"] for u in record["observation"]["units"] if (u["side"] == "attacker") == mine)
                )
                if self._reward_weighting == "difficulty":
                    self._difficulty = difficulty_weight(record["observation"], self.side)
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
            self.last_planes = encode_planes(record["observation"]) if self._planes else None
            return Step(encode_observation(record["observation"]), encode_mask(record["legal_actions"]), 0.0, False)

        # Terminal. The reward is defined here rather than in the environment, per ADR 0005,
        # which keeps the objective a training-configuration choice rather than engine behaviour.
        if self._reward_margin == "strength":
            reward = terminal_reward_strength(record, self.side)
        elif self._reward_margin == "two_sided":
            reward = terminal_reward_two_sided(record, self.side)
        elif self._reward_margin == "two_sided_commanded":
            reward = terminal_reward_two_sided(record, self.side, commanded=True)
        else:
            reward = terminal_reward(record, self.side, self._own_initial_hp)
        if self._reward_weighting == "difficulty":
            reward = apply_difficulty(reward, self._difficulty)
        step = Step(
            observation=np.zeros_like(encode_observation(self._pending["observation"])),
            mask=np.zeros(encode_mask([]).shape, dtype=bool),
            reward=reward,
            done=True,
            info=record,
        )
        # Nothing is pending once the battle ended, which is also what tells reset() that the
        # worker is between episodes and can simply be read forward.
        self._pending = None
        return step

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
    won = _side_won(record, side)
    if record["termination"] == "stalemate" and side == "attacker":
        survived = 0.0
    return (1.0 if won else -1.0) + survived


def terminal_reward_two_sided(record: dict, side: str, commanded: bool = False) -> float:
    """The owner's piecewise form, directed 2026-08-07: wins graded by own strength kept, losses
    graded by the damage dealt to the enemy, both priced by engine creature strength.

    The loss branch replaces own-survival credit with enemy-destruction credit, because in a
    lost fight surviving often means having fled while damage dealt measures having fought; a
    rout still reads -1.0 and a near-win loss approaches 0.0, so every loss stays below every
    win and the difficulty weighting's sign split keeps holding. The two enemy-term objections
    recorded in reward-design do not apply: the win branch never reads the foe (no collapse into
    the win bit), and the credit is terminal-only (no per-step damage-farming incentive).
    Stalemates keep their engine-grounded resolution: the defender's win branch prices its kept
    strength, and the forfeiting attacker's enemy-damage is zero by construction, reading -1.0.
    """
    own = record["attacker" if side == "attacker" else "defender"]
    foe = record["defender" if side == "attacker" else "attacker"]
    won = _side_won(record, side)
    # `commanded` prices each stack at its effective attack and defense, which include the
    # commander's. The base pricing calls two identical armies equal when one of them is led by a
    # hero worth more than any budget difference the sampler draws, so on those matchups the
    # ratio the reward sees and the ratio the battle plays out at are different numbers.
    now, start = ("strength_commanded", "initial_strength_commanded") if commanded \
        else ("strength", "initial_strength")
    if won:
        initial = float(own.get(start, own.get("initial_strength", 0.0)))
        kept = float(own.get(now, own.get("strength", 0.0))) / initial if initial > 0 else 0.0
        return 1.0 + kept
    foe_initial = float(foe.get(start, foe.get("initial_strength", 0.0)))
    destroyed = 1.0 - (float(foe.get(now, foe.get("strength", 0.0))) / foe_initial if foe_initial > 0 else 0.0)
    if record["termination"] == "stalemate" and side == "attacker":
        destroyed = 0.0
    return -1.0 + destroyed


def terminal_reward_strength(record: dict, side: str) -> float:
    """The strength-margin variant, owner-directed 2026-08-05: survival priced by engine
    creature strength rather than hit points, so losing a Champion costs what a Champion is
    worth and a win with cheap losses outscores the same win paid for in cavalry. Both totals
    are engine-computed in the terminal record; commander bonuses are not priced in."""
    own = record["attacker" if side == "attacker" else "defender"]
    initial = float(own.get("initial_strength", 0.0))
    survived = float(own.get("strength", 0.0)) / initial if initial > 0 else 0.0
    won = _side_won(record, side)
    if record["termination"] == "stalemate" and side == "attacker":
        survived = 0.0
    return (1.0 if won else -1.0) + survived


def _side_won(record: dict, side: str) -> bool:
    """Who a terminal record says won, including the stall case the owner raised.

    A battle nobody finishes is not a free draw. The engine's own AI settles it: after fifty
    turns without a death it forces the attacking hero to retreat, which loses the attacker the
    battle. The runner stops at forty no-death rounds (its captains cannot retreat, so letting
    the engine's breaker fire would abort), and this function scores that termination the way
    the engine would have resolved it: the defender outlasted the attacker and wins, the
    attacker who failed to force an engagement loses. Without this, an evading defender scored
    -1 + survival = 0.0, and a policy in a losing matchup would rationally prefer stalling at
    0.0 to fighting at -0.4; with it, evasion is worth +2.0 to a defender exactly when the real
    game would award the battle. The retreat also costs the attacker its army, so both reward
    functions zero the attacker's survival term at a stalemate, -1.0 flat, strictly worse than
    fighting and losing with anything left; the first run of the evasion demo showed that was
    not yet true, an evading attacker banking 0.0 through full survival.

    The 100-round `round_limit` cap stays a loss for both sides: it has no engine analogue, it
    never fired in 16,060 recorded episodes (61 stalemates did), and a battle still trading
    deaths at that horizon is an artifact of truncation rather than a stall anyone chose.
    """
    if record["termination"] == "stalemate":
        return side == "defender"
    return record["termination"] == ("victory" if side == "attacker" else "defeat")


class MatchupPool:
    """Rotate over several matchups, one battle at a time.

    Training on a single matchup measures that matchup, not the policy. Rotating means the
    gradient comes from a distribution, which is what makes a reported number a statement about
    the generator rather than about one army pair.

    A group-relative trainer needs more than rotation, and getting this wrong is silent. Leave-one-
    out, GRPO and Dr. GRPO all ask how one episode compares with others started from the same
    place, which is why GRPO on a language model samples several completions of one prompt. If
    every episode in a group draws a different army pair, the baseline measures which matchup was
    drawn rather than how the policy played, and the advantage is mostly scenario difficulty.

    So the matchup is resampled on `new_group` rather than on `reset`. A trainer that never calls
    `new_group` gets the old behaviour of one matchup per episode, which is correct for a method
    whose baseline is a learned critic, since a critic conditions on the observation and does not
    care which episodes sit beside it in the batch.
    """

    def __init__(self, worker: str, matchups, side: str = "attacker", seed: int = 0, home: str = "/tmp",
                 hold_within_group: bool = False, seeds: int = 1, reward_weighting: str = "none",
                 reward_margin: str = "hit_points"):
        import random

        self._worker = worker
        self._matchups = list(matchups)
        self._side = side
        self._rng = random.Random(seed)
        self._home = home
        self._env: BattleEnv | None = None
        self._hold = hold_within_group
        self._seeds = seeds
        self._reward_weighting = reward_weighting
        self._reward_margin = reward_margin
        self.side = side
        self.current = None

    def new_group(self) -> None:
        """Draw the matchup the next group of episodes will share."""
        self.current = self._rng.choice(self._matchups)

    def reset(self):
        # The environment is rebuilt only when the matchup changes. Keeping it across episodes
        # of the same matchup is what lets `seeds` rotate battlefields inside a group, since the
        # rotation lives in the worker's scenario list and dies with the process.
        previous = self.current
        if not self._hold or self.current is None:
            self.current = self._rng.choice(self._matchups)
        if self._env is None or self.current is not previous:
            self.close()
            self._env = BattleEnv(self._worker, side=self._side, attacker=self.current.attacker,
                                  defender=self.current.defender, home=self._home,
                                  attacker_hero=getattr(self.current, "attacker_hero", None),
                                  defender_hero=getattr(self.current, "defender_hero", None),
                                  allow_wide=getattr(self.current, "allow_wide", False),
                                  seeds=self._seeds, reward_weighting=self._reward_weighting,
                                  reward_margin=self._reward_margin)
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
