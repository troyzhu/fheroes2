"""Root search over the real engine: the primitives every search harness shares.

Lifted out of `agent_play/experiments/search_probe.py` on 2026-08-10, where nine scripts had been
reaching for them across a `sys.path` insert. The probe that first hosted them is still an
experiment and still lives there; what moved is only the machinery its callers depend on.

The method is one ply of explicit branching with Monte Carlo playouts, not a tree. At a decision,
candidate actions are scored by PUCT (Q from playout returns, prior from the policy, exploration
bonus from visit counts) over a budget of simulations, and the most-visited action plays. Playouts
run on the real engine by reset-continuation: a persistent side environment replays the action
prefix, applies the candidate, and samples the policy to termination. Increasing the budget
sharpens each candidate's estimate and covers more of them; it never looks further ahead, because
a playout already runs to the end of the battle.

Two properties of the side environment decide what the numbers mean, and both were configuration
accidents until 2026-08-09. What `rollout` returns is that environment's terminal reward, so its
`reward_margin` is the quantity search maximizes. And its world seed fixes both the battlefield and
the combat dice, so pinning it to the live episode is required for the prefix replay to track the
live position, but pinning it without a `combat_seed_offset` also hands search the live battle's
actual dice. The replay is bit-exact only in that shared-dice configuration, which ADR 0008 labels
a ceiling; under the nonzero offset the same ADR mandates for honest numbers, different rolls mean
different casualties, so the sim state is a resampled trajectory under the same action prefix
rather than a copy of the live one. A candidate that is legal in the live battle may therefore not
exist at the replayed position, 21.6 percent of them over 256 measured decisions and 36 percent by
depth twelve (`agent_play/experiments/replay_divergence.py`). The engine answers an illegal
selection by skipping the acting unit's turn (`agent_external_controller.cpp`), so scoring such a
candidate measures a different action. Declining to score it is nonetheless measured worse, by
$-0.139$ win rate over three paired seeds, because search picks such candidates on 30 percent of
decisions and the live battle can play them; substitution is the default and `exclude_unappliable`
keeps the losing arm reproducible. See `agent_play/docs/rl/reward-design.md`.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .env import BattleEnv
from .policy import BattlePolicy

def _plane_arg(model: BattlePolicy, source) -> tuple:
    """The planes tensor for a planes-built policy, from whichever env presented the state.

    A planes policy hard-fails without its tensor rather than silently reading zeros, so every
    search path threads the presenting environment through; entity policies get an empty tuple
    and are untouched."""
    if not getattr(model, "planes", False):
        return ()
    planes = getattr(source, "last_planes", None)
    if planes is None:
        raise ValueError("planes policy searched through an env constructed without planes=True")
    return (torch.from_numpy(planes).unsqueeze(0),)


def policy_action(model: BattlePolicy, observation: np.ndarray, mask: np.ndarray, sample: bool = True,
                  env=None) -> int:
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(observation).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0),
                          *_plane_arg(model, env))
        if sample:
            return int(torch.distributions.Categorical(logits=logits).sample())
        return int(logits.argmax())


def priors(model: BattlePolicy, observation: np.ndarray, mask: np.ndarray, env=None) -> dict[int, float]:
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(observation).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0),
                          *_plane_arg(model, env))
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    legal = np.flatnonzero(mask)
    return {int(a): float(probs[a]) for a in legal}


def sync_side_environment(sim: BattleEnv | None, live: BattleEnv, worker: str,
                          combat_seed_offset: int = 0, **kwargs) -> BattleEnv:
    """Rebuild the side environment when the live episode moves to a different battlefield.

    `rollout` replays the action prefix in `sim` and relies on that replay reproducing the live
    state exactly, which holds only when both are on the same world seed, since the obstacle layout
    and the combat seed both derive from it. Built with the harness default `seeds=4`, the side
    environment resets to variant zero on every rollout while the live episode rotates over four,
    so most episodes were searched against a battlefield the battle was not on. Rebuilding at
    `seeds=1` pinned to the live variant is what makes the prefix guarantee true.

    Cheap in the only place it is called, once per episode rather than once per rollout, and a
    no-op when the variant has not moved.

    `combat_seed_offset` decides what kind of model the search is given, and it is the difference
    between an upper bound and an honest number. The battle's random stream is derived from the same
    world seed as the battlefield, so pinning the variant alone hands the side environment the live
    battle's actual dice, and search then reads outcomes instead of estimating them. A nonzero offset
    keeps the battlefield and makes the randomness independent, which is what a perfect dynamics
    model with unknown randomness looks like. Measured on the mirror suite 2026-08-10, the searching
    agent reads 0.927 with the live dice and 0.604 without them (`search_leakage.py`), so which one
    is passed changes the conclusion and neither should be chosen by default.
    """
    wanted = live.current_battlefield
    pin = (wanted, combat_seed_offset)
    if sim is not None and getattr(sim, "_pinned_battlefield", None) == pin:
        return sim
    if sim is not None:
        sim.close()
    fresh = BattleEnv(worker, seeds=1, seed_offset=wanted,
                      combat_seed_offset=combat_seed_offset, **kwargs)
    fresh._pinned_battlefield = pin
    return fresh


def rollout_self_play(sim: BattleEnv, model: BattlePolicy, prefix: list[int], first: int,
                      agent_side: str, full_prefix: bool = False,
                      skip_unappliable: bool = False) -> float | None:
    """A playout in which the policy answers BOTH chairs, the owner's 2026-08-13 direction for
    self-play search: the opponent inside a playout should be the policy itself, not the engine's
    built-in AI, so the value estimates match the self-play objective rather than the engine.

    Needs a side environment built with side="both". The prefix holds only the agent's own live
    actions, so during replay the opponent's interleaved decisions are answered by the policy; the
    replayed position therefore tracks the live one only as far as the opponent model matches the
    live opponent, the same resampled semantics the combat offset already carries. The terminal
    reward is computed from the record for the agent's chair explicitly, because a side="both"
    environment's own step reward is not perspectived to the agent, and only record-computable
    margins are valid here (`hit_points` needs episode state a both-sides env does not keep per
    chair; `strength` is the survival-graded record-only analogue)."""
    from .env import reward_from_record
    observation, mask = sim.reset()
    i, applied = 0, False
    while True:
        ours = sim.acting_side == agent_side
        # full_prefix: the caller's prefix holds BOTH chairs' live actions (self-play collection,
        # where the live episode is itself both-sides), so replay consumes it unconditionally;
        # otherwise the prefix is the agent's own actions only (play_vs against a human).
        if i < len(prefix) and (full_prefix or ours):
            action = prefix[i]; i += 1
        elif ours and not applied:
            action = first; applied = True
        else:
            action = policy_action(model, observation, mask, env=sim)
        if not mask[action]:
            if action == first and applied and skip_unappliable:
                # The candidate does not exist at the replayed position. Declining to score it is
                # the honest reading and is measured worse (see `rollout`), so it is opt-in.
                return None
            # A stale PREFIX action: the replay has diverged. Continue under the policy so the
            # playout still reaches a terminal, which is the accepted resampled semantics.
            action = policy_action(model, observation, mask, env=sim)
        step = sim.step(action)
        if step.done:
            if not applied:
                return None  # battle ended before the candidate was ever played
            return reward_from_record(step.info, agent_side, sim._reward_margin)
        observation, mask = step.observation, step.mask


def rollout(sim: BattleEnv, model: BattlePolicy, prefix: list[int], first: int,
            skip_unappliable: bool = False) -> float | None:
    """Replay the prefix, apply the candidate, sample the policy to terminal; the return is the
    episode's terminal reward, or None when the candidate could not be applied at all.

    With the side environment on the live dice the replay reproduces the position exactly. Under
    the offset combat stream ADR 0008 mandates for honest numbers it does not: different rolls kill
    different units, so the replayed position drifts from the live one and a live-legal candidate
    may not exist there. Measured 2026-08-23 on a mirror matchup, 38 percent of live-legal
    candidates were un-appliable by depth eight under the offset, against 0 percent on shared dice.

    Returning None is what makes that visible. The engine's contract for an illegal selection is to
    skip the acting unit's turn (`agent_external_controller.cpp`), so the old code played a
    DIFFERENT action and credited its value to the candidate, and a prefix that ended the battle
    early returned a terminal the candidate never reached. Both silently corrupted the Q value the
    search compares and the label a corpus records."""
    observation, mask = sim.reset()
    for action in prefix:
        step = sim.step(action)
        if step.done:
            return None  # the resampled battle ended before the candidate's position was reached
        observation, mask = step.observation, step.mask
    if skip_unappliable and not mask[first]:
        return None
    # Otherwise the candidate is played regardless. The engine answers an illegal selection by
    # skipping the acting unit's turn, so the number that comes back is the value of skipping and
    # playing on, credited to the candidate. That is fabricated evidence and it was found on
    # 2026-08-23, but declining to score the candidate is MEASURED WORSE and by a wide margin:
    # paired over three seeds on the mirrors, excluding cost -0.139 win rate, -0.267 reward and
    # -0.128 strength margin, every seed negative. The reason is mechanical. Un-appliable
    # candidates are 19.2 percent of the legal set but the search CHOSE one on 30.4 percent of
    # decisions, so excluding them forbids the move search wants roughly a third of the time,
    # and the live battle can play them perfectly well. A noisy value for every option beats an
    # exact value for two thirds of them. `skip_unappliable` keeps the exclusion arm reproducible.
    step = sim.step(first)
    while not step.done:
        action = policy_action(model, step.observation, step.mask, env=sim)
        step = sim.step(action)
    return step.reward



def _sequential_halving(do_rollout, prior: dict[int, float], simulations: int,
                        visits: dict[int, int], total_return: dict[int, float],
                        candidates: int = 0) -> list[int]:
    """Spend the budget by Sequential Halving (Karnin et al. 2013) instead of by UCB.

    PUCB minimises cumulative regret, which is the right objective when a node's estimate feeds a
    parent. The root has no parent: only the action finally returned matters, never the ones tried
    on the way, so the root is a simple-regret problem (Bubeck et al. 2011, and Danihelka et al.
    2022 for the AlphaZero-specific argument). This search is a single ply, so the root is the
    whole search and the mismatch is total rather than partial.

    Coverage forcing, which this replaces, is the first phase of this algorithm with the rest
    missing: it buys one rollout for every candidate and then hands the remainder to UCB, paying
    the full breadth cost without the schedule that makes breadth pay. Here the budget is split
    evenly across ceil(log2(m)) phases, every survivor in a phase is measured equally often, and
    the worse half is dropped at the end of each phase.

    `candidates` caps how many of the prior's top actions enter phase one, which is the paper's
    `m`; zero admits every legal action. Returns the surviving arms, best mean first, because the
    schedule's answer is its last survivor rather than the most-visited action.
    """
    unavailable: set[int] = set()
    survivors = sorted(prior, key=prior.get, reverse=True)
    if candidates:
        survivors = survivors[:candidates]
    if not survivors:
        # Guard the empty case only. The earlier form also fired at candidates=1 and replaced the
        # top-prior candidate with list(prior)[:1], which is legal-action insertion order, so a
        # one-candidate search spent its whole budget measuring the lowest-indexed legal action
        # (2026-08-12 panel, finding 17).
        survivors = sorted(prior, key=prior.get, reverse=True)[:1]
    phases = max(1, math.ceil(math.log2(max(len(survivors), 2))))
    spent = 0
    while spent < simulations:
        # Equal visits per survivor within a phase is what makes the comparison at the end of the
        # phase fair; a floor of one keeps a wide phase from being skipped entirely on a small budget.
        per = max(1, (simulations - spent) // max(phases * len(survivors), 1)) if len(survivors) > 1 else \
            simulations - spent
        for a in list(survivors):
            for _ in range(per):
                if spent >= simulations:
                    break
                value = do_rollout(a)
                if value is None:
                    # Un-appliable at the replayed position: drop the arm rather than score it
                    # with a substitute, and do not charge the budget for a playout that never ran.
                    unavailable.add(a)
                    survivors = [x for x in survivors if x != a]
                    break
                total_return[a] += value
                visits[a] += 1
                spent += 1
        if not survivors:
            break
        if len(survivors) <= 1:
            break
        means = {a: (total_return[a] / visits[a] if visits[a] else 0.0) for a in survivors}
        # Ceil, not floor: Karnin's rule keeps the better HALF, so three survivors keep two.
        # Floor division kept one, committing the remaining budget to an arm chosen on two
        # samples each (2026-08-12 panel, finding 16).
        survivors = sorted(survivors, key=lambda a: means[a], reverse=True)[:max(1, (len(survivors) + 1) // 2)]
        phases = max(1, phases - 1)
    return survivors


def search_action_detail(sim: BattleEnv, model: BattlePolicy, prefix: list[int],
                         observation: np.ndarray, mask: np.ndarray, simulations: int,
                         c_puct: float, live: BattleEnv | None = None,
                         coverage_forced: bool = False, allocator: str = "puct",
                         candidates: int = 0, rollout_opponent: str = "ai",
                         agent_side: str | None = None, full_prefix: bool = False,
                         diagnostics: dict | None = None,
                         exclude_unappliable: bool = False) -> tuple[int, dict, dict, dict]:
    """The search decision plus its whole measurement: per-candidate mean rollout values, visit
    counts, and the prior. The values are the counterfactuals only search produces (a real
    playout per candidate it tried), which is what makes them valid soft-distillation targets
    where fitted state values and behavior Q measured 0.00; the prior anchors the target on
    support per Grill et al.

    `coverage_forced` is the demonstrated prerequisite from the soft-target program: UCB left
    alone visits about two candidates per state, which starves every downstream consumer of
    support, so the forced variant spends the first rollouts visiting every candidate once, in
    descending prior order, before UCB takes over. With more candidates than simulations the
    sweep is truncated at the simulation budget, still widest-support-first."""
    # The observation is the live environment's state, so a planes policy needs the live
    # env's tensor here; the sim's belongs to whatever state its own replay last presented.
    prior = priors(model, observation, mask, env=live if live is not None else sim)
    actions = list(prior)
    if rollout_opponent == "policy":
        if agent_side not in ("attacker", "defender"):
            raise ValueError("rollout_opponent='policy' needs agent_side, and a side='both' sim")
        do_rollout = lambda a: rollout_self_play(sim, model, prefix, a, agent_side, full_prefix,
                                                 exclude_unappliable)  # noqa: E731
    elif rollout_opponent == "ai":
        do_rollout = lambda a: rollout(sim, model, prefix, a, exclude_unappliable)  # noqa: E731
    else:
        raise ValueError(f"unknown rollout_opponent {rollout_opponent!r}")
    if len(actions) == 1:
        return actions[0], {actions[0]: 0.0}, {actions[0]: 1}, prior
    if simulations <= 0:
        # No budget means no search, so the prior's own pick stands. Without this the tie-break
        # below sees an all-zero visit count and an all-zero value, ties on every candidate, and
        # returns the lowest legal action index: not the policy, not an error, just array order.
        # `search_strength.py` sidesteps it by branching to `policy_action` before calling here,
        # but a harness that simply passes `--simulations 0` would otherwise measure nothing.
        top = max(prior, key=prior.get)
        return top, {a: 0.0 for a in actions}, {a: 0 for a in actions}, prior
    visits = {a: 0 for a in actions}
    total_return = {a: 0.0 for a in actions}
    if candidates and allocator == "puct":
        # The control that separates the two things a capped halving arm changes at once. Sequential
        # halving with a cap both restricts the candidate set and reallocates the budget within it,
        # so a gain over plain PUCT is not attributable until PUCT has been run on the same
        # restricted set. Without this arm the 2026-08-11 result would repeat the terrain-versus-dice
        # attribution error: two variables moved, one credited.
        actions = sorted(prior, key=prior.get, reverse=True)[:candidates]
    if allocator == "sequential_halving":
        survivors = _sequential_halving(do_rollout, prior, simulations, visits, total_return,
                                        candidates)
        means = {a: (total_return[a] / visits[a] if visits[a] else 0.0) for a in actions}
        # The answer is the best surviving arm, not the most-visited action overall. Under a
        # truncated final phase the visit maximum can sit on whichever survivor the loop reached
        # first, or on an arm eliminated earlier, so the (visits, means) rule returned an action
        # the schedule never selected (2026-08-12 panel, finding 18). The visits>0 guard covers
        # budgets smaller than the survivor count, where an unvisited survivor's 0.0 mean must not
        # outrank a measured one.
        best = max(survivors, key=lambda a: (visits[a] > 0, means[a]))
        return best, means, visits, prior
    if allocator != "puct":
        raise ValueError(f"unknown allocator {allocator!r}")
    sweep = sorted(actions, key=lambda a: -prior[a]) if coverage_forced else []
    # Candidates the replayed position cannot offer. They are excluded from selection rather than
    # scored with a substitute's value, and they cost no budget, because no playout ran.
    unavailable: set[int] = set()
    n = 0
    while n < simulations:
        pool = [a for a in actions if a not in unavailable]
        if not pool:
            break
        unvisited = [a for a in sweep if visits[a] == 0 and a not in unavailable]
        if unvisited:
            chosen = unvisited[0]
        else:
            scores = {}
            for a in pool:
                q = total_return[a] / visits[a] if visits[a] else 0.0
                u = c_puct * prior[a] * math.sqrt(n + 1) / (1 + visits[a])
                scores[a] = q + u
            chosen = max(scores, key=scores.get)
        value = do_rollout(chosen)
        if value is None:
            unavailable.add(chosen)
            continue
        visits[chosen] += 1
        total_return[chosen] += value
        n += 1
    means = {a: (total_return[a] / visits[a] if visits[a] else 0.0) for a in actions}
    if diagnostics is not None:
        diagnostics["unavailable"] = sorted(unavailable)
        diagnostics["candidates"] = len(actions)
        diagnostics["playouts_run"] = n
    if not any(visits[a] for a in actions):
        # Every candidate was un-appliable, so nothing was measured; the prior's pick stands rather
        # than an argmax over an all-zero table, which would return the lowest legal index.
        return max(prior, key=prior.get), means, visits, prior
    # Ties on visit count are broken by the mean rollout value, not by action index. `visits` is
    # keyed in ascending legal-action order, so a plain argmax over it returns the lowest index,
    # and under coverage forcing ties are the common case rather than the rare one: 11.12 percent
    # of the 15,007 decisions in the first scaled corpus tied at the top and every one of them
    # was labeled by array position. The value is the signal the search actually measured.
    best = max(actions, key=lambda a: (visits[a], means[a]))
    return best, means, visits, prior


def search_action(sim: BattleEnv, model: BattlePolicy, prefix: list[int],
                  observation: np.ndarray, mask: np.ndarray, simulations: int, c_puct: float,
                  live: BattleEnv | None = None, coverage_forced: bool = False) -> int:
    action, _, _, _ = search_action_detail(sim, model, prefix, observation, mask, simulations, c_puct,
                                           live=live, coverage_forced=coverage_forced)
    return action
