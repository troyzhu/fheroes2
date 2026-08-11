#!/usr/bin/env python3
"""The searching agent on the battery's own suites, with the quality columns.

Root search over the cloned prior is the only mechanism here measured above the built-in AI, but
that was only ever measured on the held-out pool. This runs it on any subset of the battery's
suites, from either chair, reporting the same columns policies report, so the agent regime can be
read against the engine on the same scoreboard rather than on one number.

The question it exists to answer: the 2026-08-08 scoreboard leaves three suites at par and the
gap concentrated in the held-out pool and the two mirror chairs. If search crosses the engine on
the mirror chairs too, the distillation target is well defined everywhere and the remaining work
is transfer into weights. If search does not, the mirror deficit is not a policy-quality problem
at all, and no amount of distillation will move it.

Search costs roughly half a second per decision, so budget episodes accordingly; the defaults
here are deliberately smaller than the battery's, and the report records them.

Usage:
    ./search_agent_battery.py WORKER CHECKPOINT [--suites held_out_pool mirrors_defender ...]
                              [--episodes 8] [--simulations 32] [--report R.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import zlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fheroes2_agent.env import REWARD_MARGINS, BattleEnv, ScenarioRejected, _side_won  # noqa: E402
from fheroes2_agent.policy import load_policy  # noqa: E402
from fheroes2_agent.search import policy_action, search_action_detail, sync_side_environment  # noqa: E402
from fheroes2_agent.suites import SUITE_SIDE, build_suites  # noqa: E402

DEFAULT_SUITES = ("held_out_pool", "mirrors_attacker", "mirrors_defender", "held_out_as_defender")


def play_episodes(worker: str, matchup, side: str, model, episodes: int, simulations: int,
                  c_puct: float, seeds: int, searched: bool, reward_margin: str = "two_sided",
                  allocator: str = "puct", candidates: int = 0,
                  combat_seed_offset: int = 0, report_margin: str = "two_sided") -> dict:
    """One matchup, measured with the same columns `scenarios.measure` reports for policies."""
    kwargs = dict(side=side, attacker=matchup.attacker, defender=matchup.defender,
                  attacker_hero=matchup.attacker_hero, defender_hero=matchup.defender_hero,
                  allow_wide=matchup.allow_wide)
    # Two different jobs that had been sharing one flag. The live environment's margin decides what
    # the `rw` column MEASURES; the side environment's decides what search MAXIMIZES. Passing one
    # value to both meant asking for a search objective silently changed the reported reward, and
    # the report went on stamping "two_sided" regardless, so it mislabelled its own column.
    env = BattleEnv(worker, seeds=seeds, reward_margin=report_margin, **kwargs)
    # The side environment is pinned to whatever battlefield the live episode is on, and rebuilt
    # when that rotates. Sharing the live `seeds` here silently searched the wrong terrain.
    sim = None
    wins, rewards, lengths, survival, damage, margins = [], [], [], [], [], []
    # The same deployment-side diagnostics `scenarios.measure` reports, which this harness had
    # never carried, so a searched arm could only ever be compared to a policy arm on the rate.
    # Entropy is kept in both forms: raw, in nats, and normalized against the uniform maximum over
    # the state's own legal set so a five-action state and a thirty-action state read on one scale.
    entropies, normalized, perplexities, supports, legal_counts, rounds = [], [], [], [], [], []
    reward_wins, reward_losses, visit_entropies = [], [], []
    try:
        for _ in range(episodes):
            observation, mask = env.reset()
            # After the reset, not before: `current_battlefield` reads the live worker's
            # `scenario_id`, which names the episode that has just started.
            if searched:
                sim = sync_side_environment(sim, env, worker, reward_margin=reward_margin,
                                            combat_seed_offset=combat_seed_offset, **kwargs)
            prefix, steps = [], 0
            while True:
                # The diagnostics describe the network, not the wrapper around it: under a greedy
                # rule the acting distribution is one-hot for every checkpoint. `Sampler` keeps the
                # wrapped model, so the distribution is read from the inner network either way.
                inner = getattr(model, "model", model)
                with torch.no_grad():
                    raw_logits, _ = inner(torch.from_numpy(observation).unsqueeze(0),
                                          torch.from_numpy(mask).unsqueeze(0))
                diagnostic = torch.distributions.Categorical(logits=raw_logits)
                entropy = float(diagnostic.entropy())
                legal = int(mask.sum())
                entropies.append(entropy)
                normalized.append(entropy / float(np.log(max(legal, 2))))
                perplexities.append(float(np.exp(entropy)))
                supports.append(int((diagnostic.probs.squeeze(0) >= 0.01).sum()))
                legal_counts.append(legal)
                if searched:
                    action, _, visits, _ = search_action_detail(
                        sim, model, prefix, observation, mask, simulations, c_puct, live=env,
                        allocator=allocator, candidates=candidates)
                    # How decided the search itself was, which is a different question from how
                    # decided the network was, and the one that says whether the budget resolved
                    # anything. Zero means every playout went to one candidate.
                    counts = np.asarray([v for v in visits.values() if v > 0], dtype=float)
                    share = counts / counts.sum() if counts.sum() else counts
                    visit_entropies.append(float(-(share * np.log(share)).sum()) if len(share) else 0.0)
                else:
                    action = policy_action(model, observation, mask, env=env)
                prefix.append(action)
                step = env.step(action)
                steps += 1
                if step.done:
                    record = step.info
                    # `_side_won`, not a termination-string comparison, so a stalemate is scored
                    # the way the engine's own breaker resolves it rather than as a loss for both.
                    won = bool(_side_won(record, side))
                    own = record["attacker" if side == "attacker" else "defender"]
                    foe = record["defender" if side == "attacker" else "attacker"]
                    own_initial = float(own.get("initial_strength", 0.0))
                    foe_initial = float(foe.get("initial_strength", 0.0))
                    own_kept = float(own.get("strength", 0.0)) / own_initial if own_initial > 0 else 0.0
                    foe_kept = float(foe.get("strength", 0.0)) / foe_initial if foe_initial > 0 else 0.0
                    wins.append(won)
                    rewards.append(step.reward)
                    lengths.append(steps)
                    margins.append(own_kept - foe_kept)
                    (survival if won else damage).append(own_kept if won else 1.0 - foe_kept)
                    (reward_wins if won else reward_losses).append(step.reward)
                    if record.get("rounds") is not None:
                        rounds.append(record["rounds"])
                    break
                observation, mask = step.observation, step.mask
    finally:
        env.close()
        if sim is not None:
            sim.close()
    mean = lambda xs: float(np.mean(xs)) if len(xs) else None  # noqa: E731
    return {"win_rate": float(np.mean(wins)), "mean_reward": float(np.mean(rewards)),
            # The owner's standing reporting requirement: the rate is never the report.
            "entropy": mean(entropies), "normalized_entropy": mean(normalized),
            "effective_actions": mean(perplexities), "support_at_1pct": mean(supports),
            "legal_actions": mean(legal_counts), "mean_rounds": mean(rounds),
            "reward_on_wins": mean(reward_wins), "reward_on_losses": mean(reward_losses),
            "search_visit_entropy": mean(visit_entropies),
            "mean_length": float(np.mean(lengths)), "strength_margin": float(np.mean(margins)),
            "surviving_strength": float(np.mean(survival)) if survival else None,
            "loss_damage": float(np.mean(damage)) if damage else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--allocator", default="puct", choices=("puct", "sequential_halving"),
                        help="how the root spends its budget. puct optimises cumulative regret, "
                             "which the root does not need; sequential_halving optimises simple "
                             "regret, which is what only-the-final-action-matters means")
    parser.add_argument("--candidates", type=int, default=0,
                        help="cap the candidate set entering sequential halving (the paper's m). "
                             "Zero admits every legal action, which at this budget leaves one visit "
                             "per candidate in phase one and degenerates into uniform coverage")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--deployment", default="sample", choices=("sample", "greedy", "adaptive"),
                        help="how the network acts, both on its own and as the search prior. The policy "
                             "battery has had this since the deployment sweep; the search battery never "
                             "did, so search had only ever been measured under sampled actions")
    parser.add_argument("--seed", type=int, default=None,
                        help="seeds the policy sampling per matchup, so a run is reproducible and two "
                             "configurations can be compared paired. Until 2026-08-10 this harness seeded "
                             "nothing at all, so every number it produced carried unmeasured run-to-run "
                             "variance: two runs of one configuration on held_out_pool read 0.619 and 0.750")
    parser.add_argument("--search-combat-offset", type=int, default=0,
                        help="perturbs the side environment's random stream while keeping its battlefield. "
                             "Zero gives search the live battle's actual dice, which is an upper bound "
                             "rather than a fair number; nonzero is a perfect dynamics model with unknown "
                             "randomness. On the mirror suite the two read 0.927 and 0.604")
    parser.add_argument("--report-margin", default="two_sided", choices=REWARD_MARGINS,
                        help="which objective the rw column MEASURES, matching validation_battery. "
                             "Separate from what search maximizes, because they are separate choices "
                             "and sharing one flag silently changed the reported reward whenever a "
                             "search objective was requested")
    parser.add_argument("--reward-margin", default="two_sided", choices=REWARD_MARGINS,
                        help="what root search maximizes, since `rollout` returns the side "
                             "environment's terminal reward. Every search number recorded before "
                             "2026-08-09 used two_sided because this was hardcoded; "
                             "`search_objective.py` measured hit_points +0.191 over it on the "
                             "mirror suite, so the default is kept only for comparability"),
    parser.add_argument("--fresh", type=int, default=24)
    parser.add_argument("--baseline", action="store_true",
                        help="also measure the same policy without search, the paired control")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    model = load_policy(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    if args.deployment != "sample":
        # A deployment rule is a logits transform wrapped around the checkpoint, matching the
        # policy battery, so the same rule governs the played move and the search prior.
        from sampling_policies import Sampler
        model = Sampler(model, "greedy" if args.deployment == "greedy" else "adaptive")
    model.eval()
    suites = build_suites(args.fresh)
    started = time.time()
    report = {"checkpoint": args.checkpoint, "episodes": args.episodes, "eval_seeds": args.seeds,
              "search_objective": args.reward_margin,
              "search_combat_offset": args.search_combat_offset, "seed": args.seed,
              # Self-labelling, because the offset defaults to the leaky value and a run that
              # forgets it produces a report indistinguishable from an honest one at a glance. A
              # printed warning is not enough: these are routinely run with stdout redirected, and
              # this harness was in fact run once that way on 2026-08-11 before the report's own
              # offset field caught it. Any table built from the JSON now carries the label.
              "dice": ("shared-with-live (CEILING, not comparable to the built-in AI)"
                       if args.simulations > 0 and args.search_combat_offset == 0
                       else "independent-of-live" if args.simulations > 0 else "unsearched"),
              "deployment": args.deployment, "allocator": args.allocator,
              "candidates": args.candidates,
              "simulations": args.simulations, "reward_margin": args.report_margin, "arms": {}}

    arms = [("search", True)] + ([("policy", False)] if args.baseline else [])
    for arm, searched in arms:
        report["arms"][arm] = {}
        for suite in args.suites:
            side = SUITE_SIDE.get(suite, "attacker")
            measured = []
            for index, m in enumerate(suites[suite]):
                if args.seed is not None:
                    # Per matchup rather than once per run, so a suite's numbers do not depend on
                    # which suites ran before it and two arms meet identical sampling noise.
                    # `zlib.crc32`, not `hash`: Python randomizes string hashing per process unless
                    # PYTHONHASHSEED is pinned, so the first version of this line produced a
                    # different stream in every process and the runs it labelled "seed 11" were not
                    # reproducible at all. They remain valid as independent samples, which is what
                    # the spread measurement used them for, but not as a repeatable configuration.
                    salt = zlib.crc32(suite.encode()) % 997
                    torch.manual_seed(args.seed + 1000 * salt + index)
                try:
                    measured.append(play_episodes(args.worker, m, side, model, args.episodes,
                                                  args.simulations, args.c_puct, args.seeds, searched,
                                                  args.reward_margin,
                                                  allocator=args.allocator, candidates=args.candidates,
                                                  combat_seed_offset=args.search_combat_offset,
                                                  report_margin=args.report_margin))
                except ScenarioRejected as error:
                    print(f"  {suite}: matchup rejected ({str(error)[:70]})", flush=True)
            if not measured:
                continue
            def column(key):
                vals = [d[key] for d in measured if isinstance(d.get(key), (int, float))]
                return float(np.mean(vals)) if vals else float("nan")
            # Every numeric column the episodes produced, aggregated generically. The hardcoded
            # list this replaced is why adding a diagnostic to `play_episodes` used to require
            # remembering to add it here as well, and why the entropy columns never appeared.
            keys = sorted({k for d in measured for k, v in d.items() if isinstance(v, (int, float))})
            report["arms"][arm][suite] = {"per_matchup": measured} | {k: column(k) for k in keys}
            r = report["arms"][arm][suite]
            fmt = lambda k, s="+.2f": (f"{r[k]:{s}}" if r.get(k) == r.get(k) and r.get(k) is not None else "  --")  # noqa: E731
            print(f"{arm:7s} {suite:22s} rate {r['win_rate']:.3f}  wq {fmt('surviving_strength','.2f')} "
                  f"lq {fmt('loss_damage','.2f')} mg {fmt('strength_margin')} rw {fmt('mean_reward')} "
                  f"rwW {fmt('reward_on_wins')} rwL {fmt('reward_on_losses')} "
                  f"H {fmt('entropy','.2f')} Hn {fmt('normalized_entropy','.2f')} "
                  f"effA {fmt('effective_actions','.1f')} sup {fmt('support_at_1pct','.1f')} "
                  f"legal {fmt('legal_actions','.0f')} dec {fmt('mean_length','.0f')} "
                  f"rnds {fmt('mean_rounds','.1f')} Hvis {fmt('search_visit_entropy','.2f')}", flush=True)

    report["seconds"] = round(time.time() - started, 1)
    print(f"\ntotal {report['seconds']:.0f}s")
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
