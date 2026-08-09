#!/usr/bin/env python3
"""Write a readable index for each vendored run-report directory.

The archive holds hundreds of report files, which is correct as provenance and useless as a
reading surface: the owner's complaint on 2026-08-09 was that a directory of a hundred and more
JSONs cannot be read. Every number in them is already quoted in a dated log, so the index does not
repeat numbers; it says what each family of files answers and which script produced it, so a
reader can find the one artifact behind a claim without opening any of them.

The index is generated rather than hand-written so it cannot drift from the directory, and
`lint_docs.sh` fails when a file matches no family, which is what stops a new artifact from
landing unexplained. Adding a family here is the intended way to introduce a new report type.

Usage:
    ./index_run_reports.py [DIR ...]        # defaults to every dated directory under files/
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

FILES = pathlib.Path(__file__).resolve().parents[1] / "docs" / "archive" / "experiments" / "files"

# Ordered: the first pattern that matches a file names it, so put specific prefixes first.
FAMILIES: list[tuple[str, str, str]] = [
    ("builtin_ai_baseline*", "builtin_ai_baseline.py",
     "The engine's own AI on every validation suite, the bar the policies must clear; the `_v2` form adds the quality columns"),
    ("battery_deploy_*", "validation_battery.py --deployment",
     "The same checkpoints under sampled, greedy and entropy-adaptive action rules, which is how the deployment rule was settled"),
    ("battery_greedy_*", "validation_battery.py --deployment greedy",
     "Greedy evaluation across seeds, the three-seed confirmation of a single-seed reading"),
    ("battery_massmatched*", "validation_battery.py",
     "Corpora compared at identical total soft mass, so the arm differs only in which decisions carry the weight"),
    ("battery_band*", "validation_battery.py",
     "The regret-band corpus against the unscreened one and their combination"),
    ("battery_*", "validation_battery.py",
     "A report card over the suites for a named arm: rates per suite, per-rung ladders, and the quality columns"),
    ("symmetry_*", "symmetry_gap.py",
     "The same army from either chair against the engine's own asymmetry, which is the neutral point"),
    ("duels_*", "selfplay_probe.py duel tables",
     "Head-to-head rates against each pool opponent and the built-in AI"),
    ("convergence_*", "convergence_report.py",
     "Per-metric settlement verdicts over a run's heartbeat: converged, trending or oscillating"),
    ("heartbeat_*", "the trainers themselves",
     "One JSON line per training iteration or epoch: losses decomposed, per-term per-module gradient norms, reward split over wins and losses, terminations, and any armed mechanism's own column"),
    ("deviation_probe*", "deviation_probe.py",
     "How often search disagrees with the prior, split by whether the policy wins the matchup"),
    ("regret_density*", "computed from the corpora",
     "Regret per decision in each corpus, which is what the collection screen was aimed at"),
    ("coverage_corpus_stats*", "computed from the corpus",
     "Support completeness of a coverage-forced corpus: what fraction of decisions priced every candidate"),
    ("coverage_*_manifest*", "search_teacher.py",
     "Which matchups a coverage-forced collection shard kept or dropped, with the flags it ran under"),
    ("regret_band_shard*", "search_teacher.py --policy-max-win",
     "The screened collection's manifest, recording each matchup's prior win rate and why it was kept"),
    ("*_manifest*", "the recording scripts",
     "A dataset's matchup list and settings; corpora re-record from these rather than being vendored whole"),
    ("*distill*", "soft_distill.py, awr_distill.py",
     "A paired distillation run: the arm, its twin, target entropy and per-epoch history"),
    ("*fidelity*", "fidelity_report.py",
     "Top-k agreement with the teacher, the probability given to its move, entropy and the reliability table"),
    ("rounds_probe*", "rounds_probe.py",
     "Round-count histograms and stalemate incidence per checkpoint"),
    ("search_*", "search_probe.py, search_teacher.py, search_agent_battery.py",
     "What root search measures or plays: per-candidate values, agreement, or suite results"),
    ("*value*", "the value-estimation lab",
     "A fitted value estimator and what it explained, kept because the thread is an educational record"),
    ("pool*", "the scenario generator",
     "A standing matchup pool that later runs evaluate against"),
    ("*.pt", "the trainers",
     "An anchor checkpoint other artifacts are calibrated against"),
    ("dagger*", "dagger_iteration.py",
     "A DAgger round: student-played states relabeled by the planner probe, and what the retrained clone scored"),
    ("thunk_*", "thunk_validation.py",
     "The Thunk opening fight as a standing ladder, per checkpoint and per rung"),
    ("critic*", "critic_pre_fitting.py, critic_calibration.py, critic_on_pool.py",
     "A critic fit and where it actually works: explained variance and bias on teacher against student states"),
    ("ppo_*", "ppo_from_strongest.py and the PPO harnesses",
     "A reinforcement arm's own report: what it gained on the pool and what it did to held-out play"),
    ("selfplay_*", "selfplay_probe.py",
     "A self-play round's duels before and after, and its transfer against the built-in AI"),
    ("sharp_*", "planes_ablation.py sharpness arms",
     "The imitation-sharpness arms: entropy bonus, label smoothing and the early-stop budget cut"),
    ("fid_*", "fidelity_report.py",
     "Fidelity for one sharpness arm, the diagnostic behind the softness verdicts"),
    ("planes_ablation*", "planes_ablation.py",
     "The spatial-observation ablation with its width-matched capacity control"),
    ("ability_ablation*", "ability_ablation.py",
     "Whether explicit per-creature ability features earn their place"),
    ("softplus_*", "planes_ablation.py softplus arm",
     "The activation change measured against its ReLU twin"),
    ("pooling_*", "planes_ablation.py pooling arms",
     "Mean pooling against concatenation over the entity slots"),
    ("champ_planes_*", "planes_ablation.py",
     "The champion mixture trained with the planes arm, per seed"),
    ("owner_gen_*", "search_teacher.py with the owner objective",
     "Labels chosen by search scoring the two-sided reward, per seed"),
    ("gen*", "search_teacher.py generations",
     "A search-taught generation's collection or evaluation"),
    ("arms*", "advantage_and_trust_region.py",
     "Advantage estimators and trust regions compared, on one matchup or on a pool"),
    ("thr*", "advantage_and_trust_region.py --threshold",
     "The divergence trust region at a given threshold"),
    ("floor*", "advantage_floor.py",
     "Whether flooring the advantage-normalization divisor stops a converged run destroying itself"),
    ("telem_*", "advantage_floor.py telemetry",
     "Per-iteration telemetry behind a floor verdict"),
    ("census_*", "the corpus census",
     "What a recorded corpus contains, by matchup and by decision"),
    ("capacity*", "capacity.py",
     "Whether network size binds, for cloning or for reinforcement"),
    ("bc*", "train_bc.py",
     "A behavior-cloning run's history and its best-agreement checkpoint"),
    ("ablation*", "encoding_ablation.py",
     "Which encoding features earn their place, on an episode split and a count-extrapolation split"),
    ("battlefield_spread*", "battlefield_spread.py",
     "How much the obstacle layout alone moves a matchup's win rate"),
    ("planner_query*", "planner_query.py",
     "Whether querying the built-in planner perturbs the battle, paired terminal digests"),
    ("credit_assignment*", "credit_assignment.py",
     "How often trajectory-level credit mis-signs a decision, judged by search rollout values"),
    ("difficulty_reward*", "difficulty_reward.py",
     "Whether difficulty-weighting the terminal reward changes what pool training learns"),
    ("generaliz*", "generalization.py, generalization_battlefields.py",
     "Whether training transfers to matchups or battlefields it never trained on"),
    ("margin*", "the reward experiments",
     "A margin-form reward measured against its alternative"),
    ("mirror_symmetry*", "symmetry_gap.py predecessor",
     "An early chair-swap measurement"),
    ("side_swap*", "the chair experiments",
     "A chair-swap check on recorded episodes"),
    ("replay_*", "capture_replay.py, render_replay.py",
     "A recorded episode kept so it can be rendered again"),
    ("evasion_stalemate*", "evasion_stalemate.py",
     "The stalling exploit demonstrated and priced"),
    ("tabula_rasa_pilot*", "tabula_rasa_pilot.py",
     "Whether search bootstraps a policy from random initialization"),
    ("real_map*", "real_map_fights.py",
     "Opening fights harvested from the shipped maps, with the built-in AI column"),
    ("sampling_policies*", "sampling_policies.py",
     "Deployment sampling schemes measured against each other"),
    ("q_one_step*", "a vendored one-off script and its log",
     "The behavior-Q one-step rerank measurement, kept with the script that produced it"),
    ("inference_walkthrough*", "capture_replay.py",
     "The single decision the inference primer walks through, vendored so its numbers are checkable"),
    ("reverified_numbers*", "the honesty audit",
     "Numbers re-measured after an audit found them unvendored"),
    ("teacher_winrates*", "the teacher census",
     "The built-in planner's own rates on the pool"),
    ("diverse_pick*", "record_diverse.py",
     "Which matchups the diverse recording kept"),
    ("smoke*", "any harness",
     "A smoke run kept because a later claim cites it"),
    ("validation_battery*", "validation_battery.py",
     "An early battery report predating the per-arm naming"),
    ("*matchups*", "the scenario generator",
     "The exact matchup list a round trained on, so the round re-creates"),
    ("round*_bothchair*", "selfplay_round.py",
     "A self-play round's configuration and per-seed outcome, including chair skips"),
    ("*.py", "vendored one-off scripts",
     "A script kept beside its output because it was never promoted to the experiments directory"),
    ("*.log", "vendored run logs",
     "Standard output kept beside the report it explains"),
    ("*.md", "hand-written",
     "A note vendored beside the runs it belongs to"),
    ("*.html", "the dashboard or viewer",
     "A rendered view kept as it appeared"),
]


def family_of(name: str):
    for pattern, script, description in FAMILIES:
        regex = "^" + re.escape(pattern).replace(r"\*", ".*") + "$"
        if re.match(regex, name):
            return pattern, script, description
    return None


def index_directory(directory: pathlib.Path) -> tuple[str, list[str]]:
    groups: dict[tuple, list[str]] = {}
    unknown: list[str] = []
    for path in sorted(directory.iterdir()):
        if path.name == "README.md" or path.name.startswith("."):
            continue
        found = family_of(path.name)
        if found is None:
            unknown.append(path.name)
            continue
        groups.setdefault(found, []).append(path.name)

    lines = [f"# Vendored run reports, {directory.name}", "",
             "Provenance, not a reading path: every number these files carry is already quoted in a"
             " dated log under `../..`, and this index exists so a reader can find the one artifact"
             " behind a claim without opening a hundred of them. Generated by"
             " `agent_play/experiments/index_run_reports.py`; the documentation gate fails when a"
             " file here matches no family below.", "",
             "| Files | Count | Produced by | What it answers |", "|---|---|---|---|"]
    for (pattern, script, description), names in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| `{pattern}` | {len(names)} | `{script}` | {description} |")
    lines.append("")
    return "\n".join(lines) + "\n", unknown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="*", default=None)
    parser.add_argument("--check", action="store_true", help="report unfamilied files and exit nonzero")
    args = parser.parse_args()

    directories = [pathlib.Path(d) for d in args.directories] if args.directories else \
        sorted(p for p in FILES.iterdir() if p.is_dir())
    failures = 0
    for directory in directories:
        text, unknown = index_directory(directory)
        if unknown:
            failures += len(unknown)
            print(f"{directory.name}: {len(unknown)} files match no family: {unknown[:6]}", file=sys.stderr)
        if not args.check:
            (directory / "README.md").write_text(text)
            print(f"{directory.name}: indexed {len(text.splitlines()) - 6} families")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
