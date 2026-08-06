# Experiment scripts

Runnable measurements that are too slow for a verification gate and too specific to be a unit test. Each answers one question, prints a table, and writes a JSON report.

These are separate from `../tests/` and the `../verify_*.sh` gates on purpose. A gate asserts that something still works and must stay fast enough to run on every change. An experiment measures how well something works, takes minutes, and produces a number that goes into `../docs/archive/experiments/`.

The scripts live here rather than being typed at a shell because the earlier round of this work ran from a temporary directory and lost every script when the directory was cleaned. The results survived only because they had been written into the archive, which left the conclusions standing on numbers nobody could reproduce.

| Script | Question | Runtime |
|---|---|---|
| `generalization.py` | Does group-relative training transfer to matchups it never trained on? | about 8 min at the default split |
| `critic_pre_fitting.py` | Does a value head fitted on teacher play improve reinforcement learning, or only look better on paper? | about 12 min over 20 seeds |
| `critic_on_pool.py` | The same question where a single matchup cannot answer it, because every run solves that one | about 12 min over 3 paired seeds |
| `advantage_floor.py` | Does flooring the advantage-normalization divisor stop a converged run destroying itself? Reports dips and terminal collapses separately | about 11 min over 20 seeds |
| `advantage_and_trust_region.py` | Do the advantage estimator and the trust region matter, on one matchup or on a pool, at which threshold? | about 15 min over 10 seeds |
| `solved_region_width.py` | How fast does a solved policy's win rate degrade under parameter noise, matchup by matchup? | about 7 min over 6 matchups |
| `capacity.py` | Does the network size bind, for cloning or for reinforcement learning? | about 12 min over 3 widths |
| `record_diverse.py` | Record teacher demonstrations over the whole bestiary: three count regimes, commanders, wide units, coverage-audited | about 25 min for 12,000 episodes |
| `encoding_ablation.py` | Which encoding earns its features, on an episode split and a count-extrapolation split | about 1 h over 4 variants |
| `thunk_validation.py` | The real opening fight as a standing validation ladder, run against any checkpoints after any change | about 2 min per checkpoint |
| `capture_replay.py` | Record one policy episode as a frame-by-frame replay JSON, encoded per the checkpoint's stamped version, retrying for a wanted outcome | seconds per episode |
| `render_replay.py` | Replay a recording through the real engine: headless verification, rendered capture, digest equality, ffmpeg assembly at true animation timing | about 2 min per episode, opens a game window |
| `planner_query.py` | Does querying the built-in planner at a decision perturb the battle? Paired terminal digests with the probe on and off, plus the DAgger label-resolution rate | about 3 min over 100 paired episodes |
| `battlefield_spread.py` | How much does the obstacle layout alone move a matchup's win rate? Per-battlefield rates against binomial noise | about 5 min over 12 matchups at 6 battlefields |
| `difficulty_reward.py` | Does difficulty-weighting the terminal reward change what pool training learns, on raw win rate? Paired arms from one clone | about 40 min over 3 paired seeds |
| `generalization_battlefields.py` | Does training over rotated battlefields transfer better, judged by evaluation over battlefields? Also re-grounds every transfer number on seed-spread evaluation | about 75 min over 3 paired seeds |
| `dagger_iteration.py` | One DAgger round: student-played states relabeled by the planner probe, clone retrained on the aggregate, judged on pool win rate over battlefields | about 10 min at 1,000 collected episodes |
| `ppo_from_strongest.py` | Does PPO still earn anything from a strong supervised anchor, or erode it? Paired against the anchor's vendored evaluation | about 35 min over 3 seeds |
| `critic_calibration.py` | Where does the behavior value actually work? Explained variance and bias on teacher holdout against student-played states | about 8 min including the refit |
| `search_probe.py` | Does root-PUCT with rollout values lift play on the matchups the policy loses? | about 15 min at 32 simulations |
| `search_teacher.py` | Collection half of a search-taught round: search plays and labels in the dataset schema, sharded across cores | about 30 min at 6 shards |
| `validation_battery.py` | The report card: fresh generator samples, held-out pool, OOD stress suites, and the Thunk ladder, per checkpoint over battlefields | about 40 s per checkpoint |
| `credit_assignment.py` | How often does trajectory-level credit mis-sign a decision, judged by search rollout values? The critique's two error rates, measured | about 2 min at 32 simulations |
| `play_vs.py` | Battle a checkpoint yourself: you command one side in the real battle window, the policy answers the other over the line protocol | interactive, needs a display |
| `ability_ablation.py` | Do explicit per-creature ability features earn their place, paired on one corpus and judged by the battery? | about 25 min for both arms |
| `builtin_ai_baseline.py` | The engine's own AI on every validation suite: the baseline a player recognizes, and the bar the policies must clear | about 70 s for all suites |
| `search_value.py` | Can a value network replace rollouts at search leaves, and is it calibrated where it would be used? | about 10 min including the fit |
| `symmetry_gap.py` | Does the same army win as often from the other chair, and how much of any gap is the game rather than the policy? | about 6 min over 10 matchups |
| `tabula_rasa_pilot.py` | Could search bootstrap a policy from random initialization, with no demonstrations at all? Rounds of search play, distil, evaluate on a fixed matchup set | about 20 min for three rounds |

## Conventions

Every script takes the worker binary as an argument rather than finding it, so a stale build is a visible choice.

Every script accepts `--seed` and reports across seeds where the answer could plausibly be noise. A single-seed comparison is reported as a single-seed comparison.

Errors are taken across the unit that will vary in use. For a pool of matchups that means across matchups rather than across episodes, because episodes inside one matchup share an army pair and pooling them understates the spread.

That choice is not cosmetic and it caught a near-miss. On the generalization run the same difference measured $\pm 0.012$ across seeds and $\pm 0.056$ across matchups, a factor of five, and only the second answers whether a result holds on matchups the generator has not produced yet. Quoting the tighter error would have reported a discovery at 4.1 standard errors where there was nothing.

A group-relative trainer needs every episode in a group to start from the same position, which `MatchupPool(..., hold_within_group=True)` provides. Rotating per episode makes the baseline measure which army pair was drawn, and the failure is silent: it fits the training set normally and simply does not transfer.

Store the full per-iteration history in every report, never just the headline metric. The first versions of two scripts here kept win rates alone and threw away the clip fractions, shifted fractions and advantage spreads the trainers had already computed, which made the next day's questions unanswerable from the record and forced reruns.

Cloning rolls carry seed noise of about a tenth of win rate on the hardest validation rungs, measured on 2026-08-06 by replicating the champion recipe across seeds, so a single-roll comparison on one rung resolves nothing there. Judge arms on many suites moving coherently, and gate any adoption on a multi-seed battery.

Comparisons at a fixed iteration budget are statements about that budget. The divergence trust region read as worse than the ratio clip at 30 iterations on the pool and better at 60, on runs whose first 30 iterations were bit-identical; where feasible, settle a comparison by continuing the same runs rather than by rerunning them, since determinism makes the continuation free of re-run variance.

Results belong in `../docs/archive/experiments/`, which is provenance. Conclusions belong in `../docs/rl/`. Decisions belong in `../docs/decisions/`, and only once the evidence supports one.

## Artifacts outlive the session, or they are lost

The scratchpad a session works in is presumed destroyed when the session ends, and the working convention follows from that. Report JSONs whose numbers are quoted anywhere in the documentation are vendored under `../docs/archive/experiments/files/` in a dated directory, along with recording manifests (a dataset re-records bit-identically from its manifest, so the hundreds of megabytes of episodes themselves stay out) and the few checkpoints other artifacts are calibrated against, currently the clone anchors and the curriculum result the pools and replay recordings reference. Everything else in scratch is disposable by definition, and if it turns out not to be, that is the signal it should have been vendored.

Scripts follow a lifecycle. They start here as measurements with a README row, which the documentation gate now enforces, so an unindexed script fails the build rather than fading from memory. A script that stops being a measurement and becomes something training depends on graduates into `python/fheroes2_agent/` with unit tests and a gate; `capture_replay.py` and `render_replay.py` sit at that boundary today and move the day something in training consumes a replay.

## Do not rebuild while an experiment is running

The verification gates relink `src/agent_worker/fheroes2_agent_worker`, and a running experiment spawns that binary once per episode. A gate run during a sweep killed one at seed 34 of 60 with a `FileNotFoundError`, because the file did not exist during the relink window.

Either wait, or copy the binary somewhere else and pass that path, since every script takes it as an argument for exactly this reason.

The copy must be private to the run, not shared. A pool build died silently when the "pinned" copy it was spawning per episode was itself overwritten by a fresh pin for a different experiment. One binary copy per concurrently running experiment, named for it.

