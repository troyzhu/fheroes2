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
| `awr_distill.py` | Advantage-weighted distillation against its unweighted twin, one improvement step that never queries an unseen action | about 8 min for both arms |
| `planes_ablation.py` | Do spatial planes earn their place, with capacity controlled? Entity, planes, and a width-matched control arm from one seed on a planes-recorded corpus | about 15 min for three arms |
| `soft_distill.py` | Does distilling search's whole per-candidate measurement beat distilling its argmax, at identical data? Prior-anchored soft targets against a hard-label twin | about 8 min for both arms |
| `real_map_fights.py` | Real opening fights harvested from the shipped maps by dump, nearest stack per hero, stratified across maps, evaluated with the built-in AI column attached | about 1 min end to end |
| `fidelity_report.py` | Fidelity beyond exact match: top-1/3/5 accuracy, the probability given to the teacher's move, and the policy's entropy, on a corpus holdout | about 30 s per checkpoint |
| `training_dashboard.py` | Live training health: heartbeat sparklines per run, per-term gradient norms, the advantage floor drawn where the collapse lives, pipeline log tails, self-refreshing HTML | continuous; --once for a snapshot |
| `rollout_value.py` | The owner's proposal: a value trained on search's own branch rollouts rather than played episodes; fit, search agreement, and greedy play in one run | about 12 min |
| `sampling_policies.py` | Deployment sampling schemes, full against greedy against nucleus against entropy-adaptive nucleus, the same checkpoint under each | about 8 min |
| `selfplay_probe.py` | The first self-play training probe: PPO against a checkpoint pool with the AI as anchor, judged by pool duels before and after plus the AI columns | about 30 min |
| `builtin_ai_baseline.py` | The engine's own AI on every validation suite: the baseline a player recognizes, and the bar the policies must clear | about 70 s for all suites |
| `search_value.py` | Can a value network replace rollouts at search leaves, and is it calibrated where it would be used? | about 10 min including the fit |
| `symmetry_gap.py` | Does the same army win as often from the other chair, and how much of any gap is the game rather than the policy? | about 6 min over 10 matchups |
| `evasion_stalemate.py` | Can a policy stall a battle by pure evasion, and what does a stall pay? The owner-predicted exploit, demonstrated and priced | seconds |
| `tabula_rasa_pilot.py` | Could search bootstrap a policy from random initialization, with no demonstrations at all? Rounds of search play, distil, evaluate on a fixed matchup set | about 20 min for three rounds |
| `convergence_report.py` | Had a training run settled when it stopped? Per-metric trend verdicts over a heartbeat's trailing third: converged, trending, or oscillating | seconds per heartbeat |
| `trust_region_rematch.py` | DPPO's divergence gates against the ratio clip inside the actor-critic trainer: exact total variation and the paper's binary lower bound, matched to the self-play control trio | about 50 min per arm over 3 seeds |
| `rounds_probe.py` | Round-count histograms and stalemate incidence per checkpoint on the training and held-out matchup slices, from exact terminal records | about 2 min per checkpoint |

## Conventions

Every script takes the worker binary as an argument rather than finding it, so a stale build is a visible choice.

Every script accepts `--seed` and reports across seeds where the answer could plausibly be noise. A single-seed comparison is reported as a single-seed comparison.

Errors are taken across the unit that will vary in use. For a pool of matchups that means across matchups rather than across episodes, because episodes inside one matchup share an army pair and pooling them understates the spread.

That choice is not cosmetic and it caught a near-miss. On the generalization run the same difference measured $\pm 0.012$ across seeds and $\pm 0.056$ across matchups, a factor of five, and only the second answers whether a result holds on matchups the generator has not produced yet. Quoting the tighter error would have reported a discovery at 4.1 standard errors where there was nothing.

A group-relative trainer needs every episode in a group to start from the same position, which `MatchupPool(..., hold_within_group=True)` provides. Rotating per episode makes the baseline measure which army pair was drawn, and the failure is silent: it fits the training set normally and simply does not transfer.

Store the full per-iteration history in every report, never just the headline metric. The first versions of two scripts here kept win rates alone and threw away the clip fractions, shifted fractions and advantage spreads the trainers had already computed, which made the next day's questions unanswerable from the record and forced reruns.

Cloning rolls carry seed noise of about a tenth of win rate on the hardest validation rungs, measured on 2026-08-06 by replicating the champion recipe across seeds, so a single-roll comparison on one rung resolves nothing there. Judge arms on many suites moving coherently, and gate any adoption on a multi-seed battery.

Graded suites are never collapsed to one number: the Thunk ladder reports its four rungs, because a mean of 1.00/1.00/0.96/0.67 hides that the last rung is the only hard one. Win rate alone is also not the report, per the owner's 2026-08-07 reporting requirements: every battery suite carries four numbers beside the rate, win quality (wq, engine strength kept when winning), loss quality (lq, fraction of the enemy destroyed when losing, the counterpart the owner asked for because a win-conditioned number alone is survivorship-biased), the unconditional strength margin (mg, own kept minus enemy kept over every episode, which no conditioning can flatter), the reward the run optimized (rw, and the battery stamps which margin it measured under: `reward_margin` in the report, defaulting to the two-sided owner objective since 2026-08-08; reports without that key predate the fix and their rw is the hit-point margin whatever the checkpoint trained on), and mean episode length.

The reward column earns a caution the owner's question surfaced: the terminal reward prices the agent's own surviving strength and not the enemy's remaining strength, deliberately, because a decided battle almost always ends with the loser wiped out, so an enemy-side term collapses into the win bit on wins and vanishes on losses; the enemy's fate enters the reward only through winning at all. That is exactly the blind spot lq and mg cover, damage dealt in losses and the unconditional two-sided margin, which is why the decomposition stays primary and the reward rides along as the compact summary the trainer actually optimized. Optimization targets make suspect headline metrics, so rate stays the adjudicator against the built-in AI.

Agreement, wherever quoted alone, is top-1 exact match against the teacher's index. The diagnostic form is `fidelity_report.py`: top-1/3/5, the mean and median probability the policy gives the teacher's move, and the policy's mean entropy, which together separate confidently-wrong from undecided and near-miss from nowhere-close.

Comparisons at a fixed iteration budget are statements about that budget. The divergence trust region read as worse than the ratio clip at 30 iterations on the pool and better at 60, on runs whose first 30 iterations were bit-identical; where feasible, settle a comparison by continuing the same runs rather than by rerunning them, since determinism makes the continuation free of re-run variance.

Every training verdict states whether the run had settled, read from its heartbeat by `convergence_report.py` rather than eyeballed, because a budget comparison between a settled run and one still trending is a comparison of different things. Two readings need care. Training-pool convergence says nothing about generalization: the 2026-08-07 fixed-AI control converged on its training win rate while its held-out ladder had collapsed, so a settled curve licenses "more iterations would not have changed this", never "this is good". And under self-play the opponent mixture is nonstationary, so the training win rate is not a convergence signal at all there; the loss terms and entropy are what can settle.

### The metric ledger

What a training claim must be able to cite, accumulated from the owner's reporting requirements. During training, from the heartbeat and `training_dashboard.py`: the total loss and its policy, value and entropy terms separately; the per-term gradient norms measured before the weighted sum and split per module, because a converged total can hide one term still moving; the trained reward decomposed per iteration over won and lost episodes (`reward_on_wins`, `reward_on_losses`), the live counterparts of wq and lq, since the rate alone cannot see quality on either side; the supervised runs' per-epoch holdout agreement (reinforcement runs carry no per-iteration probe; their before-and-after probes live in each run's report); normalized entropy $H/\log K$ as the sharpness diagnostic; and the settlement verdict above, which reads the reward first, the rate last, and every gradient-norm series.

For a final performance claim: win rate with the built-in AI column beside it, the four quality columns (wq, lq, mg, rw) and episode length per suite, graded suites reported per rung, the real-map suite, the symmetry gauge with the engine's own gap as reference, and for imitation checkpoints the fidelity trio (top-1/3/5, teacher probability, entropy) with the reliability table and its deterministic-teacher caveat. Self-play adds per-opponent duel tables at an episode count that clears the measured $\pm 0.06$ noise band. The one owner-requested read still unbuilt is outcome-grounded calibration, predicted win probability against realized outcomes.

Results belong in `../docs/archive/experiments/`, which is provenance. Conclusions belong in `../docs/rl/`. Decisions belong in `../docs/decisions/`, and only once the evidence supports one.

## Artifacts outlive the session, or they are lost

The scratchpad a session works in is presumed destroyed when the session ends, and the working convention follows from that. Report JSONs whose numbers are quoted anywhere in the documentation are vendored under `../docs/archive/experiments/files/` in a dated directory, along with recording manifests (a dataset re-records bit-identically from its manifest, so the hundreds of megabytes of episodes themselves stay out) and the few checkpoints other artifacts are calibrated against, currently the clone anchors and the curriculum result the pools and replay recordings reference. Everything else in scratch is disposable by definition, and if it turns out not to be, that is the signal it should have been vendored.

Scripts follow a lifecycle. They start here as measurements with a README row, which the documentation gate now enforces, so an unindexed script fails the build rather than fading from memory. A script that stops being a measurement and becomes something training depends on graduates into `python/fheroes2_agent/` with unit tests and a gate; `capture_replay.py` and `render_replay.py` sit at that boundary today and move the day something in training consumes a replay.

## Do not rebuild while an experiment is running

The verification gates relink `src/agent_worker/fheroes2_agent_worker`, and a running experiment spawns that binary once per episode. A gate run during a sweep killed one at seed 34 of 60 with a `FileNotFoundError`, because the file did not exist during the relink window.

Either wait, or copy the binary somewhere else and pass that path, since every script takes it as an argument for exactly this reason.

The copy must be private to the run, not shared. A pool build died silently when the "pinned" copy it was spawning per episode was itself overwritten by a fresh pin for a different experiment. One binary copy per concurrently running experiment, named for it.


Load checkpoints only through `load_policy`, never a bare `BattlePolicy()` plus `load_state_dict`: the state dict is self-describing (ability table, planes conv, widths), and the direct pattern crashes on any architecture it predates, which killed a full overnight collection round on 2026-08-07. For the same reason, smoke-test the exact invocation path that will run, since a library-path smoke proved nothing about the CLI path that night.

Pin worker binaries per feature, not per day. A pinned copy that predates a feature silently fails any experiment whose evaluation path requests that feature; the first planes replication chains died on a pre-planes worker rejecting `--planes`.

A metric must name its own objective, and the report must stamp it. Until 2026-08-08 `measure` built its environment without passing `reward_margin`, so every battery `rw` column was the hit-point margin no matter what the checkpoint had trained on, while this file claimed the column followed the run's configuration. The mislabel survived a day of verdicts because the two rewards correlate; it was caught only when the built-in AI's own quality columns were measured for the first time and the two sides of a comparison had to be put on one scale. Nothing that reads a configurable objective may be reported without the configuration beside it.

No training verdict from single-suite reads: every reinforcement or self-play conclusion runs the full battery (quality columns and per-rung ladders included) and the symmetry gauge before any claim, with the built-in AI columns beside it. Two long-budget headline claims on 2026-08-07 died on first contact with the full apparatus after being drawn from one held-out number and eight-episode duels; the apparatus exists to be used, not to be available.
