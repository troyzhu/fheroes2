# Experiment scripts

Runnable measurements that are too slow for a verification gate and too specific to be a unit test. Each answers one question, prints a table, and writes a JSON report.

These are separate from `../tests/` and the `../verify_*.sh` gates on purpose. A gate asserts that something still works and must stay fast enough to run on every change. An experiment measures how well something works, takes minutes, and produces a number that goes into `../docs/archive/experiments/`.

The scripts live here rather than being typed at a shell because the earlier round of this work ran from a temporary directory and lost every script when the directory was cleaned. The results survived only because they had been written into the archive, which left the conclusions standing on numbers nobody could reproduce.

| Script | Question | Runtime |
|---|---|---|
| `generalization.py` | Does group-relative training transfer to matchups it never trained on? | about 8 min at the default split |
| `critic_pre_fitting.py` | Does a value head fitted on teacher play improve reinforcement learning, or only look better on paper? | about 12 min over 20 seeds |
| `critic_on_pool.py` | The same question where a single matchup cannot answer it, because every run solves that one | about 12 min over 3 paired seeds |
| `advantage_floor.py` | Does flooring the advantage-normalization divisor stop a converged run destroying itself? | about 11 min over 20 seeds |
| `advantage_and_trust_region.py` | Do the advantage estimator and the trust region matter, on one matchup or on a pool? | about 15 min over 10 seeds |

## Conventions

Every script takes the worker binary as an argument rather than finding it, so a stale build is a visible choice.

Every script accepts `--seed` and reports across seeds where the answer could plausibly be noise. A single-seed comparison is reported as a single-seed comparison.

Errors are taken across the unit that will vary in use. For a pool of matchups that means across matchups rather than across episodes, because episodes inside one matchup share an army pair and pooling them understates the spread.

That choice is not cosmetic and it caught a near-miss. On the generalization run the same difference measured $\pm 0.012$ across seeds and $\pm 0.056$ across matchups, a factor of five, and only the second answers whether a result holds on matchups the generator has not produced yet. Quoting the tighter error would have reported a discovery at 4.1 standard errors where there was nothing.

A group-relative trainer needs every episode in a group to start from the same position, which `MatchupPool(..., hold_within_group=True)` provides. Rotating per episode makes the baseline measure which army pair was drawn, and the failure is silent: it fits the training set normally and simply does not transfer.

Results belong in `../docs/archive/experiments/`, which is provenance. Conclusions belong in `../docs/rl/`. Decisions belong in `../docs/decisions/`, and only once the evidence supports one.

## Do not rebuild while an experiment is running

The verification gates relink `src/agent_worker/fheroes2_agent_worker`, and a running experiment spawns that binary once per episode. A gate run during a sweep killed one at seed 34 of 60 with a `FileNotFoundError`, because the file did not exist during the relink window.

Either wait, or copy the binary somewhere else and pass that path, since every script takes it as an argument for exactly this reason.

