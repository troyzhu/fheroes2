# Experiment scripts

Runnable measurements that are too slow for a verification gate and too specific to be a unit test. Each answers one question, prints a table, and writes a JSON report.

These are separate from `../tests/` and the `../verify_*.sh` gates on purpose. A gate asserts that something still works and must stay fast enough to run on every change. An experiment measures how well something works, takes minutes, and produces a number that goes into `../docs/archive/experiments/`.

The scripts live here rather than being typed at a shell because the earlier round of this work ran from a temporary directory and lost every script when the directory was cleaned. The results survived only because they had been written into the archive, which left the conclusions standing on numbers nobody could reproduce.

| Script | Question | Runtime |
|---|---|---|
| `generalization.py` | Does group-relative training transfer to matchups it never trained on? | about 15 min at the default split |
| `critic_pre_fitting.py` | Does a value head fitted on teacher play improve reinforcement learning, or only look better on paper? | about 12 min over 20 seeds |

## Conventions

Every script takes the worker binary as an argument rather than finding it, so a stale build is a visible choice.

Every script accepts `--seed` and reports across seeds where the answer could plausibly be noise. A single-seed comparison is reported as a single-seed comparison.

Errors are taken across the unit that will vary in use. For a pool of matchups that means across matchups rather than across episodes, because episodes inside one matchup share an army pair and pooling them understates the spread.

Results belong in `../docs/archive/experiments/`, which is provenance. Conclusions belong in `../docs/rl/`. Decisions belong in `../docs/decisions/`, and only once the evidence supports one.
