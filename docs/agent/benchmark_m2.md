# fheroes2 agent — Mac mini M2 benchmark

> Deliverable required by spec §19.4 and START_HERE §6 step 2. **Initial version: Mode A only**
> (spec §19.1-A — pure engine baseline, built-in AI vs built-in AI, no per-decision JSON, one
> worker). Modes B (serialization overhead) and C (Python end-to-end) require the protocol worker
> and Python client from Milestones 4–5; extend this document when they exist.

## Environment

| Field | Value |
|---|---|
| Date | 2026-07-27 |
| Machine | `Mac14,3` — Mac mini, Apple M2, 8 cores (4P + 4E), 16 GB |
| macOS | 26.5.1 (25F80) |
| Compiler | Apple clang 21.0.0 (clang-2100.1.1.101) |
| SDL | 2.32.10 (Homebrew), SDL2_mixer 2.8.2 |
| Commit | `b16e6f698` (branch `agent-env`) plus the spike portability fix |
| Engine build | `make -C src/dist -j8` — `-O3`, **asserts live** (see note below) |
| Spike build | `build_spike.sh` — spike TU at `-O2`, relinked against the 220 Release game objects |

**Assert note.** The `src/dist` Makefile never defines `-DNDEBUG`, so the `-O3` "Release" build
keeps `assert()` active (verified: `battle_arena.o`, `battle_action.o`, `battle_board.o` all
reference `__assert_rtn`). Every number below — and every Phase 0 run on both machines — therefore
already includes assert overhead. A CMake `Release` build (which does add `-DNDEBUG`) would be
slightly faster and is not what is measured here.

## Commands

```bash
make -C src/dist clean && make -C src/dist -j"$(sysctl -n hw.ncpu)"
./agent_play/spike/build_spike.sh
./agent_play/spike/verify_phase0.sh     # 7/7 PASS before benchmarking
./agent_play/spike/bench_m2.sh          # this report's numbers, machine otherwise idle
```

## Workloads and scenario identity

Single-stack armies only — the most the Phase 0 spike can express. All use world seed `20260726`
→ map seed `2227197244`, tile 1, grass.

| Workload | Composition | Rounds | Combat seed | Terminal digest |
|---|---|---|---|---|
| `tiny_melee` | Peasant 50 vs Peasant 50 | 3 | 1356111745 | `2cfd42cb104aa5e7` |
| `ranged_fast` | Archer 20 vs Peasant 60 | 1 | 1381489788 | `21e34dfd5fbe595e` |
| `ranger_duel` | Ranger 100 vs Ranger 100 | 2 | 1274517553 | `8264f5a23e72796b` |
| `melee_large` | Peasant 1000 vs Peasant 1000 | 3 | 3437871903 | `8f02852176509095` |

Two structural caveats:

- **Every expressible battle is 1–3 rounds.** Slow melee units spend up to two rounds walking,
  then mass damage resolves combat almost instantly (1000 peasants deal exactly 1000 damage into
  1000 × 1 HP — the mirror matchup is a deterministic first-strike one-shot; symmetric armies
  always favor the attacker). The spec's "longer balanced battle" workload (§19.2) needs
  multi-stack scenarios and is blocked on the Milestone 1 runner.
- Stack size is nearly free: `melee_large` (2000 creatures) costs about the same per episode as
  `tiny_melee` (100 creatures). Per-round fixed cost dominates.

## Results

### Throughput and memory (10 000 episodes/rep × 3 reps; `melee_large` 2000/rep)

| Workload | Median episodes/s | CPU | Peak RSS |
|---|---|---|---|
| `tiny_melee` | **4 566** | ~100 % one core | 12 MB |
| `ranged_fast` | **12 195** | ~99 % | 12 MB |
| `ranger_duel` | **11 111** | ~99 % | 12 MB |
| `melee_large` | **4 444** | ~98 % | 12 MB |

Rep-to-rep spread was ≤ 1.5 % everywhere. For comparison the M3 MacBook measured ~5 000 eps/s on
`tiny_melee` — the M2 is ~9 % slower, i.e. the two machines are the same class and none of the
Phase 0 conclusions shift.

### Process startup

`--episodes 1` full process lifetime (spawn, world generation, one battle, teardown), 20 runs:
**median 10 ms**, p95 10 ms. Worker restarts are effectively free at this stage; per-process
episode reuse remains preferable but is not load-bearing for throughput.

### Multi-process scaling (`tiny_melee`, 5 000 episodes per process)

| Concurrent workers | Aggregate eps/s | Per-worker eps/s | Efficiency vs 1 worker |
|---|---|---|---|
| 1 | 3 976 | 3 976 | 100 % |
| 2 | 7 976 | 3 988 | 100 % |
| 4 | 14 551 | 3 638 | 91 % |
| 8 | 16 958 | 2 120 | 53 % |

(Aggregate figures include ~0.1–0.2 s of harness spawn/wait overhead, so absolute values slightly
understate; the relative scaling is the signal.) Scaling is linear across the 4 performance cores
and collapses on the efficiency cores: going 4 → 8 workers buys only +17 % aggregate.

**Recommendation: default worker count 4.** Use 8 only for bulk throughput where +17 % matters
more than per-episode latency; use 1–2 for latency-sensitive or interactive runs. Memory is a
non-issue: 8 workers × 12 MB ≈ 100 MB against 16 GB.

## Determinism cross-checks recorded during this session

- Map seed `2227197244` and `tiny_melee` digest `2cfd42cb104aa5e7` **reproduce exactly** on the
  M2 — identical to the M3 baseline and across three prior working trees.
- The digest is also **identical between `-O0 -g -DWITH_DEBUG` and `-O3` builds** (7/7
  verification passed on both), so battle resolution is stable across optimization levels — no
  floating-point-ordering or UB-dependent behavior in these paths.
- Two fresh processes produce identical digests; 500-arena sequential reuse yields one distinct
  digest.

## Not measured yet

- Median/p95 **per-episode** latency (spike has no per-episode timer; add timing to the real
  runner rather than instrumenting the throwaway spike).
- Decisions per second, external-decision round-trip latency, stdout bytes per decision — all
  need the Milestone 2+ decision hook and Milestone 4 protocol.
- Modes B and C, including 1/2/4-worker Python end-to-end.
- Multi-stack and genuinely long battles (need the Milestone 1 scenario runner).
