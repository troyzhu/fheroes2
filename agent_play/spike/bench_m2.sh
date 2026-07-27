#!/bin/bash
# Mode A benchmark (spec §19.1-A: pure engine baseline, built-in AI vs built-in AI,
# no per-decision JSON, single worker) for docs/agent/benchmark_m2.md.
#
# Modes B and C require the protocol layer / Python client (Milestones 4-5) and are
# therefore NOT covered here. Workloads are limited to what the Phase 0 spike can
# express: single-stack armies. Multi-stack workloads (spec §19.2) arrive with the
# real runner.
#
# Usage:  ./agent_play/spike/bench_m2.sh
# Prerequisites: Release build (`make -C src/dist -j…`) and `build_spike.sh` done.
# Run on an otherwise idle machine; results go to stdout.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPIKE="${REPO_ROOT}/agent_play/spike/smoke_battle"

EPISODES="${EPISODES:-10000}"        # per timed repetition
HEAVY_EPISODES="${HEAVY_EPISODES:-2000}"
REPS="${REPS:-3}"
SCALE_EPISODES="${SCALE_EPISODES:-5000}"   # per process in the scaling probe

if [ ! -x "${SPIKE}" ]; then
    echo "error: ${SPIKE} not built" >&2
    exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT
cd "${WORKDIR}"
export HOME="${WORKDIR}"    # keep engine config files out of the real home

# ---- helpers ---------------------------------------------------------------

# timed <output-prefix> <spike args...>
# Writes <prefix>.time with `/usr/bin/time -l` output; echoes "wall user sys maxrss_bytes".
timed() {
    local prefix="$1"; shift
    /usr/bin/time -l "${SPIKE}" "$@" > /dev/null 2> "${prefix}.time"
    awk '
        /real/  { wall=$1; user=$3; sys=$5 }
        /maximum resident set size/ { rss=$1 }
        END { printf "%s %s %s %s\n", wall, user, sys, rss }
    ' "${prefix}.time"
}

median3() { printf '%s\n%s\n%s\n' "$1" "$2" "$3" | sort -n | sed -n 2p; }

echo "== bench_m2: environment =="
echo "host:     $(sysctl -n hw.model) ($(sysctl -n machdep.cpu.brand_string)), $(sysctl -n hw.ncpu) cores, $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
echo "macos:    $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
echo "compiler: $(c++ --version | head -1)"
echo "commit:   $(cd "${REPO_ROOT}" && git rev-parse --short HEAD) ($(cd "${REPO_ROOT}" && git branch --show-current))"
echo "episodes: ${EPISODES}/rep (heavy: ${HEAVY_EPISODES}), reps: ${REPS}"
echo

# ---- workload characterization + throughput --------------------------------

WORKLOADS=(
    "tiny_melee|--monster-a 1 --count-a 50 --monster-b 1 --count-b 50"
    "ranged_fast|--monster-a 2 --count-a 20 --monster-b 1 --count-b 60"
    "ranger_duel|--monster-a 3 --count-a 100 --monster-b 3 --count-b 100"
    "melee_large|--monster-a 1 --count-a 1000 --monster-b 1 --count-b 1000"
)

echo "== workloads (characterization: 1 episode each) =="
for w in "${WORKLOADS[@]}"; do
    name="${w%%|*}"; args="${w#*|}"
    # shellcheck disable=SC2086
    line="$("${SPIKE}" --episodes 1 ${args} 2>/dev/null | head -1)"
    echo "${name}: ${line}"
done
echo

echo "== throughput (episodes/s, ${REPS} reps, median) =="
for w in "${WORKLOADS[@]}"; do
    name="${w%%|*}"; args="${w#*|}"
    eps="${EPISODES}"
    case "${name}" in melee_large) eps="${HEAVY_EPISODES}";; esac

    rates=()
    rss=0
    for r in $(seq 1 "${REPS}"); do
        # shellcheck disable=SC2086
        read -r wall user sys maxrss <<< "$(timed "${name}.${r}" --episodes "${eps}" --quiet ${args})"
        rate="$(awk -v e="${eps}" -v t="${wall}" 'BEGIN { printf "%.0f", e / t }')"
        cpu="$(awk -v u="${user}" -v s="${sys}" -v t="${wall}" 'BEGIN { printf "%.0f", (u + s) * 100 / t }')"
        rates+=( "${rate}" )
        rss="${maxrss}"
        echo "  ${name} rep${r}: wall=${wall}s eps/s=${rate} cpu=${cpu}% maxrss=$(( maxrss / 1048576 ))MB"
    done
    echo "  ${name} MEDIAN eps/s: $(median3 "${rates[@]}")"
done
echo

# ---- worker startup --------------------------------------------------------

echo "== process startup + 1 episode (20 runs, seconds) =="
walls=()
for i in $(seq 1 20); do
    read -r wall _ _ _ <<< "$(timed "startup.${i}" --episodes 1 --quiet)"
    walls+=( "${wall}" )
done
printf '%s\n' "${walls[@]}" | sort -n | awk '
    { v[NR] = $1 }
    END {
        printf "  median=%.3fs p95=%.3fs min=%.3fs max=%.3fs\n",
               v[int(NR * 0.5)], v[int(NR * 0.95)], v[1], v[NR]
    }'
echo

# ---- multi-process scaling -------------------------------------------------

echo "== multi-process scaling (tiny_melee, ${SCALE_EPISODES} eps/process) =="
for n in 1 2 4 8; do
    start="$(python3 -c 'import time; print(time.time())')"
    pids=()
    for i in $(seq 1 "${n}"); do
        "${SPIKE}" --episodes "${SCALE_EPISODES}" --quiet --world-seed "$(( 20260726 + i ))" > /dev/null 2>&1 &
        pids+=( "$!" )
    done
    wait "${pids[@]}"
    end="$(python3 -c 'import time; print(time.time())')"
    awk -v n="${n}" -v e="${SCALE_EPISODES}" -v s="${start}" -v t="${end}" \
        'BEGIN { w = t - s; printf "  workers=%d wall=%.2fs aggregate_eps_per_s=%.0f per_worker=%.0f\n", n, w, n * e / w, e / w }'
done
