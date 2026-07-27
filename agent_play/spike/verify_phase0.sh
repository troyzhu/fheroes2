#!/bin/bash
# Run every Phase 0 experiment from spec Section 2.4 that can be automated, and print PASS/FAIL.
#
# Usage:  ./agent_play/spike/verify_phase0.sh
#
# Prerequisites: `make -C src/dist -j10` and `./agent_play/spike/build_spike.sh` have both run.
# Everything executes in a scratch directory so no config lands in $HOME.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPIKE="${REPO_ROOT}/agent_play/spike/smoke_battle"

if [ ! -x "${SPIKE}" ]; then
    echo "error: ${SPIKE} not built. Run ./agent_play/spike/build_spike.sh first." >&2
    exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT
cd "${WORKDIR}"

PASS=0
FAIL=0

report() {
    # report <name> <condition-result> <detail>
    if [ "$2" = "0" ]; then
        printf '  \033[32mPASS\033[0m  %-34s %s\n' "$1" "$3"
        PASS=$((PASS + 1))
    else
        printf '  \033[31mFAIL\033[0m  %-34s %s\n' "$1" "$3"
        FAIL=$((FAIL + 1))
    fi
}

run() { HOME="${WORKDIR}" "${SPIKE}" "$@" 2>/dev/null; }

echo "fheroes2 agent Phase 0 verification"
echo "  repo:   ${REPO_ROOT}"
echo "  commit: $(cd "${REPO_ROOT}" && git rev-parse --short HEAD) ($(cd "${REPO_ROOT}" && git branch --show-current))"
echo "  host:   $(sysctl -n hw.model 2>/dev/null || uname -m), $(sysctl -n hw.ncpu 2>/dev/null || echo '?') cores, $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 )) GB"
echo

# 1. Headless startup with no game assets present.
out="$(run --episodes 1)"
echo "${out}" | grep -q "SUMMARY episodes=1"
report "headless startup / no assets" "$?" "ran with no display, audio, or game data"

# 2. Determinism with the thread-local RNG reseeded.
d="$(run --episodes 10 | grep -c 'distinct_digests=1')"
[ "${d}" = "1" ]
report "determinism (RNG reseeded)" "$?" "10 runs -> 1 distinct digest"

# 3. Control: without the reseed the engine must be nondeterministic.
n="$(run --episodes 10 --no-global-seed | sed -n 's/.*distinct_digests=\([0-9]*\).*/\1/p')"
[ "${n}" -gt 1 ] 2>/dev/null
report "control: no reseed diverges" "$?" "10 runs -> ${n} distinct digests"

# 4. Sequential reuse: one process, many fresh arenas, single arena invariant respected.
s="$(run --episodes 500 --quiet | sed -n 's/.*distinct_digests=\([0-9]*\).*/\1/p')"
[ "${s}" = "1" ]
report "sequential reuse (500 arenas)" "$?" "no crash/assert, 1 distinct digest"

# 5. Cross-process determinism.
a="$(run --episodes 1 | head -1 | sed -n 's/.*digest=\([0-9a-f]*\).*/\1/p')"
b="$(run --episodes 1 | head -1 | sed -n 's/.*digest=\([0-9a-f]*\).*/\1/p')"
[ -n "${a}" ] && [ "${a}" = "${b}" ]
report "cross-process determinism" "$?" "two fresh processes -> ${a:0:12}…"

# 6. Combat seed must react to army composition; map seed must not.
m1="$(run --episodes 1 | sed -n 's/.*map_seed=\([0-9]*\).*/\1/p')"
c1="$(run --episodes 1 | sed -n 's/.*combat_seed=\([0-9]*\).*/\1/p')"
m2="$(run --episodes 1 --monster-a 2 --count-a 20 | sed -n 's/.*map_seed=\([0-9]*\).*/\1/p')"
c2="$(run --episodes 1 --monster-a 2 --count-a 20 | sed -n 's/.*combat_seed=\([0-9]*\).*/\1/p')"
[ "${m1}" = "${m2}" ] && [ "${c1}" != "${c2}" ]
report "seed visibility" "$?" "map seed stable (${m1}), combat seed varies"

# 7. Ranged creature path resolves.
run --episodes 1 --monster-a 2 --count-a 20 --monster-b 1 --count-b 60 | grep -q "winner=attacker"
report "ranged (Archer) path" "$?" "Archer 20 beats Peasant 60"

echo
echo "  ${PASS} passed, ${FAIL} failed"
echo
echo "  Not covered here (must be done manually -- see docs/agent/local_source_audit.md):"
echo "    - Debug build assertion run (FHEROES2_WITH_DEBUG)"
echo "    - sanitizers (ASan/UBSan)"
echo "    - CMake normal-game regression"
echo "    - real Battle Only UI run"

[ "${FAIL}" -eq 0 ]
