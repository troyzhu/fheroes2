#!/bin/bash
# Milestone 1 verification (spec §20, "deterministic runner foundation").
#
# Checks, in order:
#   1. the agent unit tests pass (seed helper goldens, SHA-256 vectors, scenario validation);
#   2. the worker builds against the current Makefile game objects;
#   3. every Milestone 1 fixture reproduces one identical canonical digest across ten runs in
#      one process (the milestone's exit criterion);
#   4. two fresh worker processes produce byte-identical output (cross-process determinism).
#
# Prerequisite: `make -C src/dist -j…` has been run (Release or Debug both work).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER="${REPO_ROOT}/src/agent_worker/fheroes2_agent_worker"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

PASS=0
FAIL=0

report() {
    if [ "$2" = "0" ]; then
        printf '  \033[32mPASS\033[0m  %-36s %s\n' "$1" "$3"
        PASS=$((PASS + 1))
    else
        printf '  \033[31mFAIL\033[0m  %-36s %s\n' "$1" "$3"
        FAIL=$((FAIL + 1))
    fi
}

echo "fheroes2 agent Milestone 1 verification"
echo "  repo:   ${REPO_ROOT}"
echo "  commit: $(cd "${REPO_ROOT}" && git rev-parse --short HEAD) ($(cd "${REPO_ROOT}" && git branch --show-current))"
echo "  host:   $(sysctl -n hw.model 2>/dev/null || uname -m)"
echo

"${REPO_ROOT}/agent_play/tests/build_and_run_tests.sh" > "${WORKDIR}/tests.log" 2>&1
report "agent unit tests" "$?" "$(grep -c '^  PASS' "${WORKDIR}/tests.log") checks (see tests.log on failure)"

"${REPO_ROOT}/src/agent_worker/build_worker.sh" > "${WORKDIR}/build.log" 2>&1
report "worker build" "$?" "relink against game objects"

if [ -x "${WORKER}" ]; then
    HOME="${WORKDIR}" "${WORKER}" --runs 10 > "${WORKDIR}/ten.log" 2> /dev/null
    rc=$?
    verdict="$(tail -1 "${WORKDIR}/ten.log")"
    report "ten-run determinism (M1 exit)" "${rc}" "${verdict}"

    HOME="${WORKDIR}" "${WORKER}" --runs 1 > "${WORKDIR}/proc_a.log" 2> /dev/null
    HOME="${WORKDIR}" "${WORKER}" --runs 1 > "${WORKDIR}/proc_b.log" 2> /dev/null
    cmp -s "${WORKDIR}/proc_a.log" "${WORKDIR}/proc_b.log"
    report "cross-process determinism" "$?" "two fresh processes, byte-identical output"

    # Terminal-state invariants (spec §18.2 item 7, "state-core canonicalization").
    #
    # Every other check in this file proves the extracted state is STABLE: identical across
    # runs, processes, machines and optimization levels. None of them proves it is CORRECT.
    # An extraction that read counts before deaths were applied, or that kept a dead stack,
    # would be perfectly deterministic and would pass all of them. These assert properties
    # that must hold whatever the battle was, so they need no golden value and no oracle;
    # a golden value would only lock in whatever the current implementation happens to do.
    invariants="$(awk '
        /^fixture=/ {
            for ( i = 1; i <= NF; ++i ) { split($i, kv, "="); v[kv[1]] = kv[2] }
            n = v["fixture"]

            # A living stack holds at least one creature, and a living creature at least one
            # hit point. Catches an inverted or unit-confused aggregation.
            if ( v["a_creatures"] < v["a_stacks"] || v["a_hp"] < v["a_creatures"] ) bad[n] = bad[n] " attacker-ordering"
            if ( v["d_creatures"] < v["d_stacks"] || v["d_hp"] < v["d_creatures"] ) bad[n] = bad[n] " defender-ordering"

            # A side with no stacks must be zero everywhere. This is the check that catches
            # reading the force before deaths are applied.
            if ( v["a_stacks"] == 0 && (v["a_creatures"] != 0 || v["a_hp"] != 0) ) bad[n] = bad[n] " attacker-wiped-nonzero"
            if ( v["d_stacks"] == 0 && (v["d_creatures"] != 0 || v["d_hp"] != 0) ) bad[n] = bad[n] " defender-wiped-nonzero"

            # The reported termination and the surviving forces must agree.
            if ( v["termination"] == "victory" && !(v["a_stacks"] > 0 && v["d_stacks"] == 0) ) bad[n] = bad[n] " victory-disagrees"
            if ( v["termination"] == "defeat"  && !(v["d_stacks"] > 0 && v["a_stacks"] == 0) ) bad[n] = bad[n] " defeat-disagrees"

            # A decided battle cannot leave both sides standing.
            if ( (v["termination"] == "victory" || v["termination"] == "defeat") && v["a_stacks"] > 0 && v["d_stacks"] > 0 ) bad[n] = bad[n] " both-sides-alive"

            if ( v["rounds"] < 1 ) bad[n] = bad[n] " rounds<1"
            if ( v["decisions"] < 1 ) bad[n] = bad[n] " decisions<1"
            ++seen
        }
        END {
            for ( f in bad ) printf "%s:%s ", f, bad[f]
            if ( seen == 0 ) printf "no-fixture-lines "
        }' "${WORKDIR}/proc_a.log")"
    [ -z "${invariants}" ]
    report "terminal-state invariants" "$?" "${invariants:-$(grep -c '^fixture=' "${WORKDIR}/proc_a.log") fixtures, 8 properties each}"

    echo
    sed -n 's/^fixture=/  /p' "${WORKDIR}/proc_a.log"
else
    report "worker binary present" "1" "${WORKER} missing"
fi

echo
echo "  ${PASS} passed, ${FAIL} failed"
[ "${FAIL}" -eq 0 ]
