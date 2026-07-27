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

    echo
    sed -n 's/^fixture=/  /p' "${WORKDIR}/proc_a.log"
else
    report "worker binary present" "1" "${WORKER} missing"
fi

echo
echo "  ${PASS} passed, ${FAIL} failed"
[ "${FAIL}" -eq 0 ]
