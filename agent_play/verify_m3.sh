#!/bin/bash
# Milestone 3 verification (spec §20, "simple_v1 legal actions").
#
# Exit criteria checked here:
#   1. agent unit tests pass (indexing math, capability audit rules, plus all earlier suites);
#   2. the worker builds;
#   3. behavior is unchanged with enumeration running at every decision: golden state digests
#      still hold and the determinism verdict covers state + decision streams;
#   4. every supported fixture has valid candidates at every decision (min_candidates >= 1)
#      and built-in teacher coverage is 100% (spec §10.6 acceptance);
#   5. coverage output is deterministic across fresh processes;
#   6. the machine-generated capability audit is written and sane.
#
# Prerequisite: `make -C src/dist -j…` has been run.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER="${REPO_ROOT}/src/agent_worker/fheroes2_agent_worker"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

PASS=0
FAIL=0

report() {
    if [ "$2" = "0" ]; then
        printf '  \033[32mPASS\033[0m  %-44s %s\n' "$1" "$3"
        PASS=$((PASS + 1))
    else
        printf '  \033[31mFAIL\033[0m  %-44s %s\n' "$1" "$3"
        FAIL=$((FAIL + 1))
    fi
}

echo "fheroes2 agent Milestone 3 verification"
echo "  repo:   ${REPO_ROOT}"
echo "  commit: $(cd "${REPO_ROOT}" && git rev-parse --short HEAD) ($(cd "${REPO_ROOT}" && git branch --show-current))"
echo

"${REPO_ROOT}/agent_play/tests/build_and_run_tests.sh" > "${WORKDIR}/tests.log" 2>&1
report "agent unit tests" "$?" "$(grep -c '^  PASS' "${WORKDIR}/tests.log") checks"

"${REPO_ROOT}/src/agent_worker/build_worker.sh" > "${WORKDIR}/build.log" 2>&1
report "worker build" "$?" "relink against game objects"

if [ ! -x "${WORKER}" ]; then
    report "worker binary present" "1" "${WORKER} missing"
    echo
    echo "  ${PASS} passed, ${FAIL} failed"
    exit 1
fi

GOLDEN_tiny="0985937ec02c27bc605a70d9b26e57ec8a9e921408db7a1043532000cbaff463"
GOLDEN_three="51a390dbef464e57a69a8cd3cc90bbf9424376e4edd23454a14dee538413b3a3"
GOLDEN_five="b893b0589479dbb3a069ac9a4bcc1c91eb300bbd7cc3799cbdd5e4678ab04ada"
GOLDEN_ranged="29fae534c93145af56bef0845b46476eef129116de9ada912c6d86a1e07a0e05"
GOLDEN_longer="ec8268417c2f75312cab33defb3255eeb7c18e0b765074f789b65da57905230e"

HOME="${WORKDIR}" "${WORKER}" --runs 10 --audit-coverage > "${WORKDIR}/ten.log" 2> /dev/null
rc=$?
report "ten-run determinism + coverage verdict" "${rc}" "$(tail -1 "${WORKDIR}/ten.log")"

golden_ok=0
for pair in "m1_tiny_melee:${GOLDEN_tiny}" "m1_three_stack:${GOLDEN_three}" "m1_five_stack:${GOLDEN_five}" "m1_ranged_heavy:${GOLDEN_ranged}" \
            "m1_longer_balanced:${GOLDEN_longer}"; do
    fixture="${pair%%:*}"
    golden="${pair#*:}"
    grep -q "fixture=${fixture} .* digest=${golden}$" "${WORKDIR}/ten.log" || golden_ok=1
done
report "state digests equal Milestone 1 goldens" "${golden_ok}" "enumeration at every decision perturbs nothing"

cov_ok=0
cov_lines="$(grep -c '^COVERAGE' "${WORKDIR}/ten.log")"
[ "${cov_lines}" = "5" ] || cov_ok=1
grep '^COVERAGE' "${WORKDIR}/ten.log" | grep -vq "coverage=100.0%" && cov_ok=1
report "teacher coverage 100% on all fixtures" "${cov_ok}" "${cov_lines} fixtures x 10 runs"

cand_ok=0
grep '^COVERAGE' "${WORKDIR}/ten.log" | awk -F'min_candidates=' '{ if ($2 + 0 < 1) exit 1 }' || cand_ok=1
report "every decision had at least one candidate" "${cand_ok}" "min_candidates >= 1"

HOME="${WORKDIR}" "${WORKER}" --runs 1 --audit-coverage > "${WORKDIR}/proc_a.log" 2> /dev/null
HOME="${WORKDIR}" "${WORKER}" --runs 1 --audit-coverage > "${WORKDIR}/proc_b.log" 2> /dev/null
cmp -s "${WORKDIR}/proc_a.log" "${WORKDIR}/proc_b.log"
report "coverage output byte-identical across processes" "$?" "two fresh processes"

HOME="${WORKDIR}" "${WORKER}" --capability-audit "${WORKDIR}/caps.json" > "${WORKDIR}/caps.log" 2> /dev/null
caps_ok=0
[ -s "${WORKDIR}/caps.json" ] || caps_ok=1
head -1 "${WORKDIR}/caps.json" | grep -q '^\[' || caps_ok=1
grep -q '"monster_id": 1, "name": "Peasant".*"simple_v1_supported": true' "${WORKDIR}/caps.json" || caps_ok=1
records="$(grep -c '"monster_id":' "${WORKDIR}/caps.json")"
[ "${records}" -ge 60 ] || caps_ok=1
report "capability audit written and sane" "${caps_ok}" "${records} monster records"

# Behaviour-cloning samples: observation, legal set and teacher index together, not an action
# alone. See agent_play/tests/check_bc_samples.py for what is asserted and why.
mkdir -p "${WORKDIR}/bc"
HOME="${WORKDIR}" "${WORKER}" --runs 1 --audit-coverage --trajectory-dir "${WORKDIR}/bc" > /dev/null 2>&1
bc_detail="$("${REPO_ROOT}/agent_play/tests/check_bc_samples.py" "${WORKDIR}/bc" 2>&1)"
report "cloning samples are complete" "$?" "${bc_detail}"

echo
echo "  ${PASS} passed, ${FAIL} failed"
[ "${FAIL}" -eq 0 ]
