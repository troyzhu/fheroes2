#!/bin/bash
# Milestone 2 verification (spec §20, "decision hook and passive logging").
#
# Exit criteria checked here:
#   1. agent unit tests pass (now including CommandSnapshot decoding);
#   2. the worker builds;
#   3. built-in behavior is unchanged: state digests with the passive recorder attached equal
#      the Milestone 1 goldens (recorded before the DecisionController seam existed), and the
#      spike (which passes no controller at all) still reproduces its historical digest;
#   4. passive logs replay deterministically: one distinct decision digest per fixture across
#      ten runs, and trajectory files from two fresh processes are byte-identical.
#
# Prerequisite: `make -C src/dist -j…` has been run.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER="${REPO_ROOT}/src/agent_worker/fheroes2_agent_worker"
SPIKE="${REPO_ROOT}/agent_play/spike/smoke_battle"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

PASS=0
FAIL=0

report() {
    if [ "$2" = "0" ]; then
        printf '  \033[32mPASS\033[0m  %-40s %s\n' "$1" "$3"
        PASS=$((PASS + 1))
    else
        printf '  \033[31mFAIL\033[0m  %-40s %s\n' "$1" "$3"
        FAIL=$((FAIL + 1))
    fi
}

echo "fheroes2 agent Milestone 2 verification"
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

# Milestone 1 golden state digests: unchanged terminal outcomes prove the observer is inert.
GOLDEN_tiny="0985937ec02c27bc605a70d9b26e57ec8a9e921408db7a1043532000cbaff463"
GOLDEN_three="51a390dbef464e57a69a8cd3cc90bbf9424376e4edd23454a14dee538413b3a3"
GOLDEN_five="b893b0589479dbb3a069ac9a4bcc1c91eb300bbd7cc3799cbdd5e4678ab04ada"
GOLDEN_ranged="29fae534c93145af56bef0845b46476eef129116de9ada912c6d86a1e07a0e05"
GOLDEN_longer="ec8268417c2f75312cab33defb3255eeb7c18e0b765074f789b65da57905230e"

HOME="${WORKDIR}" "${WORKER}" --runs 10 > "${WORKDIR}/ten.log" 2> /dev/null
rc=$?
report "ten-run determinism (state + decisions)" "${rc}" "$(tail -1 "${WORKDIR}/ten.log")"

golden_ok=0
for pair in "m1_tiny_melee:${GOLDEN_tiny}" "m1_three_stack:${GOLDEN_three}" "m1_five_stack:${GOLDEN_five}" "m1_ranged_heavy:${GOLDEN_ranged}" \
            "m1_longer_balanced:${GOLDEN_longer}"; do
    fixture="${pair%%:*}"
    golden="${pair#*:}"
    grep -q "fixture=${fixture} .* digest=${golden}$" "${WORKDIR}/ten.log" || golden_ok=1
done
report "state digests equal Milestone 1 goldens" "${golden_ok}" "recorder attached, outcomes unchanged"

decisions_ok=0
grep -q "distinct_decision_digests=1" "${WORKDIR}/ten.log" || decisions_ok=1
grep -vq "distinct_decision_digests=1" <(grep "^RESULT" "${WORKDIR}/ten.log") && decisions_ok=1
report "one decision digest per fixture" "${decisions_ok}" "$(grep -c '^RESULT' "${WORKDIR}/ten.log") fixtures x 10 runs"

# Null-controller engine path: the spike constructs its arena without any controller.
spike_ok=1
if [ -x "${SPIKE}" ]; then
    HOME="${WORKDIR}" "${SPIKE}" --episodes 1 2> /dev/null | grep -q "digest=2cfd42cb104aa5e7" && spike_ok=0
    report "null-controller spike digest" "${spike_ok}" "2cfd42cb104aa5e7"
else
    report "null-controller spike digest" "1" "spike not built (run build_spike.sh)"
fi

mkdir -p "${WORKDIR}/trajA" "${WORKDIR}/trajB"
HOME="${WORKDIR}" "${WORKER}" --runs 1 --quiet --trajectory-dir "${WORKDIR}/trajA" > /dev/null 2>&1
HOME="${WORKDIR}" "${WORKER}" --runs 1 --quiet --trajectory-dir "${WORKDIR}/trajB" > /dev/null 2>&1

traj_ok=0
file_count=0
for fileA in "${WORKDIR}"/trajA/*.jsonl; do
    fileB="${WORKDIR}/trajB/$(basename "${fileA}")"
    cmp -s "${fileA}" "${fileB}" || traj_ok=1
    file_count=$((file_count + 1))
done
[ "${file_count}" -ge 5 ] || traj_ok=1
report "trajectories byte-identical across processes" "${traj_ok}" "${file_count} files compared"

record_ok=0
grep -q '"record":"episode_header"' "${WORKDIR}"/trajA/m1_tiny_melee-run00.jsonl || record_ok=1
grep -q '"record":"decision"' "${WORKDIR}"/trajA/m1_tiny_melee-run00.jsonl || record_ok=1
grep -q '"record":"terminal"' "${WORKDIR}"/trajA/m1_tiny_melee-run00.jsonl || record_ok=1
report "trajectory contains header/decision/terminal" "${record_ok}" "m1_tiny_melee-run00.jsonl"

echo
echo "  ${PASS} passed, ${FAIL} failed"
[ "${FAIL}" -eq 0 ]
