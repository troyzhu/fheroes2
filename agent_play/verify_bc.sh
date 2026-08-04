#!/usr/bin/env bash
# Behaviour-cloning gate (stage 1 of decisions/0005-training-and-reward).
#
# Checks, in order:
#   1. the Python unit tests pass (observation encoding, dataset loading, policy masking);
#   2. the worker records complete cloning samples across many seeds;
#   3. the dataset loads, and every teacher action lies inside its own legal mask;
#   4. a short training run beats both trivial baselines by a wide margin;
#   5. the checkpoint carries the encoding version it was trained against.
#
# Kept fast enough to run on every change: the whole gate is well under a minute, because
# recording 200 episodes takes about a third of a second and five epochs takes a few.
#
# Prerequisite: `make -C src/dist -j…` and `src/agent_worker/build_worker.sh`.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER="${REPO_ROOT}/src/agent_worker/fheroes2_agent_worker"
PY="${REPO_ROOT}/python"

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

echo "fheroes2 agent behaviour-cloning verification"
echo "  repo:   ${REPO_ROOT}"
echo "  commit: $(cd "${REPO_ROOT}" && git rev-parse --short HEAD) ($(cd "${REPO_ROOT}" && git branch --show-current))"
echo

python3 "${PY}/tests/test_encoding.py" > "${WORKDIR}/enc.log" 2>&1
report "encoding and dataset unit tests" "$?" "$(grep -c '^  PASS' "${WORKDIR}/enc.log") checks"

python3 "${PY}/tests/test_policy.py" > "${WORKDIR}/pol.log" 2>&1
report "policy and masking unit tests" "$?" "$(grep -c '^  PASS' "${WORKDIR}/pol.log") checks"

if [ ! -x "${WORKER}" ]; then
    report "worker binary present" "1" "${WORKER} missing"
    echo
    echo "  ${PASS} passed, ${FAIL} failed"
    exit 1
fi

mkdir -p "${WORKDIR}/data"
HOME="${WORKDIR}" "${WORKER}" --runs 1 --seeds 40 --audit-coverage --trajectory-dir "${WORKDIR}/data" --quiet > /dev/null 2>&1
episodes="$(find "${WORKDIR}/data" -name '*.jsonl' | wc -l | tr -d ' ')"
[ "${episodes}" -ge 200 ]
report "teacher episodes recorded" "$?" "${episodes} episodes over 40 world seeds per fixture"

"${REPO_ROOT}/agent_play/tests/check_bc_samples.py" "${WORKDIR}/data" > "${WORKDIR}/samples.log" 2>&1
report "samples are complete and consistent" "$?" "$(cat "${WORKDIR}/samples.log")"

# Training. Fifteen epochs on this dataset measures 0.66 agreement and takes under a second, so
# the floor below sits at 0.55 with room for run-to-run variation. The published figure comes
# from a longer run on more seeds and is recorded in the training-design document.
( cd "${PY}" && python3 -m fheroes2_agent.train_bc "${WORKDIR}/data" --epochs 15 \
    --out "${WORKDIR}/policy.pt" --report "${WORKDIR}/report.json" ) > "${WORKDIR}/train.log" 2>&1
train_rc=$?

verdict="$(python3 - "${WORKDIR}/report.json" "${train_rc}" <<'PYEOF'
import json, pathlib, sys
if sys.argv[2] != "0" or not pathlib.Path(sys.argv[1]).exists():
    print("training did not complete")
    sys.exit(1)
r = json.loads(pathlib.Path(sys.argv[1]).read_text())
best = r["best"]["agreement"]
uniform = r["baseline_uniform_over_legal"]
majority = r["baseline_majority_action"]
# A wide margin, because the failure this catches is a broken pipeline rather than a weak model:
# a shuffled label, a mask misalignment or a dead encoder all land near the baselines.
ok = best > 0.55 and best > 5 * uniform and best > 3 * majority
print(f"agreement {best:.3f} vs uniform {uniform:.3f} and majority {majority:.3f}")
sys.exit(0 if ok else 1)
PYEOF
)"
report "cloning beats trivial baselines" "$?" "${verdict}"

timeout 300 "${REPO_ROOT}/agent_play/tests/test_protocol.py" "${WORKER}" > "${WORKDIR}/proto.log" 2>&1
report "external control drives a battle" "$?" "$(grep -c '^  PASS' "${WORKDIR}/proto.log") checks, scripted stdin and stdout"

stamp="$(python3 -c "
import torch, sys
c = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
print(c.get('encoding_version', 'MISSING'))
" "${WORKDIR}/policy.pt" 2>/dev/null)"
[ "${stamp}" = "obs_encoding_v1" ]
report "checkpoint stamps its encoding version" "$?" "${stamp:-unreadable}"

echo
echo "  ${PASS} passed, ${FAIL} failed"
[ "${FAIL}" -eq 0 ]
