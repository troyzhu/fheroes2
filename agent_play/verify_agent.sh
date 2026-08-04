#!/usr/bin/env bash
# Behaviour-cloning gate (stage 1 of decisions/0005-training-and-reward).
#
# Checks, in order:
#   1. the Python unit tests pass (observation encoding, dataset loading, policy masking);
#   2. the worker records complete cloning samples across many seeds;
#   3. the dataset loads, and every teacher action lies inside its own legal mask;
#   4. a short training run beats both trivial baselines by a wide margin;
#   5. the value head fits on teacher returns without moving the cloned policy;
#   6. the checkpoint carries the encoding version it was trained against.
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

echo "fheroes2 agent training verification (cloning and PPO)"
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

# Stage 2b, the critic pre-fitted on teacher play. Two properties are checked, and the second
# matters more than the first: fitting the value head must leave the cloned policy bit-identical,
# because the trunk is shared and an unfrozen fit measurably damages it (0.887 -> 0.701 agreement
# on the full dataset). The explained-variance floor is deliberately low. On this gate's 200
# episodes the fit reaches only about 0.11, since both the data volume and the weak trunk bind;
# the published 0.835 comes from ten times the data behind a trunk cloned to 0.887.
( cd "${PY}" && python3 -m fheroes2_agent.train_critic "${WORKDIR}/data" "${WORKDIR}/policy.pt" \
    --epochs 20 --out "${WORKDIR}/critic.pt" --report "${WORKDIR}/critic.json" ) > "${WORKDIR}/critic.log" 2>&1
critic_rc=$?

critic_detail="$(cd "${PY}" && python3 - "${WORKDIR}" "${critic_rc}" <<'PYEOF'
import json, pathlib, sys, torch
work = pathlib.Path(sys.argv[1])
if sys.argv[2] != "0" or not (work / "critic.json").exists():
    print("critic fitting did not complete")
    sys.exit(1)
r = json.loads((work / "critic.json").read_text())
before, after = r["before"]["explained_variance"], r["after"]["explained_variance"]

# The frozen fit must not move one policy parameter. Compared tensor by tensor rather than by a
# metric, because an agreement score can be unchanged while the weights have drifted.
old = torch.load(work / "policy.pt", map_location="cpu", weights_only=True)["state_dict"]
new = torch.load(work / "critic.pt", map_location="cpu", weights_only=True)["state_dict"]
moved = [k for k in old if not k.startswith("value_head") and not torch.equal(old[k], new[k])]
frozen = not moved
print(f"explained variance {before:+.3f} -> {after:+.3f}, policy weights "
      + ("untouched" if frozen else f"MOVED in {len(moved)} tensors"))
sys.exit(0 if (before < 0 and after > 0.05 and frozen) else 1)
PYEOF
)"
report "critic fits on teacher play, policy frozen" "$?" "${critic_detail}"

timeout 300 "${REPO_ROOT}/agent_play/tests/test_protocol.py" "${WORKER}" > "${WORKDIR}/proto.log" 2>&1
report "external control drives a battle" "$?" "$(grep -c '^  PASS' "${WORKDIR}/proto.log") checks, scripted stdin and stdout"

python3 "${PY}/tests/test_ppo.py" > "${WORKDIR}/ppo.log" 2>&1
report "reward, GAE and truncation unit tests" "$?" "$(grep -c '^  PASS' "${WORKDIR}/ppo.log") checks"

python3 "${PY}/tests/test_objectives.py" > "${WORKDIR}/obj.log" 2>&1
report "advantage and trust-region unit tests" "$?" "$(grep -c '^  PASS' "${WORKDIR}/obj.log") checks"

# PPO end to end on a matchup measured to sit inside the difficulty band. Five iterations is
# enough to show the loop closes; the published improvement comes from a longer run.
( cd "${PY}" && timeout 600 python3 -m fheroes2_agent.train_ppo "${WORKER}" \
    --checkpoint "${WORKDIR}/policy.pt" --attacker 1:5 --defender 1:5 \
    --iterations 5 --episodes 16 --report "${WORKDIR}/ppo_report.json" ) > "${WORKDIR}/ppo_run.log" 2>&1
ppo_rc=$?
ppo_detail="$(python3 - "${WORKDIR}/ppo_report.json" "${ppo_rc}" <<'PYEOF'
import json, pathlib, sys
if sys.argv[2] != "0" or not pathlib.Path(sys.argv[1]).exists():
    print("PPO did not complete")
    sys.exit(1)
r = json.loads(pathlib.Path(sys.argv[1]).read_text())
wins = [h["win_rate"] for h in r["history"]]
# The gate proves the loop closes and does not collapse, not that it reaches a given number:
# a five-iteration run is far too short to hold to a target, and a threshold on it would be a
# coin flip dressed as a check.
ok = len(wins) == r["iterations"] and max(wins) >= r["initial_win_rate"]
print(f"win rate {r['initial_win_rate']:.3f} -> best {max(wins):.3f} over {r['iterations']} iterations")
sys.exit(0 if ok else 1)
PYEOF
)"
report "PPO closes the loop without collapsing" "$?" "${ppo_detail}"

stamp="$(python3 -c "
import torch, sys
c = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
print(c.get('encoding_version', 'MISSING'))
" "${WORKDIR}/policy.pt" 2>/dev/null)"
# Compared against what the code declares, not a frozen literal, so bumping the encoding does
# not require editing this gate. The property under test is that the two agree.
expected="$(cd "${PY}" && python3 -c "from fheroes2_agent.encoding import ENCODING_VERSION; print(ENCODING_VERSION)")"
[ -n "${stamp}" ] && [ "${stamp}" = "${expected}" ]
report "checkpoint stamps its encoding version" "$?" "${stamp:-unreadable} matches ${expected}"

echo
echo "  ${PASS} passed, ${FAIL} failed"
[ "${FAIL}" -eq 0 ]
