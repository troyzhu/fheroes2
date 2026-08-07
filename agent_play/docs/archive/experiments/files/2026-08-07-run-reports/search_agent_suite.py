"""The search agent measured on the AI's own columns: held-out pool, rollout-scored, planes prior."""
import sys, json, pathlib, time
sys.path.insert(0, "python"); sys.path.insert(0, "agent_play/experiments")
import numpy as np, torch
from fheroes2_agent.env import BattleEnv
from fheroes2_agent.policy import load_policy
from search_probe import search_action
W = "/private/tmp/claude-501/-Volumes-External-Drive-GitHub-fheroes2/f819124c-6010-4c4d-80fe-e094e5e9e0b0/scratchpad/work"
model = load_policy(torch.load(W + "/champ_planes_s0/policy_planes.pt", map_location="cpu", weights_only=True)["state_dict"]); model.eval()
entries = json.loads(pathlib.Path("agent_play/docs/archive/experiments/files/2026-08-05-run-reports/pool_value.json").read_text())["matchups"][40:60]
rates, started = [], time.time()
for i, e in enumerate(entries):
    kw = dict(attacker=e["attacker"], defender=e["defender"], attacker_hero=e.get("attacker_hero"),
              defender_hero=e.get("defender_hero"), allow_wide=bool(e.get("allow_wide")), planes=True)
    wins = 0
    for ep in range(4):
        env = BattleEnv(W + "/worker_planes", **kw, seeds=1, seed_offset=ep)
        sim = BattleEnv(W + "/worker_planes", **kw, seeds=1, seed_offset=ep)
        try:
            obs, mask = env.reset(); prefix = []
            while True:
                a = search_action(sim, model, prefix, obs, mask, 32, 1.5, live=env)
                prefix.append(a); step = env.step(a)
                if step.done:
                    wins += step.info["termination"] == "victory"; break
                obs, mask = step.observation, step.mask
        finally:
            env.close(); sim.close()
    rates.append(wins / 4)
    print(f"matchup {i:02d}: {wins}/4  (elapsed {round(time.time()-started)}s)", flush=True)
arr = np.array(rates)
print(f"\nSEARCH AGENT held-out: {arr.mean():.3f} +/- {arr.std(ddof=1)/np.sqrt(len(arr)):.3f}   (built-in AI: 0.660)")
json.dump({"rates": rates, "mean": float(arr.mean())}, open(W + "/search_agent_heldout.json", "w"), indent=2)
