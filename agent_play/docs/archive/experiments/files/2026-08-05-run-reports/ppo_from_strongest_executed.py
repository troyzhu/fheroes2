import json, pathlib, sys, tempfile
import numpy as np
import torch
sys.path.insert(0, "python")
from fheroes2_agent.env import MatchupPool
from fheroes2_agent.policy import BattlePolicy
from fheroes2_agent.scenarios import Matchup, measure
from fheroes2_agent import train_ppo

W = pathlib.Path("/private/tmp/claude-501/-Volumes-External-Drive-GitHub-fheroes2/f819124c-6010-4c4d-80fe-e094e5e9e0b0/scratchpad/work")
CK = "agent_play/docs/archive/experiments/files/2026-08-05-checkpoints/policy_combined_v2.pt"
entries = json.loads(pathlib.Path("agent_play/docs/archive/experiments/files/2026-08-05-run-reports/pool_value.json").read_text())["matchups"]
mk = lambda e: Matchup(e["attacker"], e["defender"], attacker_hero=e.get("attacker_hero"), defender_hero=e.get("defender_hero"), allow_wide=bool(e.get("allow_wide")))
train_set, held_set = [mk(e) for e in entries[:40]], [mk(e) for e in entries[40:60]]
workdir = pathlib.Path(tempfile.mkdtemp(prefix="ppo_strong_"))
runs = []
for seed in range(3):
    pool = MatchupPool(str(W / "worker_dagger"), train_set, seed=seed)
    out = workdir / f"s{seed}.pt"
    train_ppo.train(str(W / "worker_dagger"), checkpoint=CK, iterations=40, seed=seed, env=pool, quiet=True, out=str(out))
    m = BattlePolicy(); m.load_state_dict(torch.load(out, map_location="cpu", weights_only=True)["state_dict"]); m.eval()
    r = {"seed": seed,
         "train": [measure(m, str(W / "worker_dagger"), x, episodes=24, seeds=4)["win_rate"] for x in train_set],
         "held": [measure(m, str(W / "worker_dagger"), x, episodes=24, seeds=4)["win_rate"] for x in held_set]}
    runs.append(r)
    print(f"seed {seed}: train {np.mean(r['train']):.3f}, held {np.mean(r['held']):.3f}", flush=True)
prev = json.loads((W / "dagger_combined_v2.json").read_text())["evals"]
for split in ("train", "held"):
    arm = np.array([np.mean(r[split]) for r in runs])
    print(f"{split}: ppo-from-combined {arm.mean():.3f} +/- {arm.std(ddof=1)/np.sqrt(3):.3f} (anchor {np.mean(prev[split]):.3f})")
(W / "ppo_from_strongest.json").write_text(json.dumps({"runs": runs, "anchor": prev, "checkpoints": str(workdir)}, indent=2))
print("checkpoints:", workdir)
