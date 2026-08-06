"""Fit a behavior Q on (state, action, outcome) and take one improvement step inside support.

The Unplugged recipe at our scale: Q(s,a) by Monte-Carlo regression on unfiltered corpora,
then act by re-ranking the policy prior's top candidates by Q. Move-level where the failed
leaf probe was state-level; educational contrast is the point as much as the win rate.
"""
import sys, json, pathlib
sys.path.insert(0, "python"); sys.path.insert(0, "agent_play/experiments")
W = "/private/tmp/claude-501/-Volumes-External-Drive-GitHub-fheroes2/f819124c-6010-4c4d-80fe-e094e5e9e0b0/scratchpad/work"
sys.path.insert(0, W)
import numpy as np, torch, torch.nn as nn
from value_target_test import outcome_targets
from fheroes2_agent.encoding import SLOT_COUNT, SLOT_FEATURES, GLOBAL_FEATURES
from fheroes2_agent.policy import load_policy
from fheroes2_agent.scenarios import Matchup, measure

def action_targets(roots):
    """(state, taken action, actor outcome) triples; the teacher's resolved canonical index is
    the action, and rows whose action never resolved into the schema are dropped in lockstep
    with outcome_targets' row order."""
    from fheroes2_agent.encoding import encode_mask, encode_observation
    obs, masks, acts, ys = [], [], [], []
    for root in roots:
        for path in sorted(pathlib.Path(root).rglob("*.jsonl")):
            recs = [json.loads(l) for l in path.read_text().splitlines()]
            term = next((r for r in recs if r.get("record") == "terminal"), None)
            if term is None: continue
            att_won = term["termination"] == "victory"
            for r in recs:
                if r.get("record") != "decision" or "observation" not in r: continue
                if not r.get("teacher_resolved") or r.get("teacher_action") is None: continue
                actor_att = bool(r["observation"]["active_is_attacker"])
                obs.append(encode_observation(r["observation"])); masks.append(encode_mask(r["legal_actions"]))
                acts.append(int(r["teacher_action"])); ys.append(1.0 if (actor_att == att_won) else -1.0)
    return (np.stack(obs), np.stack(masks), np.asarray(acts), np.asarray(ys, dtype=np.float32))

class QNet(nn.Module):
    def __init__(self, slot_hidden=96, trunk=192):
        super().__init__()
        self.slot = nn.Sequential(nn.Linear(SLOT_FEATURES, slot_hidden), nn.ReLU(), nn.Linear(slot_hidden, slot_hidden), nn.ReLU())
        self.glob = nn.Sequential(nn.Linear(GLOBAL_FEATURES, 32), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(SLOT_COUNT * slot_hidden + 32, trunk), nn.ReLU(), nn.Linear(trunk, trunk), nn.ReLU())
        self.head = nn.Linear(trunk, 793)
    def forward(self, x):
        b = x.shape[0]
        slots = x[:, :SLOT_COUNT * SLOT_FEATURES].view(b, SLOT_COUNT, SLOT_FEATURES)
        e = self.slot(slots) * slots[:, :, :1]
        return torch.tanh(self.head(self.trunk(torch.cat([e.flatten(1), self.glob(x[:, SLOT_COUNT * SLOT_FEATURES:])], dim=1))))

roots = [f"{W}/data_diverse", f"{W}/data_dagger", f"{W}/data_teacher_control"]
X, M, A, y = action_targets(roots)
print(f"corpus: {len(y)} decisions, win share {(y>0).mean():.3f}", flush=True)
rng = np.random.default_rng(0); idx = np.arange(len(y)); rng.shuffle(idx)
split = int(0.8 * len(y)); tr, ho = idx[:split], idx[split:]
torch.manual_seed(0)
q = QNet(); opt = torch.optim.Adam(q.parameters(), lr=1e-3)
Xt, At, yt = torch.from_numpy(X[tr]), torch.from_numpy(A[tr]), torch.from_numpy(y[tr])
Xh, Ah, yh = torch.from_numpy(X[ho]), torch.from_numpy(A[ho]), torch.from_numpy(y[ho])
for epoch in range(8):
    perm = torch.randperm(len(yt))
    for s in range(0, len(yt), 512):
        b = perm[s:s+512]
        pred = q(Xt[b]).gather(1, At[b].unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.mse_loss(pred, yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ph = q(Xh).gather(1, Ah.unsqueeze(1)).squeeze(1)
    ev = 1.0 - np.var(yh.numpy() - ph.numpy()) / np.var(yh.numpy())
    print(f"epoch {epoch}: taken-action holdout EV {ev:+.3f}", flush=True)
torch.save({"state_dict": q.state_dict()}, W + "/q_behavior.pt")

policy = load_policy(torch.load(W + "/policy_gen1.pt", map_location="cpu", weights_only=True)["state_dict"]); policy.eval()

class OneStep(nn.Module):
    """Policy prior's top-k re-ranked by Q: one improvement step, inside support."""
    def __init__(self, policy, q, k=5):
        super().__init__(); self.policy = policy; self.q = q; self.k = k
    def forward(self, obs, mask):
        logits, value = self.policy(obs, mask)
        qv = self.q(obs)
        top = torch.topk(logits, self.k, dim=-1).indices
        boost = torch.full_like(logits, -1e8)
        boost.scatter_(1, top, qv.gather(1, top) * 50.0)
        return boost, value

pool = json.loads(pathlib.Path("agent_play/docs/archive/experiments/files/2026-08-05-run-reports/pool_value.json").read_text())["matchups"]
suite = [Matchup(e["attacker"], e["defender"], attacker_hero=e.get("attacker_hero"), defender_hero=e.get("defender_hero"), allow_wide=bool(e.get("allow_wide"))) for e in pool[40:50]]
thunk = Matchup("11:1,11:1,11:1,10:2,9:2", "1:334,1:333,1:333", attacker_hero="13:12", allow_wide=True)
for name, model in (("gen1 raw", policy), ("one-step Q rerank", OneStep(policy, q))):
    model.eval()
    held = float(np.mean([measure(model, W + "/worker_value", m, episodes=8, seeds=4)["win_rate"] for m in suite]))
    tk = measure(model, W + "/worker_value", thunk, episodes=12, seeds=4)["win_rate"]
    print(f"{name:20s} held-out(10) {held:.3f} | Thunk-1000 {tk:.2f}", flush=True)
