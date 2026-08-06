"""Does the value target's shape explain the failure? Discounted margin vs AlphaZero outcome."""
import sys, json, pathlib
sys.path.insert(0, "python"); sys.path.insert(0, "agent_play/experiments")
import numpy as np, torch
from fheroes2_agent.dataset import load_dir, split_by_episode
from fheroes2_agent.encoding import encode_mask, encode_observation
from fheroes2_agent.policy import load_policy
W = "/private/tmp/claude-501/-Volumes-External-Drive-GitHub-fheroes2/f819124c-6010-4c4d-80fe-e094e5e9e0b0/scratchpad/work"

def outcome_targets(roots):
    """AlphaZero-style: undiscounted +1/-1 for the acting side, no margin term."""
    obs, masks, ys = [], [], []
    for root in roots:
        for path in sorted(pathlib.Path(root).rglob("*.jsonl")):
            recs = [json.loads(l) for l in path.read_text().splitlines()]
            term = next((r for r in recs if r.get("record") == "terminal"), None)
            if term is None: continue
            att_won = term["termination"] == "victory"
            for r in recs:
                if r.get("record") != "decision" or "observation" not in r: continue
                actor_att = bool(r["observation"]["active_is_attacker"])
                obs.append(encode_observation(r["observation"])); masks.append(encode_mask(r["legal_actions"]))
                ys.append(1.0 if (actor_att == att_won) else -1.0)
    return np.stack(obs), np.stack(masks), np.asarray(ys, dtype=np.float32)

def fit_head(model, X, M, y, epochs=20):
    for p in model.parameters(): p.requires_grad_(False)
    model.value_head.weight.requires_grad_(True); model.value_head.bias.requires_grad_(True)
    opt = torch.optim.Adam(model.value_head.parameters(), lr=3e-3)
    n = len(y); idx = np.arange(n); rng = np.random.default_rng(0)
    split = int(n * 0.8); rng.shuffle(idx); tr, ho = idx[:split], idx[split:]
    Xt, Mt, yt = torch.from_numpy(X[tr]), torch.from_numpy(M[tr]), torch.from_numpy(y[tr])
    Xh, Mh, yh = torch.from_numpy(X[ho]), torch.from_numpy(M[ho]), torch.from_numpy(y[ho])
    for _ in range(epochs):
        perm = torch.randperm(len(yt))
        for s in range(0, len(yt), 256):
            b = perm[s:s+256]
            _, v = model(Xt[b], Mt[b]); loss = torch.nn.functional.mse_loss(v, yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _, vh = model(Xh, Mh)
    pred = vh.numpy()
    ev = 1.0 - np.var(yh.numpy() - pred) / max(np.var(yh.numpy()), 1e-9)
    return float(ev), float(np.mean(pred - yh.numpy()))

roots = [f"{W}/data_dagger", f"{W}/data_teacher_control"]
X, M, y = outcome_targets(roots)
print(f"outcome targets: n={len(y)}, win share {float((y>0).mean()):.3f}, variance {float(np.var(y)):.3f}")
m = load_policy(torch.load(W + "/policy_gen1.pt", map_location="cpu", weights_only=True)["state_dict"])
ev, bias = fit_head(m, X, M, y)
print(f"AlphaZero-style (undiscounted, outcome only): holdout EV {ev:+.3f}, bias {bias:+.3f}")
s = load_dir(roots)
keep = np.isfinite(s.returns)
print(f"current target (discounted, margin-mixed): variance {float(np.var(s.returns[keep])):.3f}")
torch.save({"state_dict": m.state_dict(), "encoding_version": "obs_encoding_v3"}, W + "/value_outcome.pt")
