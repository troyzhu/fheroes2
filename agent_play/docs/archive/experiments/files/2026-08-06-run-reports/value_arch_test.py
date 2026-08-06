"""Is the frozen imitation trunk the binding constraint on value quality?"""
import sys, json, pathlib
sys.path.insert(0, "python"); sys.path.insert(0, "agent_play/experiments")
import numpy as np, torch, torch.nn as nn
from fheroes2_agent.encoding import OBSERVATION_SIZE, SLOT_COUNT, SLOT_FEATURES, GLOBAL_FEATURES
sys.path.insert(0, str(pathlib.Path("/private/tmp/claude-501/-Volumes-External-Drive-GitHub-fheroes2/f819124c-6010-4c4d-80fe-e094e5e9e0b0/scratchpad/work")))
from value_target_test import outcome_targets
W = "/private/tmp/claude-501/-Volumes-External-Drive-GitHub-fheroes2/f819124c-6010-4c4d-80fe-e094e5e9e0b0/scratchpad/work"

class ValueNet(nn.Module):
    """A value network of its own, same slot structure as the policy but trained only to predict
    the outcome, so nothing is shared with action imitation."""
    def __init__(self, slot_hidden=96, trunk=192):
        super().__init__()
        self.slot = nn.Sequential(nn.Linear(SLOT_FEATURES, slot_hidden), nn.ReLU(),
                                  nn.Linear(slot_hidden, slot_hidden), nn.ReLU())
        self.glob = nn.Sequential(nn.Linear(GLOBAL_FEATURES, 32), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(SLOT_COUNT * slot_hidden + 32, trunk), nn.ReLU(),
                                   nn.Linear(trunk, trunk), nn.ReLU())
        self.head = nn.Linear(trunk, 1)
    def forward(self, x):
        b = x.shape[0]
        slots = x[:, :SLOT_COUNT * SLOT_FEATURES].view(b, SLOT_COUNT, SLOT_FEATURES)
        g = x[:, SLOT_COUNT * SLOT_FEATURES:]
        e = self.slot(slots) * slots[:, :, :1]
        return torch.tanh(self.head(self.trunk(torch.cat([e.flatten(1), self.glob(g)], dim=1)))).squeeze(-1)

X, M, y = outcome_targets([f"{W}/data_dagger", f"{W}/data_teacher_control"])
rng = np.random.default_rng(0); idx = np.arange(len(y)); rng.shuffle(idx)
split = int(0.8 * len(y)); tr, ho = idx[:split], idx[split:]
Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
Xh, yh = torch.from_numpy(X[ho]), torch.from_numpy(y[ho])
m = ValueNet(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for epoch in range(12):
    perm = torch.randperm(len(yt))
    for s in range(0, len(yt), 256):
        b = perm[s:s+256]
        loss = torch.nn.functional.mse_loss(m(Xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = m(Xh).numpy()
    ev = 1.0 - np.var(yh.numpy() - pred) / np.var(yh.numpy())
    if epoch % 3 == 2 or epoch == 11:
        print(f"  epoch {epoch:2d}: holdout EV {ev:+.3f}")
torch.save({"state_dict": m.state_dict()}, W + "/value_dedicated.pt")
print(f"dedicated value network ({sum(p.numel() for p in m.parameters()):,} params): final holdout EV {ev:+.3f}")
