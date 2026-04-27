"""Analyze emergent strategy codes from trained model."""
import sys
import torch
import numpy as np
from collections import deque, defaultdict, Counter
sys.path.insert(0, ".")
from config import *
from environment.pacman_env import PacmanEnv
from model.hdt import HierarchicalAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("checkpoints/final_model.pt", map_location=device, weights_only=False)
config = ckpt["config"]
agent = HierarchicalAgent(config).to(device)
agent.load_state_dict(ckpt["model_state_dict"])
agent.eval()
print(f"Loaded model at step {ckpt.get('total_steps', '?')}")
print(f"Gumbel temp: {agent.temperature:.2f}")

stage = CURRICULUM[-1]
env = PacmanEnv(config, stage)

cd = defaultdict(lambda: {
    "n": 0, "gd": [], "fr": [], "pp": [], "rm": [],
    "pe": 0, "ge": 0, "de": 0, "a": [],
    "switches_from": [],  # What code was active before switching to this one
})

total_switches = 0
total_ghost_kills = 0
total_deaths = 0

for ep in range(100):
    obs = env.reset()
    agent.reset_inference()
    hb = deque(maxlen=config.high_seq_len)
    lb = deque(maxlen=config.low_seq_len)
    z = np.zeros(FEATURE_DIM, dtype=np.float32)
    for _ in range(config.high_seq_len):
        hb.append(z.copy())
    for _ in range(config.low_seq_len):
        lb.append(z.copy())
    ppe = 0
    prev_code = None

    for step in range(stage.max_steps):
        hb.append(obs)
        lb.append(obs)
        hs = torch.FloatTensor(np.array(list(hb))).unsqueeze(0).to(device)
        ls = torch.FloatTensor(np.array(list(lb))).unsqueeze(0).to(device)
        code, action = agent.act(hs, ls)

        if prev_code is not None and code != prev_code:
            total_switches += 1
            cd[code]["switches_from"].append(prev_code)
        prev_code = code

        d = cd[code]
        d["n"] += 1
        d["a"].append(action)

        mg = 999
        af = False
        for i in range(4):
            b = 2 + i * 5
            if obs[b + 4] > 0.5:
                if obs[b + 2] < mg:
                    mg = obs[b + 2]
                if obs[b + 3] > 0.5:
                    af = True
        if mg < 999:
            d["gd"].append(mg)
        d["fr"].append(1.0 if af else 0.0)
        d["pp"].append(obs[27])
        d["rm"].append(obs[28])

        obs, reward, done, info = env.step(action)
        p2 = info.get("pellets_eaten", 0)
        if p2 > ppe:
            d["pe"] += p2 - ppe
        ppe = p2
        c = info.get("collision", "")
        if c == "eat_ghost":
            d["ge"] += 1
            total_ghost_kills += 1
        elif c in ("death", "game_over"):
            d["de"] += 1
            total_deaths += 1
        if done:
            break

total = sum(v["n"] for v in cd.values())
AN = ["UP", "DOWN", "LEFT", "RIGHT"]

sep = "=" * 65
print(f"\n{sep}")
print(f"EMERGENT STRATEGY CODES (100 ep, 4 ghosts, {total} steps)")
print(sep)
print(f"Active codes: {len(cd)}/8")
print(f"Total code switches: {total_switches}")
print(f"Total ghost kills: {total_ghost_kills}")
print(f"Total deaths: {total_deaths}")
print()

for cid in sorted(cd.keys()):
    d = cd[cid]
    n = d["n"]
    if n == 0:
        continue

    gd = np.mean(d["gd"]) if d["gd"] else -1
    gm = np.percentile(d["gd"], 10) if len(d["gd"]) > 5 else -1
    fp = 100 * np.mean(d["fr"])
    pp = np.mean(d["pp"])
    rm = 100 * np.mean(d["rm"])
    ac = Counter(d["a"]).most_common(4)
    astr = " ".join(f"{AN[a]}:{c}" for a, c in ac)

    # Auto-label based on context
    lbl = "?"
    if gd < 0:
        lbl = "FORAGE (no ghost)"
    elif fp > 30:
        lbl = "HUNT (ghosts blue)"
    elif gd < 0.10:
        lbl = "EVADE (danger!)"
    elif gd < 0.18 and pp < 0.15:
        lbl = "SEEK POWER"
    elif gd >= 0.20:
        lbl = "FORAGE (safe)"
    else:
        lbl = "NAVIGATE"

    # Transition info
    from_codes = Counter(d["switches_from"])
    trans_str = ""
    if from_codes:
        trans_str = " <- " + ", ".join(f"Code{c}({n})" for c, n in from_codes.most_common(3))

    print(f"Code {cid} [{lbl}] | {100*n/total:.1f}% ({n} steps)")
    print(f"  Ghost dist: avg={gd:.3f} p10={gm:.3f} | Fright: {fp:.1f}%")
    print(f"  PP dist: {pp:.3f} | Remaining: {rm:.0f}%")
    print(f"  Pellets: {d['pe']} | Ghost kills: {d['ge']} | Deaths: {d['de']}")
    print(f"  Actions: {astr}")
    if trans_str:
        print(f"  Transitions{trans_str}")
    print()
