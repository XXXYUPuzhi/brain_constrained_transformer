# -*- coding: utf-8 -*-
"""
run_topk_48d2h.py
=================
Run TopK activation sparsity experiments only for the 48d2h model.
Useful for re-running the TopK sweep independently of L1 experiments.

Author: Puzhi YU
Date:   January 2026
"""

import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

PY = sys.executable
S  = os.path.join(SCRIPT_DIR, "train_resource_limited.py")
D  = os.path.join(SCRIPT_DIR, "junction_predict_resortind")
O  = os.path.join(SCRIPT_DIR, "results_48d2h")

os.makedirs(O, exist_ok=True)

experiments = [
    # TopK sweep (MLP hidden = 4*48=192)
    f"{PY} {S} --reg topk --topk_k 8   --layers 2 --dim 48 --heads 2 --data_dir {D} --out_dir {O}",
    f"{PY} {S} --reg topk --topk_k 16  --layers 2 --dim 48 --heads 2 --data_dir {D} --out_dir {O}",
    f"{PY} {S} --reg topk --topk_k 32  --layers 2 --dim 48 --heads 2 --data_dir {D} --out_dir {O}",
    f"{PY} {S} --reg topk --topk_k 64  --layers 2 --dim 48 --heads 2 --data_dir {D} --out_dir {O}",
    f"{PY} {S} --reg topk --topk_k 96  --layers 2 --dim 48 --heads 2 --data_dir {D} --out_dir {O}",
    f"{PY} {S} --reg topk --topk_k 128 --layers 2 --dim 48 --heads 2 --data_dir {D} --out_dir {O}",
]

sep = "=" * 60
print(f"Starting {len(experiments)} TopK experiments at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{sep}\n")
sys.stdout.flush()

for i, cmd in enumerate(experiments):
    print(f"\n{sep}")
    tag = cmd.split("--reg")[1].split("--layers")[0].strip()
    print(f"[{i+1}/{len(experiments)}] {tag}")
    print(sep)
    sys.stdout.flush()

    t0 = time.time()
    ret = subprocess.run(cmd.split(), cwd=SCRIPT_DIR)
    elapsed = (time.time() - t0) / 60

    status = "OK" if ret.returncode == 0 else f"FAILED (code {ret.returncode})"
    print(f"\n  => {status} ({elapsed:.1f} min)")
    sys.stdout.flush()

print(f"\n{sep}")
print(f"All TopK experiments done at {time.strftime('%Y-%m-%d %H:%M:%S')}")
