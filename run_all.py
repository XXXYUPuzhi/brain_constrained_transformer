# -*- coding: utf-8 -*-
"""
run_all.py
==========
Run all resource-limitation experiments sequentially for the 12d1h model.
Covers baseline, L1 lambda sweep, and Top-K activation sparsity sweep.

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
O  = os.path.join(SCRIPT_DIR, "results")

experiments = [
    # Baseline (12d1h)
    f"{PY} {S} --reg none --layers 2 --dim 12 --heads 1 --data_dir {D} --out_dir {O}",
    # L1 sweep
    f"{PY} {S} --reg l1 --lambda_l1 1e-5 --layers 2 --dim 12 --heads 1 --data_dir {D} --out_dir {O} --snapshot_every 100",
    f"{PY} {S} --reg l1 --lambda_l1 5e-5 --layers 2 --dim 12 --heads 1 --data_dir {D} --out_dir {O} --snapshot_every 100",
    f"{PY} {S} --reg l1 --lambda_l1 1e-4 --layers 2 --dim 12 --heads 1 --data_dir {D} --out_dir {O} --snapshot_every 100",
    f"{PY} {S} --reg l1 --lambda_l1 5e-4 --layers 2 --dim 12 --heads 1 --data_dir {D} --out_dir {O} --snapshot_every 100",
    f"{PY} {S} --reg l1 --lambda_l1 1e-3 --layers 2 --dim 12 --heads 1 --data_dir {D} --out_dir {O} --snapshot_every 100",
    f"{PY} {S} --reg l1 --lambda_l1 1e-2 --layers 2 --dim 12 --heads 1 --data_dir {D} --out_dir {O} --snapshot_every 100",
    # TopK sweep (MLP hidden = 4*12=48)
    f"{PY} {S} --reg topk --topk_k 4  --layers 2 --dim 12 --heads 1 --data_dir {D} --out_dir {O}",
    f"{PY} {S} --reg topk --topk_k 8  --layers 2 --dim 12 --heads 1 --data_dir {D} --out_dir {O}",
    f"{PY} {S} --reg topk --topk_k 16 --layers 2 --dim 12 --heads 1 --data_dir {D} --out_dir {O}",
    f"{PY} {S} --reg topk --topk_k 24 --layers 2 --dim 12 --heads 1 --data_dir {D} --out_dir {O}",
    f"{PY} {S} --reg topk --topk_k 32 --layers 2 --dim 12 --heads 1 --data_dir {D} --out_dir {O}",
]

print(f"Starting {len(experiments)} experiments at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 60}\n")
sys.stdout.flush()

for i, cmd in enumerate(experiments):
    print(f"\n{'=' * 60}")
    print(f"[{i+1}/{len(experiments)}] {cmd.split('--reg')[1].split('--layers')[0].strip()}")
    print(f"{'=' * 60}")
    sys.stdout.flush()

    t0 = time.time()
    ret = subprocess.run(cmd.split(), cwd=SCRIPT_DIR)
    elapsed = (time.time() - t0) / 60

    status = "OK" if ret.returncode == 0 else f"FAILED (code {ret.returncode})"
    print(f"\n  => {status} ({elapsed:.1f} min)")
    sys.stdout.flush()

print(f"\n{'=' * 60}")
print(f"All experiments done at {time.strftime('%Y-%m-%d %H:%M:%S')}")
