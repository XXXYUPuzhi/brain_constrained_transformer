# -*- coding: utf-8 -*-
"""
summarize_all.py
================
Summarize training results for all resource-limitation experiments.

Loads checkpoint .pkl files from the results directory and prints a
consolidated table of accuracy, sparsity, and training statistics for
each regularization condition (baseline, L1 sweep, TopK sweep).

Author: Puzhi YU
Date:   January 2026
"""

import pickle
import torch
import numpy as np
import os
import sys

results_dir = "./results_48d2h"


def load_model(path):
    with open(path, "rb") as f:
        logger, state_dict, configs = pickle.load(f)
    return logger, state_dict, configs


def get_sparsity(state_dict, threshold=1e-4):
    """Calculate weight sparsity for MLP layers."""
    mlp_total = 0
    mlp_zero = 0
    all_total = 0
    all_zero = 0

    for name, param in state_dict.items():
        w = param.cpu().numpy().flatten()
        total = len(w)
        zero = np.sum(np.abs(w) < threshold)
        all_total += total
        all_zero += zero

        if "mlp" in name or "fc1" in name or "fc2" in name:
            mlp_total += total
            mlp_zero += zero

    return {
        "mlp_sparsity": mlp_zero / mlp_total if mlp_total > 0 else 0,
        "mlp_total_params": mlp_total,
        "mlp_zero_params": mlp_zero,
        "all_sparsity": all_zero / all_total if all_total > 0 else 0,
        "all_total_params": all_total,
        "all_zero_params": all_zero,
    }


def get_weight_stats(state_dict):
    """Get weight magnitude statistics for MLP layers."""
    mlp_weights = []
    for name, param in state_dict.items():
        if "mlp" in name or "fc1" in name or "fc2" in name:
            mlp_weights.append(param.cpu().numpy().flatten())

    if mlp_weights:
        w = np.concatenate(mlp_weights)
        return {
            "mlp_mean_abs": float(np.mean(np.abs(w))),
            "mlp_median_abs": float(np.median(np.abs(w))),
            "mlp_max_abs": float(np.max(np.abs(w))),
            "mlp_std": float(np.std(w)),
            "mlp_l1_norm": float(np.sum(np.abs(w))),
        }
    return {}


def summarize_model(path, label):
    """Print training summary for a single model checkpoint."""
    try:
        logger, state_dict, configs = load_model(path)
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        return

    train_losses = logger.train_loss if hasattr(logger, "train_loss") else []
    test_accs = logger.accuracy if hasattr(logger, "accuracy") else []
    test_losses = logger.test_loss if hasattr(logger, "test_loss") else []

    best_acc = max(test_accs) if test_accs else 0
    best_epoch = np.argmax(test_accs) + 1 if test_accs else 0
    final_acc = test_accs[-1] if test_accs else 0
    total_epochs = len(test_accs)

    final_train_loss = train_losses[-1] if train_losses else 0
    final_test_loss = test_losses[-1] if test_losses else 0
    best_test_loss = min(test_losses) if test_losses else 0

    sparsity = get_sparsity(state_dict)
    weight_stats = get_weight_stats(state_dict)

    reg_type = configs.get("reg_type", "none")
    lambda_l1 = configs.get("lambda_l1", 0)
    topk_k = configs.get("topk_k", 0)

    sep = '=' * 70
    print(f"\n{sep}")
    print(f"  {label}")
    print(f"{sep}")
    print(f"  File: {os.path.basename(path)}")
    reg_info = f"  Reg: {reg_type}"
    if lambda_l1:
        reg_info += f", lambda={lambda_l1}"
    if topk_k:
        reg_info += f", k={topk_k}"
    print(reg_info)
    print(f"  Total epochs: {total_epochs}")
    print(f"  Best test acc: {best_acc:.4f} (epoch {best_epoch})")
    print(f"  Final test acc: {final_acc:.4f}")
    print(f"  Final train loss: {final_train_loss:.6f}")
    print(f"  Final test loss: {final_test_loss:.6f}")
    print(f"  Best test loss: {best_test_loss:.6f}")
    print(f"  --- Weight Stats (final epoch) ---")
    print(f"  MLP sparsity (|w|<1e-4): {sparsity['mlp_sparsity']:.4f} "
          f"({sparsity['mlp_zero_params']}/{sparsity['mlp_total_params']})")
    print(f"  All sparsity (|w|<1e-4): {sparsity['all_sparsity']:.4f} "
          f"({sparsity['all_zero_params']}/{sparsity['all_total_params']})")
    if weight_stats:
        print(f"  MLP mean|w|: {weight_stats['mlp_mean_abs']:.6f}")
        print(f"  MLP median|w|: {weight_stats['mlp_median_abs']:.6f}")
        print(f"  MLP max|w|: {weight_stats['mlp_max_abs']:.4f}")
        print(f"  MLP L1 norm: {weight_stats['mlp_l1_norm']:.2f}")

    # Accuracy at key training checkpoints
    checkpoints = [10, 20, 50, 100, 200, 300, 500]
    acc_at = []
    for ep in checkpoints:
        if ep <= len(test_accs):
            acc_at.append(f"ep{ep}={test_accs[ep-1]:.4f}")
    if acc_at:
        print(f"  Acc trajectory: {', '.join(acc_at)}")


# ────────────────────────────────────────────────────────────
# Define experiment configurations
# ────────────────────────────────────────────────────────────

experiments = []
base = "140sess_2layers_focalloss_48d2h_noEmbLN_0768b"

# Baseline
experiments.append((f"{results_dir}/{base}.pkl", "Baseline (none)"))

# L1 sweep
for lam in ["1e-05", "5e-05", "1e-04", "5e-04", "1e-03", "1e-02"]:
    path = f"{results_dir}/{base}_L1{lam}.pkl"
    if os.path.exists(path):
        experiments.append((path, f"L1 lambda={lam}"))

# TopK sweep
for k in [8, 16, 32, 64, 96, 128]:
    path = f"{results_dir}/{base}_topk{k}.pkl"
    es_path = f"{results_dir}/{base}_topk{k}_earlystop.pkl"
    if os.path.exists(path):
        experiments.append((path, f"TopK k={k} (final)"))
    elif os.path.exists(es_path):
        experiments.append((es_path, f"TopK k={k} (earlystop only)"))


# ────────────────────────────────────────────────────────────
# Print summaries
# ────────────────────────────────────────────────────────────

print("\n" + "#" * 70)
print("# FINAL EPOCH MODELS")
print("#" * 70)
for path, label in experiments:
    summarize_model(path, label)

# Early-stop models
print("\n\n" + "#" * 70)
print("# EARLY STOP MODELS (best test acc checkpoint)")
print("#" * 70)

es_experiments = []
es_experiments.append((f"{results_dir}/{base}_earlystop.pkl",
                        "Baseline (none) - earlystop"))
for lam in ["1e-05", "5e-05", "1e-04", "5e-04", "1e-03", "1e-02"]:
    path = f"{results_dir}/{base}_L1{lam}_earlystop.pkl"
    if os.path.exists(path):
        es_experiments.append((path, f"L1 lambda={lam} - earlystop"))

for k in [8, 16, 32, 64, 96, 128]:
    path = f"{results_dir}/{base}_topk{k}_earlystop.pkl"
    if os.path.exists(path):
        es_experiments.append((path, f"TopK k={k} - earlystop"))

for path, label in es_experiments:
    summarize_model(path, label)
