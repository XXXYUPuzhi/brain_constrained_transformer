# -*- coding: utf-8 -*-
"""
compare_baseline_topk.py
========================
Head-to-head comparison between the baseline and TopK-constrained models.

Includes:
  1. Structural metrics: modularity Q, hierarchy H, weight sparsity
  2. Neuron-feature correlation (49 game-state features)
  3. Behavioral clustering (k-means on CLS + game feature profiling)
  4. NMF decomposition of MLP activations
  5. TopK-specific: neuron usage frequency analysis
  6. Optional: channel ablation comparison

Usage:
  python compare_baseline_topk.py \\
    --baseline results_48d2h/..._earlystop.pkl \\
    --topk results_48d2h/..._topk8_earlystop.pkl \\
    --data_dir ./junction_predict_resortind \\
    --out_dir ./comparison_results

Author: Puzhi YU
Date:   January 2026
"""

import argparse
import importlib.util
import os
import pickle
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# 0. Import project modules
# ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(module_name, filepath):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


utils = _load_module('utils', os.path.join(_HERE, 'utils.py'))
bb = _load_module('building_blocks', os.path.join(_HERE, 'building_blocks.py'))
visual2action_3 = bb.visual2action_3

train_mod = _load_module('train_resource_limited',
                         os.path.join(_HERE, 'train_resource_limited.py'))
load_dataset = train_mod.load_dataset
TopKMlp = train_mod.TopKMlp
apply_topk_to_model = train_mod.apply_topk_to_model
Logger = train_mod.Logger  # needed for pickle deserialization

sd_mod = _load_module('strategy_decode',
                      os.path.join(_HERE, 'strategy_decode.py'))
extract_game_features = sd_mod.extract_game_features

IDX2DIR = {0: 'down', 1: 'left', 2: 'right', 3: 'up'}


# ──────────────────────────────────────────────────────────────
# 1. Enhanced hidden-state extraction (captures post-topk + masks)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_hidden_states(model, s, a, device, batch_size=2048):
    """
    Extract CLS token hidden states after each transformer layer.

    Returns
    -------
    preds          : (N, 4) softmax predictions
    layer_cls      : list of (N, dim) — CLS after each layer
    mlp_acts_pre   : list of (N, hidden_dim) — MLP activations post-GELU, pre-topk
    mlp_acts_post  : list of (N, hidden_dim) — MLP activations post-topk (same as pre for baseline)
    topk_masks     : list of (N, hidden_dim) binary — which neurons were selected (None for baseline)
    """
    model.eval()
    loader = DataLoader(TensorDataset(s, a), batch_size=batch_size, shuffle=False)

    n_layers = len(model.encode)
    all_preds = []
    layer_cls_acc = [[] for _ in range(n_layers)]
    mlp_pre_acc = [[] for _ in range(n_layers)]
    mlp_post_acc = [[] for _ in range(n_layers)]
    topk_mask_acc = [[] for _ in range(n_layers)]
    is_topk = isinstance(model.encode[0].mlp, TopKMlp)

    for s_b, a_b in tqdm(loader, desc='  Extracting hidden states', ncols=80):
        s_b = s_b.to(device)
        B = s_b.shape[0]

        x = model.to_patch_embedding(s_b)
        x = x + model.input_pos_emb.expand(B, -1, -1)
        x = torch.cat((model.act_token.expand(B, -1, -1), x), dim=1)

        for i, block in enumerate(model.encode):
            # Attention
            x1 = block.norm1(x)
            attn_out, _, _, _, _ = block.attn(x1)
            x = x + block.drop_path(attn_out)

            # MLP (manual, to capture activations)
            x2 = block.norm2(x)
            mlp = block.mlp

            if isinstance(mlp, TopKMlp):
                h = mlp.fc1(x2)
                h = mlp.act(h)
                mlp_pre_acc[i].append(h[:, 0, :].cpu().numpy())

                if mlp.k is not None and mlp.k < h.shape[-1]:
                    topk_idx = torch.topk(h.abs(), k=mlp.k, dim=-1).indices
                    mask = torch.zeros_like(h).scatter_(-1, topk_idx, 1.0)
                    h_masked = h * mask
                    topk_mask_acc[i].append(mask[:, 0, :].cpu().numpy())
                else:
                    h_masked = h
                    topk_mask_acc[i].append(
                        np.ones((B, h.shape[-1]), dtype=np.float32))

                mlp_post_acc[i].append(h_masked[:, 0, :].cpu().numpy())
                h = mlp.drop1(h_masked)
                h = mlp.fc2(h)
                h = mlp.drop2(h)
            else:
                # Standard timm Mlp
                h = mlp.fc1(x2)
                h = mlp.act(h)
                mlp_pre_acc[i].append(h[:, 0, :].cpu().numpy())
                mlp_post_acc[i].append(h[:, 0, :].cpu().numpy())
                h = mlp.drop1(h)
                h = mlp.fc2(h)
                h = mlp.drop2(h)

            x = x + block.drop_path(h)
            layer_cls_acc[i].append(x[:, 0, :].cpu().numpy())

        y = F.softmax(model.predhead(x[:, 0, :]), dim=-1)
        all_preds.append(y.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    layer_cls = [np.concatenate(lc, axis=0) for lc in layer_cls_acc]
    mlp_pre = [np.concatenate(mp, axis=0) for mp in mlp_pre_acc]
    mlp_post = [np.concatenate(mp, axis=0) for mp in mlp_post_acc]
    topk_masks = ([np.concatenate(tm, axis=0) for tm in topk_mask_acc]
                  if is_topk else None)

    return preds, layer_cls, mlp_pre, mlp_post, topk_masks


# ──────────────────────────────────────────────────────────────
# 2. Structural metrics (from analyze_resource_limited.py)
# ──────────────────────────────────────────────────────────────

def compute_modularity_hierarchy(features, threshold_pct=90):
    """Modularity Q (Louvain) and hierarchy score from hidden states."""
    import networkx as nx
    import community.community_louvain as community_louvain

    corr = np.abs(np.corrcoef(features.T))
    corr = np.nan_to_num(corr, nan=0.0)
    threshold = np.percentile(corr, threshold_pct)
    adj = corr.copy()
    adj[adj < threshold] = 0
    np.fill_diagonal(adj, 0)

    G = nx.from_numpy_array(adj)
    partition = community_louvain.best_partition(G)
    q_score = community_louvain.modularity(partition, G)
    n_communities = len(set(partition.values()))

    degrees = np.array([d for _, d in G.degree()])
    n = len(degrees)
    if n > 1 and degrees.max() > 0:
        c_max = degrees.max()
        hierarchy_score = np.sum(c_max - degrees) / (n - 1) / c_max
    else:
        hierarchy_score = 0.0

    return {
        'q_score': q_score,
        'hierarchy_score': hierarchy_score,
        'n_communities': n_communities,
        'partition': partition,
    }


def compute_weight_sparsity(model, threshold=1e-6):
    """Fraction of MLP weights with |w| < threshold."""
    zeros, total = 0, 0
    for name, p in model.named_parameters():
        if 'encode' in name and ('mlp' in name or 'fc1' in name or 'fc2' in name) and 'weight' in name:
            zeros += (p.abs() < threshold).sum().item()
            total += p.numel()
    return zeros / total if total > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# 3. Neuron-feature correlation (vectorized for speed)
# ──────────────────────────────────────────────────────────────

def neuron_feature_correlation(activations, feat_matrix, feat_names):
    """
    Pearson r between each neuron and each game feature.
    Returns r_matrix (n_neurons, n_features), p_matrix.
    """
    n_neurons = activations.shape[1]
    n_feats = feat_matrix.shape[1]
    r_mat = np.zeros((n_neurons, n_feats))
    p_mat = np.ones((n_neurons, n_feats))

    for j in range(n_feats):
        fj = feat_matrix[:, j]
        valid = ~np.isnan(fj)
        if valid.sum() < 10:
            continue
        fj_valid = fj[valid]
        if fj_valid.std() < 1e-10:
            continue
        for i in range(n_neurons):
            ai = activations[valid, i]
            if ai.std() < 1e-10:
                continue
            r, p = stats.pearsonr(ai, fj_valid)
            r_mat[i, j] = r
            p_mat[i, j] = p

    return r_mat, p_mat


# ──────────────────────────────────────────────────────────────
# 4. Behavioral clustering with game-feature profiling
# ──────────────────────────────────────────────────────────────

def behavioral_clustering(cls_features, labels, feat_dict, n_clusters=6):
    """K-means on CLS, profile each cluster by game features."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(cls_features)

    profiles = {}
    for c in range(n_clusters):
        mask = cluster_ids == c
        dir_counts = np.bincount(labels[mask], minlength=4)
        feat_means = {}
        for fname, fvals in feat_dict.items():
            valid = ~np.isnan(fvals[mask])
            if valid.sum() > 0:
                feat_means[fname] = float(np.nanmean(fvals[mask]))
            else:
                feat_means[fname] = np.nan
        profiles[c] = {
            'count': int(mask.sum()),
            'dir_counts': {IDX2DIR[i]: int(dir_counts[i]) for i in range(4)},
            'dir_dominant': IDX2DIR[dir_counts.argmax()],
            'feat_means': feat_means,
        }

    return {
        'cluster_ids': cluster_ids,
        'centers': km.cluster_centers_,
        'inertia': km.inertia_,
        'profiles': profiles,
    }


def strategy_diversity_score(clustering_result, feature_name):
    """Variance of a game feature's mean across clusters — higher = more strategic differentiation."""
    means = []
    for p in clustering_result['profiles'].values():
        v = p['feat_means'].get(feature_name, np.nan)
        if not np.isnan(v):
            means.append(v)
    return float(np.var(means)) if len(means) > 1 else 0.0


# ──────────────────────────────────────────────────────────────
# 5. NMF decomposition
# ──────────────────────────────────────────────────────────────

def strategy_nmf(mlp_activations, n_components=8):
    """NMF on post-GELU activations (already non-negative)."""
    X = np.maximum(mlp_activations, 0)
    if X.sum() == 0:
        return None
    nmf = NMF(n_components=min(n_components, X.shape[1]),
              init='nndsvd', random_state=42, max_iter=500)
    W = nmf.fit_transform(X)
    H = nmf.components_
    return {'W': W, 'H': H, 'reconstruction_error': nmf.reconstruction_err_}


# ──────────────────────────────────────────────────────────────
# 6. Channel ablation
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def channel_ablation(model, s, a, device, batch_size=4096):
    """Zero out channel groups and measure accuracy drop."""
    model.eval()
    loader = DataLoader(TensorDataset(s, a), batch_size=batch_size, shuffle=False)

    def get_accuracy(s_tensor, a_tensor):
        correct, total = 0, 0
        for s_b, a_b in DataLoader(TensorDataset(s_tensor, a_tensor),
                                    batch_size=batch_size, shuffle=False):
            s_b, a_b = s_b.to(device), a_b.to(device)
            y, _ = model((s_b, a_b))
            correct += (y.argmax(1) == a_b).sum().item()
            total += len(a_b)
        return correct / total

    baseline_acc = get_accuracy(s, a)

    # Channel groups (both frames)
    groups = {
        'ghost1': [8, 9, 10, 11, 25, 26, 27, 28],
        'ghost2': [12, 13, 14, 15, 29, 30, 31, 32],
        'ALL_ghosts': [8, 9, 10, 11, 12, 13, 14, 15,
                       25, 26, 27, 28, 29, 30, 31, 32],
        'beans': [1, 18],
        'walls': [0, 17],
        'pacman': [16, 33],
        'energizer': [2, 19],
    }

    results = {'baseline_acc': baseline_acc}
    for name, channels in groups.items():
        s_abl = s.clone()
        for ch in channels:
            s_abl[:, :, :, ch] = 0.0
        acc = get_accuracy(s_abl, a)
        results[name] = {
            'acc': acc,
            'drop': baseline_acc - acc,
            'drop_pct': (baseline_acc - acc) / baseline_acc * 100,
        }
    return results


# ──────────────────────────────────────────────────────────────
# 7. Model loading
# ──────────────────────────────────────────────────────────────

def load_model(model_path, device):
    """Load a trained model checkpoint."""
    with open(model_path, 'rb') as f:
        log, state_dict, configs = pickle.load(f)

    configs['device'] = device
    model = visual2action_3(configs)

    if configs.get('regularization') == 'topk' and configs.get('topk_k') is not None:
        model = apply_topk_to_model(model, k=configs['topk_k'],
                                     dim=configs['patch_emb_dim'])

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, configs, log


# ──────────────────────────────────────────────────────────────
# 8. Visualization
# ──────────────────────────────────────────────────────────────

def plot_all(baseline_res, topk_res, out_dir):
    """Generate all comparison figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    colors_4dir = {'down': '#e74c3c', 'left': '#3498db',
                   'right': '#2ecc71', 'up': '#f39c12'}

    # ── 8a. Structural comparison bar chart ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    labels_m = ['Baseline', 'TopK=8']

    for layer_idx in range(2):
        q_vals = [baseline_res['structural'][f'layer_{layer_idx+1}']['q_score'],
                  topk_res['structural'][f'layer_{layer_idx+1}']['q_score']]
        h_vals = [baseline_res['structural'][f'layer_{layer_idx+1}']['hierarchy_score'],
                  topk_res['structural'][f'layer_{layer_idx+1}']['hierarchy_score']]

        x = np.arange(2)
        w = 0.35
        axes[layer_idx].bar(x - w/2, q_vals, w, label='Modularity Q', color='steelblue')
        axes[layer_idx].bar(x + w/2, h_vals, w, label='Hierarchy H', color='coral')
        axes[layer_idx].set_xticks(x)
        axes[layer_idx].set_xticklabels(labels_m)
        axes[layer_idx].set_title(f'Layer {layer_idx+1}')
        axes[layer_idx].legend()
        axes[layer_idx].set_ylim(0, 1)

    # Accuracy + sparsity
    acc_vals = [baseline_res['test_acc'], topk_res['test_acc']]
    spar_vals = [baseline_res['weight_sparsity'], topk_res['weight_sparsity']]
    x = np.arange(2)
    axes[2].bar(x - w/2, acc_vals, w, label='Test Acc', color='steelblue')
    axes[2].bar(x + w/2, spar_vals, w, label='Weight Sparsity', color='coral')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels_m)
    axes[2].set_title('Accuracy & Sparsity')
    axes[2].legend()
    axes[2].set_ylim(0, 1)

    fig.suptitle('Structural Metrics Comparison', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'structural_comparison.png'), dpi=150)
    plt.close(fig)
    print('  Saved structural_comparison.png')

    # ── 8b. Neuron-feature correlation heatmaps (side by side per layer) ──
    feat_names = baseline_res['feat_names']
    for layer_idx in range(2):
        r_base = baseline_res['r_matrices'][layer_idx]
        r_topk = topk_res['r_matrices'][layer_idx]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 8))
        sns.heatmap(r_base, cmap='RdBu_r', center=0, vmin=-0.5, vmax=0.5,
                    xticklabels=feat_names, ax=ax1,
                    yticklabels=[f'N{i}' for i in range(r_base.shape[0])],
                    cbar_kws={'label': 'Pearson r'})
        ax1.set_title(f'Baseline — Layer {layer_idx+1}')
        ax1.tick_params(axis='x', rotation=60, labelsize=6)
        ax1.tick_params(axis='y', labelsize=6)

        sns.heatmap(r_topk, cmap='RdBu_r', center=0, vmin=-0.5, vmax=0.5,
                    xticklabels=feat_names, ax=ax2,
                    yticklabels=[f'N{i}' for i in range(r_topk.shape[0])],
                    cbar_kws={'label': 'Pearson r'})
        ax2.set_title(f'TopK=8 — Layer {layer_idx+1}')
        ax2.tick_params(axis='x', rotation=60, labelsize=6)
        ax2.tick_params(axis='y', labelsize=6)

        fig.suptitle(f'Neuron-Feature Correlation — Layer {layer_idx+1}', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'neuron_feature_corr_L{layer_idx+1}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved neuron_feature_corr_L{layer_idx+1}.png')

    # ── 8c. TopK neuron usage frequency ──
    if topk_res.get('topk_masks') is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for layer_idx in range(2):
            usage = topk_res['topk_masks'][layer_idx].mean(axis=0)
            r_topk = topk_res['r_matrices'][layer_idx]
            max_r_idx = np.abs(r_topk).argmax(axis=1)
            max_r_val = np.array([np.abs(r_topk[i, max_r_idx[i]])
                                  for i in range(len(max_r_idx))])

            # Color by feature category
            ghost_feats = [i for i, n in enumerate(feat_names)
                           if n.startswith('g1') or n.startswith('g2')
                           or n in ('max_threat', 'max_opportunity')]
            colors = []
            for i in range(len(max_r_idx)):
                if max_r_idx[i] in ghost_feats and max_r_val[i] > 0.15:
                    colors.append('#e74c3c')  # red = ghost-related
                elif max_r_val[i] > 0.15:
                    colors.append('#3498db')  # blue = other feature
                else:
                    colors.append('#95a5a6')  # grey = no strong correlation

            axes[layer_idx].bar(range(len(usage)), usage, color=colors, width=1.0)
            axes[layer_idx].axhline(y=8/192, color='black', linestyle='--',
                                     alpha=0.5, label='Uniform (8/192)')
            axes[layer_idx].set_xlabel('Neuron Index')
            axes[layer_idx].set_ylabel('Selection Frequency')
            axes[layer_idx].set_title(f'Layer {layer_idx+1} Neuron Usage')
            axes[layer_idx].legend()
            # Custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#e74c3c', label='Ghost-related'),
                Patch(facecolor='#3498db', label='Other feature'),
                Patch(facecolor='#95a5a6', label='No strong corr'),
            ]
            axes[layer_idx].legend(handles=legend_elements, fontsize=8)

        fig.suptitle('TopK=8: Neuron Selection Frequency', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'neuron_usage_frequency.png'), dpi=150)
        plt.close(fig)
        print('  Saved neuron_usage_frequency.png')

    # ── 8d. Cluster comparison ──
    key_feats = ['max_threat', 'max_opportunity', 'g1_dist', 'g2_dist',
                 'bean_total', 'nearest_bean_dist', 'pac_row', 'pac_col',
                 'energizer_present', 'n_passable']

    for tag, res in [('baseline', baseline_res), ('topk', topk_res)]:
        clust = res['clustering']
        n_c = len(clust['profiles'])

        # Direction distribution
        fig, axes_row = plt.subplots(1, n_c, figsize=(3*n_c, 3), sharey=True)
        if n_c == 1:
            axes_row = [axes_row]
        for c in range(n_c):
            d = clust['profiles'][c]['dir_counts']
            dirs = ['down', 'left', 'right', 'up']
            counts = [d[dn] for dn in dirs]
            axes_row[c].bar(dirs, counts, color=[colors_4dir[dn] for dn in dirs])
            axes_row[c].set_title(f'C{c} (n={clust["profiles"][c]["count"]})')
        fig.suptitle(f'{tag.upper()}: Cluster Direction Distribution')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'cluster_directions_{tag}.png'), dpi=150)
        plt.close(fig)

        # Feature profile heatmap
        feat_mat = np.array([[clust['profiles'][c]['feat_means'].get(f, np.nan)
                              for f in key_feats] for c in range(n_c)])
        # Z-score normalize per feature
        with np.errstate(divide='ignore', invalid='ignore'):
            feat_z = (feat_mat - np.nanmean(feat_mat, axis=0)) / (np.nanstd(feat_mat, axis=0) + 1e-10)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(feat_z, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                    xticklabels=key_feats,
                    yticklabels=[f'C{c} ({clust["profiles"][c]["dir_dominant"]})'
                                 for c in range(n_c)],
                    ax=ax)
        ax.set_title(f'{tag.upper()}: Cluster Feature Profiles (z-scored)')
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'cluster_features_{tag}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    print('  Saved cluster comparison figures')

    # ── 8e. NMF comparison ──
    if baseline_res.get('nmf') and topk_res.get('nmf'):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        sns.heatmap(baseline_res['nmf']['H'], cmap='viridis', ax=ax1,
                    xticklabels=False,
                    yticklabels=[f'Comp {i}' for i in range(baseline_res['nmf']['H'].shape[0])])
        ax1.set_title('Baseline: NMF Basis Vectors')
        ax1.set_xlabel('Neuron Index')

        sns.heatmap(topk_res['nmf']['H'], cmap='viridis', ax=ax2,
                    xticklabels=False,
                    yticklabels=[f'Comp {i}' for i in range(topk_res['nmf']['H'].shape[0])])
        ax2.set_title('TopK=8: NMF Basis Vectors')
        ax2.set_xlabel('Neuron Index')

        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'nmf_comparison.png'), dpi=150)
        plt.close(fig)
        print('  Saved nmf_comparison.png')

    # ── 8f. Ablation comparison ──
    if baseline_res.get('ablation') and topk_res.get('ablation'):
        groups = ['ghost1', 'ghost2', 'ALL_ghosts', 'beans', 'walls',
                  'pacman', 'energizer']
        base_drops = [baseline_res['ablation'][g]['drop_pct'] for g in groups]
        topk_drops = [topk_res['ablation'][g]['drop_pct'] for g in groups]

        x = np.arange(len(groups))
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - w/2, base_drops, w, label='Baseline', color='steelblue')
        ax.bar(x + w/2, topk_drops, w, label='TopK=8', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.set_ylabel('Accuracy Drop (%)')
        ax.set_title('Channel Ablation Comparison')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'ablation_comparison.png'), dpi=150)
        plt.close(fig)
        print('  Saved ablation_comparison.png')


# ──────────────────────────────────────────────────────────────
# 9. Print comparison tables
# ──────────────────────────────────────────────────────────────

def print_structural_table(base, topk):
    """Print side-by-side structural metrics."""
    print('\n' + '='*70)
    print('  STRUCTURAL METRICS COMPARISON')
    print('='*70)
    print(f'{"Metric":<30} {"Baseline":>12} {"TopK=8":>12} {"Delta":>12}')
    print('-'*70)

    rows = [
        ('Test Accuracy',
         base['test_acc'], topk['test_acc']),
        ('Weight Sparsity',
         base['weight_sparsity'], topk['weight_sparsity']),
    ]
    for layer_idx in range(2):
        bl = base['structural'][f'layer_{layer_idx+1}']
        tl = topk['structural'][f'layer_{layer_idx+1}']
        rows.extend([
            (f'Layer {layer_idx+1} Modularity Q',
             bl['q_score'], tl['q_score']),
            (f'Layer {layer_idx+1} Hierarchy H',
             bl['hierarchy_score'], tl['hierarchy_score']),
            (f'Layer {layer_idx+1} Communities',
             bl['n_communities'], tl['n_communities']),
        ])

    for name, bv, tv in rows:
        delta = tv - bv
        sign = '+' if delta >= 0 else ''
        if isinstance(bv, int):
            print(f'{name:<30} {bv:>12d} {tv:>12d} {sign}{delta:>11d}')
        else:
            print(f'{name:<30} {bv:>12.4f} {tv:>12.4f} {sign}{delta:>11.4f}')

    print('='*70)


def print_neuron_specialization(base, topk, feat_names):
    """Print neuron specialization summary."""
    ghost_feats = set(i for i, n in enumerate(feat_names)
                      if n.startswith('g1') or n.startswith('g2')
                      or n in ('max_threat', 'max_opportunity'))
    bean_feats = set(i for i, n in enumerate(feat_names)
                     if 'bean' in n or 'energizer' in n)

    print('\n' + '='*70)
    print('  NEURON SPECIALIZATION COMPARISON')
    print('='*70)

    for layer_idx in range(2):
        r_base = base['r_matrices'][layer_idx]
        r_topk = topk['r_matrices'][layer_idx]

        print(f'\n  --- Layer {layer_idx+1} ---')
        print(f'  {"Metric":<45} {"Baseline":>10} {"TopK=8":>10}')
        print(f'  {"-"*65}')

        # Neurons with |r| > 0.2 for ghost features
        for threshold in [0.2, 0.3]:
            base_ghost = sum(1 for i in range(r_base.shape[0])
                             if any(abs(r_base[i, j]) > threshold for j in ghost_feats))
            topk_ghost = sum(1 for i in range(r_topk.shape[0])
                             if any(abs(r_topk[i, j]) > threshold for j in ghost_feats))
            print(f'  Neurons w/ ghost |r|>{threshold:<4}           {base_ghost:>10d} {topk_ghost:>10d}')

        # Max |r| for ghost features across all neurons
        base_max_ghost = max(abs(r_base[:, list(ghost_feats)]).max(), 0)
        topk_max_ghost = max(abs(r_topk[:, list(ghost_feats)]).max(), 0)
        print(f'  Max ghost |r|                                {base_max_ghost:>10.4f} {topk_max_ghost:>10.4f}')

        # Mean |r| across top-5 ghost-correlated neurons
        base_ghost_r = np.sort(abs(r_base[:, list(ghost_feats)]).max(axis=1))[::-1][:5]
        topk_ghost_r = np.sort(abs(r_topk[:, list(ghost_feats)]).max(axis=1))[::-1][:5]
        print(f'  Top-5 ghost neurons avg |r|                  {base_ghost_r.mean():>10.4f} {topk_ghost_r.mean():>10.4f}')

        # Top-3 strongest neuron-feature pairs
        print(f'\n  Top-5 neuron-feature pairs (Baseline L{layer_idx+1}):')
        flat = np.abs(r_base).flatten()
        top_idx = flat.argsort()[::-1][:5]
        for idx in top_idx:
            ni, fi = divmod(idx, r_base.shape[1])
            print(f'    N{ni:>3d} ↔ {feat_names[fi]:<25} r={r_base[ni, fi]:>+.4f}')

        print(f'  Top-5 neuron-feature pairs (TopK=8 L{layer_idx+1}):')
        flat = np.abs(r_topk).flatten()
        top_idx = flat.argsort()[::-1][:5]
        for idx in top_idx:
            ni, fi = divmod(idx, r_topk.shape[1])
            print(f'    N{ni:>3d} ↔ {feat_names[fi]:<25} r={r_topk[ni, fi]:>+.4f}')

    print('='*70)


def print_cluster_comparison(base, topk, feat_names_for_profile):
    """Print cluster strategy diversity comparison."""
    print('\n' + '='*70)
    print('  BEHAVIORAL CLUSTERING COMPARISON')
    print('='*70)

    for tag, res in [('Baseline', base), ('TopK=8', topk)]:
        clust = res['clustering']
        print(f'\n  --- {tag} ---')
        for c in sorted(clust['profiles'].keys()):
            p = clust['profiles'][c]
            d = p['dir_counts']
            print(f'  Cluster {c}: n={p["count"]:>5d}  '
                  f'D={d["down"]:>4d} L={d["left"]:>4d} '
                  f'R={d["right"]:>4d} U={d["up"]:>4d}  '
                  f'dominant={p["dir_dominant"]}  '
                  f'threat={p["feat_means"].get("max_threat", 0):.3f}  '
                  f'opportunity={p["feat_means"].get("max_opportunity", 0):.3f}  '
                  f'g1_dist={p["feat_means"].get("g1_dist", 0):.1f}')

    # Strategy diversity scores
    print(f'\n  --- Strategy Diversity Scores (higher = more strategic differentiation) ---')
    print(f'  {"Feature":<25} {"Baseline":>10} {"TopK=8":>10} {"Delta":>10}')
    print(f'  {"-"*55}')
    for feat in feat_names_for_profile:
        sv_base = strategy_diversity_score(base['clustering'], feat)
        sv_topk = strategy_diversity_score(topk['clustering'], feat)
        delta = sv_topk - sv_base
        sign = '+' if delta >= 0 else ''
        print(f'  {feat:<25} {sv_base:>10.6f} {sv_topk:>10.6f} {sign}{delta:>9.6f}')

    print('='*70)


def print_nmf_interpretation(base, topk, feat_dict, feat_names):
    """Interpret NMF components by correlating with game features."""
    print('\n' + '='*70)
    print('  NMF COMPONENT INTERPRETATION')
    print('='*70)

    feat_matrix = np.column_stack([feat_dict[k] for k in feat_names])

    for tag, res in [('Baseline', base), ('TopK=8', topk)]:
        nmf = res.get('nmf')
        if nmf is None:
            continue
        W = nmf['W']
        print(f'\n  --- {tag} (recon_err={nmf["reconstruction_error"]:.2f}) ---')

        for comp_idx in range(W.shape[1]):
            w_col = W[:, comp_idx]
            if w_col.std() < 1e-10:
                continue
            correlations = []
            for fi, fn in enumerate(feat_names):
                fv = feat_matrix[:, fi]
                valid = ~np.isnan(fv)
                if valid.sum() < 10:
                    continue
                r, _ = stats.pearsonr(w_col[valid], fv[valid])
                correlations.append((fn, r))
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            top3 = correlations[:3]
            desc = ', '.join(f'{n}({r:+.3f})' for n, r in top3)
            print(f'  Comp {comp_idx}: {desc}')

    print('='*70)


# ──────────────────────────────────────────────────────────────
# 10. TopK neuron usage analysis
# ──────────────────────────────────────────────────────────────

def print_neuron_usage(topk_res, feat_names):
    """Analyze which neurons are most/least used in TopK."""
    if topk_res.get('topk_masks') is None:
        return

    print('\n' + '='*70)
    print('  TOPK NEURON USAGE FREQUENCY ANALYSIS')
    print('='*70)

    for layer_idx in range(2):
        masks = topk_res['topk_masks'][layer_idx]
        usage = masks.mean(axis=0)
        uniform = 8 / 192

        print(f'\n  --- Layer {layer_idx+1} ---')
        print(f'  Uniform expected: {uniform:.4f}')
        print(f'  Actual range: [{usage.min():.4f}, {usage.max():.4f}]')
        print(f'  Std: {usage.std():.4f}')
        print(f'  Neurons never used (<0.1%): {(usage < 0.001).sum()}/192')
        print(f'  Neurons rarely used (<1%):  {(usage < 0.01).sum()}/192')
        print(f'  "Core" neurons (>10%):      {(usage > 0.10).sum()}/192')

        # Top-10 most used
        top10 = np.argsort(usage)[::-1][:10]
        r_mat = topk_res['r_matrices'][layer_idx]
        print(f'\n  Top-10 most used neurons:')
        for rank, ni in enumerate(top10):
            best_fi = np.abs(r_mat[ni]).argmax()
            best_fn = feat_names[best_fi]
            best_r = r_mat[ni, best_fi]
            print(f'    #{rank+1} N{ni:>3d}: usage={usage[ni]:.4f}  '
                  f'best_feat={best_fn:<20s} r={best_r:>+.4f}')

    print('='*70)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compare baseline vs TopK model')
    parser.add_argument('--baseline', type=str, required=True)
    parser.add_argument('--topk', type=str, required=True)
    parser.add_argument('--data_dir', type=str,
                        default='./junction_predict_resortind')
    parser.add_argument('--out_dir', type=str,
                        default='./comparison_results')
    parser.add_argument('--n_clusters', type=int, default=6)
    parser.add_argument('--n_nmf', type=int, default=8)
    parser.add_argument('--do_ablation', action='store_true',
                        help='Run channel ablation (slower)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load data (shared) ──
    print('\n── Loading test data ──')
    test_indices = [140, 141, 142, 143, 144]
    s_test, a_test, _ = load_dataset(args.data_dir, test_indices, desc='Test')
    print(f'Test samples: {s_test.shape[0]:,}')

    print('Extracting game features...')
    feat_dict = extract_game_features(s_test.numpy())
    feat_names = sorted(feat_dict.keys())
    feat_matrix = np.column_stack([feat_dict[k] for k in feat_names])
    print(f'Game features: {len(feat_names)}')

    # ── Analyze both models ──
    results = {}
    for tag, path in [('baseline', args.baseline), ('topk', args.topk)]:
        print(f'\n{"="*60}')
        print(f'  Analyzing: {tag.upper()} ({os.path.basename(path)})')
        print(f'{"="*60}')

        model, configs, log = load_model(path, device)
        print(f'  Reg: {configs.get("regularization", "none")}, '
              f'topk_k={configs.get("topk_k", "N/A")}, '
              f'lambda_l1={configs.get("lambda_l1", "N/A")}')

        # Extract hidden states
        preds, layer_cls, mlp_pre, mlp_post, topk_masks = \
            extract_hidden_states(model, s_test, a_test, device)

        pred_labels = preds.argmax(axis=1)
        true_labels = a_test.numpy()
        test_acc = (pred_labels == true_labels).mean()
        print(f'  Test accuracy: {test_acc:.4f}')

        # Structural metrics
        print('  Computing structural metrics...')
        structural = {}
        for i, cls_feat in enumerate(layer_cls):
            m = compute_modularity_hierarchy(cls_feat)
            structural[f'layer_{i+1}'] = {
                'q_score': m['q_score'],
                'hierarchy_score': m['hierarchy_score'],
                'n_communities': m['n_communities'],
            }
            print(f'    Layer {i+1}: Q={m["q_score"]:.4f}, '
                  f'H={m["hierarchy_score"]:.4f}, '
                  f'communities={m["n_communities"]}')

        weight_sparsity = compute_weight_sparsity(model)
        print(f'    Weight sparsity: {weight_sparsity:.4%}')

        # Neuron-feature correlation (use post-topk activations)
        print('  Computing neuron-feature correlations...')
        r_matrices = []
        p_matrices = []
        for i in range(len(mlp_post)):
            print(f'    Layer {i+1}...')
            r_mat, p_mat = neuron_feature_correlation(
                mlp_post[i], feat_matrix, feat_names)
            r_matrices.append(r_mat)
            p_matrices.append(p_mat)

        # Behavioral clustering
        print(f'  K-means clustering (k={args.n_clusters})...')
        clustering = behavioral_clustering(
            layer_cls[-1], true_labels, feat_dict, n_clusters=args.n_clusters)

        # NMF
        print(f'  NMF decomposition (n={args.n_nmf})...')
        nmf_result = strategy_nmf(mlp_post[-1], n_components=args.n_nmf)

        # Channel ablation (optional)
        ablation_result = None
        if args.do_ablation:
            print('  Channel ablation...')
            ablation_result = channel_ablation(model, s_test, a_test, device)

        results[tag] = {
            'model_path': path,
            'configs': configs,
            'test_acc': test_acc,
            'weight_sparsity': weight_sparsity,
            'structural': structural,
            'layer_cls': layer_cls,
            'mlp_acts_pre': mlp_pre,
            'mlp_acts_post': mlp_post,
            'topk_masks': [tm for tm in topk_masks] if topk_masks is not None else None,
            'r_matrices': r_matrices,
            'p_matrices': p_matrices,
            'feat_names': feat_names,
            'clustering': clustering,
            'nmf': nmf_result,
            'ablation': ablation_result,
        }

        del model
        torch.cuda.empty_cache()

    base = results['baseline']
    topk = results['topk']

    # ── Print comparison ──
    print_structural_table(base, topk)
    print_neuron_specialization(base, topk, feat_names)

    key_profile_feats = ['max_threat', 'max_opportunity', 'g1_dist', 'g2_dist',
                         'bean_total', 'nearest_bean_dist', 'pac_row', 'pac_col']
    print_cluster_comparison(base, topk, key_profile_feats)
    print_nmf_interpretation(base, topk, feat_dict, feat_names)
    print_neuron_usage(topk, feat_names)

    # ── Save results ──
    # Don't save large arrays to keep pkl manageable
    save_results = {}
    for tag in ['baseline', 'topk']:
        r = results[tag]
        save_results[tag] = {
            'model_path': r['model_path'],
            'test_acc': r['test_acc'],
            'weight_sparsity': r['weight_sparsity'],
            'structural': r['structural'],
            'r_matrices': r['r_matrices'],
            'feat_names': r['feat_names'],
            'clustering': {
                'profiles': r['clustering']['profiles'],
                'inertia': r['clustering']['inertia'],
            },
            'nmf': {'H': r['nmf']['H'],
                     'reconstruction_error': r['nmf']['reconstruction_error']}
                   if r['nmf'] else None,
            'ablation': r['ablation'],
        }
        if r.get('topk_masks') is not None:
            # Save usage frequency only (not full masks)
            save_results[tag]['neuron_usage'] = [
                tm.mean(axis=0) for tm in r['topk_masks']]

    save_path = os.path.join(args.out_dir, 'comparison_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_results, f)
    print(f'\nResults saved: {save_path}')

    # ── Plot ──
    print('\nGenerating figures...')
    plot_all(base, topk, args.out_dir)

    print('\nDone!')


if __name__ == '__main__':
    main()
