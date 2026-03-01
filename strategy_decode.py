# -*- coding: utf-8 -*-
"""
strategy_decode.py
==================
Decode high-level behavioral strategies from the transformer's FFN activations.

This module extracts game-state features from 34-channel Pac-Man input and
correlates them with MLP hidden activations to reveal what each neuron encodes
(ghost proximity, bean density, Pac-Man position, etc.).

Pipeline:
  1. Load model + session data (streaming, memory-efficient)
  2. Extract game-state features from 34-channel input
  3. Forward pass -> FFN activations + CLS representations
  4. Neuron-feature correlation analysis (Pearson r)
  5. Behavioral clustering (k-means on CLS)
  6. UMAP visualization
  7. Save results + figures

Usage:
  python strategy_decode.py
  python strategy_decode.py --sessions 140-144        # test set only
  python strategy_decode.py --sessions 0-189 --no_umap  # skip UMAP

Author: Puzhi YU
Date:   January 2026
"""

import argparse
import importlib.util
import os
import pickle
import sys
import warnings

import io
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# 0. Import project modules
# ══════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════
# 1. Data loading (one session at a time for memory efficiency)
# ══════════════════════════════════════════════════════════════════════
from sklearn.preprocessing import LabelEncoder
import einops

_label_enc = LabelEncoder()
_label_enc.fit(['down', 'left', 'right', 'up'])
IDX2DIR = {0: 'down', 1: 'left', 2: 'right', 3: 'up'}


def load_session(file_path):
    """Load one session. Returns (s, a) where s: (N,32,28,34), a: (N,)."""
    with open(file_path, 'rb') as f:
        tiles, actions, _ = pickle.load(f)

    tiles_cat = np.array([x[0] for x in tiles])
    N, T, H, W, C = tiles_cat.shape

    x_flat = torch.tensor(tiles_cat.reshape(-1, C))
    props_idx = torch.where(torch.all(utils.v1 == x_flat[:, None, :8], dim=2))[1]
    chara_idx = torch.where(torch.all(utils.v2 == x_flat[:, None, 8:], dim=2))[1]
    del x_flat

    game_info = np.stack(
        (props_idx.reshape(N, T, H, W).numpy(),
         chara_idx.reshape(N, T, H, W).numpy()),
        axis=-1,
    )

    obs = utils.label2voc(game_info)
    s = einops.rearrange(obs[:, :, 2:34, :, :], 'n t h w c -> n h w (t c)')
    a = _label_enc.transform(actions).astype(np.int64)
    return s.astype(np.float32), a


# ══════════════════════════════════════════════════════════════════════
# 2. Game-state feature extraction (vectorized)
# ══════════════════════════════════════════════════════════════════════

# Channel layout per frame (17 channels):
#   0:wall  1:bean  2:energizer  3-7:fruit1-5
#   8:g1_normal 9:g1_eaten 10:g1_scared 11:g1_flash
#   12:g2_normal 13:g2_eaten 14:g2_scared 15:g2_flash
#   16:pacman
# Full 34 channels: frame0 = ch[0:17], frame1 = ch[17:34]

def _find_positions(channel_map, H, W):
    """Find (row, col) from a (N, H, W) binary map. Returns (N,2) positions and (N,) exists mask."""
    N = channel_map.shape[0]
    flat = channel_map.reshape(N, -1)
    has_obj = flat.max(axis=1) > 0
    flat_idx = flat.argmax(axis=1)
    row = flat_idx // W
    col = flat_idx % W
    # Zero out positions where object doesn't exist
    row = row * has_obj
    col = col * has_obj
    pos = np.stack([row, col], axis=1).astype(np.float64)
    return pos, has_obj.astype(np.float64)


def extract_game_features(s):
    """
    Extract interpretable game-state features from raw input.

    Parameters
    ----------
    s : np.ndarray (N, 32, 28, 34)

    Returns
    -------
    feat_dict : dict  {feature_name: np.ndarray (N,)}
    """
    N, H, W, C = s.shape
    f0 = s[:, :, :, :17]    # earlier frame
    f1 = s[:, :, :, 17:]    # later frame (current)

    feat = {}

    # ── Grids for vectorized distance/quadrant computation ──
    row_grid = np.arange(H)[None, :, None]   # (1, H, 1)
    col_grid = np.arange(W)[None, None, :]   # (1, 1, W)

    # ── Pacman position (frame 1 = current) ──
    pac_pos1, _ = _find_positions(f1[:, :, :, 16], H, W)
    pac_r = pac_pos1[:, 0]
    pac_c = pac_pos1[:, 1]
    feat['pac_row'] = pac_r
    feat['pac_col'] = pac_c

    # Pacman position (frame 0) for movement
    pac_pos0, _ = _find_positions(f0[:, :, :, 16], H, W)
    feat['pac_move_row'] = pac_r - pac_pos0[:, 0]
    feat['pac_move_col'] = pac_c - pac_pos0[:, 1]

    # Broadcast pacman pos for spatial computations
    pr = pac_r[:, None, None]   # (N, 1, 1)
    pc = pac_c[:, None, None]

    # Distance from pacman to every cell
    dist_grid = np.abs(row_grid - pr) + np.abs(col_grid - pc)  # (N, H, W)

    # ── Ghost features (for each ghost) ──
    for g_idx, g_prefix in enumerate(['g1', 'g2']):
        base_ch = 8 + g_idx * 4
        # Ghost presence in frame 1: any of 4 state channels
        g_map1 = np.sum(f1[:, :, :, base_ch:base_ch+4], axis=-1)  # (N, H, W)
        g_pos1, g_exists1 = _find_positions(g_map1, H, W)

        # Ghost presence in frame 0
        g_map0 = np.sum(f0[:, :, :, base_ch:base_ch+4], axis=-1)
        g_pos0, g_exists0 = _find_positions(g_map0, H, W)

        # Distance to pacman
        g_dist = np.abs(g_pos1[:, 0] - pac_r) + np.abs(g_pos1[:, 1] - pac_c)
        g_dist_f0 = np.abs(g_pos0[:, 0] - pac_pos0[:, 0]) + np.abs(g_pos0[:, 1] - pac_pos0[:, 1])
        g_dist[g_exists1 == 0] = np.nan
        g_dist_f0[g_exists0 == 0] = np.nan

        feat[f'{g_prefix}_dist'] = g_dist
        feat[f'{g_prefix}_exists'] = g_exists1

        # Relative position
        feat[f'{g_prefix}_rel_row'] = g_pos1[:, 0] - pac_r
        feat[f'{g_prefix}_rel_col'] = g_pos1[:, 1] - pac_c

        # Ghost movement
        feat[f'{g_prefix}_move_row'] = g_pos1[:, 0] - g_pos0[:, 0]
        feat[f'{g_prefix}_move_col'] = g_pos1[:, 1] - g_pos0[:, 1]

        # Ghost state (binary)
        for j, state in enumerate(['normal', 'eaten', 'scared', 'flash']):
            ch = base_ch + j
            feat[f'{g_prefix}_{state}'] = (f1[:, :, :, ch].sum(axis=(1, 2)) > 0).astype(np.float64)

        # Approaching? (distance decreasing between frames)
        feat[f'{g_prefix}_approaching'] = np.where(
            (g_exists1 > 0) & (g_exists0 > 0),
            (g_dist_f0 > g_dist).astype(np.float64),
            np.nan,
        )

        # Composite: threat level  (normal & close → high)
        max_dist = H + W
        safe_dist = np.where(np.isnan(g_dist), max_dist, g_dist)
        feat[f'{g_prefix}_threat'] = feat[f'{g_prefix}_normal'] * (1 - safe_dist / max_dist)

        # Composite: opportunity  (scared & close → high)
        feat[f'{g_prefix}_opportunity'] = (
            np.maximum(feat[f'{g_prefix}_scared'], feat[f'{g_prefix}_flash'])
            * (1 - safe_dist / max_dist)
        )

    # Combined threat/opportunity
    feat['max_threat'] = np.nanmax(
        np.stack([feat['g1_threat'], feat['g2_threat']], axis=1), axis=1)
    feat['max_opportunity'] = np.nanmax(
        np.stack([feat['g1_opportunity'], feat['g2_opportunity']], axis=1), axis=1)

    # ── Bean features ──
    bean1 = f1[:, :, :, 1]   # (N, H, W)
    bean0 = f0[:, :, :, 1]

    feat['bean_total'] = bean1.sum(axis=(1, 2))
    feat['beans_eaten'] = bean0.sum(axis=(1, 2)) - bean1.sum(axis=(1, 2))

    # Directional bean counts (relative to pacman)
    up_mask = (row_grid < pr)
    down_mask = (row_grid > pr)
    left_mask = (col_grid < pc)
    right_mask = (col_grid > pc)

    feat['beans_up'] = (bean1 * up_mask).sum(axis=(1, 2))
    feat['beans_down'] = (bean1 * down_mask).sum(axis=(1, 2))
    feat['beans_left'] = (bean1 * left_mask).sum(axis=(1, 2))
    feat['beans_right'] = (bean1 * right_mask).sum(axis=(1, 2))

    # Nearest bean distance
    bean_dists = np.where(bean1 > 0, dist_grid, 999)
    feat['nearest_bean_dist'] = bean_dists.reshape(N, -1).min(axis=1).astype(np.float64)
    feat['nearest_bean_dist'][feat['bean_total'] == 0] = np.nan

    # ── Energizer features ──
    ener1 = f1[:, :, :, 2]
    feat['energizer_present'] = (ener1.sum(axis=(1, 2)) > 0).astype(np.float64)

    ener_dists = np.where(ener1 > 0, dist_grid, 999)
    nearest_ener = ener_dists.reshape(N, -1).min(axis=1).astype(np.float64)
    nearest_ener[feat['energizer_present'] == 0] = np.nan
    feat['nearest_energizer_dist'] = nearest_ener

    # ── Fruit features ──
    fruit_sum = f1[:, :, :, 3:8].sum(axis=(1, 2, 3))
    feat['fruit_present'] = (fruit_sum > 0).astype(np.float64)

    # ── Wall / passability features ──
    wall1 = f1[:, :, :, 0]   # (N, H, W)
    idx_n = np.arange(N)
    pr_int = pac_r.astype(int)
    pc_int = pac_c.astype(int)

    feat['can_up'] = 1 - wall1[idx_n, np.clip(pr_int - 1, 0, H - 1), pc_int]
    feat['can_down'] = 1 - wall1[idx_n, np.clip(pr_int + 1, 0, H - 1), pc_int]
    feat['can_left'] = 1 - wall1[idx_n, pr_int, np.clip(pc_int - 1, 0, W - 1)]
    feat['can_right'] = 1 - wall1[idx_n, pr_int, np.clip(pc_int + 1, 0, W - 1)]
    feat['n_passable'] = feat['can_up'] + feat['can_down'] + feat['can_left'] + feat['can_right']

    # ── Ghost state changes between frames ──
    for g_idx, g_prefix in enumerate(['g1', 'g2']):
        base_ch = 8 + g_idx * 4
        scared_f0 = (f0[:, :, :, base_ch + 2].sum(axis=(1, 2)) > 0).astype(np.float64)
        scared_f1 = (f1[:, :, :, base_ch + 2].sum(axis=(1, 2)) > 0).astype(np.float64)
        feat[f'{g_prefix}_just_scared'] = ((scared_f1 > 0) & (scared_f0 == 0)).astype(np.float64)

    return feat


# ══════════════════════════════════════════════════════════════════════
# 3. Forward pass with FFN activation extraction
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_activations(model, s_tensor, device='cpu', batch_size=512):
    """
    Forward pass with manual layer-by-layer extraction.

    Returns
    -------
    preds      : np.ndarray (N, 4)     softmax predictions
    cls_layers : list of (N, dim)      CLS token after each layer
    mlp_hidden : list of (N, 4*dim)    MLP hidden activations (post-GELU) for CLS token
    """
    model.eval()
    N = s_tensor.shape[0]
    n_layers = len(model.encode)
    dim = model.encode[0].mlp.fc1.weight.shape[1]
    hidden_dim = model.encode[0].mlp.fc1.weight.shape[0]

    all_preds = []
    cls_acc = [[] for _ in range(n_layers)]
    mlp_acc = [[] for _ in range(n_layers)]

    loader = DataLoader(TensorDataset(s_tensor), batch_size=batch_size, shuffle=False)

    for (s_b,) in loader:
        s_b = s_b.to(device)
        B = s_b.shape[0]

        # Patch embedding
        x = model.to_patch_embedding(s_b)
        x = x + model.input_pos_emb.expand(B, -1, -1)
        x = torch.cat((model.act_token.expand(B, -1, -1), x), dim=1)

        for i, block in enumerate(model.encode):
            # Attention
            x1 = block.norm1(x)
            attn_out, _, _, _, _ = block.attn(x1)
            x = x + block.drop_path(attn_out)

            # MLP (manual, to capture hidden activations)
            x2 = block.norm2(x)
            h = block.mlp.fc1(x2)
            h = block.mlp.act(h)
            mlp_acc[i].append(h[:, 0, :].cpu().numpy())   # CLS token's MLP hidden
            h = block.mlp.drop1(h)
            h = block.mlp.fc2(h)
            h = block.mlp.drop2(h)
            x = x + block.drop_path(h)

            cls_acc[i].append(x[:, 0, :].cpu().numpy())

        y = F.softmax(model.predhead(x[:, 0, :]), dim=-1)
        all_preds.append(y.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    cls_layers = [np.concatenate(c, axis=0) for c in cls_acc]
    mlp_hidden = [np.concatenate(m, axis=0) for m in mlp_acc]

    return preds, cls_layers, mlp_hidden


# ══════════════════════════════════════════════════════════════════════
# 4. Neuron–feature correlation analysis
# ══════════════════════════════════════════════════════════════════════

def neuron_feature_correlation(activations, feat_matrix, feat_names):
    """
    Compute Pearson r between each neuron's activation and each game feature.

    Parameters
    ----------
    activations : np.ndarray (N, n_neurons)
    feat_matrix : np.ndarray (N, n_features)
    feat_names  : list of str

    Returns
    -------
    r_matrix : (n_neurons, n_features)
    p_matrix : (n_neurons, n_features)
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
        for i in range(n_neurons):
            ai = activations[valid, i]
            if ai.std() < 1e-10 or fj_valid.std() < 1e-10:
                continue
            r, p = stats.pearsonr(ai, fj_valid)
            r_mat[i, j] = r
            p_mat[i, j] = p

    return r_mat, p_mat


# ══════════════════════════════════════════════════════════════════════
# 5. Behavioral clustering
# ══════════════════════════════════════════════════════════════════════

def behavioral_clustering(cls_features, labels, feat_dict, n_clusters=6):
    """
    K-means on CLS embeddings, then profile each cluster by game-state features.

    Returns
    -------
    dict with cluster_ids, centers, profiles (per-cluster feature means and direction dist)
    """
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(cls_features)

    profiles = {}
    for c in range(n_clusters):
        mask = cluster_ids == c
        n_c = mask.sum()

        # Direction distribution
        dir_counts = np.bincount(labels[mask], minlength=4)

        # Game-state feature means
        feat_means = {}
        for fname, fvals in feat_dict.items():
            valid = ~np.isnan(fvals[mask])
            if valid.sum() > 0:
                feat_means[fname] = float(np.nanmean(fvals[mask]))
            else:
                feat_means[fname] = np.nan

        profiles[c] = {
            'count': int(n_c),
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


# ══════════════════════════════════════════════════════════════════════
# 6. UMAP
# ══════════════════════════════════════════════════════════════════════

def compute_umap(features, n_neighbors=15, min_dist=0.1):
    try:
        import umap
    except ImportError:
        print('  [WARN] umap-learn not installed, skipping UMAP.')
        return None
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=42)
    return reducer.fit_transform(features)


# ══════════════════════════════════════════════════════════════════════
# 7. Visualization
# ══════════════════════════════════════════════════════════════════════

def plot_results(results, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(out_dir, exist_ok=True)
    colors_4dir = {'down': '#e74c3c', 'left': '#3498db',
                   'right': '#2ecc71', 'up': '#f39c12'}

    # ── 7a. Neuron-feature correlation heatmap (per layer) ──
    for layer_idx in range(len(results['r_matrices'])):
        r_mat = results['r_matrices'][layer_idx]
        feat_names = results['feat_names']

        fig, ax = plt.subplots(figsize=(max(14, len(feat_names) * 0.35), 8))
        sns.heatmap(r_mat, cmap='RdBu_r', center=0, vmin=-0.5, vmax=0.5,
                    xticklabels=feat_names, yticklabels=[f'N{i}' for i in range(r_mat.shape[0])],
                    ax=ax, cbar_kws={'label': 'Pearson r'})
        ax.set_title(f'Layer {layer_idx+1} MLP Hidden Neurons vs Game Features')
        ax.set_xlabel('Game Feature')
        ax.set_ylabel('Neuron')
        plt.xticks(rotation=60, ha='right', fontsize=7)
        plt.yticks(fontsize=7)
        fig.savefig(os.path.join(out_dir, f'neuron_feature_corr_layer{layer_idx+1}.png'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved neuron_feature_corr_layer{layer_idx+1}.png')

    # ── 7b. Top correlated features per neuron ──
    for layer_idx in range(len(results['r_matrices'])):
        r_mat = results['r_matrices'][layer_idx]
        feat_names = results['feat_names']
        n_neurons = r_mat.shape[0]

        fig, ax = plt.subplots(figsize=(10, max(6, n_neurons * 0.4)))
        # For each neuron, find the feature with highest |r|
        max_r_idx = np.abs(r_mat).argmax(axis=1)
        max_r_vals = np.array([r_mat[i, max_r_idx[i]] for i in range(n_neurons)])
        max_r_names = [feat_names[j] for j in max_r_idx]

        colors = ['#e74c3c' if v < 0 else '#3498db' for v in max_r_vals]
        y_pos = np.arange(n_neurons)
        ax.barh(y_pos, max_r_vals, color=colors, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'N{i} ({max_r_names[i]})' for i in range(n_neurons)], fontsize=7)
        ax.set_xlabel('Pearson r')
        ax.set_title(f'Layer {layer_idx+1}: Strongest Feature per Neuron')
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.invert_yaxis()
        fig.savefig(os.path.join(out_dir, f'top_feature_per_neuron_layer{layer_idx+1}.png'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved top_feature_per_neuron_layer{layer_idx+1}.png')

    # ── 7c. Cluster profiles ──
    km_result = results['clustering']
    n_clusters = len(km_result['profiles'])

    # Direction distribution per cluster
    fig, axes = plt.subplots(1, n_clusters, figsize=(3.5 * n_clusters, 3.5), sharey=True)
    if n_clusters == 1:
        axes = [axes]
    for c in range(n_clusters):
        p = km_result['profiles'][c]
        dirs = ['down', 'left', 'right', 'up']
        counts = [p['dir_counts'][d] for d in dirs]
        axes[c].bar(dirs, counts, color=[colors_4dir[d] for d in dirs])
        axes[c].set_title(f'Cluster {c}\nn={p["count"]}, dom={p["dir_dominant"]}', fontsize=9)
    fig.suptitle('K-means Clusters: Direction Distribution', fontsize=12)
    fig.savefig(os.path.join(out_dir, 'cluster_direction_dist.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved cluster_direction_dist.png')

    # Key feature comparison across clusters
    key_features = [
        'g1_dist', 'g2_dist', 'max_threat', 'max_opportunity',
        'bean_total', 'nearest_bean_dist', 'nearest_energizer_dist',
        'beans_up', 'beans_down', 'beans_left', 'beans_right',
        'n_passable',
    ]
    available_keys = [k for k in key_features if k in results['feat_dict']]

    cluster_feat_matrix = np.zeros((n_clusters, len(available_keys)))
    for c in range(n_clusters):
        for j, k in enumerate(available_keys):
            cluster_feat_matrix[c, j] = km_result['profiles'][c]['feat_means'].get(k, np.nan)

    # Normalize columns for visualization
    scaler = StandardScaler()
    normed = scaler.fit_transform(np.nan_to_num(cluster_feat_matrix, nan=0))

    fig, ax = plt.subplots(figsize=(max(10, len(available_keys) * 0.6), 5))
    sns.heatmap(normed, cmap='RdYlGn', center=0,
                xticklabels=available_keys,
                yticklabels=[f'C{c} (n={km_result["profiles"][c]["count"]})' for c in range(n_clusters)],
                ax=ax, annot=np.round(cluster_feat_matrix, 2), fmt='',
                cbar_kws={'label': 'z-score'})
    ax.set_title('Cluster Game-State Profiles')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    fig.savefig(os.path.join(out_dir, 'cluster_feature_profiles.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved cluster_feature_profiles.png')

    # ── 7d. UMAP scatter ──
    if results.get('umap_embedding') is not None:
        emb = results['umap_embedding']
        labels = results['labels']

        # Color by direction
        fig, ax = plt.subplots(figsize=(8, 7))
        for d_idx, d_name in IDX2DIR.items():
            mask = labels == d_idx
            ax.scatter(emb[mask, 0], emb[mask, 1], s=3, alpha=0.3,
                       label=d_name, color=colors_4dir[d_name])
        ax.legend(markerscale=5, fontsize=10)
        ax.set_title('UMAP of CLS Embeddings (colored by direction)')
        fig.savefig(os.path.join(out_dir, 'umap_by_direction.png'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
        print('  Saved umap_by_direction.png')

        # Color by cluster
        fig, ax = plt.subplots(figsize=(8, 7))
        cluster_ids = km_result['cluster_ids']
        cmap = plt.get_cmap('tab10')
        for c in range(n_clusters):
            mask = cluster_ids == c
            ax.scatter(emb[mask, 0], emb[mask, 1], s=3, alpha=0.3,
                       label=f'C{c}', color=cmap(c))
        ax.legend(markerscale=5, fontsize=10)
        ax.set_title('UMAP of CLS Embeddings (colored by cluster)')
        fig.savefig(os.path.join(out_dir, 'umap_by_cluster.png'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
        print('  Saved umap_by_cluster.png')

        # Color by max_threat
        fig, ax = plt.subplots(figsize=(8, 7))
        threat = results['feat_dict']['max_threat']
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=3, alpha=0.3,
                        c=threat, cmap='YlOrRd', vmin=0)
        plt.colorbar(sc, ax=ax, label='Max Ghost Threat')
        ax.set_title('UMAP colored by Ghost Threat Level')
        fig.savefig(os.path.join(out_dir, 'umap_by_threat.png'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
        print('  Saved umap_by_threat.png')

    # ── 7e. CLS neuron correlation heatmap ──
    for layer_idx, cls_feat in enumerate(results['cls_layers']):
        corr = np.abs(np.corrcoef(cls_feat.T))
        corr = np.nan_to_num(corr, nan=0.0)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, cmap='viridis', square=True, ax=ax,
                    xticklabels=False, yticklabels=False)
        ax.set_title(f'Layer {layer_idx+1} CLS Neuron Correlation')
        fig.savefig(os.path.join(out_dir, f'cls_corr_layer{layer_idx+1}.png'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved cls_corr_layer{layer_idx+1}.png')


# ══════════════════════════════════════════════════════════════════════
# 8. Print summary
# ══════════════════════════════════════════════════════════════════════

def print_summary(results):
    """Print key findings to console."""
    print('\n' + '=' * 70)
    print('STRATEGY DECODING SUMMARY')
    print('=' * 70)

    labels = results['labels']
    preds = results['preds'].argmax(axis=1)
    acc = (preds == labels).mean()
    print(f'\nModel accuracy on analyzed data: {acc:.4f}')
    print(f'Total samples: {len(labels):,}')

    # Top neuron-feature associations per layer
    for layer_idx, r_mat in enumerate(results['r_matrices']):
        feat_names = results['feat_names']
        print(f'\n-- Layer {layer_idx+1} MLP: Top neuron-feature associations --')
        for i in range(r_mat.shape[0]):
            top_j = np.abs(r_mat[i]).argmax()
            r_val = r_mat[i, top_j]
            if abs(r_val) > 0.1:
                print(f'  Neuron {i:2d} <-> {feat_names[top_j]:<25s}  r={r_val:+.3f}')

    # Cluster profiles
    km = results['clustering']
    print(f'\n-- Behavioral clusters (k={len(km["profiles"])}) --')
    for c, p in km['profiles'].items():
        dc = p['dir_counts']
        dom = p['dir_dominant']
        threat = p['feat_means'].get('max_threat', np.nan)
        opp = p['feat_means'].get('max_opportunity', np.nan)
        g1d = p['feat_means'].get('g1_dist', np.nan)
        beans = p['feat_means'].get('bean_total', np.nan)
        print(f'  Cluster {c}: n={p["count"]:5d} | dom={dom:>5s} | '
              f'D={dc["down"]:4d} L={dc["left"]:4d} R={dc["right"]:4d} U={dc["up"]:4d} | '
              f'threat={threat:.3f} opp={opp:.3f} g1_dist={g1d:.1f} beans={beans:.0f}')

    print('\n' + '=' * 70)


# ══════════════════════════════════════════════════════════════════════
# 9. Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Decode strategies from FFN activations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', type=str,
                        default='./omega_models_fromcluster/140sess_2layers_focalloss_12d1h_noEmbLN_0768b.pkl',
                        help='Model checkpoint path')
    parser.add_argument('--data_dir', type=str,
                        default='./junction_predict_resortind',
                        help='Directory with session pickle files')
    parser.add_argument('--sessions', type=str, default='0-189',
                        help='Session range, e.g. "0-189" or "140-144"')
    parser.add_argument('--n_clusters', type=int, default=6,
                        help='Number of k-means clusters')
    parser.add_argument('--no_umap', action='store_true',
                        help='Skip UMAP (can be slow)')
    parser.add_argument('--out_dir', type=str, default='./strategy_results',
                        help='Output directory')
    args = parser.parse_args()

    # Parse session range
    parts = args.sessions.split('-')
    session_start, session_end = int(parts[0]), int(parts[1])
    session_indices = list(range(session_start, session_end + 1))

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load model ──
    print('Loading model...')
    model_path = os.path.join(_HERE, args.model)

    # Original models were saved with pickle.dump on CUDA; use custom unpickler
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
            return super().find_class(module, name)

    with open(model_path, 'rb') as f:
        _, state_dict, configs = CPU_Unpickler(f).load()

    configs['device'] = 'cpu'
    model = visual2action_3(configs).to('cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print(f'  Model: {os.path.basename(args.model)}')
    print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')

    # ── Load data + extract features (streaming) ──
    files = sorted(os.listdir(args.data_dir))
    print(f'\nLoading {len(session_indices)} sessions and extracting features...')

    all_features = {}
    all_labels = []
    all_preds = []
    cls_layers_acc = None
    mlp_hidden_acc = None

    for idx in tqdm(session_indices, desc='Sessions', ncols=80):
        if idx >= len(files):
            print(f'  [WARN] Session index {idx} out of range (only {len(files)} files), skipping.')
            continue
        path = os.path.join(args.data_dir, files[idx])
        s, a = load_session(path)

        # Extract game features
        feat = extract_game_features(s)

        # Forward pass
        s_tensor = torch.tensor(s)
        preds, cls_layers, mlp_hidden = extract_activations(model, s_tensor)

        # Accumulate
        all_labels.append(a)
        all_preds.append(preds)

        if cls_layers_acc is None:
            cls_layers_acc = [[] for _ in cls_layers]
            mlp_hidden_acc = [[] for _ in mlp_hidden]
        for i in range(len(cls_layers)):
            cls_layers_acc[i].append(cls_layers[i])
            mlp_hidden_acc[i].append(mlp_hidden[i])

        for fname, fvals in feat.items():
            if fname not in all_features:
                all_features[fname] = []
            all_features[fname].append(fvals)

        # Free memory
        del s, s_tensor, preds, cls_layers, mlp_hidden, feat

    # Concatenate
    print('\nConcatenating...')
    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    cls_layers_final = [np.concatenate(c, axis=0) for c in cls_layers_acc]
    mlp_hidden_final = [np.concatenate(m, axis=0) for m in mlp_hidden_acc]
    feat_dict = {k: np.concatenate(v) for k, v in all_features.items()}
    del all_labels, all_preds, cls_layers_acc, mlp_hidden_acc, all_features

    N = len(labels)
    print(f'Total samples: {N:,}')

    # ── Build feature matrix ──
    feat_names = sorted(feat_dict.keys())
    feat_matrix = np.column_stack([feat_dict[k] for k in feat_names])
    print(f'Game features: {len(feat_names)}')

    # ── Neuron-feature correlation ──
    print('\nComputing neuron-feature correlations...')
    r_matrices = []
    p_matrices = []
    for layer_idx, mlp_act in enumerate(mlp_hidden_final):
        print(f'  Layer {layer_idx+1}: {mlp_act.shape[1]} neurons x {len(feat_names)} features')
        r_mat, p_mat = neuron_feature_correlation(mlp_act, feat_matrix, feat_names)
        r_matrices.append(r_mat)
        p_matrices.append(p_mat)

    # ── Clustering ──
    print(f'\nK-means clustering (k={args.n_clusters}) on last-layer CLS...')
    km_result = behavioral_clustering(
        cls_layers_final[-1], labels, feat_dict, n_clusters=args.n_clusters)

    # ── UMAP ──
    umap_emb = None
    if not args.no_umap:
        print('\nComputing UMAP (this may take a few minutes)...')
        umap_emb = compute_umap(cls_layers_final[-1])

    # ── Collect results ──
    results = {
        'labels': labels,
        'preds': preds,
        'cls_layers': cls_layers_final,
        'mlp_hidden': mlp_hidden_final,
        'feat_dict': feat_dict,
        'feat_names': feat_names,
        'r_matrices': r_matrices,
        'p_matrices': p_matrices,
        'clustering': km_result,
        'umap_embedding': umap_emb,
    }

    # ── Print summary ──
    print_summary(results)

    # ── Save ──
    print('\nSaving results...')
    result_path = os.path.join(args.out_dir, 'strategy_results.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'  Results: {result_path}')

    # ── Plot ──
    print('\nGenerating figures...')
    plot_results(results, args.out_dir)

    print(f'\nDone! All outputs in {args.out_dir}/')


if __name__ == '__main__':
    main()
