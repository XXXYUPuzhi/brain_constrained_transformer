# -*- coding: utf-8 -*-
"""
analyze_resource_limited.py
===========================
Post-training analysis for resource-limited transformer models.

Given a trained model checkpoint (.pkl), this script performs:
  1. Hidden-state extraction (CLS token at each layer)
  2. Structural metrics (Louvain modularity Q, hierarchy H, weight sparsity)
  3. Strategy decoding:
     a) K-means clustering on CLS embeddings + game-state statistics
     b) NMF on MLP activations (parts-based decomposition)
     c) Input channel ablation (which game elements matter?)
     d) UMAP visualization colored by direction
  4. Generates analysis figures

Usage:
  python analyze_resource_limited.py --model results/140sess_2layers_..._earlystop.pkl
  python analyze_resource_limited.py --model_dir results/   # batch mode

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
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder
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

# Reuse data loading from training script
train_mod = _load_module('train_resource_limited',
                         os.path.join(_HERE, 'train_resource_limited.py'))
load_session = train_mod.load_session
load_dataset = train_mod.load_dataset
TopKMlp = train_mod.TopKMlp

_label_enc = LabelEncoder()
_label_enc.fit(['down', 'left', 'right', 'up'])
IDX2DIR = {0: 'down', 1: 'left', 2: 'right', 3: 'up'}


# ──────────────────────────────────────────────────────────────
# 1. Hidden-layer extraction (CLS token at each layer)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_hidden_states(model, s, a, device, batch_size=2048):
    """
    Extract CLS token hidden states after each transformer layer.

    Returns
    -------
    preds     : np.ndarray (N, 4)  softmax predictions
    layer_cls : list of np.ndarray, each (N, dim)
                layer_cls[i] = CLS token output after layer i (post-MLP residual)
    mlp_acts  : list of np.ndarray, each (N, hidden_dim)
                MLP intermediate activations (post-GELU, pre-fc2) for CLS token
    """
    model.eval()
    loader = DataLoader(TensorDataset(s, a), batch_size=batch_size, shuffle=False)

    all_preds = []
    # Accumulate per-layer CLS states
    n_layers = len(model.encode)
    layer_cls_acc = [[] for _ in range(n_layers)]
    mlp_act_acc = [[] for _ in range(n_layers)]

    for s_b, a_b in tqdm(loader, desc='Extracting hidden states', ncols=80):
        s_b = s_b.to(device)
        B = s_b.shape[0]

        # Patch embedding
        x = model.to_patch_embedding(s_b)
        x = x + model.input_pos_emb.expand(B, -1, -1)
        act_token = model.act_token.expand(B, -1, -1)
        x = torch.cat((act_token, x), dim=1)

        # Forward through each block manually
        for i, block in enumerate(model.encode):
            # Pre-norm attention
            x1 = block.norm1(x)
            attn_out, _, _, _, _ = block.attn(x1)
            x = x + block.drop_path(attn_out)

            # Pre-norm MLP
            x2 = block.norm2(x)

            # Capture MLP intermediate activation
            mlp_module = block.mlp
            if isinstance(mlp_module, TopKMlp):
                h = mlp_module.fc1(x2)
                h = mlp_module.act(h)
                mlp_act_acc[i].append(h[:, 0, :].cpu().numpy())
                if mlp_module.k is not None and mlp_module.k < h.shape[-1]:
                    topk_idx = torch.topk(h.abs(), k=mlp_module.k, dim=-1).indices
                    mask = torch.zeros_like(h).scatter_(-1, topk_idx, 1.0)
                    h = h * mask
                h = mlp_module.drop1(h)
                h = mlp_module.fc2(h)
                h = mlp_module.drop2(h)
                mlp_out = h
            else:
                # timm Mlp: fc1 -> act -> drop1 -> fc2 -> drop2
                h = mlp_module.fc1(x2)
                h = mlp_module.act(h)
                mlp_act_acc[i].append(h[:, 0, :].cpu().numpy())
                h = mlp_module.drop1(h)
                h = mlp_module.fc2(h)
                h = mlp_module.drop2(h)
                mlp_out = h

            x = x + block.drop_path(mlp_out)
            layer_cls_acc[i].append(x[:, 0, :].cpu().numpy())

        # Prediction
        y = model.predhead(x[:, 0, :])
        y = F.softmax(y, dim=-1)
        all_preds.append(y.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    layer_cls = [np.concatenate(lc, axis=0) for lc in layer_cls_acc]
    mlp_acts = [np.concatenate(ma, axis=0) for ma in mlp_act_acc]

    return preds, layer_cls, mlp_acts


# ──────────────────────────────────────────────────────────────
# 2. Structural metrics
# ──────────────────────────────────────────────────────────────

def compute_modularity_hierarchy(features, threshold_pct=90):
    """
    Compute modularity Q (Louvain) and hierarchy score from hidden states.

    Parameters
    ----------
    features : np.ndarray (N, D)  — N samples, D neurons
    threshold_pct : int — percentile threshold for adjacency matrix

    Returns dict with keys: q_score, hierarchy_score, n_communities, partition
    """
    import networkx as nx
    import community.community_louvain as community_louvain

    # Correlation-based adjacency
    corr = np.abs(np.corrcoef(features.T))
    corr = np.nan_to_num(corr, nan=0.0)
    threshold = np.percentile(corr, threshold_pct)
    adj = corr.copy()
    adj[adj < threshold] = 0
    np.fill_diagonal(adj, 0)

    G = nx.from_numpy_array(adj)

    # Modularity
    partition = community_louvain.best_partition(G)
    q_score = community_louvain.modularity(partition, G)
    n_communities = len(set(partition.values()))

    # Hierarchy (degree-based heterogeneity)
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
        if 'encode' in name and 'mlp' in name and 'weight' in name:
            zeros += (p.abs() < threshold).sum().item()
            total += p.numel()
    return zeros / total if total > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# 3. Strategy decoding
# ──────────────────────────────────────────────────────────────

def strategy_kmeans(cls_features, labels, n_clusters=6):
    """
    K-means clustering on CLS embeddings.

    Returns
    -------
    dict with cluster_labels, cluster_centers, direction_distribution (per cluster)
    """
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(cls_features)

    # Direction distribution per cluster
    dir_dist = {}
    for c in range(n_clusters):
        mask = cluster_ids == c
        counts = np.bincount(labels[mask], minlength=4)
        dir_dist[c] = {
            'count': int(mask.sum()),
            'down': counts[0], 'left': counts[1],
            'right': counts[2], 'up': counts[3],
        }

    return {
        'cluster_labels': cluster_ids,
        'cluster_centers': km.cluster_centers_,
        'direction_distribution': dir_dist,
        'inertia': km.inertia_,
    }


def strategy_nmf(mlp_activations, n_components=8):
    """
    NMF decomposition of MLP activations (post-GELU, already non-negative).

    Returns W (samples x components), H (components x neurons)
    """
    # Clamp negatives (should be minimal after GELU but just in case)
    X = np.maximum(mlp_activations, 0)
    if X.sum() == 0:
        return None

    nmf = NMF(n_components=min(n_components, X.shape[1]),
              init='nndsvd', random_state=42, max_iter=500)
    W = nmf.fit_transform(X)
    H = nmf.components_

    return {'W': W, 'H': H, 'reconstruction_error': nmf.reconstruction_err_}


@torch.no_grad()
def strategy_channel_ablation(model, s, a, device, batch_size=4096):
    """
    Zero out each input channel and measure prediction confidence change.

    The 34 input channels correspond to 2 frames x 17 channels:
      Frame t: [wall, bean, energizer, fruit1-5, g1_normal, g1_eaten,
                g1_scared, g1_flash, g2_normal, g2_eaten, g2_scared,
                g2_flash, pacman]
      Frame t-1: same 17 channels

    Returns dict mapping channel_name → mean confidence drop (higher = more important)
    """
    channel_names = [
        'wall', 'bean', 'energizer', 'fruit1', 'fruit2', 'fruit3',
        'fruit4', 'fruit5', 'g1_normal', 'g1_eaten', 'g1_scared',
        'g1_flash', 'g2_normal', 'g2_eaten', 'g2_scared', 'g2_flash',
        'pacman'
    ]
    # Full names for 34 channels (2 frames)
    full_names = [f't0_{n}' for n in channel_names] + \
                 [f't1_{n}' for n in channel_names]

    model.eval()
    loader = DataLoader(TensorDataset(s, a), batch_size=batch_size, shuffle=False)

    # Baseline confidence
    all_conf = []
    for s_b, a_b in loader:
        s_b, a_b = s_b.to(device), a_b.to(device)
        y, _ = model((s_b, a_b))
        conf = y.max(dim=1).values
        all_conf.append(conf.cpu().numpy())
    baseline_conf = np.concatenate(all_conf).mean()

    # Per-channel ablation
    importance = {}
    for ch in range(34):
        all_conf_abl = []
        for s_b, a_b in loader:
            s_b_abl = s_b.clone()
            s_b_abl[:, :, :, ch] = 0.0
            s_b_abl, a_b = s_b_abl.to(device), a_b.to(device)
            y, _ = model((s_b_abl, a_b))
            conf = y.max(dim=1).values
            all_conf_abl.append(conf.cpu().numpy())
        abl_conf = np.concatenate(all_conf_abl).mean()
        importance[full_names[ch]] = float(baseline_conf - abl_conf)

    return {'baseline_confidence': baseline_conf, 'channel_importance': importance}


def strategy_umap(cls_features, labels, n_neighbors=15, min_dist=0.1):
    """UMAP 2D embedding of CLS features, colored by direction."""
    try:
        import umap
    except ImportError:
        print('  [WARN] umap-learn not installed, skipping UMAP.')
        return None

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=42)
    embedding = reducer.fit_transform(cls_features)
    return {'embedding': embedding, 'labels': labels}


# ──────────────────────────────────────────────────────────────
# 4. Visualization helpers
# ──────────────────────────────────────────────────────────────

def plot_all(results, out_dir, model_name):
    """Generate and save all analysis figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    prefix = os.path.join(fig_dir, model_name)

    colors_4dir = {'down': '#e74c3c', 'left': '#3498db',
                   'right': '#2ecc71', 'up': '#f39c12'}

    # --- 4a. Correlation heatmap per layer ---
    for i, cls in enumerate(results['layer_cls']):
        corr = np.abs(np.corrcoef(cls.T))
        corr = np.nan_to_num(corr, nan=0.0)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, cmap='viridis', square=True, cbar=True,
                    xticklabels=False, yticklabels=False, ax=ax)
        ax.set_title(f'Layer {i+1} Neuron Correlation')
        fig.savefig(f'{prefix}_layer{i+1}_corr.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # --- 4b. Channel ablation bar chart ---
    if 'channel_ablation' in results and results['channel_ablation'] is not None:
        imp = results['channel_ablation']['channel_importance']
        # Group by game element (aggregate across frames)
        element_imp = {}
        for ch_name, val in imp.items():
            elem = ch_name.split('_', 1)[1]  # strip t0_/t1_
            element_imp[elem] = element_imp.get(elem, 0) + val

        sorted_elems = sorted(element_imp.items(), key=lambda x: x[1], reverse=True)
        names, vals = zip(*sorted_elems)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(len(names)), vals, color='steelblue')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Confidence drop when ablated')
        ax.set_title('Channel Importance (ablation)')
        ax.invert_yaxis()
        fig.savefig(f'{prefix}_channel_ablation.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # --- 4c. UMAP scatter ---
    if 'umap' in results and results['umap'] is not None:
        emb = results['umap']['embedding']
        labs = results['umap']['labels']
        fig, ax = plt.subplots(figsize=(8, 7))
        for d_idx, d_name in IDX2DIR.items():
            mask = labs == d_idx
            ax.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.4,
                       label=d_name, color=colors_4dir[d_name])
        ax.legend(markerscale=5)
        ax.set_title('UMAP of CLS embeddings by direction')
        fig.savefig(f'{prefix}_umap.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # --- 4d. K-means cluster direction distribution ---
    if 'kmeans' in results:
        km = results['kmeans']
        n_c = len(km['direction_distribution'])
        fig, axes = plt.subplots(1, n_c, figsize=(3*n_c, 3), sharey=True)
        if n_c == 1:
            axes = [axes]
        for c in range(n_c):
            d = km['direction_distribution'][c]
            dirs = ['down', 'left', 'right', 'up']
            counts = [d[dn] for dn in dirs]
            axes[c].bar(dirs, counts, color=[colors_4dir[dn] for dn in dirs])
            axes[c].set_title(f'Cluster {c}\n(n={d["count"]})')
        fig.suptitle('K-means clusters: direction distribution')
        fig.savefig(f'{prefix}_kmeans.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # --- 4e. NMF basis heatmap ---
    if 'nmf' in results and results['nmf'] is not None:
        H = results['nmf']['H']
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(H, cmap='viridis', ax=ax,
                    xticklabels=False, yticklabels=[f'Comp {i}' for i in range(H.shape[0])])
        ax.set_xlabel('Neuron index')
        ax.set_title('NMF basis vectors (MLP activations)')
        fig.savefig(f'{prefix}_nmf.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f'  Figures saved to {fig_dir}/')


# ──────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────

def analyze_one_model(model_path, data_dir, device, out_dir,
                      n_kmeans=6, n_nmf=8, do_ablation=True, do_umap=True):
    """Full analysis pipeline for a single model."""

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f'\n{"="*60}')
    print(f'Analyzing: {model_name}')
    print(f'{"="*60}')

    # Load model
    with open(model_path, 'rb') as f:
        log, state_dict, configs = pickle.load(f)

    # Ensure device is set correctly
    configs['device'] = device
    model = visual2action_3(configs).to(device)

    # Handle TopK models
    if configs.get('regularization') == 'topk' and configs.get('topk_k') is not None:
        from train_resource_limited import apply_topk_to_model
        model = apply_topk_to_model(model, k=configs['topk_k'],
                                     dim=configs['patch_emb_dim'])

    model.load_state_dict(state_dict)
    model.eval()

    # Load test data
    test_indices = configs.get('test_files', [140, 141, 142, 143, 144])
    print(f'  Loading test data (sessions {test_indices})...')
    s_test, a_test, _ = load_dataset(data_dir, test_indices, desc='Test')
    print(f'  Test samples: {s_test.shape[0]:,}')

    # ── Extract hidden states
    print('  Extracting hidden states...')
    preds, layer_cls, mlp_acts = extract_hidden_states(
        model, s_test, a_test, device)

    pred_labels = preds.argmax(axis=1)
    true_labels = a_test.numpy()
    test_acc = (pred_labels == true_labels).mean()
    print(f'  Test accuracy: {test_acc:.4f}')

    # ── Structural metrics
    print('  Computing structural metrics...')
    struct_metrics = {}
    for i, cls_feat in enumerate(layer_cls):
        m = compute_modularity_hierarchy(cls_feat)
        struct_metrics[f'layer_{i+1}'] = {
            'q_score': m['q_score'],
            'hierarchy_score': m['hierarchy_score'],
            'n_communities': m['n_communities'],
        }
        print(f'    Layer {i+1}: Q={m["q_score"]:.4f}, '
              f'H={m["hierarchy_score"]:.4f}, '
              f'communities={m["n_communities"]}')

    sparsity = compute_weight_sparsity(model)
    struct_metrics['weight_sparsity'] = sparsity
    print(f'    Weight sparsity: {sparsity:.4%}')

    # ── Strategy decoding
    # Use last layer CLS for clustering/UMAP
    last_cls = layer_cls[-1]

    # K-means
    print(f'  K-means clustering (k={n_kmeans})...')
    km_results = strategy_kmeans(last_cls, true_labels, n_clusters=n_kmeans)
    for c, d in km_results['direction_distribution'].items():
        print(f'    Cluster {c}: n={d["count"]}, '
              f'D={d["down"]} L={d["left"]} R={d["right"]} U={d["up"]}')

    # NMF on last layer MLP activations
    print(f'  NMF (n_components={n_nmf})...')
    last_mlp_act = mlp_acts[-1]
    nmf_results = strategy_nmf(last_mlp_act, n_components=n_nmf)

    # Channel ablation
    ablation_results = None
    if do_ablation:
        print('  Channel ablation...')
        ablation_results = strategy_channel_ablation(
            model, s_test, a_test, device)
        # Print top-5 most important
        sorted_imp = sorted(ablation_results['channel_importance'].items(),
                            key=lambda x: x[1], reverse=True)
        print('    Top-5 important channels:')
        for name, val in sorted_imp[:5]:
            print(f'      {name}: {val:.4f}')

    # UMAP
    umap_results = None
    if do_umap:
        print('  UMAP embedding...')
        umap_results = strategy_umap(last_cls, true_labels)

    # ── Collect results
    results = {
        'model_name': model_name,
        'model_path': model_path,
        'configs': configs,
        'test_acc': test_acc,
        'structural_metrics': struct_metrics,
        'layer_cls': layer_cls,
        'mlp_acts': mlp_acts,
        'preds': preds,
        'true_labels': true_labels,
        'kmeans': km_results,
        'nmf': nmf_results,
        'channel_ablation': ablation_results,
        'umap': umap_results,
        'training_log': {
            'train_loss': getattr(log, 'train_loss', []),
            'test_loss': getattr(log, 'test_loss', []),
            'accuracy': getattr(log, 'accuracy', []),
        },
    }

    # ── Save results
    result_path = os.path.join(out_dir, f'{model_name}_analysis.pkl')
    os.makedirs(out_dir, exist_ok=True)
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'  Results saved: {result_path}')

    # ── Plot
    plot_all(results, out_dir, model_name)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Post-training analysis for resource-limited transformers')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to a single .pkl model file')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory of .pkl files to analyze in batch')
    parser.add_argument('--data_dir', type=str,
                        default='./junction_predict_resortind',
                        help='Training data directory')
    parser.add_argument('--out_dir', type=str, default='./analysis_results',
                        help='Output directory for results and figures')
    parser.add_argument('--n_kmeans', type=int, default=6,
                        help='Number of k-means clusters')
    parser.add_argument('--n_nmf', type=int, default=8,
                        help='Number of NMF components')
    parser.add_argument('--no_ablation', action='store_true',
                        help='Skip channel ablation (slow)')
    parser.add_argument('--no_umap', action='store_true',
                        help='Skip UMAP')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    if args.model:
        analyze_one_model(
            args.model, args.data_dir, device, args.out_dir,
            n_kmeans=args.n_kmeans, n_nmf=args.n_nmf,
            do_ablation=not args.no_ablation,
            do_umap=not args.no_umap,
        )
    elif args.model_dir:
        pkl_files = sorted([
            f for f in os.listdir(args.model_dir)
            if f.endswith('.pkl') and 'analysis' not in f
        ])
        print(f'Found {len(pkl_files)} model files in {args.model_dir}')

        summaries = []
        for pkl_f in pkl_files:
            path = os.path.join(args.model_dir, pkl_f)
            r = analyze_one_model(
                path, args.data_dir, device, args.out_dir,
                n_kmeans=args.n_kmeans, n_nmf=args.n_nmf,
                do_ablation=not args.no_ablation,
                do_umap=not args.no_umap,
            )
            summaries.append({
                'model': r['model_name'],
                'test_acc': r['test_acc'],
                'sparsity': r['structural_metrics']['weight_sparsity'],
                **{f'{k}_Q': v['q_score']
                   for k, v in r['structural_metrics'].items() if k.startswith('layer')},
                **{f'{k}_H': v['hierarchy_score']
                   for k, v in r['structural_metrics'].items() if k.startswith('layer')},
            })

        # Print summary table
        print(f'\n{"="*80}')
        print('SUMMARY')
        print(f'{"="*80}')
        header = f'{"Model":<50} {"Acc":>6} {"Spar%":>6}'
        for i in range(len(summaries[0]) - 3):
            pass
        print(header)
        for s in summaries:
            line = f'{s["model"]:<50} {s["test_acc"]:>6.4f} {s["sparsity"]:>5.1%}'
            print(line)

        # Save summary
        summary_path = os.path.join(args.out_dir, 'summary.pkl')
        with open(summary_path, 'wb') as f:
            pickle.dump(summaries, f)
        print(f'\nSummary saved: {summary_path}')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
