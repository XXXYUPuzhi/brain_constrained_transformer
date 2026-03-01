# -*- coding: utf-8 -*-
"""
hidden_layer_output.py
======================
Extract intermediate hidden-layer representations from the trained ViT model.

Performs manual layer-by-layer forward pass to capture:
  - CLS token states after each encoder block
  - Attention scores (per head)
  - MLP hidden activations (post-GELU)
  - Top-K important tokens based on CLS attention weights

Also computes network topology metrics (modularity Q and hierarchy H) on the
extracted CLS representations using Louvain community detection.

Author: Puzhi YU
Date:   January 2026
"""

import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import community.community_louvain as community_louvain
from building_blocks import visual2action_3
import utils


# ────────────────────────────────────────────────────────────
# 1. Configuration
# ────────────────────────────────────────────────────────────

model_paths = './omega_models_fromcluster'
model_file = '140sess_2layers_focalloss_12d1h_noEmbLN_0768b.pkl'
game_data_path = './junction_predict_resortind'
device = 'cuda'


# ────────────────────────────────────────────────────────────
# 2. Load model
# ────────────────────────────────────────────────────────────

load_file = os.path.join(model_paths, model_file)
with open(load_file, 'rb') as file:
    _, state_dict, configs = pickle.load(file)

model = visual2action_3(configs).to(device)
model.load_state_dict(state_dict)
model.eval()


# ────────────────────────────────────────────────────────────
# 3. Load a session and prepare input
# ────────────────────────────────────────────────────────────

from attn_eye_correlation_analyses import prepare_data

fns_board = os.listdir(game_data_path)
eye_data_path = './eyecorrection_fromcluster/linear/'
fns_eye = os.listdir(eye_data_path)

i_eyefile = 5
fn = fns_eye[i_eyefile]
i_boardfile = np.where([fn[:16] in x for x in fns_board])[0][0]
game_info, action, board_inds, observation, _, _, s_, a_ = prepare_data(
    i_boardfile, '', game_data_path, device=device
)


# ────────────────────────────────────────────────────────────
# 4. Manual layer-by-layer forward pass
# ────────────────────────────────────────────────────────────

B, H, W, D = s_.shape

# Patch embedding with positional encoding
emb = model.to_patch_embedding(s_)
emb_with_pos = emb + model.input_pos_emb.expand(B, -1, -1)
emb_with_CLS = torch.cat((model.act_token.expand(B, -1, -1), emb_with_pos), dim=1)

# Layer 1: attention + MLP
emb_LN = model.encode[0].norm1(emb_with_CLS)
attn1, attn1_score, q1, k1, v1 = model.encode[0].attn(emb_LN)
attn1_res = attn1 + emb_with_CLS

attn1_res_LN = model.encode[0].norm2(attn1_res)
mlp1 = model.encode[0].mlp(attn1_res_LN)
mlp1_res = attn1_res + mlp1

# Layer 2: attention + MLP
mlp1_res_LN = model.encode[1].norm1(mlp1_res)
attn2, attn2_score, q2, k2, v2 = model.encode[1].attn(mlp1_res_LN)
attn2_res = attn2 + mlp1_res

attn2_res_LN = model.encode[1].norm2(attn2_res)
mlp2 = model.encode[1].mlp(attn2_res_LN)
mlp2_res = attn2_res + mlp2

# Final prediction
y = model.predhead(mlp2_res[:, 0, :])
y = F.softmax(y, dim=-1)

# Extract CLS token representations from both layers
cls_state_layer1 = mlp1_res[:, 0, :]
cls_state_layer2 = mlp2_res[:, 0, :]
print(f"CLS Layer 2 shape: {cls_state_layer2.shape}")


# ────────────────────────────────────────────────────────────
# 5. Network topology metrics
# ────────────────────────────────────────────────────────────

def calculate_network_metrics(hidden_states):
    """
    Compute modularity Q and hierarchy H from neuron activation correlations.

    Parameters
    ----------
    hidden_states : torch.Tensor (batch_size, hidden_dim)
        CLS token representations (e.g. mlp2_res[:, 0, :])

    Returns
    -------
    q_score : float
        Louvain modularity, higher means more functionally modular
    hierarchy_score : float
        Degree-based heterogeneity, higher means more hierarchical
    """
    features = hidden_states.detach().cpu().numpy().T
    adj_matrix = np.abs(np.corrcoef(features))

    # Retain top-10% strongest correlations
    threshold = np.percentile(adj_matrix, 90)
    adj_matrix[adj_matrix < threshold] = 0

    G = nx.from_numpy_array(adj_matrix)
    partition = community_louvain.best_partition(G)
    q_score = community_louvain.modularity(partition, G)

    degrees = np.array([d for _, d in G.degree()])
    n = len(degrees)
    if n > 1:
        c_max = np.max(degrees)
        hierarchy_score = np.sum(c_max - degrees) / (n - 1)
        hierarchy_score /= (c_max if c_max > 0 else 1)
    else:
        hierarchy_score = 0

    return q_score, hierarchy_score


q, h = calculate_network_metrics(cls_state_layer2)
print(f"Modularity Q: {q:.4f}")
print(f"Hierarchy H:  {h:.4f}")


# ────────────────────────────────────────────────────────────
# 6. Visualization: correlation heatmap and community graph
# ────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_network_structure(hidden_states, save_prefix='layer_analysis'):
    """
    Generate correlation heatmap and community graph for neuron activations.

    Parameters
    ----------
    hidden_states : np.ndarray (batch_size, hidden_dim)
    save_prefix : str
        Prefix for saved figure filenames
    """
    features = hidden_states.T
    corr_matrix = np.corrcoef(features)
    corr_matrix = np.nan_to_num(corr_matrix)

    adj_matrix = np.abs(corr_matrix)
    threshold = np.percentile(adj_matrix, 70)
    adj_matrix[adj_matrix < threshold] = 0
    np.fill_diagonal(adj_matrix, 0)

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.abs(corr_matrix), cmap="viridis",
                xticklabels=False, yticklabels=False, square=True,
                cbar_kws={"label": "Correlation Strength"})
    plt.title("Neuron Correlation Matrix (Heatmap)")
    plt.xlabel("Neuron Index")
    plt.ylabel("Neuron Index")
    plt.savefig(f'{save_prefix}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Community graph
    G = nx.from_numpy_array(adj_matrix)
    pos = nx.spring_layout(G, k=0.3, seed=42)
    partition = community_louvain.best_partition(G)

    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('tab10')
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=500,
                           cmap=cmap, node_color=list(partition.values()),
                           alpha=0.9)
    edges = G.edges(data=True)
    weights = [d['weight'] * 2 for (u, v, d) in edges]
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=weights, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='white')
    plt.title("Functional Modularity Graph")
    plt.axis('off')
    plt.savefig(f'{save_prefix}_graph.png', dpi=300, bbox_inches='tight')
    plt.show()


visualize_network_structure(
    cls_state_layer2.detach().cpu().numpy(),
    save_prefix='egtoken_analysis'
)


# ────────────────────────────────────────────────────────────
# 7. Top-K important tokens via attention
# ────────────────────────────────────────────────────────────

def get_topk_important_features(attn_score, hidden_states, k=5):
    """
    Identify the k most-attended tokens based on the CLS token's attention
    weights and extract their feature vectors from the hidden layer.

    Parameters
    ----------
    attn_score : torch.Tensor
        Attention weights, shape (B, H, L, L) or (B, L, L)
    hidden_states : torch.Tensor
        Hidden representations, shape (B, L, D)
    k : int
        Number of top tokens to extract

    Returns
    -------
    important_features : torch.Tensor (B, k, D)
    topk_indices : torch.Tensor (B, k)
    topk_vals : torch.Tensor (B, k)
    """
    # Average across heads if multi-head
    if attn_score.dim() == 4:
        attn_avg = attn_score.mean(dim=1)
    else:
        attn_avg = attn_score

    # CLS token's attention to all other tokens
    cls_attention = attn_avg[:, 0, :]
    cls_attention[:, 0] = -1e9  # mask self-attention

    topk_vals, topk_indices = torch.topk(cls_attention, k=k, dim=1)

    # Gather the corresponding feature vectors
    B_size, K = topk_indices.shape
    D_size = hidden_states.shape[-1]
    indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D_size)
    important_features = torch.gather(hidden_states, 1, indices_expanded)

    return important_features, topk_indices, topk_vals


TOP_K = 5
important_feats, important_inds, important_scores = get_topk_important_features(
    attn2_score, mlp2_res, k=TOP_K
)

print(f"\nTop-{TOP_K} important token features shape: {important_feats.shape}")
print(f"Top-{TOP_K} token indices shape: {important_inds.shape}")

sample_idx = 0
print(f"Sample {sample_idx} most-attended token indices: "
      f"{important_inds[sample_idx].cpu().numpy()}")
print(f"Corresponding attention scores: "
      f"{important_scores[sample_idx].detach().cpu().numpy()}")

# Average feature vector across the top-k tokens
important_feats_avg = important_feats.mean(dim=1)
print(f"Averaged important features shape: {important_feats_avg.shape}")
