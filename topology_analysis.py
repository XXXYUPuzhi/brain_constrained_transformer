# -*- coding: utf-8 -*-
"""
Network Topology Analysis Toolkit for Neural Representations.

This module provides tools to quantify and visualize the functional topology 
of neural networks (specifically Transformer MLPs) using graph theory metrics.

Key Metrics:
1. Modularity (Q-Score): Measures functional specialization / clustering.
2. Hierarchy (H-Score): Measures influence heterogeneity / centralized control.

Author: yupuzhi
Date:   Jan 2026
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Handle the common confusion between 'community' and 'python-louvain'
try:
    import community.community_louvain as community_louvain
except ImportError:
    try:
        import community as community_louvain
    except ImportError:
        raise ImportError("Please install 'python-louvain' via pip: pip install python-louvain")

class TopologyAnalyzer:
    def __init__(self, output_dir='./analysis_results'):
        """
        Initialize the analyzer with an output directory for plots.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _build_adjacency_matrix(self, features, threshold_percentile=90):
        """
        Constructs a sparse adjacency matrix from neuron activation correlations.
        
        Args:
            features (np.ndarray): Shape (N_samples, N_neurons).
            threshold_percentile (int): Percentile to threshold weak connections.
            
        Returns:
            np.ndarray: Thresholded adjacency matrix (N_neurons, N_neurons).
        """
        # Transpose to correlate neurons (columns), not samples (rows)
        # Result shape: (N_neurons, N_neurons)
        corr_matrix = np.corrcoef(features.T)
        
        # Handle potential NaNs (e.g., dead neurons with 0 variance)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Use absolute correlation strength
        adj_matrix = np.abs(corr_matrix)
        
        # Apply thresholding to keep only strong functional connections
        threshold_val = np.percentile(adj_matrix, threshold_percentile)
        adj_matrix[adj_matrix < threshold_val] = 0
        
        # Remove self-loops
        np.fill_diagonal(adj_matrix, 0)
        
        return adj_matrix, corr_matrix

    def calculate_metrics(self, features, threshold_percentile=90):
        """
        Computes Modularity (Q) and Hierarchy (H) scores.
        
        Args:
            features (np.ndarray): Input hidden states (N_samples, N_neurons).
            
        Returns:
            dict: Dictionary containing 'modularity' and 'hierarchy' scores.
        """
        adj_matrix, _ = self._build_adjacency_matrix(features, threshold_percentile)
        
        # Build Graph
        G = nx.from_numpy_array(adj_matrix)
        
        # 1. Calculate Modularity (Q)
        # Using Louvain method for community detection
        try:
            partition = community_louvain.best_partition(G)
            q_score = community_louvain.modularity(partition, G)
        except ValueError:
            # Handle edge case where graph might be empty or fully disconnected
            q_score = 0.0

        # 2. Calculate Hierarchy (H)
        # Based on degree heterogeneity (proxy for influence reachability)
        degrees = np.array([d for _, d in G.degree()])
        n_nodes = len(degrees)
        
        if n_nodes > 1:
            # H = (Sum(Max_degree - Degree_i)) / (N - 1)
            # Normalized by Max_degree to keep it between [0, 1] usually
            c_max = np.max(degrees)
            if c_max > 0:
                h_score = np.sum(c_max - degrees) / (n_nodes - 1)
                h_score = h_score / c_max 
            else:
                h_score = 0.0
        else:
            h_score = 0.0

        return {
            'modularity': q_score,
            'hierarchy': h_score
        }

    def plot_graph(self, features, label='layer', threshold_percentile=80):
        """
        Visualizes the functional modularity graph with community coloring.
        """
        adj_matrix, _ = self._build_adjacency_matrix(features, threshold_percentile)
        G = nx.from_numpy_array(adj_matrix)
        
        # Detect communities for coloring
        partition = community_louvain.best_partition(G)
        
        # Visualization setup
        plt.figure(figsize=(10, 8), dpi=150)
        
        # Spring layout usually gives the best "cluster" visualization
        pos = nx.spring_layout(G, k=0.5, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=800,
            cmap=plt.get_cmap('tab10'),
            node_color=list(partition.values()),
            alpha=0.9
        )
        
        # Draw edges (thickness based on weight)
        edges = G.edges(data=True)
        if edges:
            weights = [d['weight'] * 2 for (u, v, d) in edges]
            nx.draw_networkx_edges(
                G, pos,
                alpha=0.3,
                width=weights,
                edge_color='gray'
            )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_color='white', font_weight='bold')
        
        title = f"Functional Modularity Graph - {label}"
        plt.title(title, fontsize=14)
        plt.axis('off')
        
        save_path = os.path.join(self.output_dir, f"{label}_graph.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[Info] Graph saved to: {save_path}")

    def plot_heatmap(self, features, label='layer'):
        """
        Visualizes the raw correlation matrix as a heatmap.
        """
        _, corr_matrix = self._build_adjacency_matrix(features, threshold_percentile=0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            np.abs(corr_matrix),
            cmap="viridis",
            square=True,
            cbar_kws={"label": "|Correlation|"}
        )
        plt.title(f"Neuron Correlation Matrix - {label}")
        plt.xlabel("Neuron Index")
        plt.ylabel("Neuron Index")
        
        save_path = os.path.join(self.output_dir, f"{label}_heatmap.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[Info] Heatmap saved to: {save_path}")

# =============================================================================
# Usage Example
# =============================================================================
if __name__ == "__main__":
    # Mock data generation for testing purposes
    # Replace this with your actual tensor data: mlp2_res[:, 0, :].detach().cpu().numpy()
    
    print("Running topology analysis on mock data...")
    
    # Simulate 1246 samples, 12 neurons
    np.random.seed(42)
    mock_features = np.random.rand(1246, 12) 
    
    # Initialize analyzer
    analyzer = TopologyAnalyzer(output_dir='./results_test')
    
    # 1. Calculate Metrics
    metrics = analyzer.calculate_metrics(mock_features, threshold_percentile=90)
    print(f"\n--- Metrics Results ---")
    print(f"Modularity (Q): {metrics['modularity']:.4f}")
    print(f"Hierarchy (H):  {metrics['hierarchy']:.4f}")
    
    # 2. Generate Plots
    analyzer.plot_heatmap(mock_features, label='mock_layer_cls')
    analyzer.plot_graph(mock_features, label='mock_layer_cls', threshold_percentile=80)
    
    print("\nDone.")