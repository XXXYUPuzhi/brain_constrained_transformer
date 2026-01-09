#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSA Analysis Tool (Single File Version)
=======================================
A tool for performing Representational Similarity Analysis (RSA) on neural or model data.
It calculates Representational Dissimilarity Matrices (RDMs) and compares them between
two datasets (e.g., Biological vs. Artificial Neural Networks).

Features:
1. Loads neural/model data in .npy format.
2. Calculates RDMs (supports Correlation, Euclidean, Cosine).
3. Statistically compares RDMs (Spearman, Pearson, Kendall).
4. Generates and saves visualization plots.

Usage:
    Command Line:
    python rsa_analysis.py --file1 path/to/brain.npy --file2 path/to/model.npy

    Or modify the `DEFAULT_CONFIG` dictionary below to set default paths.
"""

import argparse
import os
import sys
from typing import Dict, Tuple, Optional, List, Union

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# Configuration Area
# (Modify these paths if you want to run without command line arguments)
# ==========================================
DEFAULT_CONFIG = {
    'file1': 'data/brain_data.npy', 
    'file2': 'data/model_data.npy',
    'output_dir': './rsa_results',
    'metrics': ['correlation', 'euclidean', 'cosine']
}

# ==========================================
# Core Logic Class
# ==========================================

class RSAAnalyzer:
    """
    RSA Analyzer: Encapsulates data processing, RDM calculation, and visualization logic.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir)
                print(f"[INFO] Created output directory: {self.output_dir}")
            except OSError as e:
                print(f"[ERROR] Could not create directory {self.output_dir}: {e}")
                sys.exit(1)

    @staticmethod
    def load_data(file_path: str) -> np.ndarray:
        """Loads .npy data and performs basic validation."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            data = np.load(file_path, allow_pickle=True)
            # Handle pandas DataFrame or other objects if necessary
            if hasattr(data, 'values'): 
                data = data.values
            data = np.asarray(data)
            print(f"[INFO] Loaded data: {os.path.basename(file_path)} | Shape: {data.shape}")
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load file {file_path}: {e}")

    @staticmethod
    def flatten_data(data: np.ndarray) -> np.ndarray:
        """
        Flattens high-dimensional data into (n_samples, n_features).
        Example: (Conditions, Time, Neurons) -> (Conditions, Time*Neurons)
        """
        if data.ndim < 2:
            raise ValueError("Data dimension must be at least 2D (samples, features)")
        
        if data.ndim > 2:
            n_samples = data.shape[0]
            # Automatically flatten all subsequent dimensions
            return data.reshape(n_samples, -1)
        return data

    def calculate_rdm(self, data: np.ndarray, method: str = 'correlation') -> np.ndarray:
        """Calculates the Representational Dissimilarity Matrix (RDM)."""
        # 1. Preprocessing: Flatten
        flat_data = self.flatten_data(data)
        
        # 2. Calculate Distance/Dissimilarity
        if method == 'correlation':
            # RDM = 1 - Pearson Correlation
            rdm = 1 - np.corrcoef(flat_data)
            
        elif method == 'euclidean':
            # Euclidean distance
            rdm = squareform(pdist(flat_data, metric='euclidean'))
            
        elif method == 'cosine':
            # Cosine distance = 1 - Cosine Similarity
            cos_sim = cosine_similarity(flat_data)
            rdm = 1 - cos_sim
            
        else:
            raise ValueError(f"Unsupported method: {method}")

        # 3. Clean up potential NaN/Inf values
        if not np.all(np.isfinite(rdm)):
            print(f"[WARN] NaN or Inf detected in RDM ({method}), replacing with 0.")
            rdm = np.nan_to_num(rdm)
            
        # Ensure diagonal is 0
        np.fill_diagonal(rdm, 0)
        return rdm

    @staticmethod
    def compare_rdms(rdm1: np.ndarray, rdm2: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculates statistical similarity metrics between two RDMs."""
        if rdm1.shape != rdm2.shape:
            raise ValueError(f"RDM shape mismatch: {rdm1.shape} vs {rdm2.shape}")

        # Extract upper triangular part (excluding diagonal) for comparison
        rows, cols = np.triu_indices(rdm1.shape[0], k=1)
        vec1 = rdm1[rows, cols]
        vec2 = rdm2[rows, cols]

        results = {}
        # Spearman (Recommended for RSA)
        rho, p_rho = stats.spearmanr(vec1, vec2)
        results['spearman'] = {'r': rho, 'p': p_rho}
        
        # Pearson
        r, p_r = stats.pearsonr(vec1, vec2)
        results['pearson'] = {'r': r, 'p': p_r}
        
        # Kendall Tau
        tau, p_tau = stats.kendalltau(vec1, vec2)
        results['kendall'] = {'r': tau, 'p': p_tau}
        
        return results

    def plot_comparison(self, rdm1: np.ndarray, rdm2: np.ndarray, 
                       method: str, stats_res: Dict, filename_suffix: str = ""):
        """Generates a side-by-side comparison plot of the two RDMs."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Get global max value for unified color scaling
        vmax = max(np.max(rdm1), np.max(rdm2))
        vmin = 0 
        
        # Plot 1
        im1 = axes[0].imshow(rdm1, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[0].set_title("Dataset 1 RDM")
        axes[0].set_xlabel("Conditions")
        axes[0].set_ylabel("Conditions")
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot 2
        im2 = axes[1].imshow(rdm2, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[1].set_title("Dataset 2 RDM")
        axes[1].set_xlabel("Conditions")
        axes[1].set_ylabel("Conditions")
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Add main title with statistical info
        title_text = (f"Metric: {method.upper()} | "
                      f"Spearman r={stats_res['spearman']['r']:.3f} | "
                      f"Pearson r={stats_res['pearson']['r']:.3f}")
        plt.suptitle(title_text, fontsize=14, y=0.98)
        
        plt.tight_layout()
        
        # Save figure
        save_name = f"compare_rdm_{method}{filename_suffix}.png"
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Plot saved to: {save_path}")
        # plt.show() # Uncomment if running locally with a display
        plt.close()

# ==========================================
# Main Execution
# ==========================================

def main():
    # 1. Define command line arguments
    parser = argparse.ArgumentParser(description="RSA RDM Comparison Tool")
    
    parser.add_argument('--file1', type=str, help='Path to first .npy file')
    parser.add_argument('--file2', type=str, help='Path to second .npy file')
    parser.add_argument('--output', type=str, help='Output directory path')
    
    args = parser.parse_args()

    # 2. Fallback logic (Prioritize CLI args, then DEFAULT_CONFIG)
    file1_path = args.file1 if args.file1 else DEFAULT_CONFIG['file1']
    file2_path = args.file2 if args.file2 else DEFAULT_CONFIG['file2']
    output_dir = args.output if args.output else DEFAULT_CONFIG['output_dir']
    
    # 3. Check if files exist
    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        print("\n[ERROR] Input files not found.")
        print(f"File 1: {file1_path} -> {'Found' if os.path.exists(file1_path) else 'Missing'}")
        print(f"File 2: {file2_path} -> {'Found' if os.path.exists(file2_path) else 'Missing'}")
        print("\nTip: Please run via command line or modify DEFAULT_CONFIG in the script.")
        print("Example: python rsa_analysis.py --file1 data/monkey.npy --file2 data/model.npy")
        return

    # 4. Initialize Analyzer
    analyzer = RSAAnalyzer(output_dir)

    try:
        # Load Data
        data1 = analyzer.load_data(file1_path)
        data2 = analyzer.load_data(file2_path)

        # Run analysis for all metrics
        methods = DEFAULT_CONFIG['metrics']
        
        print("\n=== Starting Analysis ===")
        for method in methods:
            print(f"\n[Processing] Metric: {method}")
            
            # Calculate RDM
            rdm1 = analyzer.calculate_rdm(data1, method=method)
            rdm2 = analyzer.calculate_rdm(data2, method=method)
            
            # Statistical Comparison
            stats_res = analyzer.compare_rdms(rdm1, rdm2)
            
            # Print Results
            print(f"  > Spearman Correlation: r = {stats_res['spearman']['r']:.4f} (p = {stats_res['spearman']['p']:.4e})")
            print(f"  > Pearson Correlation : r = {stats_res['pearson']['r']:.4f}")
            
            # Plot
            analyzer.plot_comparison(rdm1, rdm2, method, stats_res)

        print("\n=== Analysis Completed Successfully ===")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Execution interrupted: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()