# Brain-Constrained Transformer (Research Vision)

[![Status](https://img.shields.io/badge/Status-Work_in_Progress-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()

## 📖 Overview

Current AI models predominantly scale performance by increasing parameters, whereas the biological brain accomplishes complex cognitive tasks under severe resource constraints. A key mechanism behind this biological efficiency is the decomposition of complex problems into **modular and hierarchical abstract structures**, which simultaneously conserves resources and enhances processing efficiency.

**This research poses a fundamental question:**
> *Can artificial neural networks (ANNs) spontaneously develop similar modular and hierarchical structures when subjected to analogous resource constraints?*

To answer this, this repository leverages three key experimental resources: a highly structured **Pac-Man paradigm**, an interpretable **Transformer baseline model**, and multi-region **electrophysiological recordings**.

### Key Research Objectives
1.  **Structural Emergence via Constraints:** We implement computationally economical constraint methods to promote structure, including **Differentiable L0 Regularization**, **Modified Mixture of Experts (MoE)**, and **Hierarchical Convergence Models (HCM)**.
2.  **Brain-AI Alignment:** We hypothesize that constrained models will develop internal representations more aligned with biological neural activity, verified via **Representational Similarity Analysis (RSA)** and **Single-Neuron Encoding Models**.
3.  **Bidirectional Interpretability:** By clustering the model's internal states, we aim to discover data-driven behavioral strategy modules that serve as novel labels to reinterpret primate behavior.

> **Note:** This project is currently under active development. The codebase is being refined to support advanced modularity and hierarchical processing.

---

## 🏗 Model Architecture

Our baseline model utilizes a pure attention-based architecture to process sequential game states.

### Input Representation
* **Input Data:** The model ingests the two game frames immediately preceding a junction decision.
* **Tensor Shape:** Each frame is represented as a tensor of size $32 \times 28 \times 17$ (maze height $\times$ width $\times$ feature channels).
* **Preprocessing:** Frames are divided into non-overlapping $2 \times 2$ patches.

### The Transformer Encoder
Following the Vision Transformer (ViT) design:
1.  **Patch Projection:** Each patch is flattened and projected into a $d$-dimensional embedding space ($d=48$).
2.  **Positional Encoding:** A 2D sinusoidal encoding is added to preserve the spatial layout of the maze.
3.  **CLS Token:** A learnable classification (CLS) token is prepended to the sequence to aggregate global context and produce the final action prediction.
4.  **Encoder Blocks:** We utilize **2 Transformer blocks**, each consisting of:
    * **Multi-Head Attention (2 heads):** Allows the model to learn multiple relational patterns in parallel.
    * **MLP Feedforward Layer:** Processes information locally.

### Performance
* **Current Baseline:** The model achieves **87.664% accuracy** on the test set, demonstrating strong predictive performance in capturing the decision strategy underlying the Pacman task.

---

## 🧠 Preliminary Neural Analysis

We rigorously evaluate the biological plausibility of our model using neural recording data from primates.

### 1. Representational Similarity Analysis (RSA)
I performed RSA between the **MLP2 layer of the Transformer’s CLS token** and the **premotor region** of the biological brain using cosine similarity.

![RSA Analysis Result](./images/RSA1.png)
*Figure 1: Representational Dissimilarity Matrices (RDMs) showing the alignment between model internal states and biological neural activity.*

### 2. Linear Decoding & Encoding Models
Different token embeddings were trained to predict activity within specific brain areas.
* **Data:** 4,428 neurons (selected from 7,495 based on firing rate stability) across a 2.3-second window centered on joystick movement.
* **Metric:** Performance is quantified as $MSE - MSE_{shuffle}$ (significance $p < 0.05$).
* **Finding:** The **CLS token** exhibits significantly higher accuracy when predicting neurons in the **Premotor cortex and Frontal Eye Fields (FEF)** compared to other tokens. This indicates that the CLS token effectively encodes the motor signal.

![Decoding Accuracy](./images/linear_encoder.png)
*Figure 2: Decoding performance of different model components over time relative to joystick action.*

---

## 🧠 Emergent Functional Structure: Modularity & Hierarchy

To investigate how the model processes game states from perception to decision-making, I analyzed the **functional network topology** of the Transformer's feed-forward layers. I specifically tracked the **CLS Token**, which serves as the information bottleneck for the final action prediction.

### 1. Methodology & Metrics

I construct a connectivity matrix $A$ based on the pairwise Pearson correlation of neuron activations. The network structure is evaluated using two key metrics from complex network theory:

* **Modularity ($Q$ Score):** Measures the degree of functional specialization into distinct neural clusters (modules).
    $$Q = \frac{1}{m}\sum_{ij}\left[A_{ij} - \frac{k_i^{in}k_j^{out}}{m}\right]\sigma_{ci,cj}$$
    *Where $A_{ij}$ is the connection weight, $k$ is the node degree, $m$ is the total edge weight, and $\sigma$ indicates if nodes $i,j$ belong to the same module.*

* **Hierarchy ($H$ Score):** Quantifies the recursive organization and influence heterogeneity of the network.
    $$H = \frac{\sum_{i \in V}[C_R^{max} - C_R(i)]}{N-1}$$
    *Where $C_R(i)$ represents the influence (reachability) of neuron $i$, indicating whether the layer relies on a few "hub" neurons or is democratically organized.*

### 2. Visualization: Topological Evolution

The graphs below illustrate the functional connectivity of the CLS token in Layer 1 vs. Layer 2. Nodes represent neurons, and edges represent strong functional correlations (Top-20%). Colors indicate detected community modules.

| **Layer 1: Information Integration** | **Layer 2: Decision Disentanglement** |
| :---: | :---: |
| ![Layer 1 Graph](./path/to/your/layer1_cls_graph.png) | ![Layer 2 Graph](./path/to/your/layer2_cls_graph.png) |
| *High connectivity with a **Single Super-Cluster**. The model integrates global spatial features into a unified context representation.* | *Structure disperses into **Three Distinct Modules**. The model disentangles the context into orthogonal decision factors (e.g., Combat, Territory, Safety).* |

### 3. Quantitative Analysis

| Metric | Layer 1 (CLS) | Layer 2 (CLS) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Modularity ($Q$)** | **0.8294** | **0.8274** | Consistently high modularity indicates the model maintains specialized "neural experts" throughout processing. |
| **Hierarchy ($H$)** | **0.4545** | **0.4545** | Stable hierarchical control suggests a consistent command structure where specific "hub" neurons coordinate information flow. |

> **Key Insight:** While the quantitative metrics ($Q$ and $H$) remain stable, the **topological structure** undergoes a dramatic shift. 
> 
> * **Layer 1** exhibits a "Hub-and-Spoke" architecture centered around a large core group, suggesting **Context Integration** (merging stone positions, liberties, and shapes).
> * **Layer 2** evolves into a "Multi-Polar" architecture with separated clusters. This suggests **Feature Orthogonalization**, where the model splits the integrated context into independent strategic components (e.g., *Attacking* vs. *Defending* vs. *Territory Expansion*) before the final linear projection to action space.

---

## 🚀 Roadmap & Future Work

We are currently extending the baseline with three major architectural improvements to test the hypothesis that **resource-constrained models develop more brain-like representations**.

### Phase 1: Differentiable L0 Regularization (Sparse Connectivity)
* **Objective:** Implement a differentiable relaxation of the L0 norm.
* **Goal:** Enable simultaneous optimization of model performance and parameter sparsity within the backpropagation framework.

### Phase 2: Modified Mixture of Experts (MoE) (Sparse Activation)
* **Objective:** Decompose the output layer into modular behavioral units.
* **Mechanism:** Inspired by Schrum & Miikkulainen, each module will have specific policy neurons and a gating unit to regulate information flow.

### Phase 3: Hierarchical Convergence Model (HCM)
* **Objective:** Introduce a dual-speed architecture to solve the "rapid convergence" issue.
    * **Low-level (L) module:** Updates rapidly to reach local equilibrium.
    * **High-level (H) module:** Updates slowly, guiding the L-module via recurrent feedback.
* **Expected Outcome:** Effective layered processing while maintaining global coherence.

### Hypothesis & Expected Impact
1.  **Structural Emergence:** Resource constraints (L0 + MoE + HCM) will induce modular and hierarchical structures automatically.
2.  **Brain-Alignment:** The constrained model will show significantly higher RSA scores with neural data compared to unconstrained models.
3.  **Behavioral Discovery:** Clustering internal states will reveal data-driven "strategy modules," offering novel labels for interpreting primate behavior.

---

## 🛠 Usage

*(Code usage instructions are currently being updated)*

---
*Created by Puzhi Yu*
