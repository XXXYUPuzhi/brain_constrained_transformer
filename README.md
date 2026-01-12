# Brain-Constrained Transformer (Research Vision)

# Neuro-Nav Transformer: Primate Decision Modeling in Pacman

[![Status](https://img.shields.io/badge/Status-Work_in_Progress-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()

## 📖 Overview

This repository contains the implementation of a **Transformer-based baseline model** designed to decode and predict primate movement decisions in a Pacman maze task. By leveraging a Vision Transformer (ViT) architecture, we aim to capture global spatial context and decision strategies from game frames.

Beyond behavioral prediction, this project serves as a testbed for investigating the alignment between Artificial Neural Networks (ANNs) and biological neural activity (specifically in the Premotor cortex and FEF), bridging the gap between AI interpretability and neuroscience.

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
We performed RSA between the **MLP2 layer of the Transformer’s CLS token** and the **premotor region** of the biological brain using cosine similarity.

![RSA Analysis Result]



*Figure 1: Representational Dissimilarity Matrices (RDMs) showing the alignment between model internal states and biological neural activity.*

### 2. Linear Decoding & Encoding Models
Different token embeddings were trained to predict activity within specific brain areas.
* **Data:** 4,428 neurons (selected from 7,495 based on firing rate stability) across a 2.3-second window centered on joystick movement.
* **Metric:** Performance is quantified as $MSE - MSE_{shuffle}$ (significance $p < 0.05$).
* **Finding:** The **CLS token** exhibits significantly higher accuracy when predicting neurons in the **Premotor cortex and Frontal Eye Fields (FEF)** compared to other tokens. This indicates that the CLS token effectively encodes the motor signal.

![Decoding Accuracy]


*Figure 2: Decoding performance of different model components over time relative to joystick action.*

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
