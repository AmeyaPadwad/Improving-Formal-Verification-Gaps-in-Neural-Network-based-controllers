# Improving Formal Verification Gaps in Neural Network-Based Controllers

Course Instructor: **Prof. Michael Everett**  
Course Name: **Verifiable Machine Learning**  
Course Number: **EECE 7268 / CS 7268**  
_Master's Coursework – Northeastern University_

---

## 📘 Project Overview

While neural networks excel at control tasks in simulation, verifying their safety with mathematical guarantees remains a critical challenge. This project investigates the gap between **empirical safety** (what works in practice) and **formal verification** (what can be mathematically proven) for neural network controllers on dynamical systems.

We train compact neural network policies using imitation learning from optimal Linear Quadratic Regulator (LQR) experts on two classical control benchmarks: **Inverted Pendulum** and **CartPole**. We then evaluate their robustness both empirically and formally using advanced verification techniques (IBP, CROWN, α-CROWN), and analyze how architectural choices affect verifiability.

---

## 🎯 Objective

The goal of this project is to understand how design choices affect the formal verifiability of neural network-based controllers. Specifically, we aim to:

1. Train compact neural controllers for the Inverted Pendulum and CartPole systems using imitation learning
2. Evaluate controllers using multiple verification approaches: IBP, CROWN, α-CROWN, and Lipschitz-based bounds
3. Analyze the gap between empirical behavior and formal verification guarantees
4. Investigate techniques (smoother activations, pruning) that reduce this gap while preserving performance

---

## 🧪 Methodology

### 📐 System Models

**Inverted Pendulum:**

- 2D state space: [θ, θ̇] (angle, angular velocity)
- Dynamics: θ̈ = (g/L)sin(θ) + u/(mL²) - μθ̇
- Safe region: |θ| ≤ 0.5 rad, |θ̇| ≤ 1.0 rad/s
- Network architecture: 2 → 16 → 8 → 1

**CartPole:**

- 4D state space: [x, ẋ, θ, θ̇] (position, velocity, angle, angular velocity)
- Dynamics: Coupled nonlinear equations (Lagrangian mechanics)
- Safe region: |x| ≤ 2.4 m, |θ| ≤ 12°
- Network architecture: 4 → 32 → 16 → 1

### 🔬 Verification Pipeline

**1. Training Neural Networks:**

- Linearize each system around upright equilibrium
- Compute optimal LQR controller
- Generate state-action trajectories across safe region starting states
- Use as supervised dataset for neural network training via imitation learning

**2. Empirical Evaluation:**

- Simulate 1000 trajectories from random safe states
- Count trajectories remaining stable within safe bounds
- Measure empirical safety percentage

**3. Formal Verification:**

- Use IBP (Interval Bound Propagation) for baseline bounds
- Apply CROWN for tighter relaxation-based bounds
- Use α-CROWN for improved bounds with better efficiency
- Compute Lipschitz-based sensitivity bounds
- Propagate bounds through dynamics and track constraint satisfaction
- Measure formal safety percentage with mathematical guarantees

### ⚙️ Improvement Strategies

**Smoother Activation Functions:**

- ReLU: Piecewise linear, efficient for verification but potentially loose upper bounds
- Tanh: Bounded output [-1,1] provides implicit regularization but more conservative lower bounds

**Magnitude Pruning:**

- Remove weights below threshold to reduce network complexity
- Decreases perturbation propagation through layers
- Expected to tighten bounds while preserving accuracy

**Activation Pruning:**

- Remove consistently inactive neurons across training data
- More effective than magnitude pruning at reducing redundancy
- Substantially sparsifies network structure

---

## 📈 Key Results

### Activation Function Comparison

| System   | Activation | IBP         | CROWN       | α-CROWN    | Lipschitz   |
| -------- | ---------- | ----------- | ----------- | ---------- | ----------- |
| Pendulum | ReLU       | **Tighter** | **Tighter** | Comparable | **Tighter** |
| Pendulum | Tanh       | Looser      | Looser      | Comparable | Looser      |
| CartPole | ReLU       | **Tighter** | **Tighter** | Comparable | **Tighter** |
| CartPole | Tanh       | Looser      | Looser      | Comparable | Looser      |

**Finding:** ReLU consistently produces tighter bounds for CartPole systems, though Tanh's provided better results for Pendulum.

### Pruning Effects

**Pendulum System:**

- ReLU networks show pronounced improvements with both pruning strategies
- Lipschitz bounds decline sharply: 6.4 → 4.2 (magnitude pruning)
- IBP bounds improve: 3.2 → 2.0
- Magnitude pruning outperforms activation pruning

**CartPole System:**

- Both ReLU and Tanh show remarkable resistance to pruning
- Lipschitz bounds respond: 150 → 100 (only notable improvement)
- Relaxation-based methods (IBP, CROWN, α-CROWN) remain flat
- Problem complexity dominates over network compression effects

### Optimal Configurations

**Pendulum (Best Balance):**

- Activation: Tanh
- Pruning: Magnitude (0.5 sparsity)
- Accuracy: 90%
- α-CROWN Bound Width: 1.768

**CartPole (Best Balance):**

- Activation: ReLU
- Pruning: Magnitude (0.5 sparsity)
- Accuracy: 91%
- α-CROWN Bound Width: 3.444

**Key Insight:** Moderate pruning (0.3-0.5 sparsity) achieves near-optimal bounds without substantial accuracy loss.

### Computational Efficiency

Pruning dramatically accelerates verification:

- **Pendulum ReLU** α-CROWN: 5.079s (base) → 1.417s (0.3 sparsity) → 0.851s (0.7 sparsity)
- **CartPole ReLU** α-CROWN: 5.401s (base) → 1.348s (0.5 sparsity)
- Magnitude pruning consistently outperforms activation pruning for speedup

---

## 🔑 Key Findings

1. **Problem Complexity Determines Activation Function:** Tanh produces tighter bounds for Pendulum (simpler), while ReLU excels for CartPole (complex). System dimensionality and inherent complexity drive this trade-off.

2. **Moderate Pruning is Superior:** Magnitude pruning at 0.3-0.5 sparsity provides substantial verification improvements and computational speedup while maintaining high accuracy. Aggressive pruning (>0.7 sparsity) yields diminishing returns.

---

## 👥 Team Contributions

- **Ameya Padwad**
- **Risa Samanta**

---

## 📄 Report

See the [`Final Project Report`](./VML%20Project%20Report.pdf) for detailed analysis, comprehensive graphs, pruning sensitivity curves, computation times, and full experimental methodology.

---
