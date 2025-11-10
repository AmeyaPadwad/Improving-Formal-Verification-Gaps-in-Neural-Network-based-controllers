# Identification of Formal Verification Gaps in Neural Network-Based Controllers

Course Instructor: Prof. Michael Everett  
Course Name: Verifiable Machine Learning  
Course Number: EECE 7268 / CS 7268  
Master’s Coursework – Northeastern University

## Overview

This project investigates the critical gap between **empirical safety** (what works in simulation) and **formally verified safety** (what can be mathematically proven) for neural network controllers on dynamical systems. We train neural network policies on two classic control benchmarks (Inverted Pendulum and CartPole), evaluate their robustness empirically and formally using CROWN/β-CROWN verification, and analyze where and why formal verification fails.

---

## Methodology

### System Models

**Inverted Pendulum:**

- 2D state space: [θ, θ̇] (angle, angular velocity)
- Dynamics: θ̈ = (g/L)sin(θ) + u/(mL²) - μθ̇
- Safe region: |θ| ≤ 0.5 rad, |θ̇| ≤ 1.0 rad/s

**CartPole:**

- 4D state space: [x, ẋ, θ, θ̇] (position, velocity, angle, angular velocity)
- Dynamics: Coupled nonlinear equations (Lagrangian mechanics)
- Safe region: |x| ≤ 2.4 m, |θ| ≤ 0.209 rad

### Verification Pipeline

1. **Empirical Evaluation:**

   - Simulate 1000 trajectories from random safe states
   - Count trajectories that remain stable (don't leave safe region)
   - Measure empirical safety percentage

2. **Formal Verification:**

   - Use CROWN to compute guaranteed bounds on NN outputs under ε-bounded perturbations
   - Propagate bounds through dynamics via interval arithmetic
   - Check if trajectories stay within safe bounds (mathematical proof)
   - Measure formal safety percentage

3. **Gap Analysis:**
   - Compare empirical vs. formal: Gap = Empirical% - Formal%
   - Classify gap states into failure modes
   - Measure CROWN bound looseness

### Improvement Strategies

1. **Smooth Activations:** Replace ReLU with GELU for smoother decision boundaries
2. **Lipschitz Regularization:** Add gradient penalty during training to reduce Lipschitz constant
3. **TRADES Training:** Minimize both clean loss and robust loss (adversarial training)
4. **Pruning:** Remove redundant neurons to reduce network complexity

---
