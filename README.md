# NumPyGrad: Pure NumPy Autograd & Deep Learning Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Zero DL Frameworks](https://img.shields.io/badge/Dependencies-Pure%20NumPy-brightgreen.svg)](#)
[![Tests: 81/81 Passed](https://img.shields.io/badge/Tests-81%2F81%20Passed-success.svg)](#)

A modular, educational deep learning library and dynamic tensor autograd engine built completely from scratch using **pure Python and NumPy**—with **zero external deep learning framework dependencies** (no PyTorch, JAX, TensorFlow, or Keras).

---

## Visual Showcase: Non-Linear Decision Boundary

NumPyGrad easily trains deep multi-layer perceptrons to resolve complex non-linear classification manifolds (e.g. Two Moons, Concentric Circles, Spirals) via topological reverse-mode backpropagation and AdamW optimization:

<p align="center">
  <img src="examples/decision_boundary.png" alt="NumPyGrad Decision Boundary & Convergence Curves" width="850"/>
</p>

---

## Key Highlights

- **Dynamic Computational DAG:** Reverse-mode automatic differentiation with topological DAG ordering and recursive gradient accumulation.
- **Broadcasting-Aware Calculus:** Exact unbroadcasting logic (`sum_to_shape`) reducing gradients across arbitrary batch and feature dimensions.
- **PyTorch-Style Modular API:** Modular `Module`, `Parameter`, `Linear`, `BatchNorm1d`, `Dropout`, `Sequential`, and activations (`ReLU`, `Tanh`, `Sigmoid`, `GELU`).
- **Numerically Stable Loss Criteria:** `CrossEntropyLoss` with the Log-Sum-Exp trick, `MSELoss`, and `BCEWithLogitsLoss`.
- **First-Order Optimizers:** Vectorized `SGD` (with Polyak momentum & Nesterov) and `AdamW` with decoupled weight decay and bias correction.
- **Mathematical Rigor:** 100% test coverage with centered finite-difference numerical gradient checks (`gradcheck` relative error $< 10^{-5}$).

---

## Quickstart Example

Train a 2-layer Multi-Layer Perceptron on 2D coordinates in under 20 lines of pure NumPyGrad:

```python
import numpy as np
from numpygrad.core.tensor import Tensor
import numpygrad.nn as nn
import numpygrad.optim as optim

# 1. Define Model Architecture
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=2)
)

# 2. Setup Loss Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)

# 3. Create Sample Batch
X = Tensor(np.random.randn(64, 2).astype(np.float32))
y = np.random.choice([0, 1], size=(64,))

# 4. Forward -> Backward -> Optimize Step
optimizer.zero_grad()
logits = model(X)
loss = criterion(logits, y)
loss.backward()
optimizer.step()

print(f"Step Loss: {loss.data:.4f}")
```

---

## Mathematical Core & Architecture

NumPyGrad implements clean, mathematically verified calculus primitives:

### 1. Reverse-Mode Topological Backpropagation
Each operation creates a dynamic node tracking its predecessor inputs (`_prev`) and local Jacobian-vector product closure (`_backward`):
$$\frac{\partial \mathcal{L}}{\partial X} = \sum_{Y \in \text{Children}(X)} \frac{\partial \mathcal{L}}{\partial Y} \frac{\partial Y}{\partial X}$$
A depth-first topological traversal resolves variable dependencies before propagating upstream gradients, ensuring correct accumulation (`grad += ...`) across multi-branch DAGs.

### 2. Broadcasting Reduction (`_unbroadcast`)
When binary operations broadcast operands across differing shapes (e.g. $(N, D) + (D,)$), the backward pass projects accumulated gradients back to the original operand shape by summing over broadcasted leading and singleton axes.

### 3. Numerical Stability (Log-Sum-Exp Trick)
Softmax and Cross-Entropy compute the Log-Sum-Exp reduction over logits $z$:
$$\text{LSE}(z) = \max_j(z_j) + \log\left(\sum_j \exp\left(z_j - \max_k(z_k)\right)\right)$$
This prevents exponential floating-point overflow and underflow in single-precision float32 arithmetic.

---

## Interactive Studio (Streamlit Web Dashboard)

NumPyGrad includes an interactive real-time visualizer studio powered by Streamlit to experiment with dataset topologies, neural network architectures, and optimizer dynamics.

### Launching the Studio
```bash
py -3.13 -m streamlit run app/app.py
```
*Or with standard Streamlit:*
```bash
streamlit run app/app.py
```

### Studio Features
- **Live 2D Decision Boundaries:** Real-time contour surfaces for Two Moons, Concentric Circles, and Spirals.
- **Dynamic Architecture Control:** Configurable hidden layers, dimensions, and activation functions (`ReLU`, `Tanh`, `Sigmoid`, `GELU`).
- **Real-Time Training Metrics:** Synchronized loss and accuracy convergence tracking.
- **Gradient Diagnostics:** Post-training layer-by-layer gradient norm ($\|\nabla_\theta\|_2$) bar charts to inspect backpropagation dynamics across depths.

---

## CPU Benchmarking (NumPyGrad vs. PyTorch)

To evaluate CPU forward/backward throughput against standard PyTorch (CPU):

```bash
py -3.13 benchmarks/benchmark_cpu.py
```

---

## Running Automated Tests & Gradient Checks

Run the complete test suite with 100% `gradcheck` finite-difference numerical validation:
```bash
pytest -v
```
