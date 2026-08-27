# NumPyGrad: Deep Learning & Autograd Engine from Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![No DL Frameworks](https://img.shields.io/badge/Dependencies-Pure%20NumPy-brightgreen.svg)](#)

A modular, educational deep learning library and dynamic tensor autograd engine built entirely from scratch using only Python and NumPy with zero external deep learning framework dependencies.

---

## ✨ Highlights
- **Dynamic Computational DAG:** Reverse-mode automatic differentiation with topological backpropagation.
- **Broadcasting-Aware Calculus:** Exact gradient accumulation across arbitrary batch and feature dimensions.
- **PyTorch-Style API:** Modular `nn.Module`, `Linear`, `BatchNorm1d`, `Dropout`, and `CrossEntropyLoss`.
- **Custom Optimizers:** Vectorized `SGD` (with Momentum & Nesterov) and `AdamW` with decoupled weight decay.
- **Strict Verification:** Built-in numerical gradient checker (`gradcheck`) using centered finite-difference approximations.

---

## 🚀 Interactive Studio (Streamlit Web App)

NumPyGrad includes an interactive real-time visualizer studio powered by Streamlit to experiment with dataset topologies, neural network architectures, and optimizer dynamics.

### Launching the Studio
```bash
# Set PYTHONPATH and launch Streamlit dashboard
streamlit run app/app.py
```

### Features
- **Live 2D Decision Boundaries:** Real-time contour surfaces for Two Moons, Concentric Circles, and Spirals.
- **Dynamic Architecture Control:** Configurable hidden layers, dimensions, and activation functions (`ReLU`, `Tanh`, `Sigmoid`, `GELU`).
- **Real-Time Training Metrics:** Synchronized loss and accuracy convergence tracking.
- **Gradient Diagnostics:** Post-training layer-by-layer gradient norm ($\|\nabla_\theta\|_2$) bar charts to inspect backpropagation dynamics.

---

## 📦 Quickstart Example

```python
import numpy as np
from numpygrad.core.tensor import Tensor
import numpygrad.nn as nn
import numpygrad.optim as optim

# 1. Define a Multi-Layer Perceptron
model = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

# 2. Setup Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)

# 3. Training Step
x = Tensor(np.random.randn(16, 2))
y = np.random.choice([0, 1], size=(16,))

optimizer.zero_grad()
logits = model(x)
loss = criterion(logits, y)
loss.backward()
optimizer.step()

print(f"Loss: {loss.data:.4f}")
```

---

## 🧪 Running Tests & Gradient Checks

Run the automated test suite with full numerical `gradcheck` validation:
```bash
pytest -v
```
