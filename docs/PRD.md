# Product Requirements Document (PRD): NumPyGrad

## 1. Executive Summary & Objective

**NumPyGrad** is a lightweight, pure Python and NumPy automatic differentiation and deep learning framework built entirely from scratch with zero external deep learning framework dependencies (no PyTorch, JAX, TensorFlow, or Keras). 

The primary objective is to build an educational yet mathematically rigorous tensor library featuring:
- A dynamic computation graph (DAG) evaluated via reverse-mode automatic differentiation (topological backpropagation).
- Broadcasting-aware analytical gradient accumulation across arbitrary tensor shapes and dimensions.
- A PyTorch-inspired object-oriented API for neural network modules, parameters, layers, activation functions, loss criteria, and optimizers.
- Numerical gradient verification (`gradcheck`) using centered finite differences to guarantee mathematical precision.
- Interactive visualization tools for inspecting dynamic computational graphs.

---

## 2. Reference Architectures & Inspiration

NumPyGrad synthesizes architectural paradigms from top-tier educational and minimalist autograd systems:

1. **`karpathy/micrograd`**:
   - Scalar autograd engine paradigm: explicit DAG tracking via parent references, dynamic closure-based `_backward()` hooks, and recursive/topological graph traversal.
   - Scaled up in NumPyGrad from scalar arithmetic to multidimensional ndarray tensors.

2. **`tinygrad` (George Hotz)**:
   - Clean separation of Tensor operations, context tracking, and lazy/eager evaluation abstractions.
   - Elegant broadcasting reduction logic to accumulate gradients across broadcasted axes.

3. **Stanford CS231n (Convolutional Neural Networks for Visual Recognition)**:
   - Modular layer design (`forward` and `backward` interfaces), numerical stability techniques (Log-Sum-Exp trick for Softmax/Cross-Entropy), and numerical gradient checks.

4. **PyTorch (`torch.autograd`, `torch.nn`, `torch.optim`)**:
   - Ergonomic API conventions (`Module`, `Parameter`, `forward`, `backward`, `zero_grad`, `step`).

---

## 3. High-Level Architecture & Component Decomposition

```
numpygrad/
├── core/               # Computational Graph & Tensor Engine
│   ├── tensor.py       # Tensor data structure, DAG nodes, operator overloading
│   ├── ops.py          # Forward & analytical backward implementations
│   └── autograd.py     # Topological sorting, backward pass engine, graph traversal
├── nn/                 # Neural Network Abstractions & Layers
│   ├── module.py       # Base Module and Parameter abstractions
│   ├── layers.py       # Linear, BatchNorm1d, Dropout, Sequential, etc.
│   ├── activations.py  # ReLU, Sigmoid, Tanh, GELU, Softmax
│   └── losses.py       # MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
├── optim/              # First-Order Optimizers
│   ├── optimizer.py    # Optimizer base class
│   ├── sgd.py          # SGD with momentum and weight decay
│   └── adamw.py        # AdamW with decoupled weight decay and bias correction
└── utils/              # Diagnostics & Utilities
    ├── gradcheck.py    # Finite-difference numerical gradient checker
    ├── visualization.py# Graphviz / ASCII / Mermaid computational graph exporter
    └── data.py         # Dataset and DataLoader utilities
```

---

## 4. API Specification & Design Contracts

### 4.1 Tensor Engine (`numpygrad.core.Tensor`)
- **Properties**:
  - `data: np.ndarray`: Underlying float64/float32 storage.
  - `grad: Optional[np.ndarray]`: Accumulated analytical gradient (same shape as `data`).
  - `creator: Optional[Op]`: The operation node that generated this tensor in the DAG.
  - `_prev: Set[Tensor]`: Immediate antecedent tensors in the dynamic graph.
  - `requires_grad: bool`: Flag indicating whether the graph must track gradients for this node.
- **Methods**:
  - `backward(gradient=None)`: Constructs the topological order of DAG ancestors and executes reverse-mode backpropagation.
  - `zero_grad()`: Resets `grad` buffer to zeros or `None`.
  - `numpy() -> np.ndarray`: Returns raw NumPy representation.
  - Overloaded Operators: `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__matmul__`, `__pow__`, `__neg__`, and right-hand variants (`__radd__`, etc.).
  - Reduction & Shape Ops: `sum`, `mean`, `max`, `reshape`, `transpose`, `squeeze`, `unsqueeze`, `slice`.

### 4.2 Broadcasting-Aware Gradient Accumulation
- Binary operations between tensors of differing shapes undergo standard NumPy broadcasting.
- Analytical backward passes must project gradients back to original input shapes by summing out added/broadcasted dimensions (unbroadcasting / `sum_to_shape`).

### 4.3 Neural Network Modules (`numpygrad.nn`)
- **`Module`**:
  - `__call__(*args, **kwargs) -> Tensor`: Dispatches to `forward(*args, **kwargs)`.
  - `parameters() -> List[Parameter]`: Recursively collects all trainable parameters.
  - `zero_grad()`: Clears gradients for all parameters.
  - `train(mode=True)` / `eval()`: Switches training/inference modes (crucial for `Dropout` and `BatchNorm1d`).
- **Layers**:
  - `Linear(in_features, out_features, bias=True)`: He/Xavier initialized weights and zero biases.
  - `BatchNorm1d(num_features, eps=1e-5, momentum=0.1)`: Running mean/variance tracking for inference; batch-level statistics during training.
  - `Dropout(p=0.5)`: Inverted dropout scaling (`1 / (1 - p)` during training, identity during eval).
  - `Sequential(*layers)`: Linear container of cascaded modules.

### 4.4 Loss Functions (`numpygrad.nn.losses`)
- `MSELoss(reduction='mean')`: Mean squared error criterion.
- `CrossEntropyLoss(reduction='mean')`: Numerically stable categorical cross entropy combining LogSoftmax + NLLLoss via Log-Sum-Exp trick.
- `BCEWithLogitsLoss(reduction='mean')`: Numerically stable binary cross entropy with integrated sigmoid.

### 4.5 Optimizers (`numpygrad.optim`)
- **`SGD(params, lr=1e-3, momentum=0.9, weight_decay=0.0)`**:
  - Standard stochastic gradient descent with Polyak momentum velocity buffers and L2 weight decay.
- **`AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)`**:
  - Adaptive moment estimation with bias correction and decoupled weight decay.

---

## 5. Verification & Quality Standards

1. **Finite-Difference Gradient Verification (`gradcheck`)**:
   - Analytical gradients computed by `.backward()` must be validated against numerical central-difference approximations:
     $$\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}$$
   - Relative error metric:
     $$\text{rel\_error} = \frac{\|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}\|}{\max(\|\nabla_{\text{analytical}}\|, \|\nabla_{\text{numerical}}\|) + 10^{-15}} < 10^{-5}$$

2. **Numerical Stability Guarantee**:
   - No raw `exp` or direct quotient in softmax/cross-entropy that can cause `NaN` or overflow under large activations.
   - Strict clipping / log-sum-exp guards in log/sqrt/reciprocal operations.

3. **Deterministic Testing**:
   - Seeded random generators across unit tests for reproducibility.
