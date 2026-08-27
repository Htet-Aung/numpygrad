# Product Requirements Document (PRD): NumPyGrad

## 1. Executive Summary & Objective

**NumPyGrad** is a lightweight, educational Python and NumPy automatic differentiation and deep learning framework built from scratch with zero external deep learning framework dependencies (no PyTorch, JAX, TensorFlow, or Keras). 

The primary objective is to create an open, readable, and mathematically rigorous tensor library to explore:
- A dynamic computation graph (DAG) evaluated via reverse-mode automatic differentiation (topological backpropagation).
- Broadcasting-aware analytical gradient accumulation across arbitrary tensor shapes and dimensions (`_unbroadcast`).
- A PyTorch-inspired object-oriented API for neural network modules, parameters, layers, spatial convolutions, activation functions, loss criteria, and optimizers.
- Numerical gradient verification (`gradcheck`) using centered finite differences to guarantee mathematical precision.
- Single-file container model serialization (`.ng` format) packaging topology and compressed weight arrays.
- An interactive multi-tab web studio for visual exploration of 2D decision boundaries and live handwritten digit recognition.

### 1.1 Scope & Deliberate Non-Goals
- **Educational Exploration First:** The project exists to demystify autograd and backpropagation by keeping every tensor calculation and closure in plain, readable NumPy.
- **No GPU / CUDA Hardware Acceleration:** Execution is strictly CPU-bound to avoid platform-dependent CUDA complexity.
- **No Distributed Scaling:** Built for single-machine conceptual understanding and experimentation.
- **Not a Production Framework Replacement:** Designed for inspection, learning, and teaching rather than industrial-scale training.

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
   - Fast spatial convolutions via `im2col` unfolding and `col2im` gradient scattering.

4. **PyTorch (`torch.autograd`, `torch.nn`, `torch.optim`)**:
   - Ergonomic API conventions (`Module`, `Parameter`, `forward`, `backward`, `zero_grad`, `step`, `state_dict`, `load_state_dict`, `no_grad`).

---

## 3. High-Level Architecture & Component Decomposition

```
numpygrad/
├── core/               # Computational Graph & Tensor Engine
│   ├── tensor.py       # Tensor data structure, DAG nodes, operator overloading, no_grad
│   ├── ops.py          # Forward & analytical backward implementations, _unbroadcast
│   └── autograd.py     # Topological sorting, backward pass engine, graph traversal
├── nn/                 # Neural Network Abstractions & Layers
│   ├── module.py       # Base Module and Parameter abstractions, state_dict
│   ├── layers.py       # Linear, BatchNorm1d, Dropout, Flatten, Sequential
│   ├── convolution.py  # Conv2D (im2col), MaxPool2D (argmax routing)
│   ├── activations.py  # ReLU, Sigmoid, Tanh, GELU, Softmax
│   └── losses.py       # MSELoss, CrossEntropyLoss (Log-Sum-Exp), BCEWithLogitsLoss
├── optim/              # First-Order Optimizers
│   ├── optimizer.py    # Optimizer base class
│   ├── sgd.py          # SGD with Polyak momentum and L2 weight decay
│   └── adamw.py        # AdamW with decoupled weight decay and bias correction
├── metrics/            # Classification & Evaluation Metrics
│   └── classification.py # Accuracy, confusion matrix calculation
├── utils/              # Diagnostics & Utilities
│   ├── gradcheck.py    # Finite-difference numerical gradient checker
│   ├── summary.py      # model.summary() ASCII table, parameter counting, memory estimates
│   └── data.py         # Dataset, TensorDataset, DataLoader mini-batch pipeline
└── serialization.py    # .ng zip container serialization (save_model, load_model)
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
- **Methods & Operators**:
  - `backward(gradient=None)`: Constructs the topological order of DAG ancestors and executes reverse-mode backpropagation.
  - `zero_grad()`: Resets `grad` buffer to zeros or `None`.
  - `numpy() -> np.ndarray`: Returns raw NumPy representation.
  - Overloaded Operators: `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__matmul__`, `__pow__`, `__neg__`, `__getitem__`, and right-hand variants.
  - Shape & Reduction Ops: `sum`, `mean`, `reshape`, `transpose`, `T`, `squeeze`, `unsqueeze`, `concat`.
  - Autograd Controls: `no_grad()`, `enable_grad()`, `is_grad_enabled()`, `set_grad_enabled()`.

### 4.2 Broadcasting-Aware Gradient Accumulation (`_unbroadcast`)
- Binary operations between tensors of differing shapes undergo standard NumPy broadcasting.
- Analytical backward passes project gradients back to original input shapes by summing out added leading and singleton dimensions (`_unbroadcast` / `sum_to_shape`).

### 4.3 Neural Network Modules (`numpygrad.nn`)
- **`Module`**:
  - `__call__(*args, **kwargs) -> Tensor`: Dispatches to `forward(*args, **kwargs)`.
  - `parameters() -> List[Parameter]`: Recursively collects all trainable parameters.
  - `buffers() -> List[np.ndarray]`: Recursively collects non-trainable state (e.g. running stats).
  - `state_dict()` / `load_state_dict()`: Exports and restores module weights and buffers.
  - `save(filepath)`: Direct single-file persistence to `.ng` container format.
  - `summary(input_shape=None)`: Formatted ASCII table with shapes, param counts, and memory footprint.
  - `train(mode=True)` / `eval()`: Switches training/inference modes.
- **Layers**:
  - `Linear(in_features, out_features, bias=True)`: He/Xavier initialized weights and zero biases.
  - `Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)`: Vectorized 2D convolution via `im2col` unfolding and `col2im` gradient scattering.
  - `MaxPool2D(kernel_size=2, stride=2)`: 2D spatial downsampling with argmax gradient routing.
  - `BatchNorm1d(num_features, eps=1e-5, momentum=0.1)`: Running mean/variance tracking for inference; batch statistics during training.
  - `Dropout(p=0.5)`: Inverted dropout scaling (`1 / (1 - p)` during training, identity during eval).
  - `Flatten()`: Flattens multi-dimensional tensor activations to `(N, -1)`.
  - `Sequential(*layers)`: Container for linear chaining of modules.

### 4.4 Loss Functions (`numpygrad.nn.losses`)
- `MSELoss(reduction='mean')`: Mean squared error criterion.
- `CrossEntropyLoss(reduction='mean')`: Numerically stable categorical cross entropy combining LogSoftmax + NLLLoss via Log-Sum-Exp trick.
- `BCEWithLogitsLoss(reduction='mean')`: Numerically stable binary cross entropy with integrated sigmoid.

### 4.5 Optimizers (`numpygrad.optim`)
- **`SGD(params, lr=1e-3, momentum=0.9, weight_decay=0.0)`**:
  - Stochastic gradient descent with Polyak momentum velocity buffers and L2 weight decay.
- **`AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)`**:
  - Adaptive moment estimation with bias correction and decoupled weight decay.

### 4.6 Model Serialization (`.ng` Container)
- Single-file ZIP container format storing:
  - `architecture.json`: Layer types, hyperparameter configurations, and topology layout.
  - `weights.npz`: Compressed parameter tensors and module buffers.
- Functions: `save_model(model, filepath)` and `load_model(filepath)`.

### 4.7 Interactive Studio (`app/app.py`)
- **Tab 1 (2D Decision Boundaries):**
  - **Single Model Studio:** Interactive 2D dataset generator (Two Moons, Circles, Spirals), real-time contour plot updates, live loss/accuracy tracking, "Model Ready" status card, interactive coordinate testing with direct **Click-to-Predict** on Plotly decision contours under `with no_grad():` with gold star marker overlay, and expandable **Engine Internals & Computation Trace** drawer (step-by-step activations, Log-Sum-Exp trick breakdown, and parameter diagnostics).
  - **Model Capacity Comparison (A vs B):** Side-by-side comparative laboratory evaluating two distinct architectures on identical mini-batches, comparing decision boundary complexity, parameter counts, and synchronized dual click-to-predict telemetry without text truncation.
- **Tab 2 (Handwritten Digit Recognition):** Interactive drawing canvas, Brush Width slider, native Undo/Redo/Clear controls, MNIST center-of-mass preprocessing, live Top-3 class predictions, and probability bar chart.

---

## 5. Verification & Quality Standards

1. **Finite-Difference Gradient Verification (`gradcheck`)**:
   - Analytical gradients computed by `.backward()` are strictly validated against numerical central-difference approximations:
     $$\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}$$
   - Relative error metric:
     $$\text{rel\_error} = \frac{\|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}\|}{\max(\|\nabla_{\text{analytical}}\|, \|\nabla_{\text{numerical}}\|) + 10^{-15}} < 10^{-5}$$

2. **Numerical Stability Guarantee**:
   - Log-Sum-Exp reduction for all Softmax and Cross-Entropy operations to prevent exponential overflow/underflow.
   - Numerically stable sigmoid formulation with strict gradient clipping.

3. **100% Test Coverage & Determinism**:
   - Complete automated test suite covering autograd, operators, layers, serialization, data loading, gradient telemetry, multi-digit segmentation, and neural pathfinding simulation (**140/140 passing**).
