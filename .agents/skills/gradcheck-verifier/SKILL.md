---
name: gradcheck-verifier
description: >-
  Automates centered finite-difference numerical gradient verification for new 
  NumPyGrad tensor operations, activation functions, loss criteria, and neural network layers.
---

# Gradcheck Verifier Skill

This skill defines the standard operating procedure (SOP) and diagnostic tooling for mathematically verifying analytical gradient implementations in **NumPyGrad** using two-sided centered finite differences.

---

## 1. Mathematical Foundation

Every differentiable primitive in NumPyGrad (operators, activations, loss criteria, and neural layers) must be validated against a central difference numerical approximation:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}$$

where $\epsilon = 10^{-5}$ (or $10^{-6}$) and $e_i$ is the standard basis unit vector for coordinate $i$.

### Error Metric Formulation
Relative error guards against scale distortion across small and large gradient values:

$$\text{rel\_error} = \frac{\|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}\|_2}{\|\nabla_{\text{analytical}}\|_2 + \|\nabla_{\text{numerical}}\|_2 + 10^{-8}} < 10^{-5}$$

Element-wise tolerance check:
$$\text{rel\_error}_i = \frac{|\nabla_{\text{analytical}, i} - \nabla_{\text{numerical}, i}|}{\max(|\nabla_{\text{analytical}, i}|, |\nabla_{\text{numerical}, i}|) + 10^{-15}} < 10^{-4}$$

---

## 2. Standard Operating Procedure (SOP)

When introducing a new differentiable operation, activation, layer, or loss function:

### Step 1: Forward Pass & Scalar Reduction
Wrap the operation in a function that outputs a scalar objective $L = \sum f(x_1, x_2, \dots, x_k)$:
```python
def f(x_t, w_t, b_t):
    out = x_t @ w_t + b_t
    return out.sum()
```

### Step 2: Analytical Backward Pass
Execute `.backward()` from the scalar reduction to populate `.grad` buffers on all leaf tensors where `requires_grad=True`:
```python
loss = f(x, w, b)
loss.backward()
grad_analytical = x.grad.copy()
```

### Step 3: Central Finite-Difference Evaluation
For every coordinate $x_i$ in each differentiable tensor:
1. Perturb coordinate positively: $x_i \leftarrow x_i + \epsilon$, compute $L^+ = f(x + \epsilon e_i)$.
2. Perturb coordinate negatively: $x_i \leftarrow x_i - \epsilon$, compute $L^- = f(x - \epsilon e_i)$.
3. Compute numerical slope: $\nabla_{\text{num}, i} = \frac{L^+ - L^-}{2\epsilon}$.

### Step 4: Verification & Assertions
1. Check max absolute difference: $\max |\nabla_{\text{analytical}} - \nabla_{\text{numerical}}| < 10^{-4}$.
2. Check relative error metric: $\text{rel\_error} < 10^{-5}$.
3. Guard against subgradient kinks (e.g. initialize inputs away from $0.0$ for `ReLU` / `max`).

### Step 5: Test Non-Contiguous & Broadcasted Shapes
Ensure your analytical `_backward()` unbroadcasts correctly:
- Broadcasted inputs: e.g. $(N, D) + (D,)$ or $(B, 1, D) \times (B, H, D)$.
- Slices and non-contiguous views: e.g. $x[1:3, :, 0:2]$.

---

## 3. Helper Script: `verify_grad.py`

Use the automated verification script located at [verify_grad.py](./scripts/verify_grad.py):

### CLI Usage:
```powershell
# Verify all registered primitives
$env:PYTHONPATH="src"
py -3.13 .agents/skills/gradcheck-verifier/scripts/verify_grad.py --layer all

# Verify specific layer or loss
py -3.13 .agents/skills/gradcheck-verifier/scripts/verify_grad.py --layer linear
py -3.13 .agents/skills/gradcheck-verifier/scripts/verify_grad.py --layer cross_entropy
py -3.13 .agents/skills/gradcheck-verifier/scripts/verify_grad.py --layer batchnorm
```

### Python API Usage:
```python
from verify_grad import run_gradcheck_diagnostic
from numpygrad.nn.layers import Linear
from numpygrad.core.tensor import Tensor
import numpy as np

layer = Linear(4, 2)
x = Tensor(np.random.randn(5, 4), requires_grad=True)

results = run_gradcheck_diagnostic(
    func=lambda x_t, w_t, b_t: (x_t @ w_t + b_t).sum(),
    inputs=[x, layer.weight, layer.bias],
    names=["input_x", "weight", "bias"]
)
```
