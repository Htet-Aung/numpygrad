# System & Development Rules: NumPyGrad

All contributors and AI agents working on **NumPyGrad** must strictly adhere to the non-negotiable architectural and engineering rules defined below.

---

## Rule 1: Absolute Dependency Ban on Deep Learning Frameworks
- **Zero DL Frameworks**: Under no circumstances may `torch`, `jax`, `tensorflow`, `keras`, `autograd`, or any external automatic differentiation library be imported or installed in the core source or test suites.
- **Allowed Libraries**:
  - `numpy` (the sole mathematical and array foundation).
  - Python standard library (`typing`, `math`, `collections`, `itertools`, `abc`, etc.).
  - `pytest` (test execution).
  - `matplotlib` / `graphviz` (optional utilities for visualization & plotting examples).

---

## Rule 2: Analytical Gradients & Mandatory `gradcheck` Verification
- **Analytical `_backward()`**: Every tensor operation, activation, layer, and loss function must implement an exact analytical gradient calculation.
- **Finite-Difference Numerical Check**:
  - Every operation must be tested using a finite-difference centered approximation:
    $$\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}, \quad \epsilon = 10^{-5} \text{ or } 10^{-6}$$
  - The relative gradient error must satisfy:
    $$\text{rel\_error} = \frac{\|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}\|}{\max(\|\nabla_{\text{analytical}}\|, \|\nabla_{\text{numerical}}\|) + 10^{-15}} < 10^{-5}$$
- **Zero Untested Ops**: No tensor operation or module is considered complete without an accompanying unit test running `gradcheck`.

---

## Rule 3: Strict Numerical Stability
- **Softmax & Cross-Entropy**: Must utilize the Log-Sum-Exp trick to prevent exponential overflow/underflow:
  $$\text{LSE}(x) = \max(x) + \log\left(\sum \exp(x - \max(x))\right)$$
- **Logarithms & Quotients**: Must guard against division-by-zero or $\log(0)$ with appropriate machine epsilon clamping ($\epsilon = 10^{-8}$ to $10^{-15}$).
- **Square Roots**: For operations involving norms or variance ($\sqrt{\sigma^2 + \epsilon}$), ensure epsilon is added before the radical.

---

## Rule 4: Dynamic DAG & Immutability Standards
- **Non-Destructive Graph Construction**: Forward operations must create new `Tensor` instances. In-place mutations of tensor buffers that belong to a computation graph must be prohibited or explicitly detached.
- **Broadcasting Awareness**: Backward implementations must properly reduce broadcasted dimensions back to the original operand shape via a unified unbroadcasting helper (`sum_to_shape`).
- **Gradient Accumulation**: When multiple downstream operations use the same tensor, gradients must accumulate additively (`grad += ...`), not overwrite (`grad = ...`).

---

## Rule 5: Task Progress & Documentation Protocol
- **Agentic Tracking**: Before marking any development milestone or task as complete, you **must update `docs/TASK_PROGRESS.md`** reflecting the exact status.
- **Status Markers**: Only use standard status indicators: `[TODO]`, `[IN PROGRESS]`, `[DONE]`.
- **Code Comments**: Every mathematical operation must feature docstrings describing the forward math formula and analytical derivative.
